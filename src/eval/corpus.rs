//! Corpus perplexity, in the units that are actually comparable across models.
//!
//! The one rule this module exists to enforce: **per-token perplexity is never
//! the headline.** It is tokenizer-dependent -- a smaller vocabulary lowers it
//! mechanically, by giving the model fewer choices per step and more steps per
//! word -- so quark's per-token PPL against GPT-2's would be meaningless and
//! flattering in our favour. See `docs/DESIGN.md` §3.
//!
//! The two numbers that *are* comparable:
//!
//! * **Word-level perplexity**, `exp(total_NLL / n_words)`. The sum runs over
//!   subword tokens; the divisor counts whitespace-delimited words. This is
//!   GPT-2's own protocol (Radford et al.: "computing the log-probability of a
//!   dataset ... and dividing by the number of canonical units") -- GPT-2's
//!   50257-entry BPE is not word-level either, and faces the same problem.
//! * **Bits per byte**, `total_NLL / (n_bytes * ln 2)`, tokenizer-independent by
//!   construction. This is only meaningful because the tokenizer is byte-level
//!   and reversible (see `crate::data::tokenizer`): every byte of the source is
//!   accounted for by exactly one token sequence.

use std::sync::Arc;

use anyhow::{bail, Result};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::Backend,
    tensor::ElementConversion,
};

use crate::{
    data::{Shard, TokenBatcher, TokenDataset},
    model::QuarkLm,
    train::output::masked_cross_entropy,
};

/// How to sweep the corpus.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Context length. Capped by the model's `max_seq_len`.
    pub seq_len: usize,
    /// How far the window advances between scored blocks. Must not exceed
    /// `seq_len`.
    ///
    /// Striding is not a detail. With `stride == seq_len` the first token of
    /// every window is predicted from no context at all and the second from one
    /// token, which inflates perplexity by an amount depending on `seq_len` --
    /// a choice, not a property of the model. `seq_len / 2` scores every token
    /// with at least `seq_len / 2` tokens of context, at 2x the compute.
    pub stride: usize,
    pub batch_size: usize,
    pub num_workers: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            stride: 256,
            batch_size: 8,
            num_workers: 2,
        }
    }
}

impl EvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.seq_len == 0 {
            bail!("seq_len must be positive");
        }
        if self.stride == 0 || self.stride > self.seq_len {
            bail!(
                "stride {} must be in 1..={} -- a stride wider than the window \
                 would leave tokens between windows unscored",
                self.stride,
                self.seq_len
            );
        }
        if self.batch_size == 0 {
            bail!("batch_size must be positive");
        }
        Ok(())
    }
}

/// What a sweep measured. Deliberately raw: the totals, not the ratios, so that
/// every reported figure is derivable here rather than accumulated in whatever
/// unit someone happened to want.
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusScore {
    /// Negative log-likelihood in nats, summed over every scored token.
    pub total_nll: f64,
    /// How many tokens that sum covers.
    pub n_scored_tokens: usize,
    /// How many tokens the shard holds.
    pub n_tokens: usize,
    /// Whitespace-delimited words in the source text.
    pub n_words: usize,
    /// UTF-8 bytes of the source text.
    pub n_bytes: usize,
}

impl CorpusScore {
    /// Fraction of the scorable tokens the sweep actually scored.
    ///
    /// Not 1.0 in general, and the shortfall is worth naming because it biases
    /// the result *downward* -- i.e. in our favour. Two causes, both structural:
    ///
    /// * Token 0 is never a target; nothing precedes it to predict it from.
    ///   Hence the `- 1`.
    /// * The final partial window is dropped rather than padded, because padding
    ///   would put tokens in the loss that the corpus does not contain. That
    ///   loses fewer than `seq_len` tokens.
    ///
    /// On WikiText-103 (~10^8 tokens) that is a relative shortfall of ~10^-5,
    /// which moves word PPL by far less than its last printed digit. On a toy
    /// shard it can be large, which is exactly when you want to see it.
    pub fn coverage(&self) -> f64 {
        if self.n_tokens <= 1 {
            return 0.0;
        }
        self.n_scored_tokens as f64 / (self.n_tokens - 1) as f64
    }

    /// **Tokenizer-dependent. Never compare this across models.** Reported only
    /// because it is the number training prints, so seeing it here closes the
    /// loop between the two.
    pub fn token_ppl(&self) -> f64 {
        if self.n_scored_tokens == 0 {
            return f64::INFINITY;
        }
        (self.total_nll / self.n_scored_tokens as f64).exp()
    }

    /// The headline. Comparable to GPT-2's published 37.50 in protocol, though
    /// see `docs/DESIGN.md` §3.1 on the de-tokenizer caveat.
    pub fn word_ppl(&self) -> f64 {
        if self.n_words == 0 {
            return f64::INFINITY;
        }
        (self.total_nll / self.n_words as f64).exp()
    }

    /// Tokenizer-independent by construction.
    pub fn bits_per_byte(&self) -> f64 {
        if self.n_bytes == 0 {
            return f64::INFINITY;
        }
        self.total_nll / (self.n_bytes as f64 * std::f64::consts::LN_2)
    }

    pub fn report(&self) -> String {
        format!(
            "word perplexity      {:>12.3}   <- the comparable number\n\
             bits per byte        {:>12.4}   <- also comparable\n\
             token perplexity     {:>12.3}   (tokenizer-dependent; do not compare)\n\
             total NLL (nats)     {:>12.1}\n\
             scored tokens        {:>12}   ({:.4}% of the corpus)\n\
             words                {:>12}\n\
             bytes                {:>12}",
            self.word_ppl(),
            self.bits_per_byte(),
            self.token_ppl(),
            self.total_nll,
            self.n_scored_tokens,
            self.coverage() * 100.0,
            self.n_words,
            self.n_bytes,
        )
    }
}

/// Sweep `shard` and accumulate its negative log-likelihood under `model`.
///
/// The NLL is summed on the host in `f64`. Per batch it is an `f32` reduction on
/// the device over at most `batch * seq` terms, which is fine; across 10^8
/// tokens it would not be, because `f32` stops being able to add 1 to a running
/// total somewhere around 10^7.
pub fn evaluate<B: Backend>(
    model: &QuarkLm<B>,
    shard: Arc<Shard>,
    config: &EvalConfig,
    device: &B::Device,
) -> Result<CorpusScore> {
    config.validate()?;

    let meta = shard.meta().clone();
    if meta.vocab_size != model.config().vocab_size {
        bail!(
            "shard vocab {} != model vocab {}: this shard was tokenized with a \
             different tokenizer, and its ids mean nothing to this model",
            meta.vocab_size,
            model.config().vocab_size
        );
    }
    if config.seq_len > model.config().max_seq_len {
        bail!(
            "seq_len {} exceeds the model's max_seq_len {}",
            config.seq_len,
            model.config().max_seq_len
        );
    }

    let dataset = TokenDataset::with_stride(shard, config.seq_len, config.stride);
    if dataset.is_empty() {
        bail!(
            "shard holds {} tokens, too few for even one {}-token window",
            meta.n_tokens,
            config.seq_len
        );
    }
    let expected_scored = dataset.n_scored_tokens();

    let loader = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(dataset);

    let mut total_nll = 0.0f64;
    let mut n_scored_tokens = 0usize;
    for batch in loader.iter() {
        let out = masked_cross_entropy(model.forward(batch.input), batch.target, batch.score_mask);
        total_nll += out.sum_nll.into_scalar().elem::<f64>();
        n_scored_tokens += out.n_tokens.into_scalar().elem::<f64>() as usize;
    }

    // The mask is built from `score_from`, and `n_scored_tokens` is computed
    // from the same offsets by a different route. If they disagree, either the
    // batcher or the window layout is wrong, and every number below would be
    // quietly off. Cheap to check, impossible to notice otherwise.
    if n_scored_tokens != expected_scored {
        bail!(
            "scored {n_scored_tokens} tokens but the window layout predicts \
             {expected_scored}: the mask and the dataset disagree"
        );
    }

    Ok(CorpusScore {
        total_nll,
        n_scored_tokens,
        n_tokens: meta.n_tokens,
        n_words: meta.n_words,
        n_bytes: meta.n_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::ShardWriter, test_util::TestBackend, ModelConfig};
    use std::path::Path;

    fn shard(dir: &Path, n_tokens: usize, vocab: usize) -> Arc<Shard> {
        let bin = dir.join("e.bin");
        let mut w = ShardWriter::create(&bin, vocab, 0).unwrap();
        // A deterministic walk, so the score is reproducible run to run.
        let tokens: Vec<u32> = (0..n_tokens as u32 - 1)
            .map(|i| (i.wrapping_mul(2654435761) >> 16) % vocab as u32)
            .collect();
        w.push_document("one two three four five", &tokens).unwrap();
        w.finish().unwrap();
        Arc::new(Shard::open(&bin).unwrap())
    }

    fn cfg() -> EvalConfig {
        EvalConfig {
            seq_len: 16,
            stride: 8,
            batch_size: 2,
            num_workers: 1,
        }
    }

    /// The three ratios must be exactly the three definitions, computed from the
    /// same NLL. This is the test that would catch someone "fixing" word PPL to
    /// divide by tokens.
    #[test]
    fn the_three_units_are_the_three_definitions() {
        let s = CorpusScore {
            total_nll: 1000.0,
            n_scored_tokens: 500,
            n_tokens: 501,
            n_words: 250,
            n_bytes: 1000,
        };
        assert!((s.token_ppl() - 2.0f64.exp()).abs() < 1e-9);
        // Twice the nats per unit, because a word is two tokens here.
        assert!((s.word_ppl() - 4.0f64.exp()).abs() < 1e-9);
        // 1000 nats over 1000 bytes = 1 nat/byte = 1/ln2 bits.
        assert!((s.bits_per_byte() - 1.0 / std::f64::consts::LN_2).abs() < 1e-9);
        assert!((s.coverage() - 1.0).abs() < 1e-9);
    }

    /// A uniform model over `V` tokens costs `ln V` per token, so its word PPL
    /// is `V^(tokens/words)`. Pinning the arithmetic end to end through a real
    /// forward pass -- not just the struct -- is what makes the reported number
    /// trustworthy.
    #[test]
    fn an_untrained_model_scores_near_the_uniform_bound() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let model_cfg = ModelConfig::tiny();
        let model = QuarkLm::<TestBackend>::new(model_cfg.clone(), &device);
        let s = shard(dir.path(), 200, model_cfg.vocab_size);

        let score = evaluate(&model, s, &cfg(), &device).unwrap();

        // Freshly initialized, the model is near-uniform (see
        // `model::lm::a_fresh_model_predicts_near_uniformly`), so every token
        // costs about ln V.
        let per_token = score.total_nll / score.n_scored_tokens as f64;
        let uniform = (model_cfg.vocab_size as f64).ln();
        assert!(
            (per_token - uniform).abs() < 0.5,
            "per-token NLL {per_token} should sit near ln(V) = {uniform}"
        );
        // ...and the reported word PPL must follow from that same NLL.
        let expected_word_ppl = (score.total_nll / score.n_words as f64).exp();
        assert!((score.word_ppl() - expected_word_ppl).abs() < 1e-6);
    }

    /// Striding must not change what is counted -- only how much context each
    /// token gets. If the two strides disagreed on the token count, one of them
    /// would be double-counting or skipping, and the perplexities would not be
    /// comparable to each other, let alone to GPT-2's.
    #[test]
    fn stride_changes_context_not_bookkeeping() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let model_cfg = ModelConfig::tiny();
        let model = QuarkLm::<TestBackend>::new(model_cfg.clone(), &device);

        let disjoint = evaluate(
            &model,
            shard(dir.path(), 200, model_cfg.vocab_size),
            &EvalConfig {
                stride: 16,
                ..cfg()
            },
            &device,
        )
        .unwrap();
        let strided = evaluate(
            &model,
            shard(dir.path(), 200, model_cfg.vocab_size),
            &EvalConfig { stride: 8, ..cfg() },
            &device,
        )
        .unwrap();

        assert_eq!(disjoint.n_words, strided.n_words);
        assert_eq!(disjoint.n_bytes, strided.n_bytes);
        // Both sweeps score every token they can reach, each exactly once. The
        // strided one reaches slightly more of the tail, so it is >= rather
        // than ==; what matters is that neither exceeds the corpus.
        assert!(disjoint.n_scored_tokens <= strided.n_scored_tokens);
        assert!(strided.n_scored_tokens < strided.n_tokens);
        assert!(strided.coverage() <= 1.0);
    }

    /// The whole point of recording `vocab_size` in the shard. A shard from
    /// another tokenizer has ids that mean nothing to this model, and would
    /// produce a plausible-looking perplexity rather than an error.
    #[test]
    fn a_shard_from_another_tokenizer_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let model = QuarkLm::<TestBackend>::new(ModelConfig::tiny(), &device);
        let s = shard(dir.path(), 200, 512);

        let err = evaluate(&model, s, &cfg(), &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different tokenizer"), "got: {err}");
    }

    /// The protocol is shared with `experiments/gpt2_baseline.py`, which has to
    /// reimplement it in Python against a model we cannot run in burn. Both
    /// assert against `experiments/protocol_fixture.json`, so a drift in either
    /// fails here or there rather than turning up as a perplexity difference and
    /// being blamed on the model. See `docs/DESIGN.md` §3.1.
    #[test]
    fn the_window_layout_matches_the_frozen_protocol() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("experiments/protocol_fixture.json").unwrap(),
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();

        for case in fixture["window_layout"]["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let n_tokens = case["n_tokens"].as_u64().unwrap() as usize;
            let seq_len = case["seq_len"].as_u64().unwrap() as usize;
            let stride = case["stride"].as_u64().unwrap() as usize;
            let expected = case["windows"].as_array().unwrap();

            let s = shard(dir.path(), n_tokens, 64);
            let ds = TokenDataset::with_stride(s.clone(), seq_len, stride);

            assert_eq!(ds.len(), expected.len(), "{name}: window count");
            for (i, want) in expected.iter().enumerate() {
                let window = ds.get(i).unwrap();
                let start = want["start"].as_u64().unwrap() as usize;
                let score_from = want["score_from"].as_u64().unwrap() as usize;
                assert_eq!(
                    window.score_from, score_from,
                    "{name}: window {i} score_from"
                );
                // `start` is private, so check it through what it decides: the
                // window's tokens are the corpus's, from `start`.
                assert_eq!(
                    window.input,
                    s.tokens(start..start + seq_len),
                    "{name}: window {i} start"
                );
            }
            assert_eq!(
                ds.n_scored_tokens(),
                case["n_scored"].as_u64().unwrap() as usize,
                "{name}: scored token count"
            );
        }
    }

    /// The denominators are a property of the text, not of any model, so both
    /// implementations must count them identically or word perplexity compares
    /// nothing.
    #[test]
    fn the_denominators_match_the_frozen_protocol() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("experiments/protocol_fixture.json").unwrap(),
        )
        .unwrap();

        for case in fixture["denominators"]["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(
                crate::data::count_words(text),
                case["n_words"].as_u64().unwrap() as usize,
                "words in {text:?}"
            );
            assert_eq!(
                text.len(),
                case["n_bytes"].as_u64().unwrap() as usize,
                "bytes in {text:?}"
            );
        }
    }

    /// And the three formulas, against values computed independently of this
    /// code.
    #[test]
    fn the_formulas_match_the_frozen_protocol() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("experiments/protocol_fixture.json").unwrap(),
        )
        .unwrap();

        for case in fixture["formulas"]["cases"].as_array().unwrap() {
            let num = |k: &str| case[k].as_f64().unwrap();
            let s = CorpusScore {
                total_nll: num("total_nll"),
                n_scored_tokens: num("n_scored_tokens") as usize,
                n_tokens: num("n_scored_tokens") as usize + 1,
                n_words: num("n_words") as usize,
                n_bytes: num("n_bytes") as usize,
            };
            assert!((s.word_ppl() - num("word_ppl")).abs() < 1e-9);
            assert!((s.bits_per_byte() - num("bits_per_byte")).abs() < 1e-9);
            assert!((s.token_ppl() - num("token_ppl")).abs() < 1e-9);
        }
    }

    #[test]
    fn a_stride_wider_than_the_window_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let device = Default::default();
        let model = QuarkLm::<TestBackend>::new(ModelConfig::tiny(), &device);
        let s = shard(dir.path(), 200, ModelConfig::tiny().vocab_size);

        let err = evaluate(
            &model,
            s,
            &EvalConfig {
                stride: 32,
                ..cfg()
            },
            &device,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unscored"), "got: {err}");
    }
}
