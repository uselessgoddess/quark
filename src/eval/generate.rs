//! The fixed generation evaluation: a frozen prompt set, decoded reproducibly.
//!
//! Neither perplexity nor BLiMP can tell you that the model emits `the the the`
//! forever, or that it has memorised one Wikipedia article and reproduces it
//! regardless of the prompt. Both are cheap to notice by reading twenty samples,
//! and impossible to notice from a scalar. That is the entire justification for
//! this module: it exists to be read by a human.
//!
//! Two properties make it an *evaluation* rather than a demo:
//!
//! * **The prompts are frozen** ([`PROMPTS`]). Samples from two checkpoints are
//!   comparable because the input did not move between them. A prompt set that
//!   drifts to flatter the current model measures nothing.
//! * **Decoding is reproducible.** Greedy is deterministic outright; sampling
//!   runs on a seeded ChaCha8 stream, which is reproducible across platforms and
//!   across backends, unlike the GPU's own RNG.

use anyhow::{bail, Result};
use burn::{
    prelude::Backend,
    tensor::{activation::softmax, ElementConversion, Int, Tensor, TensorData},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tokenizers::Tokenizer;

use crate::{data::tokenizer, model::QuarkLm};

/// The frozen prompt set.
///
/// Chosen to separate failure modes rather than to look good. In order: an
/// encyclopaedic opener in WikiText's own register (the model's best case, and
/// if this fails nothing else matters); a factual continuation that needs more
/// than syntax; long-range subject-verb agreement across an intervening clause;
/// a list, where a degenerate model's repetition shows up immediately; a
/// syntactically demanding subordinate clause; and an out-of-register prompt,
/// because a model trained only on Wikipedia should visibly *not* know how to
/// continue dialogue -- if it does, the eval shard is contaminated.
pub const PROMPTS: &[&str] = &[
    "The history of the city begins",
    "In 1994 , the team won",
    "The scientist who wrote the papers about the migration of birds",
    "The three largest islands are",
    "Although the bridge had been rebuilt twice , it",
    "\" I don 't think so , \" she said",
];

/// How to decode. The defaults are the reference settings: greedy, because a
/// deterministic sample is the one worth committing to a report.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    /// `0.0` means greedy (argmax). Above that, logits are divided by it before
    /// sampling: below 1 sharpens, above 1 flattens.
    pub temperature: f64,
    /// Sample only from the `k` most likely tokens. `0` disables the cutoff.
    pub top_k: usize,
    /// Nucleus sampling: keep the most likely tokens up to `p` cumulative
    /// probability. `1.0` disables it. Applied after `top_k`.
    pub top_p: f64,
    /// Seeds the sampler. Ignored when greedy, which has nothing to seed.
    pub seed: u64,
    /// Stop when the model emits the document separator, rather than decoding
    /// through it. A model that never emits one is itself worth seeing, so the
    /// sample records why it stopped.
    pub stop_at_eos: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            seed: 42,
            stop_at_eos: true,
        }
    }
}

/// Why decoding stopped. Worth recording: "hit the token limit" and "emitted a
/// separator after four words" say very different things about a model, and both
/// look like a short completion in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted the document separator.
    Eos,
    /// `max_new_tokens` reached.
    Limit,
    /// The context filled up.
    ContextFull,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub prompt: String,
    pub completion: String,
    pub n_new_tokens: usize,
    pub stop_reason: StopReason,
}

/// Continue `prompt` under `config`.
///
/// The prompt is prefixed with the document separator, because that is the
/// context the model was trained in: [`crate::data::shard::ShardWriter`] writes
/// EOS *after* each document, so in the training stream every document's first
/// token is preceded by one. Starting a generation without it would put the
/// model somewhere its training data never went.
///
/// Decoding is incremental via the KV cache -- one cache per layer application,
/// since a shared layer sees different activations on each loop and their keys
/// cannot be pooled. That makes the cost linear in the length rather than
/// quadratic.
pub fn generate<B: Backend>(
    model: &QuarkLm<B>,
    tok: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    device: &B::Device,
) -> Result<Sample> {
    if config.temperature < 0.0 {
        bail!("temperature {} must not be negative", config.temperature);
    }
    if !(0.0..=1.0).contains(&config.top_p) {
        bail!("top_p {} must be in 0..=1", config.top_p);
    }

    let eos = tokenizer::eos_id(tok)?;
    let max_seq_len = model.config().max_seq_len;

    let mut context = vec![eos];
    context.extend(tokenizer::encode(tok, prompt)?);
    if context.len() >= max_seq_len {
        bail!(
            "the prompt is {} tokens and the model's context is {max_seq_len}: there is no room \
             to generate",
            context.len()
        );
    }

    let mut caches = model.new_caches();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut generated = Vec::new();

    // Prefill: the whole prompt in one pass, filling the caches. Only the last
    // position's logits matter -- the rest predict tokens we already have.
    let mut logits = last_logits(model, &context, &mut caches, device);

    let stop_reason = loop {
        if generated.len() == config.max_new_tokens {
            break StopReason::Limit;
        }
        let next = pick(&logits, config, &mut rng);
        if next == eos && config.stop_at_eos {
            break StopReason::Eos;
        }
        generated.push(next);
        if context.len() + generated.len() >= max_seq_len {
            break StopReason::ContextFull;
        }
        logits = last_logits(model, &[next], &mut caches, device);
    };

    Ok(Sample {
        prompt: prompt.to_string(),
        completion: tokenizer::decode(tok, &generated)?,
        n_new_tokens: generated.len(),
        stop_reason,
    })
}

/// Feed `tokens` and read back the logits at the final position.
fn last_logits<B: Backend>(
    model: &QuarkLm<B>,
    tokens: &[u32],
    caches: &mut [crate::model::KvCache<B>],
    device: &B::Device,
) -> Vec<f32> {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let n = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, n]), device);

    let logits = model.forward_cached(input, caches);
    let [_, seq, vocab] = logits.dims();
    logits
        .slice([0..1, seq - 1..seq, 0..vocab])
        .reshape([vocab])
        .into_data()
        .iter::<f32>()
        .collect()
}

/// Choose the next token.
///
/// On the host: the vocabulary is 8192 entries, so a sort costs microseconds
/// against a forward pass's milliseconds, and top-p on the device would be a
/// cumulative sum and a masked renormalization for no gain. Keeping it here also
/// keeps the RNG off the backend, which is what makes the sample reproducible
/// between wgpu and ndarray.
fn pick(logits: &[f32], config: &GenerationConfig, rng: &mut ChaCha8Rng) -> u32 {
    if config.temperature == 0.0 {
        return argmax(logits);
    }

    // Descending by probability, which both cutoffs are defined in terms of.
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));

    let k = if config.top_k == 0 {
        order.len()
    } else {
        config.top_k.min(order.len())
    };
    let order = &order[..k];

    // softmax over the survivors, with the max subtracted for the usual reason.
    let max = logits[order[0] as usize] as f64;
    let scaled: Vec<f64> = order
        .iter()
        .map(|&i| ((logits[i as usize] as f64 - max) / config.temperature).exp())
        .collect();
    let total: f64 = scaled.iter().sum();

    // Nucleus: the shortest prefix whose probability reaches top_p. The
    // `<` before the accumulate, not after, is what keeps the token that crosses
    // the threshold inside the set -- so top_p can never yield an empty one.
    let mut cutoff = order.len();
    if config.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, s) in scaled.iter().enumerate() {
            acc += s / total;
            if acc >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
    }

    let kept: f64 = scaled[..cutoff].iter().sum();
    let mut target = rng.random::<f64>() * kept;
    for (i, s) in scaled[..cutoff].iter().enumerate() {
        target -= s;
        if target <= 0.0 {
            return order[i];
        }
    }
    // Only reachable through floating-point drift in the accumulation above.
    order[cutoff - 1]
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Decode every prompt in [`PROMPTS`].
pub fn run_suite<B: Backend>(
    model: &QuarkLm<B>,
    tok: &Tokenizer,
    config: &GenerationConfig,
    device: &B::Device,
) -> Result<Vec<Sample>> {
    PROMPTS
        .iter()
        .map(|p| generate(model, tok, p, config, device))
        .collect()
}

pub fn report(samples: &[Sample]) -> String {
    let mut s = String::new();
    for sample in samples {
        s += &format!(
            "> {}\n  {}\n  [{} tokens, {:?}]\n\n",
            sample.prompt, sample.completion, sample.n_new_tokens, sample.stop_reason
        );
    }
    s
}

/// The next-token distribution at the end of `tokens`, on the host. Exposed for
/// tests and for anyone wanting to inspect what the model actually believes
/// rather than what it sampled.
pub fn next_token_probs<B: Backend>(
    model: &QuarkLm<B>,
    tokens: &[u32],
    device: &B::Device,
) -> Vec<f32> {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let n = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, n]), device);
    let logits = model.forward(input);
    let [_, seq, vocab] = logits.dims();
    softmax(
        logits
            .slice([0..1, seq - 1..seq, 0..vocab])
            .reshape([vocab]),
        0,
    )
    .into_data()
    .iter::<f32>()
    .map(|v| v.elem())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::tokenizer::EOS_TOKEN, test_util::TestBackend, ModelConfig};

    /// A tokenizer over a tiny corpus, so the tests need no fixture on disk.
    fn toy_tokenizer(dir: &std::path::Path) -> Tokenizer {
        use std::io::Write;
        let text = dir.join("t.txt");
        let mut f = std::fs::File::create(&text).unwrap();
        for _ in 0..200 {
            writeln!(f, "the city begins with a bridge and three islands here").unwrap();
        }
        drop(f);
        tokenizer::train(
            &[text.to_str().unwrap().to_string()],
            256,
            &dir.join("tok.json"),
        )
        .unwrap()
    }

    fn model(vocab: usize) -> (QuarkLm<TestBackend>, burn::prelude::Device<TestBackend>) {
        let device = Default::default();
        let cfg = ModelConfig {
            vocab_size: vocab,
            max_seq_len: 64,
            ..ModelConfig::tiny()
        };
        (QuarkLm::new(cfg, &device), device)
    }

    /// Greedy has no RNG, so two runs must agree exactly. If they ever did not,
    /// every sample in every report would be unreproducible and the eval would
    /// be worthless as a comparison between checkpoints.
    #[test]
    fn greedy_decoding_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let tok = toy_tokenizer(dir.path());
        let (m, device) = model(tok.get_vocab_size(true));
        let cfg = GenerationConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            ..Default::default()
        };

        let a = generate(&m, &tok, "the city", &cfg, &device).unwrap();
        let b = generate(&m, &tok, "the city", &cfg, &device).unwrap();

        assert_eq!(a, b);
    }

    /// Sampling is reproducible from its seed, and actually depends on it.
    /// Half of that is the guarantee; the other half proves the guarantee is not
    /// being met by ignoring the seed and going greedy.
    #[test]
    fn sampling_is_reproducible_and_seed_dependent() {
        let dir = tempfile::tempdir().unwrap();
        let tok = toy_tokenizer(dir.path());
        let (m, device) = model(tok.get_vocab_size(true));
        let sample = |seed| {
            generate(
                &m,
                &tok,
                "the city",
                &GenerationConfig {
                    max_new_tokens: 16,
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    seed,
                    ..Default::default()
                },
                &device,
            )
            .unwrap()
        };

        assert_eq!(sample(1), sample(1));
        // A fresh model is near-uniform over ~256 tokens, so 16 unseeded draws
        // colliding would be a ~256^-16 coincidence.
        assert_ne!(sample(1), sample(2));
    }

    /// The cache exists to make decoding linear rather than quadratic. It is
    /// only allowed to do that if it computes the same thing as the uncached
    /// path -- and a cache bug shows up as slightly-worse samples, which is
    /// invisible without this test.
    #[test]
    fn cached_decoding_matches_a_full_forward_pass() {
        let (m, device) = model(256);
        let context = [7u32, 3, 11, 5, 2];

        let full = next_token_probs(&m, &context, &device);

        let mut caches = m.new_caches();
        // Feed the same context incrementally, one token at a time.
        let mut incremental = Vec::new();
        for (i, &t) in context.iter().enumerate() {
            let logits = last_logits(&m, &[t], &mut caches, &device);
            if i + 1 == context.len() {
                incremental = logits;
            }
        }
        let z: f32 = incremental.iter().map(|l| l.exp()).sum();
        let incremental: Vec<f32> = incremental.iter().map(|l| l.exp() / z).collect();

        let worst = full
            .iter()
            .zip(&incremental)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "cached and uncached differ by {worst}");
    }

    /// `top_k = 1` is argmax by another route, so it must agree with greedy.
    /// This is the test that catches an off-by-one in the ordering: if `pick`
    /// sorted ascending, `top_k = 1` would return the *least* likely token and
    /// nothing else here would notice.
    #[test]
    fn top_k_of_one_is_greedy() {
        let mut logits = vec![0.0f32; 32];
        logits[9] = 5.0;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = GenerationConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            ..Default::default()
        };

        assert_eq!(pick(&logits, &cfg, &mut rng), 9);
        assert_eq!(argmax(&logits), 9);
    }

    /// A vanishing `top_p` must still leave one token, not zero. `p = 0` is the
    /// degenerate case that would divide by an empty nucleus.
    #[test]
    fn top_p_never_empties_the_nucleus() {
        let mut logits = vec![0.0f32; 32];
        logits[4] = 3.0;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = GenerationConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            ..Default::default()
        };

        assert_eq!(pick(&logits, &cfg, &mut rng), 4);
    }

    #[test]
    fn stopping_at_eos_is_recorded_and_the_separator_is_not_decoded() {
        let dir = tempfile::tempdir().unwrap();
        let tok = toy_tokenizer(dir.path());
        let (m, device) = model(tok.get_vocab_size(true));

        let s = generate(
            &m,
            &tok,
            "the city",
            &GenerationConfig {
                max_new_tokens: 4,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        assert!(!s.completion.contains(EOS_TOKEN));
        match s.stop_reason {
            StopReason::Limit => assert_eq!(s.n_new_tokens, 4),
            StopReason::Eos => assert!(s.n_new_tokens < 4),
            StopReason::ContextFull => panic!("4 tokens cannot fill a 64-token context"),
        }
    }

    #[test]
    fn a_prompt_that_fills_the_context_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let tok = toy_tokenizer(dir.path());
        let (m, device) = model(tok.get_vocab_size(true));
        let long = "the city begins with a bridge and three islands here ".repeat(40);

        let err = generate(&m, &tok, &long, &Default::default(), &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no room to generate"), "got: {err}");
    }

    /// The prompt set is the eval. Changing it invalidates every comparison
    /// against a previously reported sample, so it should take a deliberate
    /// edit here -- not an accident.
    #[test]
    fn the_prompt_set_is_frozen() {
        assert_eq!(PROMPTS.len(), 6);
        assert_eq!(PROMPTS[0], "The history of the city begins");
    }
}
