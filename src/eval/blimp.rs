//! BLiMP: grammatical knowledge as a forced choice between a minimal pair.
//!
//! Warstadt et al. 2020, *BLiMP: The Benchmark of Linguistic Minimal Pairs for
//! English* (arXiv:1912.00582). 67 paradigms of 1000 pairs each, one grammatical
//! sentence and one minimally different ungrammatical one. The model is correct
//! on a pair when it assigns the grammatical sentence the higher probability.
//!
//! Perplexity and BLiMP fail differently, which is why the issue asks for both.
//! A model can buy perplexity with frequency statistics -- get the common words
//! roughly right and never mind the syntax -- and at 3M parameters that is
//! exactly the shortcut we should expect it to take. BLiMP does not reward it:
//! both sentences in a pair use nearly the same words, so unigram statistics
//! score at chance.
//!
//! # The scoring rule
//!
//! Full-sentence log-probability, **not normalized by length**, per
//! `docs/DESIGN.md` §3.2. This is BLiMP's own `simple_LM_method` and it is the
//! only rule under which the numbers are comparable to published ones. It is
//! also defensible on its own: most pairs are the same length in words, and
//! normalizing would make the metric depend on the tokenizer, which is precisely
//! what we are at pains to avoid elsewhere.

use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

use anyhow::{bail, Context, Result};
use burn::{
    prelude::Backend,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{data::tokenizer, model::QuarkLm, train::output::token_log_probs};

/// One line of a BLiMP `.jsonl`. The suite carries more fields than these; we
/// deserialize the ones the metric is defined in terms of and ignore the rest.
#[derive(Debug, Clone, Deserialize)]
struct RawPair {
    sentence_good: String,
    sentence_bad: String,
    /// e.g. `anaphor_gender_agreement`. Identifies the paradigm.
    #[serde(rename = "UID")]
    uid: String,
    /// The broad phenomenon, e.g. `morphology`, `syntax`, `semantics`.
    #[serde(default)]
    field: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimalPair {
    pub good: String,
    pub bad: String,
}

#[derive(Debug, Clone)]
pub struct Paradigm {
    pub uid: String,
    pub field: String,
    pub pairs: Vec<MinimalPair>,
}

/// The suite as loaded from disk.
#[derive(Debug, Clone, Default)]
pub struct BlimpSuite {
    pub paradigms: Vec<Paradigm>,
}

impl BlimpSuite {
    /// Load every `.jsonl` under `dir`.
    ///
    /// The upstream release is one file per paradigm, so a paradigm is a file --
    /// but we group by the `UID` field rather than by filename, because the
    /// filename is not part of the data and a rename should not silently split a
    /// paradigm in two.
    pub fn load(dir: &Path) -> Result<Self> {
        let mut by_uid: BTreeMap<String, Paradigm> = BTreeMap::new();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading BLiMP directory {}", dir.display()))?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .collect();
        // Directory order is filesystem order, i.e. arbitrary. Sorting makes the
        // report reproducible run to run.
        files.sort();

        if files.is_empty() {
            bail!(
                "no .jsonl files in {}: download the suite from \
                 https://github.com/alexwarstadt/blimp (data/ directory)",
                dir.display()
            );
        }

        for path in &files {
            let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            for (i, line) in std::io::BufRead::lines(BufReader::new(file)).enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let raw: RawPair = serde_json::from_str(&line)
                    .with_context(|| format!("{}:{}", path.display(), i + 1))?;
                by_uid
                    .entry(raw.uid.clone())
                    .or_insert_with(|| Paradigm {
                        uid: raw.uid,
                        field: raw.field,
                        pairs: Vec::new(),
                    })
                    .pairs
                    .push(MinimalPair {
                        good: raw.sentence_good,
                        bad: raw.sentence_bad,
                    });
            }
        }

        Ok(Self {
            paradigms: by_uid.into_values().collect(),
        })
    }

    pub fn n_pairs(&self) -> usize {
        self.paradigms.iter().map(|p| p.pairs.len()).sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParadigmScore {
    pub uid: String,
    pub field: String,
    pub n_pairs: usize,
    pub n_correct: usize,
}

impl ParadigmScore {
    pub fn accuracy(&self) -> f64 {
        if self.n_pairs == 0 {
            return f64::NAN;
        }
        self.n_correct as f64 / self.n_pairs as f64
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BlimpScore {
    pub per_paradigm: Vec<ParadigmScore>,
}

impl BlimpScore {
    /// Accuracy over every pair in the suite.
    ///
    /// Pair-weighted, not paradigm-weighted. The released suite has 1000 pairs
    /// in every paradigm so the two agree there; they diverge on a subset, and
    /// pair-weighting is the one that keeps meaning what it says.
    pub fn accuracy(&self) -> f64 {
        let n: usize = self.per_paradigm.iter().map(|p| p.n_pairs).sum();
        if n == 0 {
            return f64::NAN;
        }
        let correct: usize = self.per_paradigm.iter().map(|p| p.n_correct).sum();
        correct as f64 / n as f64
    }

    /// Accuracy grouped by broad phenomenon.
    pub fn by_field(&self) -> BTreeMap<String, f64> {
        let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for p in &self.per_paradigm {
            let e = totals.entry(p.field.clone()).or_default();
            e.0 += p.n_correct;
            e.1 += p.n_pairs;
        }
        totals
            .into_iter()
            .filter(|(_, (_, n))| *n > 0)
            .map(|(f, (c, n))| (f, c as f64 / n as f64))
            .collect()
    }

    pub fn report(&self) -> String {
        let mut s = format!(
            "BLiMP accuracy {:>7.2}%   (chance is 50.00%)\n\nby field:\n",
            self.accuracy() * 100.0
        );
        for (field, acc) in self.by_field() {
            s += &format!("  {field:<28} {:>6.2}%\n", acc * 100.0);
        }
        // Worst paradigms rather than all 67: the tail is what a 3M model is
        // failing at, and it is the part worth reading.
        let mut sorted = self.per_paradigm.clone();
        sorted.sort_by(|a, b| a.accuracy().total_cmp(&b.accuracy()));
        s += "\nweakest paradigms:\n";
        for p in sorted.iter().take(10) {
            s += &format!(
                "  {:<44} {:>6.2}%  ({}/{})\n",
                p.uid,
                p.accuracy() * 100.0,
                p.n_correct,
                p.n_pairs
            );
        }
        s
    }
}

/// Did the model get this pair right?
///
/// A tie counts as **wrong**. Floating point makes exact ties vanishingly rare
/// for a trained model, but a *degenerate* one -- uniform over the vocabulary,
/// say -- ties on every equal-length pair, and counting those as correct would
/// report it at near 100%. The rule that flatters a broken model is the wrong
/// rule.
///
/// Pinned by `experiments/protocol_fixture.json`, because GPT-2's BLiMP baseline
/// is measured by a different program and a disagreement here would look like a
/// difference between the models.
pub fn is_correct(good: f64, bad: f64) -> bool {
    good > bad
}

/// Log-probability of each sequence, summed over its tokens: `[n_sequences]` on
/// the host.
///
/// Every token is scored, including the first: each sequence is prefixed with
/// `bos` so that `log P(t_0 | bos)` exists. This is not a formality. BLiMP pairs
/// routinely differ at the very first word ("*Whose* hat should Tom wear" vs
/// "*Who* should Tom wear the hat"), and a scorer that skipped `t_0` would be
/// blind to exactly the token the paradigm is testing.
///
/// Sequences are padded to the longest in the batch and masked, so padding
/// contributes nothing to any sum.
pub fn sentence_log_probs<B: Backend>(
    model: &QuarkLm<B>,
    sequences: &[Vec<u32>],
    bos: u32,
    device: &B::Device,
) -> Result<Vec<f64>> {
    if sequences.is_empty() {
        return Ok(Vec::new());
    }
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
    if max_len == 0 {
        bail!("cannot score an empty sequence: it has no tokens to assign probability to");
    }
    if max_len > model.config().max_seq_len {
        bail!(
            "a sentence tokenizes to {} tokens, past the model's context of {}. Truncating would \
             compare two different sentences, so this is an error rather than a warning",
            max_len,
            model.config().max_seq_len
        );
    }

    let mut input = vec![0i32; batch * max_len];
    let mut target = vec![0i32; batch * max_len];
    let mut mask = vec![0f32; batch * max_len];
    for (r, seq) in sequences.iter().enumerate() {
        let row = r * max_len;
        input[row] = bos as i32;
        for (t, &tok) in seq.iter().enumerate() {
            // Position `t` predicts `seq[t]` from `bos, seq[..t]`.
            if t + 1 < max_len {
                input[row + t + 1] = tok as i32;
            }
            target[row + t] = tok as i32;
            mask[row + t] = 1.0;
        }
    }

    let shape = [batch, max_len];
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(input, shape), device);
    let target = Tensor::<B, 2, Int>::from_data(TensorData::new(target, shape), device);
    let mask = Tensor::<B, 2>::from_data(TensorData::new(mask, shape), device);

    let per_token = token_log_probs(model.forward(input), target).mul(mask);
    let totals = per_token.sum_dim(1).into_data();
    Ok(totals
        .iter::<f32>()
        .map(|v| v.elem::<f64>())
        .collect::<Vec<_>>())
}

/// Run the suite.
///
/// `batch_size` is in *sentences*, and each pair contributes two, so the model
/// sees `batch_size` sentences per forward pass. Both members of a pair are
/// scored in the same batch, which keeps their padding identical -- not that
/// padding is supposed to matter, but if it ever did, this makes the bug show up
/// as a wrong accuracy rather than as a systematic preference for whichever
/// sentence happened to be shorter.
pub fn evaluate<B: Backend>(
    model: &QuarkLm<B>,
    tok: &Tokenizer,
    suite: &BlimpSuite,
    batch_size: usize,
    device: &B::Device,
) -> Result<BlimpScore> {
    if batch_size == 0 {
        bail!("batch_size must be positive");
    }
    let mut per_paradigm = Vec::with_capacity(suite.paradigms.len());

    for paradigm in &suite.paradigms {
        let mut n_correct = 0;
        // `.max(1)` because a batch_size of 1 would otherwise chunk zero pairs
        // at a time and loop forever.
        for chunk in paradigm.pairs.chunks((batch_size / 2).max(1)) {
            let mut sequences = Vec::with_capacity(chunk.len() * 2);
            for pair in chunk {
                sequences.push(tokenizer::encode(tok, &pair.good)?);
                sequences.push(tokenizer::encode(tok, &pair.bad)?);
            }
            let scores = sentence_log_probs(model, &sequences, tokenizer::eos_id(tok)?, device)?;
            n_correct += scores.chunks(2).filter(|s| is_correct(s[0], s[1])).count();
        }
        per_paradigm.push(ParadigmScore {
            uid: paradigm.uid.clone(),
            field: paradigm.field.clone(),
            n_pairs: paradigm.pairs.len(),
            n_correct,
        });
    }

    Ok(BlimpScore { per_paradigm })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_util::TestBackend, ModelConfig};
    use std::io::Write;

    fn write_suite(dir: &Path) {
        let mut f = File::create(dir.join("a.jsonl")).unwrap();
        for i in 0..3 {
            writeln!(
                f,
                r#"{{"sentence_good": "the cat sleeps {i}", "sentence_bad": "the cat sleep {i}", "UID": "agreement", "field": "morphology", "pair_id": {i}}}"#
            )
            .unwrap();
        }
        let mut f = File::create(dir.join("b.jsonl")).unwrap();
        writeln!(
            f,
            r#"{{"sentence_good": "who left", "sentence_bad": "who leaved", "UID": "islands", "field": "syntax"}}"#
        )
        .unwrap();
        // Not JSONL: must be ignored, not parsed and failed on.
        File::create(dir.join("README.md")).unwrap();
    }

    #[test]
    fn loading_groups_by_uid_and_ignores_other_files() {
        let dir = tempfile::tempdir().unwrap();
        write_suite(dir.path());

        let suite = BlimpSuite::load(dir.path()).unwrap();

        assert_eq!(suite.paradigms.len(), 2);
        assert_eq!(suite.n_pairs(), 4);
        // BTreeMap order, so this is stable rather than incidental.
        assert_eq!(suite.paradigms[0].uid, "agreement");
        assert_eq!(suite.paradigms[0].pairs.len(), 3);
        assert_eq!(suite.paradigms[0].field, "morphology");
        assert_eq!(suite.paradigms[1].pairs[0].good, "who left");
    }

    #[test]
    fn an_empty_directory_says_where_to_get_the_suite() {
        let dir = tempfile::tempdir().unwrap();
        let err = BlimpSuite::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("github.com/alexwarstadt/blimp"), "got: {err}");
    }

    /// Padding must be inert. If it were not, a batch's scores would depend on
    /// the longest sentence in it -- so the same sentence would score
    /// differently depending on what it was batched with, and the accuracy would
    /// silently depend on `batch_size`.
    #[test]
    fn padding_does_not_change_a_score() {
        let device = Default::default();
        let cfg = ModelConfig::tiny();
        let model = QuarkLm::<TestBackend>::new(cfg, &device);
        let short = vec![3u32, 7, 11];
        let long = vec![3u32, 7, 11, 13, 17, 19, 23];

        let alone = sentence_log_probs(&model, std::slice::from_ref(&short), 0, &device).unwrap();
        let padded = sentence_log_probs(&model, &[short.clone(), long], 0, &device).unwrap();

        assert!(
            (alone[0] - padded[0]).abs() < 1e-3,
            "same sentence scored {} alone and {} when padded",
            alone[0],
            padded[0]
        );
    }

    /// Sanity on the scale: a log-probability is a sum of `n` negative terms,
    /// each near `-ln V` for a fresh model, so a 3-token sentence should land
    /// near `-3 ln V` and never above zero.
    #[test]
    fn a_score_is_a_sum_of_per_token_log_probs() {
        let device = Default::default();
        let cfg = ModelConfig::tiny();
        let model = QuarkLm::<TestBackend>::new(cfg.clone(), &device);

        let scores = sentence_log_probs(&model, &[vec![3, 7, 11]], 0, &device).unwrap();

        let uniform = -3.0 * (cfg.vocab_size as f64).ln();
        assert!(scores[0] < 0.0, "a log-probability cannot be positive");
        assert!(
            (scores[0] - uniform).abs() < 2.0,
            "3-token score {} should sit near 3*ln(V) = {uniform}",
            scores[0]
        );
    }

    #[test]
    fn a_sentence_longer_than_the_context_is_an_error_not_a_truncation() {
        let device = Default::default();
        let cfg = ModelConfig::tiny();
        let model = QuarkLm::<TestBackend>::new(cfg.clone(), &device);
        let long: Vec<u32> = (0..cfg.max_seq_len as u32 + 1).map(|i| i % 7).collect();

        let err = sentence_log_probs(&model, &[long], 0, &device)
            .unwrap_err()
            .to_string();
        assert!(err.contains("past the model's context"), "got: {err}");
    }

    /// The accuracy arithmetic, independent of any model: pair-weighted, and
    /// grouped by field.
    #[test]
    fn accuracy_is_pair_weighted_across_uneven_paradigms() {
        let score = BlimpScore {
            per_paradigm: vec![
                ParadigmScore {
                    uid: "a".into(),
                    field: "syntax".into(),
                    n_pairs: 100,
                    n_correct: 90,
                },
                ParadigmScore {
                    uid: "b".into(),
                    field: "syntax".into(),
                    n_pairs: 900,
                    n_correct: 450,
                },
            ],
        };
        // Pair-weighted: 540/1000. Paradigm-weighted would say 70%.
        assert!((score.accuracy() - 0.54).abs() < 1e-9);
        assert!((score.by_field()["syntax"] - 0.54).abs() < 1e-9);
    }

    /// The half of BLiMP that GPT-2's baseline must match.
    ///
    /// `experiments/gpt2_baseline.py --self-test` asserts the same cases. The
    /// sentence scores themselves are each model's own business; the rule for
    /// turning two scores into a verdict, and the verdicts into an accuracy, are
    /// not -- a disagreement there would look like a difference between the
    /// models.
    #[test]
    fn the_blimp_protocol_matches_the_frozen_protocol() {
        let fixture: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("experiments/protocol_fixture.json").unwrap(),
        )
        .unwrap();

        for case in fixture["blimp"]["decision"]["cases"].as_array().unwrap() {
            let (good, bad) = (
                case["good"].as_f64().unwrap(),
                case["bad"].as_f64().unwrap(),
            );
            assert_eq!(
                is_correct(good, bad),
                case["correct"].as_bool().unwrap(),
                "{}: good {good}, bad {bad}",
                case["name"]
            );
        }

        let agg = &fixture["blimp"]["aggregation"];
        let score = BlimpScore {
            per_paradigm: agg["paradigms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| ParadigmScore {
                    uid: p["uid"].as_str().unwrap().to_string(),
                    field: p["field"].as_str().unwrap().to_string(),
                    n_pairs: p["n_pairs"].as_u64().unwrap() as usize,
                    n_correct: p["n_correct"].as_u64().unwrap() as usize,
                })
                .collect(),
        };

        assert!(
            (score.accuracy() - agg["accuracy"].as_f64().unwrap()).abs() < 1e-12,
            "accuracy {} != frozen {}",
            score.accuracy(),
            agg["accuracy"]
        );
        let by_field = score.by_field();
        for (field, want) in agg["by_field"].as_object().unwrap() {
            assert!(
                (by_field[field] - want.as_f64().unwrap()).abs() < 1e-12,
                "{field}: {} != frozen {want}",
                by_field[field]
            );
        }
    }
}
