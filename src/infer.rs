//! Runtime inference over user-provided text.
//!
//! Benchmark evaluation in [`crate::eval`] deliberately uses frozen prompts and
//! prepared shards. This module answers the different runtime question: "what
//! does this saved model do with *my* text?" It accepts the same artifact pair
//! for either a language model or a compressor and returns structured records
//! that can be rendered for a person or consumed as JSON.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use serde::Serialize;

use crate::{
    compress::{eval::tensor_ids, CompressTrainConfig, Compressor},
    data::tokenizer,
    eval::{self, generate, GenerationConfig},
};

/// One independent piece of user input and the label shown in output records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceInput {
    pub source: String,
    pub text: String,
}

/// Everything runtime inference needs, independent of the chosen backend.
#[derive(Debug, Clone)]
pub struct InferenceRun {
    pub artifact_dir: PathBuf,
    /// Defaults to `<artifact_dir>/config.json`.
    pub config_path: Option<PathBuf>,
    /// Defaults to `<artifact_dir>/model`.
    pub model_path: Option<PathBuf>,
    pub tokenizer_path: PathBuf,
    pub inputs: Vec<InferenceInput>,
    pub generation: GenerationConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LanguageModelMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub stop_reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompressorMetrics {
    /// Real input tokens only; padding never contributes to this denominator.
    pub input_tokens: usize,
    pub correct_tokens: usize,
    pub token_accuracy: f64,
    pub spans: usize,
    pub span_len: usize,
    pub token_ratio: f64,
    pub bits_per_token: f64,
}

/// A stable, serializable record for one input.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "model_type", rename_all = "snake_case")]
pub enum InferenceOutput {
    LanguageModel {
        source: String,
        input: String,
        output: String,
        metrics: LanguageModelMetrics,
    },
    Compressor {
        source: String,
        input: String,
        output: String,
        metrics: CompressorMetrics,
    },
}

/// The three output formats exposed by `quark infer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
    Jsonl,
}

impl OutputFormat {
    pub fn render(self, records: &[InferenceOutput]) -> Result<String> {
        match self {
            Self::Text => Ok(text_report(records)),
            Self::Json => serde_json::to_string_pretty(records).context("serializing JSON output"),
            Self::Jsonl => records
                .iter()
                .map(|record| serde_json::to_string(record).context("serializing JSONL output"))
                .collect::<Result<Vec<_>>>()
                .map(|lines| lines.join("\n")),
        }
    }
}

/// Load the saved model kind described by the config and infer every input.
pub fn run<B: Backend>(run: &InferenceRun, device: &B::Device) -> Result<Vec<InferenceOutput>> {
    if run.inputs.is_empty() {
        bail!("no input text was provided");
    }
    if run.inputs.iter().any(|input| input.text.is_empty()) {
        bail!("input text must not be empty");
    }

    let config_path = run
        .config_path
        .clone()
        .unwrap_or_else(|| run.artifact_dir.join("config.json"));
    let model_path = run
        .model_path
        .clone()
        .unwrap_or_else(|| run.artifact_dir.join("model"));
    let tok = tokenizer::load(&run.tokenizer_path)?;

    if eval::is_compress_config(&config_path) {
        let config = CompressTrainConfig::load(&config_path)
            .with_context(|| format!("reading the run's config {}", config_path.display()))?;
        let model = crate::compress::eval::load_compressor::<B>(&config, &model_path, device)?;
        ensure_tokenizer_matches(tok.get_vocab_size(true), config.compress.model.vocab_size)?;
        return run
            .inputs
            .iter()
            .map(|input| infer_compressor(&model, &tok, input, device))
            .collect();
    }

    let (model, config) = eval::load_model::<B>(&config_path, &model_path, device)?;
    ensure_tokenizer_matches(tok.get_vocab_size(true), config.model.vocab_size)?;
    run.inputs
        .iter()
        .map(|input| {
            let prompt_tokens = tokenizer::encode(&tok, &input.text)?.len();
            let sample = generate::generate(&model, &tok, &input.text, &run.generation, device)
                .with_context(|| format!("running inference for {}", input.source))?;
            Ok(InferenceOutput::LanguageModel {
                source: input.source.clone(),
                input: input.text.clone(),
                output: sample.completion,
                metrics: LanguageModelMetrics {
                    prompt_tokens,
                    generated_tokens: sample.n_new_tokens,
                    stop_reason: format!("{:?}", sample.stop_reason).to_lowercase(),
                },
            })
        })
        .collect()
}

fn ensure_tokenizer_matches(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        bail!(
            "tokenizer vocab {actual} != model vocab {expected}: token ids from this tokenizer \
             mean something else to the model"
        );
    }
    Ok(())
}

/// Reconstruct arbitrary-length text in disjoint configured spans.
///
/// A compressor only accepts exactly `span_len` tokens. The last span is padded
/// with EOS, but its padding is removed before decoding and excluded from every
/// metric so short inputs do not receive an artificially high score.
fn infer_compressor<B: Backend>(
    model: &Compressor<B>,
    tok: &tokenizers::Tokenizer,
    input: &InferenceInput,
    device: &B::Device,
) -> Result<InferenceOutput> {
    let ids = tokenizer::encode(tok, &input.text)?;
    if ids.is_empty() {
        bail!("{} encoded to zero tokens", input.source);
    }

    let cfg = model.config();
    let eos = tokenizer::eos_id(tok)?;
    let mut reconstructed = Vec::with_capacity(ids.len());
    let mut correct = 0usize;

    for chunk in ids.chunks(cfg.span_len) {
        let real_len = chunk.len();
        let mut padded = chunk.to_vec();
        padded.resize(cfg.span_len, eos);
        let stored: Vec<i32> = padded.into_iter().map(|id| id as i32).collect();
        let span =
            Tensor::<B, 2, Int>::from_data(TensorData::new(stored, [1, cfg.span_len]), device);
        let output = tensor_ids(model.reconstruct(span).into_data());
        let output = &output[..real_len];
        correct += chunk
            .iter()
            .zip(output)
            .filter(|(expected, actual)| expected == actual)
            .count();
        reconstructed.extend_from_slice(output);
    }

    let input_tokens = ids.len();
    Ok(InferenceOutput::Compressor {
        source: input.source.clone(),
        input: input.text.clone(),
        output: tokenizer::decode(tok, &reconstructed)?,
        metrics: CompressorMetrics {
            input_tokens,
            correct_tokens: correct,
            token_accuracy: correct as f64 / input_tokens as f64,
            spans: input_tokens.div_ceil(cfg.span_len),
            span_len: cfg.span_len,
            token_ratio: cfg.token_ratio(),
            bits_per_token: cfg.rate_bits_per_token(),
        },
    })
}

fn text_report(records: &[InferenceOutput]) -> String {
    use std::fmt::Write;

    let mut report = String::new();
    for (index, record) in records.iter().enumerate() {
        match record {
            InferenceOutput::LanguageModel {
                source,
                input,
                output,
                metrics,
            } => {
                let _ = writeln!(
                    report,
                    "--- input {index}: {source} (language model) ---\n\
                     in : {input}\n\
                     out: {output}\n\
                     metrics: {} prompt tokens, {} generated tokens, stopped: {}\n",
                    metrics.prompt_tokens, metrics.generated_tokens, metrics.stop_reason,
                );
            }
            InferenceOutput::Compressor {
                source,
                input,
                output,
                metrics,
            } => {
                let _ = writeln!(
                    report,
                    "--- input {index}: {source} (compressor) ---\n\
                     in : {input}\n\
                     out: {output}\n\
                     metrics: {}/{} tokens correct ({:.4}%), {} span(s) of {}, \
                     {:.2}x token ratio, {:.3} bits/token\n",
                    metrics.correct_tokens,
                    metrics.input_tokens,
                    metrics.token_accuracy * 100.0,
                    metrics.spans,
                    metrics.span_len,
                    metrics.token_ratio,
                    metrics.bits_per_token,
                );
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compress::CompressConfig, test_util::TestBackend};
    use std::io::Write;

    fn records() -> Vec<InferenceOutput> {
        vec![
            InferenceOutput::LanguageModel {
                source: "--text[0]".into(),
                input: "hello".into(),
                output: " world".into(),
                metrics: LanguageModelMetrics {
                    prompt_tokens: 1,
                    generated_tokens: 1,
                    stop_reason: "limit".into(),
                },
            },
            InferenceOutput::Compressor {
                source: "sample.txt".into(),
                input: "abc".into(),
                output: "abd".into(),
                metrics: CompressorMetrics {
                    input_tokens: 3,
                    correct_tokens: 2,
                    token_accuracy: 2.0 / 3.0,
                    spans: 1,
                    span_len: 16,
                    token_ratio: 4.0,
                    bits_per_token: 3.0,
                },
            },
        ]
    }

    #[test]
    fn every_output_format_contains_results_and_metrics() {
        let records = records();
        let text = OutputFormat::Text.render(&records).unwrap();
        assert!(text.contains("world"));
        assert!(text.contains("tokens correct"));

        let json = OutputFormat::Json.render(&records).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 2);
        assert_eq!(parsed[0]["metrics"]["generated_tokens"], 1);

        let jsonl = OutputFormat::Jsonl.render(&records).unwrap();
        let lines: Vec<_> = jsonl.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            serde_json::from_str::<serde_json::Value>(line).unwrap();
        }
    }

    #[test]
    fn compressor_accepts_arbitrary_length_text_without_scoring_padding() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = dir.path().join("corpus.txt");
        let mut file = std::fs::File::create(&corpus).unwrap();
        for _ in 0..100 {
            writeln!(file, "the quick brown fox crosses the old bridge").unwrap();
        }
        drop(file);
        let tok = tokenizer::train(
            &[corpus.to_string_lossy().into_owned()],
            300,
            &dir.path().join("tokenizer.json"),
        )
        .unwrap();

        let mut cfg = CompressConfig::tiny();
        cfg.model.vocab_size = tok.get_vocab_size(true);
        let device = Default::default();
        let model = Compressor::<TestBackend>::new(cfg.clone(), &device);
        let mut text = "the quick brown fox crosses the old bridge twice; ".repeat(8);
        while tokenizer::encode(&tok, &text).unwrap().len() % cfg.span_len == 0 {
            text.push('x');
        }
        let expected_tokens = tokenizer::encode(&tok, &text).unwrap().len();

        let result = infer_compressor(
            &model,
            &tok,
            &InferenceInput {
                source: "test".into(),
                text,
            },
            &device,
        )
        .unwrap();

        let InferenceOutput::Compressor { metrics, .. } = result else {
            panic!("compressor inference returned the wrong record kind");
        };
        assert_eq!(metrics.input_tokens, expected_tokens);
        assert_eq!(metrics.spans, expected_tokens.div_ceil(cfg.span_len));
        assert!(metrics.spans > 1);
        assert!(metrics.correct_tokens <= expected_tokens);
        assert!((0.0..=1.0).contains(&metrics.token_accuracy));
    }
}
