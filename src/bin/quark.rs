//! The quark CLI.
//!
//! Corpus preparation is offline and explicit: you train a tokenizer once, then
//! bake each split into a shard. Both artifacts are inputs to training and to
//! evaluation, and both are content-addressed by nothing at all -- so the shard
//! metadata records the vocabulary size, and the loader refuses a shard whose
//! vocabulary disagrees with the model's.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use quark::data;

#[derive(Parser)]
#[command(name = "quark", about = "A 3M-parameter language model family on burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a byte-level BPE tokenizer on raw text.
    Tokenizer {
        /// Text files to train on. Usually just the training split -- training a
        /// tokenizer on validation or test text leaks them into the vocabulary.
        #[arg(required = true)]
        input: Vec<PathBuf>,
        #[arg(long, default_value = "artifacts/tokenizer.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
    },
    /// Tokenize a text split into a shard for training or evaluation.
    Prepare {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "artifacts/tokenizer.json")]
        tokenizer: PathBuf,
        /// Split the input on WikiText ` = Article = ` headings so each article
        /// becomes its own document. Off for corpora that are already one
        /// document per line or per file.
        #[arg(long)]
        split_articles: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Tokenizer {
            input,
            out,
            vocab_size,
        } => {
            let files = input
                .iter()
                .map(|p| {
                    p.to_str()
                        .map(str::to_string)
                        .ok_or_else(|| anyhow::anyhow!("path is not UTF-8: {}", p.display()))
                })
                .collect::<Result<Vec<_>>>()?;
            let tok = data::tokenizer::train(&files, vocab_size, &out)?;
            tracing::info!(
                vocab = tok.get_vocab_size(true),
                "wrote tokenizer to {}",
                out.display()
            );
        }
        Command::Prepare {
            input,
            out,
            tokenizer,
            split_articles,
        } => {
            if !tokenizer.exists() {
                bail!(
                    "no tokenizer at {}; run `quark tokenizer` first",
                    tokenizer.display()
                );
            }
            let tok = data::tokenizer::load(&tokenizer)?;
            data::prepare_shard(&input, &out, &tok, split_articles)?;
        }
    }
    Ok(())
}
