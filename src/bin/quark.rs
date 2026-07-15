//! The quark CLI.
//!
//! Corpus preparation is offline and explicit: you train a tokenizer once, then
//! bake each split into a shard. Both artifacts are inputs to training and to
//! evaluation, and both are content-addressed by nothing at all -- so the shard
//! metadata records the vocabulary size, and the loader refuses a shard whose
//! vocabulary disagrees with the model's.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use quark::{data, TrainConfig};

#[derive(Parser)]
#[command(name = "quark", about = "A 3M-parameter language model family on burn")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Which backend to train on.
///
/// Every variant is always offered, even when it was not compiled in, because
/// `--backend wgpu` failing with "rebuild with `--features wgpu`" is a better
/// answer than clap reporting that the value does not exist.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    /// CPU. Correct, and far too slow for a real run -- for smoke tests.
    Ndarray,
    /// The primary backend: Vulkan/Metal/DX12 via wgpu.
    Wgpu,
    /// CUDA, if you have it. Faster than wgpu on NVIDIA, and less portable.
    Cuda,
}

impl Default for BackendChoice {
    fn default() -> Self {
        // wgpu first: it is what the project targets, and what a 16GB consumer
        // card will actually be driven with. ndarray is the fallback only
        // because a build with no GPU feature has nothing else.
        if cfg!(feature = "wgpu") {
            Self::Wgpu
        } else if cfg!(feature = "cuda") {
            Self::Cuda
        } else {
            Self::Ndarray
        }
    }
}

/// Define a training entry point per backend, and a matching stub for builds
/// that left that backend out.
///
/// The stub is the point: without it, `--backend cuda` on a wgpu build would
/// either fail to compile or silently run somewhere else. The type in the
/// invocation is only tokens until the feature is on, so naming `burn_cuda`
/// here does not require the crate to be present.
macro_rules! backend_entry {
    ($name:ident, $feature:literal, $backend:ty) => {
        #[cfg(feature = $feature)]
        fn $name(config: TrainConfig) -> Result<()> {
            quark::train::run::<burn::backend::Autodiff<$backend>>(config, Default::default())?;
            Ok(())
        }

        #[cfg(not(feature = $feature))]
        fn $name(_config: TrainConfig) -> Result<()> {
            bail!(
                "this binary was built without the `{feature}` backend; rebuild with \
                 `cargo build --release --features {feature}`",
                feature = $feature,
            )
        }
    };
}

backend_entry!(train_ndarray, "ndarray", burn_ndarray::NdArray<f32>);
backend_entry!(train_wgpu, "wgpu", burn_wgpu::Wgpu);
backend_entry!(train_cuda, "cuda", burn_cuda::Cuda);

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
    /// Train a model on prepared shards.
    Train {
        /// A `TrainConfig` JSON, as written to `<artifact-dir>/config.json` by a
        /// previous run. Omit for the built-in reference config: `quark_3m` on
        /// `artifacts/{train,valid}.bin`.
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, value_enum)]
        backend: Option<BackendChoice>,

        // Overrides. These are the fields worth reaching for from a shell; the
        // rest of `TrainConfig` (the AdamW constants, the schedule shape) is
        // deliberately file-only, because changing them from a flag and not
        // recording why is how a run becomes unreproducible.
        #[arg(long)]
        train_shard: Option<PathBuf>,
        #[arg(long)]
        valid_shard: Option<PathBuf>,
        #[arg(long)]
        artifact_dir: Option<PathBuf>,
        #[arg(long)]
        seq_len: Option<usize>,
        /// Windows per forward pass. Lower this first when VRAM runs out, and
        /// raise `--grad-accumulation` by the same factor to keep the optimizer
        /// seeing an identical batch.
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        grad_accumulation: Option<usize>,
        #[arg(long)]
        num_epochs: Option<usize>,
        #[arg(long)]
        lr: Option<f64>,
        #[arg(long)]
        seed: Option<u64>,
        /// Resume from an epoch checkpoint in `artifact_dir`.
        #[arg(long)]
        resume_from_epoch: Option<usize>,
        /// Print the resolved config and the parameter budget, then exit without
        /// touching the GPU.
        #[arg(long)]
        dry_run: bool,
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
        Command::Train {
            config,
            backend,
            train_shard,
            valid_shard,
            artifact_dir,
            seq_len,
            batch_size,
            grad_accumulation,
            num_epochs,
            lr,
            seed,
            resume_from_epoch,
            dry_run,
        } => {
            let mut cfg = match &config {
                Some(path) => TrainConfig::load(path)
                    .with_context(|| format!("reading config {}", path.display()))?,
                None => TrainConfig::default(),
            };

            // `resume_from_epoch` is deliberately not `Option`-overwritten the
            // way the rest are: `None` from the CLI means "say nothing", which
            // must not clear a value the file set.
            macro_rules! set {
                ($($field:ident),* $(,)?) => { $( if let Some(v) = $field { cfg.$field = v; } )* };
            }
            set!(
                train_shard,
                valid_shard,
                artifact_dir,
                seq_len,
                batch_size,
                grad_accumulation,
                num_epochs,
                lr,
                seed,
            );
            if resume_from_epoch.is_some() {
                cfg.resume_from_epoch = resume_from_epoch;
            }

            // Validate before dispatching, so a typo is caught here rather than
            // after a backend has spun up a GPU context.
            cfg.validate()?;

            let backend = backend.unwrap_or_default();
            if dry_run {
                println!("{}", cfg.model.budget_table());
                println!("{}", serde_json::to_string_pretty(&cfg)?);
                println!("backend: {backend:?}");
                return Ok(());
            }

            tracing::info!(?backend, "starting training");
            match backend {
                BackendChoice::Ndarray => train_ndarray(cfg)?,
                BackendChoice::Wgpu => train_wgpu(cfg)?,
                BackendChoice::Cuda => train_cuda(cfg)?,
            }
        }
    }
    Ok(())
}
