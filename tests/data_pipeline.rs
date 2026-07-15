//! End-to-end: raw text -> tokenizer -> shard -> windows -> batch.
//!
//! The unit tests in `src/data` each check one seam. This checks that the seams
//! line up, which is where the pipeline can be wrong while every part is right:
//! a tokenizer whose ids overflow the shard's `u16`, a shard whose word count
//! doesn't describe the text it holds, a dataset that scores a token twice.

use std::sync::Arc;

use burn::data::dataset::Dataset;
use quark::data::{self, Shard, TokenDataset};

/// Enough text for BPE to learn merges, with the article headings the WikiText
/// splitter keys on.
fn corpus() -> String {
    let mut s = String::from("\n");
    for i in 0..40 {
        s.push_str(&format!(" = Article {i} = \n\n"));
        s.push_str(" The quick brown fox jumps over the lazy dog .\n");
        s.push_str(" A tactical role @-@ playing game with naïve café prose .\n\n");
        s.push_str(" = = Section = = \n\n");
        s.push_str(" It was released in 2011 and sold well .\n\n");
    }
    s
}

struct Prepared {
    _dir: tempfile::TempDir,
    shard: Arc<Shard>,
    text: String,
}

fn prepare() -> Prepared {
    let dir = tempfile::tempdir().unwrap();
    let text = corpus();

    let text_path = dir.path().join("corpus.txt");
    std::fs::write(&text_path, &text).unwrap();

    let tok = data::tokenizer::train(
        &[text_path.to_str().unwrap().to_string()],
        512,
        &dir.path().join("tokenizer.json"),
    )
    .unwrap();

    let bin = dir.path().join("train.bin");
    data::prepare_shard(&text_path, &bin, &tok, true).unwrap();

    Prepared {
        shard: Arc::new(Shard::open(&bin).unwrap()),
        text,
        _dir: dir,
    }
}

/// The shard's `n_words` is the denominator of the word-level perplexity that
/// `docs/DESIGN.md` §3 reports against GPT-2. If it drifts from the text it
/// describes, the headline number is wrong and nothing else in the pipeline
/// notices.
#[test]
fn shard_word_count_describes_the_source_text() {
    let p = prepare();
    assert_eq!(
        p.shard.meta().n_words,
        data::count_words(&p.text),
        "article splitting must not add or drop words"
    );
}

#[test]
fn shard_byte_count_describes_the_source_text() {
    let p = prepare();
    // Splitting trims each article, so the shard's byte count is the sum of the
    // trimmed articles rather than of the raw file. It must at least never
    // exceed it -- an inflated denominator would understate bits-per-byte.
    assert!(p.shard.meta().n_bytes <= p.text.len());
    assert!(
        p.shard.meta().n_bytes > p.text.len() * 9 / 10,
        "trimming should shed only whitespace, not content: {} vs {}",
        p.shard.meta().n_bytes,
        p.text.len()
    );
}

#[test]
fn every_token_in_the_shard_is_within_the_vocabulary() {
    let p = prepare();
    let vocab = p.shard.meta().vocab_size;
    let tokens = p.shard.tokens(0..p.shard.len());
    assert!(!tokens.is_empty());
    assert!(
        tokens.iter().all(|&t| (t as usize) < vocab),
        "a token id outside the vocabulary would index the embedding table out of bounds"
    );
}

#[test]
fn documents_are_separated_by_exactly_one_eos_each() {
    let p = prepare();
    let eos = p.shard.meta().eos_id;
    let n = p
        .shard
        .tokens(0..p.shard.len())
        .iter()
        .filter(|&&t| t == eos)
        .count();
    assert_eq!(n, 40, "one separator per article, no more");
}

/// Compression is the cheapest end-to-end signal that BPE actually trained: if
/// merges failed to form, the shard degenerates to roughly one token per byte
/// and training would be learning character-level structure by accident.
#[test]
fn tokenizer_compresses_the_corpus() {
    let p = prepare();
    let bytes_per_token = p.shard.meta().n_bytes as f64 / p.shard.meta().n_tokens as f64;
    assert!(
        bytes_per_token > 2.0,
        "byte-level BPE should compress well past 2 bytes/token, got {bytes_per_token:.2}"
    );
}

#[test]
fn windows_cover_the_shard_and_batch_cleanly() {
    let p = prepare();
    let seq_len = 32;
    let ds = TokenDataset::train(p.shard.clone(), seq_len);
    assert!(ds.len() > 1, "corpus should yield several windows");

    // Disjoint windows score every token but the first.
    assert_eq!(ds.n_scored_tokens(), ds.len() * seq_len);

    let w = ds.get(0).unwrap();
    assert_eq!(w.input.len(), seq_len);
    assert_eq!(w.target.len(), seq_len);
    assert_eq!(
        w.input[1..],
        w.target[..seq_len - 1],
        "target trails input by one"
    );
}
