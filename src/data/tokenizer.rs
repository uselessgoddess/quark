//! Byte-level BPE, trained on the target corpus.
//!
//! Byte-level rather than word- or character-level, for two reasons that both
//! matter to the evaluation rather than to training:
//!
//!  * **No `<unk>`, ever.** Every byte sequence is representable, so the
//!    encoding is lossless and total. A model that can emit `<unk>` is being
//!    scored on a different, easier distribution than one that cannot, and the
//!    comparison against GPT-2 (also byte-level BPE) would not be controlled.
//!  * **Exactly reversible.** Bits-per-byte requires knowing the byte length of
//!    the text the log-probabilities correspond to. Any normalizer that folds
//!    Unicode or strips whitespace would silently change that denominator.
//!
//! Hence no normalizer at all. This mirrors GPT-2's own tokenizer, which is the
//! point: the baseline and the model should differ in their parameters, not in
//! how their inputs are prepared.

use std::path::Path;

use anyhow::{bail, Context, Result};
use tokenizers::{
    models::{
        bpe::{BpeTrainerBuilder, BPE},
        TrainerWrapper,
    },
    pre_tokenizers::byte_level::ByteLevel,
    AddedToken, Tokenizer,
};

/// Document separator. Named as GPT-2 names it so that corpora prepared for
/// either model read the same.
pub const EOS_TOKEN: &str = "<|endoftext|>";

/// The token id space must fit `u16`, because shards store `u16` (see
/// [`crate::data::shard`]). 8192 leaves a wide margin.
pub const MAX_VOCAB_SIZE: usize = u16::MAX as usize + 1;

/// Train a byte-level BPE over `files` and save it to `out`.
///
/// `vocab_size` counts the special token.
pub fn train(files: &[String], vocab_size: usize, out: &Path) -> Result<Tokenizer> {
    if vocab_size > MAX_VOCAB_SIZE {
        bail!(
            "vocab_size {vocab_size} exceeds {MAX_VOCAB_SIZE}: shards store u16 token ids, \
             so a larger vocab would silently truncate"
        );
    }
    if files.is_empty() {
        bail!("no input files to train the tokenizer on");
    }

    // `TrainerWrapper` rather than `BpeTrainer`: `Tokenizer` is
    // `TokenizerImpl<ModelWrapper, ..>`, and `train_from_files` demands a
    // trainer whose `Model` is that same `ModelWrapper`.
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![AddedToken::from(String::from(EOS_TOKEN), true)])
        // Seed the vocab with all 256 byte-level tokens so that every byte is
        // representable before any merge is learned. Without this a rare byte
        // could be absent from the vocab entirely and encoding would fail.
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer
        .with_pre_tokenizer(Some(byte_level()))
        .with_post_processor(Some(byte_level()))
        .with_decoder(Some(byte_level()));

    tokenizer
        .train_from_files(&mut trainer, files.to_vec())
        .map_err(|e| anyhow::anyhow!("BPE training failed: {e}"))?;

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating tokenizer directory {}", dir.display()))?;
    }
    tokenizer
        .save(out, /* pretty */ false)
        .map_err(|e| anyhow::anyhow!("saving tokenizer to {}: {e}", out.display()))?;

    Ok(tokenizer)
}

/// `add_prefix_space` is **off**, matching GPT-2's own tokenizer.
///
/// Turning it on is tempting -- it would make a word tokenize identically
/// whether it opens a document or sits mid-sentence. But it prepends a space
/// that decoding then faithfully reproduces, so `decode(encode(s)) == " " + s`
/// and the tokenizer is no longer reversible. Bits-per-byte divides by the byte
/// count of the *source* text; a tokenizer that silently adds a byte makes that
/// denominator wrong. Reversibility is the property the metric rests on, so it
/// wins. `roundtrips_ascii_and_unicode_losslessly` is what caught this.
fn byte_level() -> ByteLevel {
    ByteLevel::new(
        /* add_prefix_space */ false, /* trim_offsets */ true, /* use_regex */ true,
    )
}

pub fn load(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", path.display()))
}

/// The id of [`EOS_TOKEN`], which callers need in order to separate documents.
pub fn eos_id(tokenizer: &Tokenizer) -> Result<u32> {
    tokenizer
        .token_to_id(EOS_TOKEN)
        .with_context(|| format!("tokenizer has no {EOS_TOKEN} token"))
}

/// Encode without special tokens; document separators are inserted explicitly by
/// the shard writer, so letting the post-processor add them too would double up.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, /* add_special_tokens */ false)
        .map_err(|e| anyhow::anyhow!("encoding failed: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Ids back to text, dropping [`EOS_TOKEN`].
///
/// The byte-level decoder is lossless, so `decode(encode(s)) == s` exactly --
/// including whitespace. That property is what makes bits-per-byte a meaningful
/// number (see [`crate::eval::corpus`]); it is worth having a test on it rather
/// than an assumption.
pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, /* skip_special_tokens */ true)
        .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A corpus with enough repetition for BPE to find merges, and some
    /// non-ASCII to prove byte-level coverage.
    fn corpus() -> String {
        let mut s = String::new();
        for _ in 0..200 {
            s.push_str("the quick brown fox jumps over the lazy dog\n");
            s.push_str("the quick brown cat sleeps under the lazy log\n");
            s.push_str("naïve café résumé — 日本語 🦀\n");
        }
        s
    }

    fn train_tiny(dir: &Path, vocab_size: usize) -> Tokenizer {
        let corpus_path = dir.join("corpus.txt");
        let mut f = std::fs::File::create(&corpus_path).unwrap();
        f.write_all(corpus().as_bytes()).unwrap();
        train(
            &[corpus_path.to_str().unwrap().to_string()],
            vocab_size,
            &dir.join("tokenizer.json"),
        )
        .unwrap()
    }

    #[test]
    fn roundtrips_ascii_and_unicode_losslessly() {
        let dir = tempfile::tempdir().unwrap();
        let tok = train_tiny(dir.path(), 500);

        // Byte-level BPE must reproduce the input exactly, including text it
        // never saw during training. This is the property bits-per-byte relies
        // on -- if decoding is lossy, the byte denominator is a lie.
        for text in [
            "the quick brown fox",
            "naïve café résumé — 日本語 🦀",
            "totally unseen \u{1F600} bytes: \u{0416}\u{0417}",
        ] {
            let ids = encode(&tok, text).unwrap();
            let back = tok.decode(&ids, false).unwrap();
            assert_eq!(back, text, "byte-level BPE must roundtrip losslessly");
        }
    }

    #[test]
    fn never_emits_unk_on_unseen_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let tok = train_tiny(dir.path(), 500);
        // Text made entirely of characters absent from the training corpus.
        let ids = encode(&tok, "Ω≈ç√∫˜µ≤≥÷").unwrap();
        assert!(!ids.is_empty());
        let back = tok.decode(&ids, false).unwrap();
        assert_eq!(back, "Ω≈ç√∫˜µ≤≥÷");
    }

    #[test]
    fn eos_is_a_single_token() {
        let dir = tempfile::tempdir().unwrap();
        let tok = train_tiny(dir.path(), 500);
        let id = eos_id(&tok).unwrap();
        // The separator must be atomic; if BPE split it into pieces the shard
        // writer's document boundaries would be several tokens wide.
        let ids = encode(&tok, EOS_TOKEN).unwrap();
        assert_eq!(
            ids,
            vec![id],
            "{EOS_TOKEN} must encode to exactly one token"
        );
    }

    #[test]
    fn saved_tokenizer_reloads_identically() {
        let dir = tempfile::tempdir().unwrap();
        let tok = train_tiny(dir.path(), 500);
        let reloaded = load(&dir.path().join("tokenizer.json")).unwrap();
        let text = "the quick brown fox — 日本語";
        assert_eq!(
            encode(&tok, text).unwrap(),
            encode(&reloaded, text).unwrap()
        );
    }

    #[test]
    fn rejects_a_vocab_too_large_for_u16_shards() {
        let dir = tempfile::tempdir().unwrap();
        let err = train(
            &["/dev/null".to_string()],
            70_000,
            &dir.path().join("t.json"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("u16"),
            "error should explain the u16 limit: {err}"
        );
    }
}
