//! A tokenized corpus on disk: a flat `u16` token stream plus a JSON sidecar.
//!
//! The sidecar carries `n_words` and `n_bytes` alongside `n_tokens`, and that is
//! the whole reason this format exists rather than a bare array. `docs/DESIGN.md`
//! §3 commits to reporting word-level perplexity `exp(total_NLL / n_words)` and
//! bits-per-byte `total_NLL / (n_bytes * ln 2)`, both of which are
//! tokenizer-independent and therefore comparable against GPT-2. Neither
//! denominator can be recovered from the token stream after the fact -- `n_words`
//! and `n_bytes` are properties of the *text*, so they must be counted while the
//! text is still in hand.
//!
//! Tokens are `u16` because the vocabulary is 8192; see
//! [`crate::data::tokenizer::MAX_VOCAB_SIZE`]. Little-endian is written
//! explicitly so shards are portable rather than accidentally host-ordered.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

/// Counts describing one shard. Written next to the `.bin` as `<stem>.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardMeta {
    /// Number of `u16` tokens in the `.bin`.
    pub n_tokens: usize,
    /// Whitespace-delimited words in the source text, excluding separators we
    /// inserted ourselves. The denominator of word-level perplexity.
    pub n_words: usize,
    /// UTF-8 bytes of the source text. The denominator of bits-per-byte.
    pub n_bytes: usize,
    /// Recorded so that a shard tokenized with a different vocabulary is
    /// rejected rather than silently producing garbage logits.
    pub vocab_size: usize,
    pub eos_id: u32,
}

fn meta_path(bin: &Path) -> PathBuf {
    bin.with_extension("json")
}

/// Streams tokens to `<out>.bin` while tallying the sidecar counts.
pub struct ShardWriter {
    bin: PathBuf,
    out: BufWriter<File>,
    meta: ShardMeta,
}

impl ShardWriter {
    pub fn create(bin: &Path, vocab_size: usize, eos_id: u32) -> Result<Self> {
        if let Some(dir) = bin.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating shard directory {}", dir.display()))?;
        }
        let file =
            File::create(bin).with_context(|| format!("creating shard file {}", bin.display()))?;
        Ok(Self {
            bin: bin.to_path_buf(),
            out: BufWriter::new(file),
            meta: ShardMeta {
                n_tokens: 0,
                n_words: 0,
                n_bytes: 0,
                vocab_size,
                eos_id,
            },
        })
    }

    /// Append one document: its tokens, then an EOS separator.
    ///
    /// `text` is passed alongside `tokens` rather than re-derived from them
    /// because the byte and word counts must describe the *original* text.
    pub fn push_document(&mut self, text: &str, tokens: &[u32]) -> Result<()> {
        self.write_tokens(tokens)?;
        self.write_tokens(&[self.meta.eos_id])?;
        // The separator is ours, not the corpus's, so it counts toward neither
        // denominator -- charging the model for a token the text never
        // contained would deflate word-level perplexity for free.
        self.meta.n_words += count_words(text);
        self.meta.n_bytes += text.len();
        Ok(())
    }

    fn write_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        for &t in tokens {
            let t: u16 = t.try_into().with_context(|| {
                format!(
                    "token id {t} does not fit u16; vocab_size is {}",
                    self.meta.vocab_size
                )
            })?;
            self.out.write_all(&t.to_le_bytes())?;
        }
        self.meta.n_tokens += tokens.len();
        Ok(())
    }

    /// Flush the stream and write the sidecar. Consuming `self` makes it hard to
    /// leave a `.bin` on disk with no `.json` beside it.
    pub fn finish(mut self) -> Result<ShardMeta> {
        self.out.flush()?;
        let json = serde_json::to_string_pretty(&self.meta)?;
        let path = meta_path(&self.bin);
        std::fs::write(&path, json)
            .with_context(|| format!("writing shard metadata {}", path.display()))?;
        Ok(self.meta)
    }
}

/// Words as `str::split_whitespace` sees them.
///
/// This is deliberately the crudest possible definition, because it is the one
/// the published WikiText-103 token counts use and the one GPT-2's word-level
/// perplexity protocol assumes. A smarter tokenizer here would produce a
/// different denominator and quietly break comparability with the literature.
pub fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// A memory-mapped shard. Mapped rather than read so that a multi-hundred-MB
/// corpus costs page cache instead of resident memory, and so that workers can
/// share one mapping.
#[derive(Debug)]
pub struct Shard {
    mmap: Mmap,
    meta: ShardMeta,
}

impl Shard {
    pub fn open(bin: &Path) -> Result<Self> {
        let meta: ShardMeta = {
            let path = meta_path(bin);
            let json = std::fs::read_to_string(&path)
                .with_context(|| format!("reading shard metadata {}", path.display()))?;
            serde_json::from_str(&json)
                .with_context(|| format!("parsing shard metadata {}", path.display()))?
        };

        let file = File::open(bin).with_context(|| format!("opening shard {}", bin.display()))?;
        // SAFETY: standard mmap caveat -- undefined behaviour if the file is
        // mutated underneath us. Shards are written once and then read-only.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mapping shard {}", bin.display()))?;

        let expected = meta.n_tokens * 2;
        if mmap.len() != expected {
            bail!(
                "shard {} is {} bytes but its metadata claims {} tokens ({expected} bytes); \
                 the pair is out of sync",
                bin.display(),
                mmap.len(),
                meta.n_tokens
            );
        }
        Ok(Self { mmap, meta })
    }

    pub fn meta(&self) -> &ShardMeta {
        &self.meta
    }

    pub fn len(&self) -> usize {
        self.meta.n_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens in `range`, decoded from little-endian.
    ///
    /// Decoded per call rather than reinterpreted as a `&[u16]` slice: the
    /// mapping has no alignment guarantee, and a byteswap on big-endian hosts
    /// would be needed anyway. The cost is trivial against a forward pass.
    pub fn tokens(&self, range: std::ops::Range<usize>) -> Vec<u32> {
        assert!(
            range.end <= self.meta.n_tokens,
            "range {range:?} out of bounds for {} tokens",
            self.meta.n_tokens
        );
        self.mmap[range.start * 2..range.end * 2]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as u32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_shard(dir: &Path, docs: &[(&str, &[u32])]) -> (PathBuf, ShardMeta) {
        let bin = dir.join("train.bin");
        let mut w = ShardWriter::create(&bin, 8192, 0).unwrap();
        for (text, tokens) in docs {
            w.push_document(text, tokens).unwrap();
        }
        let meta = w.finish().unwrap();
        (bin, meta)
    }

    #[test]
    fn roundtrips_tokens_through_the_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, _) = write_shard(dir.path(), &[("a b", &[1, 2, 3]), ("c", &[4])]);
        let shard = Shard::open(&bin).unwrap();
        // Two documents, each followed by its EOS separator.
        assert_eq!(shard.tokens(0..shard.len()), vec![1, 2, 3, 0, 4, 0]);
    }

    #[test]
    fn separators_count_as_tokens_but_not_as_words_or_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let (_, meta) = write_shard(dir.path(), &[("one two", &[1, 2]), ("three", &[3])]);
        assert_eq!(
            meta.n_tokens, 5,
            "three document tokens plus two separators"
        );
        assert_eq!(meta.n_words, 3);
        assert_eq!(meta.n_bytes, "one two".len() + "three".len());
    }

    /// The perplexity denominators are the point of the sidecar, so they have to
    /// survive a write/read cycle exactly.
    #[test]
    fn metadata_survives_a_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, written) = write_shard(dir.path(), &[("hello world", &[7, 8])]);
        assert_eq!(Shard::open(&bin).unwrap().meta(), &written);
    }

    #[test]
    fn a_bin_that_disagrees_with_its_sidecar_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (bin, mut meta) = write_shard(dir.path(), &[("a", &[1])]);
        meta.n_tokens += 100;
        std::fs::write(meta_path(&bin), serde_json::to_string(&meta).unwrap()).unwrap();

        let err = Shard::open(&bin).unwrap_err().to_string();
        assert!(err.contains("out of sync"), "{err}");
    }

    #[test]
    fn a_token_too_large_for_u16_is_rejected_rather_than_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ShardWriter::create(&dir.path().join("t.bin"), 8192, 0).unwrap();
        // 65536 would wrap to 0 -- which is the EOS id -- if written blindly.
        let err = w.push_document("x", &[65_536]).unwrap_err().to_string();
        assert!(err.contains("u16"), "{err}");
    }

    #[test]
    fn word_counting_matches_the_published_convention() {
        // Whitespace-delimited, nothing cleverer: punctuation attached to a word
        // is part of it, and runs of whitespace collapse.
        assert_eq!(count_words("the quick   brown\nfox ,"), 5);
        assert_eq!(count_words(""), 0);
        assert_eq!(count_words("   \n  "), 0);
    }
}
