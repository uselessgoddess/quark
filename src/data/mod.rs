//! Corpus preparation: text on disk -> tokenizer -> shards -> batches.
//!
//! The pipeline is deliberately offline and file-backed. Tokenizing during
//! training would make every epoch pay the BPE cost again, and -- worse -- would
//! make the word and byte counts that the evaluation denominators need
//! unavailable at eval time. Shards are written once and then read-only.

pub mod dataset;
pub mod shard;
pub mod tokenizer;

pub use dataset::{TokenBatch, TokenBatcher, TokenDataset, TokenWindow};
pub use shard::{count_words, Shard, ShardMeta, ShardWriter};

use std::path::Path;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

/// Documents in a WikiText-style corpus.
///
/// WikiText ships as one flat file whose articles are delimited by ` = Title = `
/// headings. Splitting on them matters: the EOS separator tells the model where
/// a document ends, and without it the model learns that articles run into each
/// other, which is a claim about the data that is false.
pub fn split_wikitext_articles(text: &str) -> Vec<&str> {
    let mut articles = Vec::new();
    let mut start = 0;
    let mut cursor = 0;

    for line in text.split_inclusive('\n') {
        // WikiText spaces out its markup: an article heading is ` = Title = `
        // and a section heading is ` = = Gameplay = = `. So heading level is the
        // number of *space-separated* leading `=`, and only level 1 starts a new
        // document. Testing `starts_with("==")` would be right for raw wikitext
        // and wrong here -- `= = Gameplay = =` does not contain `==` at all.
        let t = line.trim();
        let is_article_heading = t.starts_with("= ") && t.ends_with(" =") && !t.starts_with("= = ");
        if is_article_heading && cursor > start {
            articles.push(text[start..cursor].trim());
            start = cursor;
        }
        cursor += line.len();
    }
    if cursor > start {
        articles.push(text[start..cursor].trim());
    }
    articles.retain(|a| !a.is_empty());
    articles
}

/// Tokenize `input` into a shard at `out`.
///
/// Returns the shard's metadata, whose `n_words` is worth checking against the
/// published corpus statistics -- a silent change in how the text is split is
/// otherwise invisible until the reported perplexity is already wrong.
pub fn prepare_shard(
    input: &Path,
    out: &Path,
    tokenizer: &tokenizers::Tokenizer,
    split_articles: bool,
) -> Result<ShardMeta> {
    let text = std::fs::read_to_string(input)
        .with_context(|| format!("reading corpus {}", input.display()))?;

    let docs = if split_articles {
        split_wikitext_articles(&text)
    } else {
        vec![text.as_str()]
    };

    let eos = tokenizer::eos_id(tokenizer)?;
    let vocab = tokenizer.get_vocab_size(true);
    let mut writer = ShardWriter::create(out, vocab, eos)?;

    let bar = ProgressBar::new(docs.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{bar:40} {pos}/{len} documents {msg}")
            .expect("static template"),
    );
    for doc in &docs {
        let ids = tokenizer::encode(tokenizer, doc)?;
        writer.push_document(doc, &ids)?;
        bar.inc(1);
    }
    bar.finish_and_clear();

    let meta = writer.finish()?;
    tracing::info!(
        tokens = meta.n_tokens,
        words = meta.n_words,
        bytes = meta.n_bytes,
        // The compression ratio is the single most useful sanity check on a
        // tokenizer: byte-level BPE at this vocab should land near 4 bytes per
        // token on English prose. Far below means the merges failed to form.
        bytes_per_token = meta.n_bytes as f64 / meta.n_tokens as f64,
        "prepared shard {}",
        out.display()
    );
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Abridged, but shaped exactly like the real file: leading blank line,
    /// space-padded headings, and section headings that must not split.
    const WIKITEXT: &str = "\n = Valkyria Chronicles III = \n\n Senjō no Valkyria 3 is a game .\n\n = = Gameplay = = \n\n It is a tactical RPG .\n\n = Tower Building = \n\n The Tower Building of the Little Rock Arsenal .\n";

    #[test]
    fn splits_on_article_headings_only() {
        let articles = split_wikitext_articles(WIKITEXT);
        assert_eq!(articles.len(), 2, "got {articles:#?}");
        assert!(articles[0].starts_with("= Valkyria Chronicles III ="));
        // The section heading stays inside its article rather than starting a
        // new one.
        assert!(articles[0].contains("= = Gameplay = ="));
        assert!(articles[1].starts_with("= Tower Building ="));
    }

    #[test]
    fn splitting_preserves_every_word() {
        // Splitting must only insert boundaries, never drop text -- the word
        // count is a perplexity denominator.
        let total: usize = split_wikitext_articles(WIKITEXT)
            .iter()
            .map(|a| count_words(a))
            .sum();
        assert_eq!(total, count_words(WIKITEXT));
    }

    #[test]
    fn text_without_headings_is_a_single_document() {
        let articles = split_wikitext_articles("just some prose\nover two lines\n");
        assert_eq!(articles, vec!["just some prose\nover two lines"]);
    }

    #[test]
    fn empty_input_yields_no_documents() {
        assert!(split_wikitext_articles("").is_empty());
        assert!(split_wikitext_articles("\n\n  \n").is_empty());
    }
}
