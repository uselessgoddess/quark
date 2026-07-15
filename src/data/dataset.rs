//! Windows over a shard, and the batcher that turns them into tensors.
//!
//! Training and evaluation want different window layouts, and conflating them
//! is a classic way to report a perplexity that is not comparable to anyone
//! else's:
//!
//! * **Training** wants disjoint windows (`stride == seq_len`): every token is
//!   seen once per epoch, so an epoch means what it says.
//! * **Evaluation** wants overlapping windows (`stride < seq_len`), because the
//!   first token of a window is predicted from no context at all and the second
//!   from one token. With `stride == seq_len` a 1024-token window spends its
//!   early positions guessing nearly blind, which inflates perplexity by an
//!   amount that depends on `seq_len` -- i.e. on a choice, not on the model.
//!   Striding scores each token with at least `seq_len - stride` tokens of
//!   context and is what GPT-2's own evaluation does.
//!
//! The stride is therefore explicit and identical for quark and for the GPT-2
//! baseline; otherwise the comparison `docs/DESIGN.md` §3 rests on is not
//! controlled.

use std::sync::Arc;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Device, Int, Tensor, TensorData},
};

use crate::data::shard::Shard;

/// One training window: `input[i]` is predicted to be `target[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWindow {
    pub input: Vec<u32>,
    pub target: Vec<u32>,
    /// Positions to score. Everything before this is context whose loss is
    /// discarded because the striding already scored it with more context.
    /// `0` during training, where windows do not overlap.
    pub score_from: usize,
}

/// A shard sliced into fixed-length windows.
pub struct TokenDataset {
    shard: Arc<Shard>,
    seq_len: usize,
    stride: usize,
}

impl TokenDataset {
    /// Disjoint windows: the layout for training.
    pub fn train(shard: Arc<Shard>, seq_len: usize) -> Self {
        Self::with_stride(shard, seq_len, seq_len)
    }

    /// # Panics
    /// If `stride` is zero, or exceeds `seq_len` (which would skip tokens
    /// entirely rather than merely overlap less).
    pub fn with_stride(shard: Arc<Shard>, seq_len: usize, stride: usize) -> Self {
        assert!(seq_len > 0, "seq_len must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(
            stride <= seq_len,
            "stride {stride} exceeds seq_len {seq_len}: tokens between windows would never be scored"
        );
        Self {
            shard,
            seq_len,
            stride,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Total positions this dataset scores, summed over windows.
    ///
    /// With `stride == seq_len` this is every token but the first: token 0 is
    /// only ever an input, since nothing precedes it to predict it from. That
    /// off-by-one is why the evaluator must use *this* count and not
    /// `shard.meta().n_tokens` as its bookkeeping check.
    pub fn n_scored_tokens(&self) -> usize {
        (0..self.len())
            .map(|i| self.seq_len - self.score_from(i))
            .sum()
    }

    /// Start offset of window `i` in the shard.
    fn start(&self, i: usize) -> usize {
        i * self.stride
    }

    /// First position of window `i` that has not already been scored by window
    /// `i - 1`, so that overlapping windows score each token exactly once.
    fn score_from(&self, i: usize) -> usize {
        if i == 0 {
            0
        } else {
            self.seq_len - self.stride
        }
    }
}

impl Dataset<TokenWindow> for TokenDataset {
    fn len(&self) -> usize {
        // Each window needs `seq_len + 1` tokens: `seq_len` inputs and the
        // target for the last of them. A window that would run past the end is
        // dropped rather than padded -- padding would put tokens in the loss
        // that the corpus does not contain.
        let n = self.shard.len();
        if n < self.seq_len + 1 {
            return 0;
        }
        (n - self.seq_len - 1) / self.stride + 1
    }

    fn get(&self, index: usize) -> Option<TokenWindow> {
        if index >= self.len() {
            return None;
        }
        let start = self.start(index);
        let raw = self.shard.tokens(start..start + self.seq_len + 1);
        Some(TokenWindow {
            input: raw[..self.seq_len].to_vec(),
            target: raw[1..].to_vec(),
            score_from: self.score_from(index),
        })
    }
}

/// A batch on the device: `[batch, seq]` ids, plus a mask marking which
/// positions contribute to the loss.
#[derive(Debug, Clone)]
pub struct TokenBatch<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub target: Tensor<B, 2, Int>,
    /// `1` where the position is scored, `0` where it is context carried only
    /// to condition later positions. All ones during training.
    pub score_mask: Tensor<B, 2>,
}

#[derive(Clone, Default)]
pub struct TokenBatcher;

impl<B: Backend> Batcher<B, TokenWindow, TokenBatch<B>> for TokenBatcher {
    fn batch(&self, items: Vec<TokenWindow>, device: &Device<B>) -> TokenBatch<B> {
        let batch = items.len();
        let seq = items[0].input.len();
        assert!(
            items.iter().all(|w| w.input.len() == seq),
            "windows in a batch must share a length"
        );

        let flat = |f: fn(&TokenWindow) -> &Vec<u32>| {
            let data: Vec<i32> = items
                .iter()
                .flat_map(|w| f(w).iter().map(|&t| t as i32))
                .collect();
            Tensor::<B, 2, Int>::from_data(TensorData::new(data, [batch, seq]), device)
        };

        let mask: Vec<f32> = items
            .iter()
            .flat_map(|w| (0..seq).map(move |p| if p >= w.score_from { 1.0 } else { 0.0 }))
            .collect();

        TokenBatch {
            input: flat(|w| &w.input),
            target: flat(|w| &w.target),
            score_mask: Tensor::<B, 2>::from_data(TensorData::new(mask, [batch, seq]), device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shard::ShardWriter;
    use crate::test_util::TestBackend;
    use std::path::Path;

    /// A shard whose tokens are `1..=n`, so a window's contents identify its
    /// offset by inspection.
    fn counting_shard(dir: &Path, n: usize) -> Arc<Shard> {
        let bin = dir.join("t.bin");
        let mut w = ShardWriter::create(&bin, 8192, 0).unwrap();
        // One "document" with no trailing separator would be ideal, but the
        // writer always appends EOS; ask for n-1 tokens so the total is n.
        let tokens: Vec<u32> = (1..n as u32).collect();
        w.push_document("x", &tokens).unwrap();
        w.finish().unwrap();
        Arc::new(Shard::open(&bin).unwrap())
    }

    #[test]
    fn training_windows_are_disjoint_and_shifted_by_one() {
        let dir = tempfile::tempdir().unwrap();
        let ds = TokenDataset::train(counting_shard(dir.path(), 13), 4);

        // 13 tokens, windows of 4+1: starts at 0, 4, 8 -- the window at 12 would
        // need tokens 12..17.
        assert_eq!(ds.len(), 3);

        let w0 = ds.get(0).unwrap();
        assert_eq!(w0.input, vec![1, 2, 3, 4]);
        assert_eq!(
            w0.target,
            vec![2, 3, 4, 5],
            "target is input shifted by one"
        );
        assert_eq!(w0.score_from, 0);

        let w1 = ds.get(1).unwrap();
        assert_eq!(w1.input, vec![5, 6, 7, 8]);
        // Window 0 ends predicting 5; window 1 starts predicting 6. Consecutive
        // windows' targets are contiguous, so nothing is scored twice or missed.
        assert_eq!(w1.target, vec![6, 7, 8, 9]);
    }

    #[test]
    fn disjoint_windows_score_every_token_but_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let n = 13;
        let ds = TokenDataset::train(counting_shard(dir.path(), n), 4);

        let scored: Vec<u32> = (0..ds.len())
            .flat_map(|i| ds.get(i).unwrap().target)
            .collect();
        // 12 windows' worth of targets = tokens 2..=13, i.e. every token except
        // token 0, each exactly once.
        assert_eq!(
            scored,
            (2..=12).chain(std::iter::once(0)).collect::<Vec<_>>()
        );
        assert_eq!(ds.n_scored_tokens(), scored.len());
        assert_eq!(ds.n_scored_tokens(), n - 1);
    }

    #[test]
    fn strided_windows_overlap_but_score_each_token_once() {
        let dir = tempfile::tempdir().unwrap();
        let ds = TokenDataset::with_stride(counting_shard(dir.path(), 13), 4, 2);

        let w0 = ds.get(0).unwrap();
        assert_eq!(
            w0.score_from, 0,
            "the first window has no earlier context to defer to"
        );

        let w1 = ds.get(1).unwrap();
        assert_eq!(w1.input, vec![3, 4, 5, 6], "stride 2 moves the window by 2");
        // Positions 0..2 of this window were already scored by window 0 with
        // more context; only 2..4 are new.
        assert_eq!(w1.score_from, 2);

        // Every scored target across the dataset, in order, must still be each
        // token exactly once -- overlap must not double-count.
        let mut scored = Vec::new();
        for i in 0..ds.len() {
            let w = ds.get(i).unwrap();
            scored.extend_from_slice(&w.target[w.score_from..]);
        }
        let mut sorted = scored.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            scored.len(),
            "a token was scored twice: {scored:?}"
        );
        assert_eq!(ds.n_scored_tokens(), scored.len());
    }

    #[test]
    fn a_shard_shorter_than_one_window_yields_nothing() {
        let dir = tempfile::tempdir().unwrap();
        // 4 tokens cannot fill a window of 4 inputs plus its final target.
        let ds = TokenDataset::train(counting_shard(dir.path(), 4), 4);
        assert_eq!(ds.len(), 0);
        assert!(ds.get(0).is_none());
    }

    #[test]
    #[should_panic(expected = "never be scored")]
    fn a_stride_wider_than_the_window_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        TokenDataset::with_stride(counting_shard(dir.path(), 16), 4, 5);
    }

    #[test]
    fn batcher_stacks_windows_and_marks_scored_positions() {
        let d = Default::default();
        let windows = vec![
            TokenWindow {
                input: vec![1, 2, 3],
                target: vec![2, 3, 4],
                score_from: 0,
            },
            TokenWindow {
                input: vec![5, 6, 7],
                target: vec![6, 7, 8],
                score_from: 2,
            },
        ];
        let b: TokenBatch<TestBackend> = TokenBatcher.batch(windows, &d);

        assert_eq!(b.input.dims(), [2, 3]);
        assert_eq!(b.target.dims(), [2, 3]);
        // Non-strict: the backend picks its own Int width (i64 on NdArray), and
        // the batch's correctness is in the values, not in that choice.
        b.input
            .to_data()
            .assert_eq(&TensorData::from([[1i32, 2, 3], [5, 6, 7]]), false);
        b.score_mask.to_data().assert_eq(
            &TensorData::from([[1.0f32, 1.0, 1.0], [0.0, 0.0, 1.0]]),
            false,
        );
    }
}
