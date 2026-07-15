//! The output of one training or validation step, and the masked loss that
//! produces it.
//!
//! # Why not burn's `SequenceOutput`
//!
//! burn ships `SequenceOutput<B>`, which carries the full `[batch, seq, vocab]`
//! logits and adapts to `PerplexityInput`. It is the obvious choice and it is
//! the wrong one here, for a reason that is structural rather than stylistic.
//!
//! Every metric input is produced from `<Output as ItemLazy>::ItemSync`, and
//! `sync()` registers its tensors in a [`Transaction`] -- i.e. it *reads them
//! back to the host*. `SequenceOutput::sync` reads back the logits. At this
//! project's shapes (batch 8 x seq 512 x vocab 8192, f32) that is 134 MB copied
//! off the GPU **per step**, after which `PerplexityMetric::update` runs a
//! 33.5M-element `log_softmax` on the CPU, also per step. The metric thread
//! would fall permanently behind the training loop and its queue would grow
//! without bound.
//!
//! So [`LmOutput`] carries three scalars instead. All three fall out of the
//! loss for free, `sync()` moves 12 bytes, and the perplexity metric becomes
//! two additions (see [`crate::train::metric`]). The logits never leave the
//! device.

use burn::{
    backend::Flex,
    prelude::Backend,
    tensor::{activation::log_softmax, Int, Tensor, Transaction},
    train::{
        metric::{Adaptor, LossInput},
        ItemLazy,
    },
};

use crate::train::metric::TokenPerplexityInput;

/// What one step reports: the scalar the optimizer descends, plus the two
/// running totals perplexity is built from.
///
/// `sum_nll` and `n_tokens` are kept separately rather than folded into `loss`
/// because perplexity does not average. `exp(mean_of_batch_means)` is not
/// `exp(total_nll / total_tokens)` unless every batch scores exactly the same
/// number of tokens -- which is false for the final partial batch of an epoch,
/// and false throughout for strided evaluation.
#[derive(Debug)]
pub struct LmOutput<B: Backend> {
    /// Mean negative log-likelihood over the scored tokens of this batch.
    pub loss: Tensor<B, 1>,
    /// Negative log-likelihood summed over this batch's scored tokens, in nats.
    pub sum_nll: Tensor<B, 1>,
    /// How many tokens that sum covers.
    pub n_tokens: Tensor<B, 1>,
}

/// Cross-entropy over the positions `score_mask` marks.
///
/// Hand-rolled rather than `CrossEntropyLoss` for two reasons: the mask (burn's
/// loss takes a pad token, not a per-position mask), and `sum_nll`, which burn's
/// loss reduces away before returning.
///
/// # Panics
/// If the three tensors disagree on batch or sequence length.
pub fn masked_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    score_mask: Tensor<B, 2>,
) -> LmOutput<B> {
    let [batch, seq, vocab] = logits.dims();
    assert_eq!(targets.dims(), [batch, seq], "targets must match logits");
    assert_eq!(score_mask.dims(), [batch, seq], "mask must match logits");

    let n = batch * seq;

    // log_softmax rather than log(softmax(..)): the former subtracts the row max
    // first, so a logit of 40 does not overflow the exponential. At 3M
    // parameters the logits stay small, but the loss is not the place to rely on
    // that.
    let log_probs = log_softmax(logits.reshape([n, vocab]), 1);
    let picked = log_probs.gather(1, targets.reshape([n, 1])).reshape([n]);

    let mask = score_mask.reshape([n]);
    let sum_nll = picked.neg().mul(mask.clone()).sum();
    let n_tokens = mask.sum();

    // A batch with no scored positions cannot arise from the loaders in
    // `crate::train::run` (training windows are disjoint, so every position is
    // scored), but 0/0 is NaN and a single NaN gradient destroys the model
    // irrecoverably. The clamp costs one kernel and removes the failure mode.
    let loss = sum_nll.clone() / n_tokens.clone().clamp_min(1.0);

    LmOutput {
        loss,
        sum_nll,
        n_tokens,
    }
}

impl<B: Backend> ItemLazy for LmOutput<B> {
    // Flex is burn's own CPU backend, and burn-train depends on it
    // unconditionally -- see `burn-train/Cargo.toml`. Mirroring the choice burn
    // makes for `SequenceOutput` also lets `LossMetric::new()` infer its backend
    // parameter at the registration site, as it does in burn's examples.
    type ItemSync = LmOutput<Flex>;

    fn sync(self) -> Self::ItemSync {
        let device = &Default::default();

        // One transaction rather than three `into_data()` calls: it reads all
        // three back in a single round trip instead of stalling the queue three
        // times.
        let [loss, sum_nll, n_tokens] = Transaction::default()
            .register(self.loss)
            .register(self.sum_nll)
            .register(self.n_tokens)
            .execute()
            .try_into()
            .expect("three registered tensors yield three data buffers");

        LmOutput {
            loss: Tensor::from_data(loss, device),
            sum_nll: Tensor::from_data(sum_nll, device),
            n_tokens: Tensor::from_data(n_tokens, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LmOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<TokenPerplexityInput<B>> for LmOutput<B> {
    fn adapt(&self) -> TokenPerplexityInput<B> {
        TokenPerplexityInput::new(self.sum_nll.clone(), self.n_tokens.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::{ElementConversion, TensorData};

    fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
        t.into_scalar().elem::<f32>()
    }

    /// A uniform distribution over `V` classes assigns every token
    /// `log(1/V)`, so the loss must be exactly `ln V`. This pins the
    /// normalization: an implementation that forgot the softmax, or divided by
    /// the wrong count, would not land here.
    #[test]
    fn uniform_logits_cost_exactly_ln_vocab() {
        let d = Default::default();
        let vocab = 8;
        let logits = Tensor::<TestBackend, 3>::zeros([2, 3, vocab], &d);
        let targets = Tensor::<TestBackend, 2, Int>::zeros([2, 3], &d);
        let mask = Tensor::<TestBackend, 2>::ones([2, 3], &d);

        let out = masked_cross_entropy(logits, targets, mask);

        let expected = (vocab as f32).ln();
        assert!((scalar(out.loss) - expected).abs() < 1e-5);
        assert_eq!(scalar(out.n_tokens), 6.0);
        // Six tokens, each costing ln 8.
        assert!((scalar(out.sum_nll) - 6.0 * expected).abs() < 1e-4);
    }

    /// The mask is the whole reason this function exists rather than
    /// `CrossEntropyLoss`. A masked position must contribute to neither the
    /// numerator nor the denominator -- if it only left the numerator, the loss
    /// would be silently deflated by the fraction of masked tokens.
    #[test]
    fn masked_positions_contribute_to_neither_sum_nor_count() {
        let d = Default::default();
        let vocab = 4;
        // Position 0 of each row is confidently right; position 1 is confidently
        // wrong. Masking position 1 must leave only the cheap positions.
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, // row 0, pos 0: target 0, right
                    0.0, 0.0, 0.0, 10.0, // row 0, pos 1: target 0, wrong
                ],
                [1, 2, vocab],
            ),
            &d,
        );
        let targets = Tensor::<TestBackend, 2, Int>::zeros([1, 2], &d);

        let all = masked_cross_entropy(
            logits.clone(),
            targets.clone(),
            Tensor::<TestBackend, 2>::ones([1, 2], &d),
        );
        let first_only = masked_cross_entropy(
            logits,
            targets,
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0f32, 0.0]]), &d),
        );

        assert_eq!(scalar(all.n_tokens), 2.0);
        assert_eq!(scalar(first_only.n_tokens), 1.0);

        let (all_loss, masked_loss) = (scalar(all.loss), scalar(first_only.loss));
        // Dropping the expensive position must move the *mean*, not just the
        // sum: the denominator has to follow the mask.
        assert!(
            masked_loss < all_loss / 2.0,
            "masking the wrong prediction should leave only the cheap one: \
             {masked_loss} vs {all_loss}"
        );
        assert!(scalar(first_only.sum_nll) < 0.01);
    }

    /// `loss` and `sum_nll / n_tokens` must agree, because the live perplexity
    /// metric reads the latter while checkpoint selection reads the former. If
    /// they drift, the two numbers on the dashboard describe different models.
    #[test]
    fn loss_is_sum_nll_over_n_tokens() {
        let d = Default::default();
        let logits = Tensor::<TestBackend, 3>::random(
            [3, 5, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &d,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[1i32, 2, 3, 4, 5], [0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]),
            &d,
        );
        let mask = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.0f32, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ]),
            &d,
        );

        let out = masked_cross_entropy(logits, targets, mask);
        let recomputed = scalar(out.sum_nll) / scalar(out.n_tokens.clone());
        assert_eq!(scalar(out.n_tokens), 12.0);
        assert!((scalar(out.loss) - recomputed).abs() < 1e-5);
    }

    /// An all-masked batch is unreachable through the training loaders, but the
    /// clamp that makes it survivable is easy to delete by accident. NaN in a
    /// loss is not recoverable: it propagates into every parameter on the next
    /// optimizer step.
    #[test]
    fn an_entirely_masked_batch_yields_zero_rather_than_nan() {
        let d = Default::default();
        let out = masked_cross_entropy(
            Tensor::<TestBackend, 3>::zeros([1, 2, 4], &d),
            Tensor::<TestBackend, 2, Int>::zeros([1, 2], &d),
            Tensor::<TestBackend, 2>::zeros([1, 2], &d),
        );
        assert_eq!(scalar(out.loss), 0.0);
        assert_eq!(scalar(out.n_tokens), 0.0);
    }
}
