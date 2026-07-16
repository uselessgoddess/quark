#!/usr/bin/env python3
"""What the two WikiText-103 runs actually measured.

Every number in docs/RESULTS.md comes from this script. Run it and the doc is
reproducible:

    python3 experiments/analyze_runs.py

The inputs are the burn `Learner` metric logs from the two runs reported in
issue #3, checked in under `experiments/runs/`:

    quark_3m_loop12/   1 unique layer x 12 loops, d_model 384  (the reference)
    quark_3m_dense/    6 unique layers x 1,     d_model 168  (the control)

Both are one epoch over the same shard, same seed, same schedule: 16,444
batches, 134,705,152 tokens.

Section 0 proves these logs are the runs the issue reports, by reconstructing
its summary table from them exactly.

Why this script exists
----------------------

The issue reports the runs through burn's Learner summary table, which prints
the EPOCH MEAN of the training loss. For a single-epoch run that statistic is
dominated by the first few thousand batches, when the model is still undoing its
initialization -- it is not a measurement of the trained model. The summary says
the loop model is 0.250 nats worse; at the end of the epoch it is 0.055 nats
worse. Reading the first number as if it were the second is the single easiest
way to misread these runs, and it changes the conclusion's magnitude by 4.6x.

So this script never reports an epoch mean without also reporting the terminal
window, and the gap trajectory that explains the difference between them.

What it deliberately does NOT do
--------------------------------

It does not fit L(D) = E + B/D^beta to a curve and extrapolate. Both runs are
cosine-annealed, so the loss at token t reflects the LEARNING RATE at token t as
much as the data seen by then. A within-run curve is not the envelope of
separately-annealed runs, and fitting one as if it were overstates what more
data buys. The honest data-efficiency statement available from these logs is the
equivalence point (section 3), which is a comparison between two curves under
the SAME schedule and so does not need the schedule to be modelled at all.
"""

from __future__ import annotations

import math
import pathlib

RUNS = pathlib.Path(__file__).parent / "runs"

# MEASURED. Transcribed from the `quark eval` output in issue #3. Both runs
# scored 320,000 tokens of artifacts/test.bin, 99.9925% of the corpus.
#
# The denominators are a property of the text alone and are identical for both
# runs and for GPT-2 -- see docs/DESIGN.md §3.1. They are verified against the
# real wiki.test.tokens (1,281,077 bytes, sha256 in docs/RESULTS.md): the words
# match exactly, and the byte count is 438 lower because `--split-articles`
# trims each of the 60 documents.
EVAL = {
    #                 word_ppl  bits_per_byte  total_nll  token_ppl  blimp
    "quark_3m_loop12": (115.163, 1.2897, 1144870.9, 35.792, 0.5705),
    "quark_3m_dense": (108.275, 1.2730, 1129995.6, 34.166, 0.5863),
}
EVAL_SCORED_TOKENS = 320_000
EVAL_WORDS = 241_211
EVAL_BYTES = 1_280_639

# DERIVED from src/config.rs::compute_equivalent_params, asserted in Rust
# against the constructed burn module by
# `lm.rs::analytic_budget_matches_the_real_module`. param_count is what the 3.0M
# budget constrains; compute_equivalent_params is what FLOPs and wall-clock
# track. Weight sharing drives these apart -- that is the entire point of it,
# and the entire cost.
CONFIGS = {
    #                  params    compute_equiv  d_model  unique  loops
    "quark_3m_loop12": (2_868_352, 20_643_840, 384, 1, 12),
    "quark_3m_dense": (2_871_880, 1_778_112, 168, 6, 1),
}

# MEASURED. Wall-clock and peak VRAM as reported by the issue author on a 16GB
# card. Approximate by their own account ("~11GB", "hour+" vs "~5.5GB", "15
# minutes"); used only for order-of-magnitude statements.
WALL_CLOCK_MIN = {"quark_3m_loop12": 60.0, "quark_3m_dense": 15.0}
VRAM_GB = {"quark_3m_loop12": 11.0, "quark_3m_dense": 5.5}

# MEASURED. Radford et al. 2019, Table 3: GPT-2 124M, WikiText-103, zero-shot,
# word-level PPL 37.50. Computed after an invertible de-tokenizer that was never
# released and that OpenAI values at 2.5-5 PPL, so it is NOT reproducible and
# NOT directly comparable -- docs/DESIGN.md §3.1. Carried here only to be shown
# next to the self-measured number, never to be subtracted from one.
GPT2_PUBLISHED_WT103_WORD_PPL = 37.50

# MEASURED. The Learner summary tables transcribed verbatim from issue #3. Every
# train-split row is reconstructed from the checked-in logs in section 0; that
# is what ties these files to those runs. The valid rows come from a pass the
# logs do not contain and are carried for corroboration only.
SUMMARY_REPORTED = {
    #                  lr       loss   ppl      valid_loss  valid_ppl
    "quark_3m_loop12": (1.632e-3, 4.515, 168.960, 3.706, 40.161),
    "quark_3m_dense": (1.632e-3, 4.266, 114.049, 3.653, 38.387),
}


def load(run: str, metric: str):
    """One `(value, count)` per batch, as burn serializes a `NumericEntry`.

    Both variants serialize to `value,count`, and the count is the whole reason
    to read the second column carefully:

    * `Loss` and `Learning_Rate` are `NumericEntry::Value`, whose count is
      always 1. Their second column carries no information at all.
    * `Perplexity` is `NumericEntry::Aggregated` -- see
      `src/train/metric.rs::TokenPerplexityMetric::entry`, which sets
      `count: self.total_tokens` so that burn weights each entry by the tokens
      behind it. Its count is therefore the CUMULATIVE token total.

    So the run's only token axis lives in `Perplexity.log`, and `token_axis()`
    is the only reader of a count. Taking `Loss.log`'s second column for a token
    count silently yields the constant 1.
    """
    path = RUNS / run / f"{metric}.log"
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value, count = line.split(",")
        out.append((float(value), int(count)))
    return out


def token_axis(run: str):
    """Cumulative tokens seen after each batch, from the perplexity counts."""
    return [count for _, count in load(run, "Perplexity")]


def burn_summary_aggregate(entries) -> float:
    """The statistic burn's summary table prints: a count-weighted mean.

    For `Loss` and `Learning_Rate` every count is 1, so this is the plain epoch
    mean. For `Perplexity` the counts are cumulative token totals, so late
    entries outweigh early ones ~16,000:1 -- which is why the reported
    perplexity (168.960) is neither the mean of the logged series (357.311) nor
    exp(mean loss) (91.42). It is not a quantity with an interpretation; it is
    an artifact of weighting a running aggregate by a running count. Reproduced
    here to identify the runs, not because it means anything.
    """
    return sum(v * c for v, c in entries) / sum(c for _, c in entries)


def window_mean(series, lo: float, hi: float) -> float:
    """Mean of `series` over the fractional slice [lo, hi) of training."""
    chunk = series[int(len(series) * lo) : int(len(series) * hi)]
    return sum(v for v, _ in chunk) / len(chunk)


def first_crossing(series, axis, target: float, smooth: int = 400):
    """Tokens seen when a `smooth`-batch trailing mean first reaches `target`.

    Trailing mean rather than raw loss: per-batch loss on a 3M model is noisy
    enough that a single lucky batch crosses any threshold early.
    """
    total = 0.0
    for i, (v, _) in enumerate(series):
        total += v
        if i >= smooth:
            total -= series[i - smooth][0]
            if total / smooth <= target:
                return axis[i], i
    return None, None


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    loss = {r: load(r, "Loss") for r in CONFIGS}
    lr = {r: load(r, "Learning_Rate") for r in CONFIGS}
    ppl = {r: load(r, "Perplexity") for r in CONFIGS}
    axis = {r: token_axis(r) for r in CONFIGS}

    rule("0a. These logs are the runs the issue reports")
    print(
        "  Reconstructing the issue's Learner summary table from the checked-in\n"
        "  logs, using burn's own aggregate (count-weighted mean):\n"
    )
    print(f"  {'run':18} {'metric':14} {'reported':>10} {'rebuilt':>10}   ok")
    ok = True
    for r in CONFIGS:
        for label, series, reported in (
            ("Learning Rate", lr[r], SUMMARY_REPORTED[r][0]),
            ("Loss", loss[r], SUMMARY_REPORTED[r][1]),
            ("Perplexity", ppl[r], SUMMARY_REPORTED[r][2]),
        ):
            got = burn_summary_aggregate(series)
            hit = abs(got - reported) <= 5e-4 * abs(reported)
            ok &= hit
            print(
                f"  {r:18} {label:14} {reported:10.6g} {got:10.6g}   "
                f"{'yes' if hit else 'NO'}"
            )
    print(f"\n  all six train-split rows reproduce: {ok}")
    print(
        "\n  Worth noting what that table's `Perplexity` row is, since it looks like\n"
        "  a headline number and is not one: burn weights each logged entry by its\n"
        "  count, and this metric's count is the CUMULATIVE token total, so the\n"
        "  reported 168.960 is a token-weighted mean of a running average. It is\n"
        "  neither exp(epoch-mean loss) (91.42) nor the mean of the logged series\n"
        "  (357.31). Nothing downstream uses it; section 1 uses the loss."
    )

    rule("0b. The two runs are the same experiment except for the architecture")
    for r in CONFIGS:
        n, tokens = len(loss[r]), axis[r][-1]
        peak, final = max(v for v, _ in lr[r]), lr[r][-1][0]
        print(
            f"  {r:18} batches {n:6}  tokens {tokens:,}  "
            f"lr peak {peak:.3e} -> final {final:.3e}"
        )
    same_tokens = len({axis[r][-1] for r in CONFIGS}) == 1
    same_batches = len({len(loss[r]) for r in CONFIGS}) == 1
    same_lr = len({round(max(v for v, _ in lr[r]), 12) for r in CONFIGS}) == 1
    same_axis = len({tuple(axis[r]) for r in CONFIGS}) == 1
    print(
        f"\n  identical token count : {same_tokens}\n"
        f"  identical batch count : {same_batches}\n"
        f"  identical peak lr     : {same_lr}\n"
        f"  identical token axis  : {same_axis}   (batch-for-batch, not just the total)"
    )

    # The one place the runs are not bit-identical, run down rather than waved
    # at: 134,705,152 tokens / 512 = 263,096 sequences = 16,443 batches of 16
    # plus one of 8. Both runs contain exactly that one 4,096-token batch, but
    # it lands at batch 16,438 in one run and 16,423 in the other, because
    # `num_workers: 4` interleaves four worker streams and the tail is scheduled
    # nondeterministically. Same data, same volume, same count; the order of the
    # last ~20 batches differs. Immaterial to every statistic here -- and worth
    # knowing, because it means `seed: 42` does NOT make these runs bitwise
    # reproducible, so a re-run will not reproduce 4.5155 to the digit.
    short = {
        r: [i for i in range(1, len(axis[r])) if axis[r][i] - axis[r][i - 1] != 8192]
        for r in CONFIGS
    }
    for r in CONFIGS:
        i = short[r][0]
        print(
            f"  {r:18} one short batch at #{i} ({axis[r][i] - axis[r][i - 1]} tokens); "
            f"all other batches 8192"
        )
    print(
        "  -> the tail is interleaved differently by the 4 dataloader workers.\n"
        "     Same data and same volume, so this changes nothing below, but it does\n"
        "     mean seed 42 alone does not make a run bitwise reproducible."
    )
    print(
        "\n  So data, steps and schedule are controlled. What is NOT controlled is\n"
        "  width: the 3.0M budget is fixed, so 6 unique layers must be paid for by\n"
        "  narrowing d_model 384 -> 168. This is a comparison of two ways to spend\n"
        "  one budget, not of one model with sharing switched off."
    )

    rule("1. The epoch mean is an artifact. The terminal loss is the measurement")
    print(f"  {'run':18} {'epoch mean':>12} {'final 5%':>12}")
    for r in CONFIGS:
        mean = sum(v for v, _ in loss[r]) / len(loss[r])
        print(f"  {r:18} {mean:12.4f} {window_mean(loss[r], 0.95, 1.0):12.4f}")
    d_mean = sum(v for v, _ in loss["quark_3m_loop12"]) / len(loss["quark_3m_loop12"]) - sum(
        v for v, _ in loss["quark_3m_dense"]
    ) / len(loss["quark_3m_dense"])
    d_final = window_mean(loss["quark_3m_loop12"], 0.95, 1.0) - window_mean(
        loss["quark_3m_dense"], 0.95, 1.0
    )
    print(
        f"\n  gap by epoch mean : {d_mean:+.4f} nats   <- what the issue's summary table shows\n"
        f"  gap at the end    : {d_final:+.4f} nats   <- what the trained models differ by\n"
        f"  the summary overstates the real gap by {d_mean / d_final:.1f}x."
    )
    print(
        "\n  Corroboration, from numbers the summary did not average:\n"
        "    valid loss (end of epoch)  : loop 3.706  dense 3.653   gap +0.053\n"
        f"    test NLL/token             : loop {EVAL['quark_3m_loop12'][2] / EVAL_SCORED_TOKENS:.4f}  "
        f"dense {EVAL['quark_3m_dense'][2] / EVAL_SCORED_TOKENS:.4f}   "
        f"gap {(EVAL['quark_3m_loop12'][2] - EVAL['quark_3m_dense'][2]) / EVAL_SCORED_TOKENS:+.4f}\n"
        "  Three independent end-of-run measurements agree on ~0.05 nats. The 0.25\n"
        "  figure is the only one that does not, because it is the only one that\n"
        "  averages over the untrained model."
    )

    rule("2. The gap closes monotonically. The loop model is slower, not weaker")
    print(f"  {'window':>10} {'loop12':>9} {'dense':>9} {'gap':>9}")
    for i in range(20):
        lo, hi = i / 20, (i + 1) / 20
        a = window_mean(loss["quark_3m_loop12"], lo, hi)
        b = window_mean(loss["quark_3m_dense"], lo, hi)
        print(f"  {lo * 100:4.0f}-{hi * 100:3.0f}% {a:9.4f} {b:9.4f} {a - b:+9.4f}")
    print(
        "\n  The gap peaks at ~+0.64 nats around 10-15% and falls monotonically to\n"
        "  +0.05. A model that were simply WORSE would hold a constant gap or\n"
        "  diverge. This one is converging toward the control from behind: 12\n"
        "  applications of one layer is a harder optimization problem, and most of\n"
        "  the reported difference is the cost of solving it, not a capacity limit.\n"
        "\n  This is the strongest available argument FOR the loop model, so it is\n"
        "  worth stating its limit: converging toward a control is not overtaking\n"
        "  it. Nothing here shows the curves cross, and section 3 prices what the\n"
        "  loop model pays for the privilege of being 0.05 nats behind."
    )

    rule("3. Data equivalence: how much of the epoch did the control need?")
    target = window_mean(loss["quark_3m_loop12"], 0.95, 1.0)
    tokens, idx = first_crossing(loss["quark_3m_dense"], axis["quark_3m_dense"], target)
    total_tokens = axis["quark_3m_dense"][-1]
    frac = tokens / total_tokens
    print(
        f"  loop12 finishes the epoch at        : {target:.4f} nats\n"
        f"  dense reaches that loss after       : {tokens:,} tokens ({frac:.1%} of the epoch)\n"
        f"  dense data advantage                : {1 / frac:.2f}x"
    )
    print(
        "\n  Caveat, and it runs in the control's favour: dense passes this loss while\n"
        f"  its lr is still {lr['quark_3m_dense'][idx][0]:.2e}, mid-anneal, against the loop model's\n"
        f"  fully-annealed {lr['quark_3m_loop12'][-1][0]:.2e}. A dense run whose cosine ENDED at {frac:.0%} of\n"
        f"  the data would land lower still. So {1 / frac:.2f}x understates the advantage."
    )

    rule("4. The bill: what the loop model paid for those 0.05 nats")
    print(
        f"  {'run':18} {'params':>10} {'compute-eq':>12} {'wall':>8} {'VRAM':>7} "
        f"{'word PPL':>9} {'BLiMP':>7}"
    )
    for r in CONFIGS:
        p, ce, *_ = CONFIGS[r]
        print(
            f"  {r:18} {p:10,} {ce:12,} {WALL_CLOCK_MIN[r]:6.0f}m {VRAM_GB[r]:6.1f}G "
            f"{EVAL[r][0]:9.3f} {EVAL[r][4]:6.1%}"
        )
    ce_ratio = CONFIGS["quark_3m_loop12"][1] / CONFIGS["quark_3m_dense"][1]
    print(
        f"\n  compute ratio     : {ce_ratio:.1f}x\n"
        f"  wall-clock ratio  : {WALL_CLOCK_MIN['quark_3m_loop12'] / WALL_CLOCK_MIN['quark_3m_dense']:.1f}x  (measured, author's card)\n"
        f"  VRAM ratio        : {VRAM_GB['quark_3m_loop12'] / VRAM_GB['quark_3m_dense']:.1f}x\n"
        f"  quality           : LOSES on every reported metric\n"
        f"\n  Combined with section 3, equal loss costs the loop model roughly\n"
        f"  {ce_ratio:.1f}x compute/token x {1 / frac:.2f}x tokens = {ce_ratio / frac:.0f}x the compute of the control."
    )
    print(
        "\n  THE HYPOTHESIS AND ITS RESULT. Weight sharing is a PARAMETER-efficiency\n"
        "  technique: the promise is that 3M stored parameters buy the quality of\n"
        "  the 20.6M-parameter dense model they unroll into. Measured at equal\n"
        "  params, the loop model delivers slightly WORSE quality than a 3M dense\n"
        "  control while costing 20.6M-parameter compute. It is dominated in the\n"
        "  one metric it was chosen for. Both runs are the same size on disk."
    )

    rule("5. Distance to the actual target")
    tokens_per_word = EVAL_SCORED_TOKENS / EVAL_WORDS
    print(
        f"  tokens per word (this tokenizer, this corpus): {tokens_per_word:.4f}\n"
        f"  PPL_word = exp(NLL_total / words), so a word-level target converts to a\n"
        f"  per-token loss by dividing by that ratio.\n"
    )
    print(f"  {'target':38} {'word PPL':>9} {'nats/token':>11} {'BpB':>7}")
    for name, wppl in [
        ("quark_3m_dense, measured", EVAL["quark_3m_dense"][0]),
        ("quark_3m_loop12, measured", EVAL["quark_3m_loop12"][0]),
        ("GPT-2 124M published (NOT comparable)", GPT2_PUBLISHED_WT103_WORD_PPL),
    ]:
        nll_tok = math.log(wppl) / tokens_per_word
        bpb = math.log(wppl) * EVAL_WORDS / (EVAL_BYTES * math.log(2))
        print(f"  {name:38} {wppl:9.3f} {nll_tok:11.4f} {bpb:7.4f}")
    need = math.log(EVAL["quark_3m_dense"][0] / GPT2_PUBLISHED_WT103_WORD_PPL)
    print(
        f"\n  To reach 37.50 from 108.275 the model must find {need:.3f} nats/word,\n"
        f"  i.e. {need / tokens_per_word:.3f} nats/token -- a {1 - math.log(GPT2_PUBLISHED_WT103_WORD_PPL) / math.log(EVAL['quark_3m_dense'][0]):.0%} cut in NLL. For scale, the entire\n"
        f"  loop-vs-dense architecture difference this issue is about is 0.05\n"
        f"  nats/token. The target is {need / tokens_per_word / 0.0465:.0f}x further away than the\n"
        f"  architecture question being debated."
    )
    print(
        "\n  And 37.50 is the number we are NOT allowed to subtract from (§3.1): it\n"
        "  includes an unreleased de-tokenizer worth 2.5-5 PPL. The controlled\n"
        "  comparison is the self-measured baseline in docs/RESULTS.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
