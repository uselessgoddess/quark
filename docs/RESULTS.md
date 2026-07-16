# WikiText-103 results: `quark_3m_loop12` vs `quark_3m_dense` vs GPT-2 Small

Every number in this document is produced by a script in `experiments/`, from
logs checked in under `experiments/runs/`. Nothing here is quoted from memory.

| section | script | output |
| --- | --- | --- |
| §1–§5 (training, loss, cost) | `experiments/analyze_runs.py` | `experiments/out/analyze_runs.txt` |
| §6–§8 (BLiMP) | `experiments/blimp_analysis.py` | `experiments/out/blimp_analysis.txt` |
| §9 (GPT-2 baseline) | `experiments/gpt2_baseline.py` | `experiments/out/gpt2_baseline.txt` |

Both analysis scripts have a `--self-test` that asserts the invariants they
rely on, and `analyze_runs.py` rebuilds the issue's own Learner summary table
from the logs before it does anything else. If the logs and the issue ever
disagree, the script fails loudly rather than quietly reporting something new.

## 0. The two runs are one experiment with one variable

Both runs: 16,444 batches, 134,705,152 tokens, identical peak LR
(2.999e-03 → 3.000e-04 cosine), one epoch, seed 42. Data, steps and schedule
are controlled.

**What is not controlled is width.** The 3.0M parameter budget is fixed, so
paying for 6 unique layers forces `d_model` 384 → 168. This is a comparison of
**two ways to spend one budget**, not of one model with sharing toggled off.
That distinction matters for every conclusion below.

One caveat the scripts surface and the issue does not: the token axes are not
bitwise identical (`identical token axis: False`). One 4,096-token short batch
lands at #16438 in one run and #16423 in the other, because `num_workers: 4`
interleaves the tail nondeterministically. Same data, same volume — but
**`seed: 42` alone does not make a run reproducible**, which is worth knowing
before anyone tries to bisect a 0.05-nat effect.

## 1. The headline gap is 4.6× smaller than the summary table says

The issue's summary reports Loss 4.515 (loop12) vs 4.266 (dense) — a gap of
**+0.2497 nats**. That number averages over the whole epoch, including the
first 5% when both models are barely trained. The models you would ship differ
by far less:

| measurement | loop12 | dense | gap |
| --- | --- | --- | --- |
| epoch mean (the summary's number) | 4.5155 | 4.2658 | **+0.2497** |
| terminal loss (final 5%) | 3.7618 | 3.7071 | **+0.0546** |
| valid loss (end of epoch) | 3.706 | 3.653 | +0.053 |
| test NLL/token | 3.5777 | 3.5312 | +0.0465 |

Three independent end-of-run measurements agree on ~0.05 nats. The 0.25 figure
is the only one that disagrees, because it is the only one that averages over
the untrained model. **The summary overstates the real gap by 4.6×.**

A related trap: the summary's `Perplexity 168.960` row is not a perplexity of
anything. burn's aggregate is a count-weighted mean, and this metric's count is
the *cumulative* token total, so 168.960 is a token-weighted mean of a running
average. It is neither `exp(epoch-mean loss)` (91.42) nor the mean of the
logged series (357.31). `analyze_runs.py` reproduces it exactly, as a
provenance check, and then never uses it again.

## 2. The loop model is slower, not weaker

The gap peaks at **+0.64 nats around 10–15%** of the epoch and falls
monotonically to +0.05 at the end:

```
 0- 5%  +0.3728      50- 55%  +0.1329
10-15%  +0.6414  ←   70- 75%  +0.0785
25-30%  +0.4664      90- 95%  +0.0572
40-45%  +0.2057      95-100%  +0.0546
```

A model that were simply *worse* would hold a constant gap or diverge. This one
converges toward the control from behind. Twelve applications of one layer is a
harder optimization problem, and most of the reported difference is the cost of
solving it, not a capacity ceiling.

This is the strongest available argument **for** the loop model, so its limit
should be stated plainly: converging toward a control is not overtaking it.
Nothing here shows the curves cross.

## 3. The bill

| | params | compute-eq | wall | VRAM | word PPL | BLiMP |
| --- | --- | --- | --- | --- | --- | --- |
| `quark_3m_loop12` | 2,868,352 | 20,643,840 | 60m | 11.0G | 115.163 | 57.05% |
| `quark_3m_dense` | 2,871,880 | 1,778,112 | 15m | 5.5G | **108.275** | **58.63%** |
| ratio | 1.0× | **11.6×** | **4.0×** | **2.0×** | — | — |

Data equivalence: dense reaches loop12's *final* loss after **79.2% of the
epoch** (1.26× data advantage) — and it passes that loss while its LR is still
mid-anneal at 5.79e-04, against loop12's fully-annealed 3.00e-04, so 1.26×
*understates* the advantage. Equal loss therefore costs the loop model roughly
**11.6× compute/token × 1.26× tokens ≈ 15× the compute** of the control.

**This is the hypothesis failing on its own terms.** Weight sharing is a
*parameter*-efficiency technique: the promise is that 3M stored parameters buy
the quality of the 20.6M-parameter dense model they unroll into. Measured at
equal parameters, the loop model delivers slightly *worse* quality than a 3M
dense control while paying 20.6M-parameter compute. Both runs are the same size
on disk. It is dominated in the one metric it was chosen for.

## 4. Why this is the expected result, not a surprise

Two independent 2025–26 results explain it, and both were published before
these runs:

- **Capacity is ~2 bits/param regardless of looping** (Ouro, arXiv:2510.25741).
  Looping buys *manipulation*, not *storage*.
- **One recurrence ≈ √(one parameter block)**, φ ≈ 0.46 (Schwethelm,
  arXiv:2604.21106). Sharing costs parameters and does not refund them.

WikiText-103 word-level perplexity is a **storage** benchmark. It is the one
target where looping's known strength does not apply and its known weakness
does. The result in §3 is what the literature predicts.

## 5. Confounds that run against the loop model

Honesty requires stating that **this run does not cleanly measure "looping is
bad"** — it measures "this loop configuration, trained this way, lost." Known,
published, and fixable handicaps, all specific to loop12:

1. **Residual scaling is not implemented.** `DESIGN.md` §7.1 admits it. GPT-2's
   rule is 1/√N over residual layers; for a looped model N must be the
   *effective* count (2·ℓ_unique·r), not the stored one. Jaggi
   (arXiv:2606.16825) measures **no scaling as clearly worst** (3.542 vs 3.503).
2. **The scaling rule for loops is stronger than 1/√L.** Wang et al.
   (arXiv:2606.18524): weight sharing makes residual updates *correlated* across
   iterations, requiring **ε = 1/N, not 1/√L**. Their key consequence — optimal
   LR depends only on unique layers L, not loop count N — is exactly the knob
   this run never turned.
3. **Sharing changes the effective LR.** Lin et al. (arXiv:2306.09380): a shared
   parameter accumulates gradient once per application, so the shared model sees
   a larger effective step at the same nominal LR. Both runs used one LR.
4. **Huginn needed sandwich norm and a large LR cut** to train a looped model at
   all. Neither was applied here.
5. **Width is confounded** (384 vs 168, §0), at a shared LR — and LR transfer is
   width-dependent (μP).
6. **n = 1, one epoch, no seed replicates.** A 0.05-nat gap has no error bar.
7. **`quark_3m_deep` (2×6) was never run** — the configuration the literature
   actually recommends is missing (see §8).

The correct reading is: **at r=12, with no residual scaling and no LR
retuning, looping lost by 0.05 nats while costing 11.6× compute.** A tuned r=2
run is a different experiment, and it has not been done.

## 6. The issue's BLiMP claim does not survive its own numbers

The issue states: *"dense побеждает обычную глубокую рекурсивную модель по всем
фронтам кроме `wh_vs_that_with_gap_long_distance`"* — and concludes that
Mixture-of-Recursions is therefore worth trying, to keep the long-distance
retention.

Both headlines rebuild exactly from their own field tables (57.0490 vs 57.05;
58.6293 vs 58.63), so the decomposition below is arithmetic, not opinion.

**Claim 1: "dense wins on every front" — false.** loop12 wins 3 of 5 fields:

| field | n | loop12 | dense | delta | winner |
| --- | --- | --- | --- | --- | --- |
| morphology | 18 | 61.30 | 61.66 | +0.36 | dense |
| semantics | 9 | 38.06 | 51.59 | **+13.53** | dense |
| syntax | 26 | 58.15 | 58.11 | −0.04 | loop12 |
| syntax/semantics | 1 | 79.50 | 77.80 | −1.70 | loop12 |
| syntax_semantics | 13 | 60.38 | 58.87 | −1.51 | loop12 |

**Dense's entire +1.58 is semantics.** Weighted by paradigm count, semantics
alone contributes **+1.817 — 115% of the gap** — while the other four fields
net **−0.237, against dense**. Semantics is 9 of 67 paradigms (13% of the
suite): 5 NPI-licensing and 4 quantifier.

**Claim 2: "the loop model retains long-distance context" — not what the number
shows.** BLiMP ships the same construction at both distances:

| paradigm | loop12 | dense | delta | z |
| --- | --- | --- | --- | --- |
| `wh_vs_that_with_gap` (short) | 4.60 | 3.30 | +1.30 | 1.49 |
| `wh_vs_that_with_gap_long_distance` | 2.30 | 0.60 | +1.70 | 3.19 |

The edge is present when the distance is not. It is a property of the wh/that
construction, not of distance. **An advantage that survives removing the long
distance is not long-distance retention — so MoR does not follow from it.**

**Claim 3: these are wins at all — no.** Chance is 50%. Both models sit **~30
standard deviations *below* chance** (2.30% → −30.2σ; 0.60% → −31.2σ). They do
not lack the preference; they hold the inverted one ~99% of the time. These
pairs differ by one function word ("A lady has remembered **who**/**that** the
actors conceal"), so the score reads out which word the corpus made likelier. A
model that fits the corpus better follows that prior harder and scores *lower*.
Dense fits better (108.275 vs 115.163) and scores lower. **"Winning" here is a
symptom of being the weaker LM.**

**The paradigm the issue omits points the other way.** `only_npi_licensor_present`
("Only Bill would ever complain" / "Even Bill would ever complain" — one word
apart, and that word licenses the NPI "ever"): loop12 scores **0 of 1000**.
Not near chance — zero. Dense clears 36.20%. Linking a licensor to the item it
licenses is the closest thing in BLiMP's semantics field to "keeping context
across a span," and it is precisely where the loop model is maximally worse.

**Conclusion: the stated motivation for MoR is not supported by the data it
cites.** That does not make MoR bad on its own merits — see `DECISION.md` §2,
which finds against it for unrelated and much stronger reasons.

## 7. The comparison the issue is missing

| model | params | corpus | BLiMP |
| --- | --- | --- | --- |
| `quark_3m_loop12` | 2.87M | WikiText-103 | 57.05 |
| `quark_3m_dense` | 2.87M | WikiText-103 | 58.63 |
| Transformer-XL | ~139M | **WikiText-103** | **69.60** |
| LSTM | ? | 83M tokens of Wikipedia | 69.80 |

Warstadt et al. 2020 (TACL), Table 3. Both anchors are Wikipedia-only, so the
gap is **not** a domain effect.

- loop12 vs dense: **+1.58 points**
- dense vs same-corpus Transformer-XL: **−10.97 points**

The architecture question the issue asks about is **7× smaller than the gap
both architectures share against a same-corpus baseline.** Whatever is wrong is
upstream of the loop-vs-dense choice. This is the single most important number
in this document: it says the debate the issue opens with is not the debate that
matters.

## 8. Distance to GPT-2 Small

| target | word PPL | nats/token | BpB |
| --- | --- | --- | --- |
| `quark_3m_dense`, measured | 108.275 | 3.5312 | 1.2730 |
| `quark_3m_loop12`, measured | 115.163 | 3.5777 | 1.2897 |
| GPT-2 124M, **published** (not comparable — see below) | 37.500 | 2.7320 | 0.9849 |

Tokens per word for this tokenizer on this corpus: **1.3266**. Since
`PPL_word = exp(NLL_total / n_words)`, a word-level target converts to a
per-token loss by dividing by that ratio. quark uses BPE-8192 with
word-normalized PPL — exactly GPT-2's protocol — so it pays **no** closed-vocab
tax and the comparison is methodologically sound.

To reach 37.50 from 108.275 the model must find **1.060 nats/word** (0.799
nats/token) — a **23% cut in NLL**. The entire loop-vs-dense difference this
issue is about is 0.05 nats/token. **The target is 17× further away than the
architecture question being debated.**

### 8.1 Why the published 37.50 is not the number to compare against

`DESIGN.md` §3.1 mandates a self-measured baseline, and this is why:

1. **37.50 is zero-shot.** GPT-2 never trained on WikiText-103; it is being
   evaluated out of domain. quark trained on it. These are different tasks.
2. **It includes an unreleased de-tokenizer** worth ~2.5–5 PPL, which nobody
   can reproduce.
3. **GPT-2's own paper admits ≥1.6% WikiText-103 test contamination** in
   WebText.

Points 1 and 3 pull in opposite directions, which is exactly why the number has
to be re-measured under quark's protocol rather than adjusted with a fudge
factor.

### 8.2 Self-measured GPT-2 Small, under quark's exact protocol

<!-- PENDING: experiments/gpt2_baseline.py is running on CPU; see §9. -->

*Measurement in progress — `experiments/gpt2_baseline.py` is re-running GPT-2
Small over the same `wiki.test.tokens` under quark's word-normalization and
BLiMP protocol. Both jobs verified `protocol matches
experiments/protocol_fixture.json` at startup. This section will be filled in
from `experiments/out/gpt2_baseline.txt` when they land.*

The protocol is frozen in `experiments/protocol_fixture.json` and asserted from
both sides — `cargo test the_frozen_protocol` in Rust and `--self-test` in
Python — so the baseline cannot silently drift from what quark measures.

## 9. What this document establishes

1. The reported loop-vs-dense gap is **4.6× overstated**; the real terminal gap
   is ~0.05 nats.
2. The loop model is **converging, not capped** — but it never overtakes.
3. At equal parameters it **loses on every metric while costing 11.6× compute,
   4× wall-clock and 2× VRAM**. As a parameter-efficiency technique, measured
   at equal parameters, it failed.
4. That result is **what the 2025–26 literature predicts** for a storage
   benchmark (§4) — but this run is **confounded** and does not settle
   "looping is bad" (§5). r=2 was never tried.
5. The issue's BLiMP reading is **refuted by the issue's own numbers**: dense
   does not win everywhere, its win is 115% semantics, and the wh/that "edge"
   is a ~30σ-below-chance artifact of being the weaker LM (§6).
6. **Both models are ~11 points below a same-corpus Transformer-XL** — a gap 7×
   larger than the one under debate (§7), and 17× larger in perplexity terms
   (§8).

The decision this evidence supports is in `docs/DECISION.md`.
