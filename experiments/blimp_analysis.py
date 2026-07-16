#!/usr/bin/env python3
"""Test the issue's central BLiMP claim against the issue's own numbers.

The claim (issue #3):

    "dense побеждает обычную глубокую рекурсивную модель по всем фронтам кроме
     wh_vs_that_with_gap_long_distance - так что возможно стоит попробовать
     что-то вроде Mixture-of-Recursions, чтобы сохранить лучшее сохранение
     контекста на дистанции"

    (dense beats the deep recursive model on every front except
     wh_vs_that_with_gap_long_distance - so maybe try something like
     Mixture-of-Recursions, to keep the better long-distance context
     retention without loading the model down)

That reading is load-bearing: it is the entire stated motivation for adding
MoR. This script checks it, and it does not survive.

Three findings, each derived below from numbers pasted in the issue itself:

  1. "wins on every front" is false. The loop model wins 3 of BLiMP's 5
     fields. Dense's entire +1.58 headline win is the semantics field
     (+1.82 on its own); the other four fields net -0.24 AGAINST dense.

  2. The wh_vs_that "win" is not about distance. The loop model is ahead on
     the SHORT-distance sibling paradigm (4.60 vs 3.30) by the same margin as
     on the long-distance one (2.30 vs 0.60). A long-distance-retention
     advantage that shows up identically without the long distance is not a
     long-distance-retention advantage.

  3. Both models are ~30 sigma BELOW chance on that paradigm. 6/1000 and
     23/1000 are not two levels of skill, they are two depths of the same
     systematic anti-preference. Ranking them measures who is less committed
     to a wrong heuristic -- and the loop model is "less committed" precisely
     because it is the worse language model (word ppl 115.163 vs 108.275).

  The kicker: the phenomenon the loop model is *distinctively* worst at is
  NPI licensing -- linking a licensor to the item it licenses. It scores
  0/1000 on only_npi_licensor_present. Dense scores above 36%. If you were
  shopping for evidence about tracking a dependency across a span, this is
  the paradigm to look at, and it points the other way.

Field weights come from the BLiMP suite itself (67 paradigms x 1000 pairs).
`--self-test` asserts them without needing the data on disk.

Usage:
    python3 experiments/blimp_analysis.py [--blimp-dir DIR]
"""

import argparse
import glob
import json
import math
import os
from collections import Counter

# Pasted from issue #3. `quark eval --blimp` output, both runs.
#                       overall,  morphology, semantics, syntax, syn/sem, syn_sem
REPORTED = {
    "quark_3m_loop12": (57.05, 61.30, 38.06, 58.15, 79.50, 60.38),
    "quark_3m_dense": (58.63, 61.66, 51.59, 58.11, 77.80, 58.87),
}
FIELDS = ["morphology", "semantics", "syntax", "syntax/semantics", "syntax_semantics"]

# Paradigm counts per field in BLiMP v1 (67 x 1000 minimal pairs).
# Verified against the suite by --blimp-dir; asserted by --self-test.
FIELD_PARADIGMS = {
    "morphology": 18,
    "semantics": 9,
    "syntax": 26,
    "syntax/semantics": 1,
    "syntax_semantics": 13,
}
PAIRS_PER_PARADIGM = 1000

# The "weakest paradigms" lists from the issue, as percentages. The eval prints
# only each run's bottom 10, so a paradigm missing from a run's list is known
# only to be above that run's 10th-worst -- recorded here as None.
WEAKEST = {
    #                                              loop12  dense
    "only_npi_licensor_present": (0.00, None),
    "matrix_question_npi_licensor_present": (1.80, 6.30),
    "wh_vs_that_with_gap_long_distance": (2.30, 0.60),
    "wh_vs_that_with_gap": (4.60, 3.30),
    "coordinate_structure_constraint_complex_left_branch": (15.00, None),
    "only_npi_scope": (21.90, 34.30),
    "existential_there_quantifiers_2": (22.40, None),
    "principle_A_reconstruction": (23.80, 12.10),
    "anaphor_gender_agreement": (26.10, None),
    "superlative_quantifiers_2": (28.60, 11.30),
    "sentential_subject_island": (None, 26.20),
    "tough_vs_raising_1": (None, 33.20),
    "npi_present_1": (None, 35.70),
    "distractor_agreement_relative_clause": (None, 36.20),
}
# Each run's 10th-worst paradigm: the bound for anything absent from its list.
CUTOFF = {"quark_3m_loop12": 28.60, "quark_3m_dense": 36.20}

# Same-corpus BLiMP anchors, from Warstadt et al. 2020 (TACL), Table 3.
# Both are trained on Wikipedia, so neither can be waved away as a domain gap.
TXL_WIKITEXT103 = 69.6  # Transformer-XL trained on WikiText-103 itself
LSTM_WIKIPEDIA_83M = 69.8  # LSTM, 83M tokens of Wikipedia


def measured_field_paradigms(blimp_dir):
    """Count paradigms per field from the BLiMP jsonl files."""
    counts, uids = Counter(), set()
    for path in sorted(glob.glob(os.path.join(blimp_dir, "*.jsonl"))):
        with open(path) as fh:
            row = json.loads(fh.readline())
        if (row["field"], row["UID"]) not in uids:
            uids.add((row["field"], row["UID"]))
            counts[row["field"]] += 1
    return dict(counts)


def weighted_overall(field_accs):
    """BLiMP's headline is the pair-weighted mean; every paradigm has 1000 pairs,
    so paradigm counts are the weights."""
    total = sum(FIELD_PARADIGMS.values())
    return sum(a * FIELD_PARADIGMS[f] for f, a in zip(FIELDS, field_accs)) / total


def two_proportion_z(p1, p2, n=PAIRS_PER_PARADIGM):
    """Unpooled z for a difference of two proportions (percent in, z out)."""
    p1, p2 = p1 / 100, p2 / 100
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return (p1 - p2) / se if se > 0 else float("nan")


def sigma_from_chance(p, n=PAIRS_PER_PARADIGM):
    """How many SDs below coin-flipping a score sits."""
    return (p / 100 - 0.5) / math.sqrt(0.25 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blimp-dir", help="dir of BLiMP *.jsonl, to verify weights")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        assert sum(FIELD_PARADIGMS.values()) == 67, "BLiMP has 67 paradigms"
        assert FIELD_PARADIGMS["semantics"] == 9
        for run, rep in REPORTED.items():
            got = weighted_overall(rep[1:])
            assert abs(got - rep[0]) < 0.005, f"{run}: {got} != {rep[0]}"
        print("self-test ok")
        return

    if args.blimp_dir:
        got = measured_field_paradigms(args.blimp_dir)
        print(f"field weights measured from {args.blimp_dir}: {got == FIELD_PARADIGMS}")
        if got != FIELD_PARADIGMS:
            raise SystemExit(f"suite disagrees: {got}")

    print("\n=== 0. Provenance: rebuild each headline from its own field table ===\n")
    print(f"  {'run':<18}{'reported':>10}{'rebuilt':>10}{'ok':>6}")
    for run, rep in REPORTED.items():
        got = weighted_overall(rep[1:])
        print(f"  {run:<18}{rep[0]:>10.2f}{got:>10.4f}{str(abs(got-rep[0])<0.005):>6}")
    print("\n  BLiMP weights fields by paradigm count, and every paradigm is 1000")
    print("  pairs. So a field is worth exactly its share of 67 -- semantics is 9.")

    print("\n=== 1. Claim: 'dense wins on every front' ===\n")
    loop, dense = REPORTED["quark_3m_loop12"], REPORTED["quark_3m_dense"]
    print(f"  {'field':<20}{'n':>4}{'loop12':>9}{'dense':>9}{'delta':>9}   winner")
    for i, f in enumerate(FIELDS):
        d = dense[i + 1] - loop[i + 1]
        who = "dense" if d > 0 else "loop12"
        print(
            f"  {f:<20}{FIELD_PARADIGMS[f]:>4}{loop[i+1]:>9.2f}{dense[i+1]:>9.2f}"
            f"{d:>+9.2f}   {who}"
        )
    won = sum(1 for i in range(5) if loop[i + 1] > dense[i + 1])
    print(f"\n  fields won by the loop model: {won} of 5  ->  claim is FALSE")

    print("\n=== 2. Where dense's +1.58 actually comes from ===\n")
    total = sum(FIELD_PARADIGMS.values())
    print(f"  {'field':<20}{'delta':>9}{'weight':>9}{'contribution':>14}")
    contribs = []
    for i, f in enumerate(FIELDS):
        d = dense[i + 1] - loop[i + 1]
        c = d * FIELD_PARADIGMS[f] / total
        contribs.append((f, c))
        print(f"  {f:<20}{d:>+9.2f}{FIELD_PARADIGMS[f]/total:>9.3f}{c:>+14.3f}")
    print(f"  {'':<20}{'':>9}{'':>9}{sum(c for _, c in contribs):>+14.3f}  (= headline gap)")
    sem = dict(contribs)["semantics"]
    rest = sum(c for f, c in contribs if f != "semantics")
    print(f"\n  semantics alone:      {sem:+.3f}   ({sem/(dense[0]-loop[0])*100:.0f}% of the gap)")
    print(f"  the other 4 fields:   {rest:+.3f}   (i.e. they favour the LOOP model)")
    print("\n  Semantics is 13% of the suite and delivers 115% of dense's win.")
    print("  BLiMP's semantics field is 9 paradigms: 5 NPI-licensing, 4 quantifier.")

    print("\n=== 3. Claim: the loop model retains long-distance context better ===\n")
    print("  If true, the advantage must be specific to the long-distance case.")
    print("  BLiMP ships the same construction at both distances. Compare:\n")
    print(f"  {'paradigm':<38}{'loop12':>8}{'dense':>8}{'delta':>8}{'z':>7}")
    for p in ["wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance"]:
        lo, de = WEAKEST[p]
        z = two_proportion_z(lo, de)
        print(f"  {p:<38}{lo:>8.2f}{de:>8.2f}{lo-de:>+8.2f}{z:>7.2f}")
    short = WEAKEST["wh_vs_that_with_gap"]
    long_ = WEAKEST["wh_vs_that_with_gap_long_distance"]
    print(f"\n  loop12's edge WITHOUT the long distance: {short[0]-short[1]:+.2f} points")
    print(f"  loop12's edge WITH the long distance:    {long_[0]-long_[1]:+.2f} points")
    print("\n  The edge is there when the distance is not. It is a property of the")
    print("  wh/that construction, not of distance -> the stated mechanism is not")
    print("  what produced the number, so MoR does not follow from it.")

    print("\n=== 4. What 0.60% and 2.30% actually mean ===\n")
    print(f"  {'paradigm':<38}{'run':<10}{'score':>7}{'sigma vs chance':>18}")
    for p in ["wh_vs_that_with_gap_long_distance", "wh_vs_that_with_gap"]:
        for run, v in zip(("loop12", "dense"), WEAKEST[p]):
            print(f"  {p:<38}{run:<10}{v:>7.2f}{sigma_from_chance(v):>18.1f}")
    print("\n  Chance is 50%. Both models sit ~30 SD BELOW it: they do not lack")
    print("  the preference, they hold the inverted one, ~99% of the time. These")
    print("  minimal pairs differ by one function word (who/that), so the score is")
    print("  a readout of which word the corpus made more likely. A model that")
    print("  fits the corpus better follows that prior harder and scores LOWER.")
    print(f"  Dense fits better (word ppl 108.275 vs 115.163), and scores lower.")
    print("  'Winning' here is a symptom of being the weaker LM, not of skill.")

    print("\n=== 5. The paradigm the issue does not mention ===\n")
    lo, de = WEAKEST["only_npi_licensor_present"]
    bound = CUTOFF["quark_3m_dense"]
    print(f"  only_npi_licensor_present   loop12 {lo:.2f}%  (0/1000)")
    print(f"                              dense  >{bound:.2f}%  (absent from its bottom 10)")
    print(f"\n  This pair is 'Only Bill would ever complain' / 'Even Bill would ever")
    print("  complain' -- one word apart, and that word licenses the NPI 'ever'.")
    print("  The loop model gets 0 of 1000. Not near chance: zero. Dense is 36%+.")
    print("  Linking a licensor to the item it licenses is the closest thing in")
    print("  BLiMP's semantics field to 'keeping context across a span', and it is")
    print("  where the loop model is distinctively, maximally worse.")

    print("\n=== 6. The comparison the issue is missing ===\n")
    print(f"  {'model':<44}{'params':>10}{'BLiMP':>8}")
    print(f"  {'quark_3m_loop12 (1 layer x 12, WikiText-103)':<44}{'2.87M':>10}{loop[0]:>8.2f}")
    print(f"  {'quark_3m_dense (6 layers, WikiText-103)':<44}{'2.87M':>10}{dense[0]:>8.2f}")
    print(f"  {'Transformer-XL (WikiText-103)':<44}{'~139M':>10}{TXL_WIKITEXT103:>8.2f}")
    print(f"  {'LSTM (83M tokens of Wikipedia)':<44}{'?':>10}{LSTM_WIKIPEDIA_83M:>8.2f}")
    print("\n  Warstadt et al. 2020 (TACL), Table 3. Both anchors are Wikipedia-only,")
    print("  so the gap below is not a domain effect.\n")
    print(f"  loop12 vs dense:                {dense[0]-loop[0]:+.2f} points")
    print(f"  dense vs same-corpus TXL:       {dense[0]-TXL_WIKITEXT103:+.2f} points")
    print(f"  -> the architecture question the issue asks about is {(TXL_WIKITEXT103-dense[0])/(dense[0]-loop[0]):.0f}x smaller")
    print("     than the gap both architectures share against the corpus baseline.")
    print("     Whatever is wrong is upstream of the loop-vs-dense choice.")


if __name__ == "__main__":
    main()
