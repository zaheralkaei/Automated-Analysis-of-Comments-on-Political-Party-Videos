#!/usr/bin/env python3
"""
iaa_analysis.py  –  Inter-annotator agreement.
Reads both annotation files, computes IAA
for all three annotation tasks, and writes a JSON report.

Tasks
-----
  interaction   InteractionScore ∈ {-1.0, -0.5, 0.0, 0.5, 1.0}   ordinal
  stance_child  StanceLabel      ∈ {0, 1, 2}                        nominal
  stance_parent ParentStanceLabel ∈ {0, 1, 2}                       nominal
  techniques    Techniques        — multi-label, 14 categories       multi-label

Metrics
-------
  Ordinal   : % exact, % adjacent (±1 step), κ unweighted, κ linear,
              κ quadratic, Krippendorff α nominal, Krippendorff α ordinal
  Nominal   : % exact, κ unweighted, Krippendorff α nominal
  Multi-label: % exact set match, mean Jaccard, per-technique binary κ,
               multi-label F1 (each annotator as reference)

Output
------
  09_manual_annotations/iaa_results.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Fix Windows cp1252 terminal encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    sys.exit("sklearn not found — install scikit-learn first.")

# Paths
ANNOT_DIR = Path("./09_manual_annotations")
FILE_A    = ANNOT_DIR / "gold_standard_a.jsonl"
FILE_Z    = ANNOT_DIR / "gold_standard_z.jsonl"
OUTPUT    = ANNOT_DIR / "iaa_results.json"

# Schema constants
INTERACTION_ORDER = [-1.0, -0.5, 0.0, 0.5, 1.0]

ALL_TECHNIQUES = [
    "Appeal_to_Authority", "Appeal_to_Time", "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum", "Black-and-White_Fallacy",
    "Causal_Oversimplification", "Doubt", "Exaggeration,Minimisation",
    "Flag-Waving", "Loaded_Language", "Name_Calling,Labeling",
    "Repetition", "Slogans/Thought-terminating_Cliches",
    "Whataboutism,Straw_Men",
]

# I/O helpers
def load_jsonl(path: Path) -> dict:
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["PairID"]] = r
    return recs


# Krippendorff's α
def _krippendorff(a_idx: list, b_idx: list, dist_fn) -> float:
    """
    Generic Krippendorff α for 2 annotators.
    a_idx / b_idx: integer rank indices for each item.
    dist_fn(i, j) -> float: squared distance between ranks i and j.
    """
    n = len(a_idx)
    combined = a_idx + b_idx
    N = len(combined)

    D_o = sum(dist_fn(a_idx[i], b_idx[i]) for i in range(n)) / n
    D_e = sum(
        dist_fn(combined[i], combined[j])
        for i in range(N) for j in range(N) if i != j
    ) / (N * (N - 1))

    return 1.0 - D_o / D_e if D_e > 0 else 1.0


def krippendorff_nominal(a_vals: list, b_vals: list) -> float:
    """Krippendorff α, nominal scale."""
    uniq = sorted(set(a_vals + b_vals))
    rank = {v: i for i, v in enumerate(uniq)}
    a_idx = [rank[v] for v in a_vals]
    b_idx = [rank[v] for v in b_vals]

    def d(i, j): return 0.0 if i == j else 1.0
    return _krippendorff(a_idx, b_idx, d)


def krippendorff_ordinal(a_vals: list, b_vals: list, ordered_values: list) -> float:
    """
    Krippendorff α, ordinal scale.
    Distance d(k,l) = (n_k/2 + Σ n_g for k<g<l + n_l/2)^2
    where n_g is the combined frequency of rank g.
    """
    rank = {v: i for i, v in enumerate(ordered_values)}
    a_idx = [rank[v] for v in a_vals]
    b_idx = [rank[v] for v in b_vals]
    combined = a_idx + b_idx
    freq = Counter(combined)
    n_ranks = len(ordered_values)

    # Precompute distance matrix once
    dist = [[0.0] * n_ranks for _ in range(n_ranks)]
    for k in range(n_ranks):
        for l in range(k + 1, n_ranks):
            s = freq[k] / 2.0
            for g in range(k + 1, l):
                s += freq[g]
            s += freq[l] / 2.0
            dist[k][l] = dist[l][k] = s * s

    return _krippendorff(a_idx, b_idx, lambda i, j: dist[i][j])


# Per-task analysis
def analyze_ordinal(a_vals: list, b_vals: list, ordered_values: list) -> dict:
    n = len(a_vals)
    rank = {v: i for i, v in enumerate(ordered_values)}

    exact    = sum(a == b for a, b in zip(a_vals, b_vals)) / n
    adjacent = sum(abs(rank[a] - rank[b]) <= 1 for a, b in zip(a_vals, b_vals)) / n

    a_int = [rank[v] for v in a_vals]
    b_int = [rank[v] for v in b_vals]
    labels = list(range(len(ordered_values)))

    kappa_uw  = cohen_kappa_score(a_int, b_int, labels=labels)
    kappa_lin = cohen_kappa_score(a_int, b_int, labels=labels, weights="linear")
    kappa_qua = cohen_kappa_score(a_int, b_int, labels=labels, weights="quadratic")

    alpha_nom = krippendorff_nominal(a_vals, b_vals)
    alpha_ord = krippendorff_ordinal(a_vals, b_vals, ordered_values)

    # Confusion matrix: rows = A, cols = Z
    conf = {
        str(v1): {str(v2): sum(1 for a, b in zip(a_vals, b_vals)
                               if a == v1 and b == v2)
                  for v2 in ordered_values}
        for v1 in ordered_values
    }

    return {
        "n":                              n,
        "n_agree":                        sum(a == b for a, b in zip(a_vals, b_vals)),
        "percent_exact_agreement":        round(exact, 4),
        "percent_adjacent_agreement":     round(adjacent, 4),
        "cohen_kappa_unweighted":         round(kappa_uw,  4),
        "cohen_kappa_linear_weighted":    round(kappa_lin, 4),
        "cohen_kappa_quadratic_weighted": round(kappa_qua, 4),
        "krippendorff_alpha_nominal":     round(alpha_nom, 4),
        "krippendorff_alpha_ordinal":     round(alpha_ord, 4),
        "distribution_a": {str(k): v for k, v in sorted(Counter(a_vals).items())},
        "distribution_z": {str(k): v for k, v in sorted(Counter(b_vals).items())},
        "confusion_matrix_a_rows_z_cols": conf,
    }


def analyze_nominal(a_vals: list, b_vals: list) -> dict:
    n = len(a_vals)
    exact  = sum(a == b for a, b in zip(a_vals, b_vals)) / n
    labels = sorted(set(a_vals + b_vals))

    kappa = cohen_kappa_score(a_vals, b_vals, labels=labels)
    alpha = krippendorff_nominal(a_vals, b_vals)

    conf = {
        str(v1): {str(v2): sum(1 for a, b in zip(a_vals, b_vals)
                               if a == v1 and b == v2)
                  for v2 in labels}
        for v1 in labels
    }

    return {
        "n":                          n,
        "n_agree":                    sum(a == b for a, b in zip(a_vals, b_vals)),
        "percent_exact_agreement":    round(exact, 4),
        "cohen_kappa_unweighted":     round(kappa,  4),
        "krippendorff_alpha_nominal": round(alpha,  4),
        "distribution_a": {str(k): v for k, v in sorted(Counter(a_vals).items())},
        "distribution_z": {str(k): v for k, v in sorted(Counter(b_vals).items())},
        "confusion_matrix_a_rows_z_cols": conf,
    }


def analyze_techniques(a_recs: dict, z_recs: dict, pair_ids: list) -> dict:
    def tset(r): return set(r.get("Techniques") or [])

    a_sets = [tset(a_recs[p]) for p in pair_ids]
    z_sets = [tset(z_recs[p]) for p in pair_ids]
    n = len(pair_ids)

    # Exact set agreement
    n_exact = sum(a == z for a, z in zip(a_sets, z_sets))

    # Jaccard per pair
    jaccards = []
    for a, z in zip(a_sets, z_sets):
        u = a | z
        jaccards.append(1.0 if len(u) == 0 else len(a & z) / len(u))
    mean_j = sum(jaccards) / n

    # Per-technique binary Cohen κ
    per_tech = {}
    for tech in ALL_TECHNIQUES:
        a_bin = [1 if tech in s else 0 for s in a_sets]
        z_bin = [1 if tech in s else 0 for s in z_sets]
        agree = sum(aa == zz for aa, zz in zip(a_bin, z_bin)) / n
        cnt_a, cnt_z = sum(a_bin), sum(z_bin)
        try:
            kappa = cohen_kappa_score(a_bin, z_bin)
        except Exception:
            # Degenerate: one annotator never used this technique
            kappa = None
        per_tech[tech] = {
            "count_a":           cnt_a,
            "count_z":           cnt_z,
            "percent_agreement": round(agree, 4),
            "cohen_kappa":       round(kappa, 4) if kappa is not None else None,
        }

    # Multi-label micro-F1 (each annotator as reference)
    def ml_f1(ref_sets, pred_sets):
        tp = fp = fn = 0
        for ref, pred in zip(ref_sets, pred_sets):
            tp += len(ref & pred)
            fp += len(pred - ref)
            fn += len(ref - pred)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(f1, 4), round(prec, 4), round(rec, 4)

    f1_a, p_a, r_a = ml_f1(a_sets, z_sets)   # A = reference
    f1_z, p_z, r_z = ml_f1(z_sets, a_sets)   # Z = reference

    return {
        "n":                          n,
        "n_exact_set_agreement":      n_exact,
        "percent_exact_set_agreement": round(n_exact / n, 4),
        "mean_jaccard":               round(mean_j, 4),
        "multi_label_f1_a_as_ref":    {"f1": f1_a, "precision": p_a, "recall": r_a},
        "multi_label_f1_z_as_ref":    {"f1": f1_z, "precision": p_z, "recall": r_z},
        "mean_f1":                    round((f1_a + f1_z) / 2, 4),
        "per_technique":              per_tech,
    }


# Main
def main():
    print("Loading annotations …")
    a_recs = load_jsonl(FILE_A)
    z_recs = load_jsonl(FILE_Z)
    pair_ids = sorted(set(a_recs) & set(z_recs))
    n = len(pair_ids)
    print(f"  {n} shared pairs")

    def n_agree(field):
        return sum(1 for p in pair_ids if a_recs[p][field] == z_recs[p][field])

    def tset(r): return set(r.get("Techniques") or [])

    n_tech_agree = sum(1 for p in pair_ids if tset(a_recs[p]) == tset(z_recs[p]))
    n_all_agree  = sum(
        1 for p in pair_ids
        if (a_recs[p]["InteractionScore"]  == z_recs[p]["InteractionScore"]
            and a_recs[p]["StanceLabel"]   == z_recs[p]["StanceLabel"]
            and a_recs[p]["ParentStanceLabel"] == z_recs[p]["ParentStanceLabel"]
            and tset(a_recs[p]) == tset(z_recs[p]))
    )

    results = {
        "summary": {
            "n_pairs":                  n,
            "annotators":               ["A", "Z"],
            "n_identical_all_tasks":    n_all_agree,
            "n_conflict_any_task":      n - n_all_agree,
            "n_identical_per_task": {
                "interaction":    n_agree("InteractionScore"),
                "stance_child":   n_agree("StanceLabel"),
                "stance_parent":  n_agree("ParentStanceLabel"),
                "techniques":     n_tech_agree,
            },
        }
    }

    # Task 1: InteractionScore (ordinal)
    print("Analyzing InteractionScore …")
    a_inter = [a_recs[p]["InteractionScore"] for p in pair_ids]
    z_inter = [z_recs[p]["InteractionScore"] for p in pair_ids]
    results["interaction"] = {
        "task":           "InteractionScore",
        "scale":          "ordinal",
        "ordered_values": INTERACTION_ORDER,
        **analyze_ordinal(a_inter, z_inter, INTERACTION_ORDER),
    }

    # Task 2: StanceLabel — child (nominal)
    print("Analyzing StanceLabel (child) …")
    a_sc = [a_recs[p]["StanceLabel"] for p in pair_ids]
    z_sc = [z_recs[p]["StanceLabel"] for p in pair_ids]
    results["stance_child"] = {
        "task":      "StanceLabel (child)",
        "scale":     "nominal",
        "label_map": {"0": "Against/Skeptic", "1": "Neutral", "2": "Pro"},
        **analyze_nominal(a_sc, z_sc),
    }

    # Task 3: ParentStanceLabel (nominal)
    print("Analyzing ParentStanceLabel …")
    a_sp = [a_recs[p]["ParentStanceLabel"] for p in pair_ids]
    z_sp = [z_recs[p]["ParentStanceLabel"] for p in pair_ids]
    results["stance_parent"] = {
        "task":      "ParentStanceLabel",
        "scale":     "nominal",
        "label_map": {"0": "Against/Skeptic", "1": "Neutral", "2": "Pro"},
        **analyze_nominal(a_sp, z_sp),
    }

    # Task 4: Techniques (multi-label)
    print("Analyzing Techniques …")
    results["techniques"] = {
        "task":          "Techniques",
        "scale":         "multi-label",
        "n_categories":  len(ALL_TECHNIQUES),
        "categories":    ALL_TECHNIQUES,
        **analyze_techniques(a_recs, z_recs, pair_ids),
    }

    # Save
    ANNOT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    s = results["summary"]
    print()
    print("═" * 60)
    print("IAA RESULTS SUMMARY")
    print("═" * 60)
    print(f"Pairs analysed       : {n}")
    print(f"Fully identical      : {s['n_identical_all_tasks']} ({100*s['n_identical_all_tasks']/n:.1f}%)")
    print(f"At least 1 conflict  : {s['n_conflict_any_task']} ({100*s['n_conflict_any_task']/n:.1f}%)")
    print()

    def row(task, pct, kappa, alpha, extra=""):
        print(f"  {task:<20} agree={pct:.1%}  κ={kappa:+.3f}  α={alpha:+.3f}{extra}")

    print("TASK                 EXACT_AGREE   κ            α")
    print("-" * 60)
    ri = results["interaction"]
    row("InteractionScore",
        ri["percent_exact_agreement"],
        ri["cohen_kappa_quadratic_weighted"],
        ri["krippendorff_alpha_ordinal"],
        f"  adj={ri['percent_adjacent_agreement']:.1%}")

    rc = results["stance_child"]
    row("Stance (child)",
        rc["percent_exact_agreement"],
        rc["cohen_kappa_unweighted"],
        rc["krippendorff_alpha_nominal"])

    rp = results["stance_parent"]
    row("Stance (parent)",
        rp["percent_exact_agreement"],
        rp["cohen_kappa_unweighted"],
        rp["krippendorff_alpha_nominal"])

    rt = results["techniques"]
    print(f"  {'Techniques':<20} set_agree={rt['percent_exact_set_agreement']:.1%}"
          f"  jaccard={rt['mean_jaccard']:.3f}  F1={rt['mean_f1']:.3f}")
    print()
    print(f"Full report saved to {OUTPUT}")


if __name__ == "__main__":
    main()
