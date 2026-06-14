"""
Eight analyses on the test-split predictions from steps 15, 16, 17.

Outputs written to  21_error_analysis:
  01_confusion_matrices.json    — full confusion matrices + off-diagonal examples
  02_stratum_errors.json        — error rate by topic × party × model family
  03_hard_cases.json            — pairs wrong by ALL three model families
  04_model_disagreement.json    — pairs where shallow / BERT / LLM families diverge
  05_boundary_cases.json        — boundary score pairs vs error rate
  06_technique_fp_fn.json       — per-technique precision / recall / FP / FN examples
  07_annotation_source.json     — gold vs LLM-annotated error rates
  08_feature_correlation.json   — Pearson r: linguistic features × is_error
  error_report.txt              — human-readable narrative summary
"""

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, mannwhitneyu, chi2_contingency
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score,
)

from utils.jsonl_io import load_jsonl

warnings.filterwarnings("ignore")

# Paths
FEAT_DIR          = Path("14_features")
PAIRS_FILE        = Path("08_pairs/sampled_pairs.jsonl")
GOLD_FILE         = Path("09_manual_annotations/gold_standard.jsonl")
S15_PRED_DIR      = Path("15_shallow_results/predictions")
S15_METRICS       = Path("15_shallow_results/metrics.json")
S16_PRED_DIR      = Path("16_bert_results/predictions")
S16_METRICS       = Path("16_bert_results/metrics.json")
S17_PRED_DIR      = Path("17_llm_results/predictions")
S17_METRICS       = Path("17_llm_results/metrics.json")
OUT_DIR           = Path("21_error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Label constants
VALID_TECHNIQUES = [
    "Appeal_to_Authority", "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum", "Black-and-White_Fallacy",
    "Causal_Oversimplification", "Doubt", "Exaggeration,Minimisation",
    "Flag-Waving", "Loaded_Language", "Name_Calling,Labeling",
    "Repetition", "Slogans/Thought-terminating_Cliches",
    "Whataboutism,Straw_Men", "Appeal_to_Time",
]

SCORE_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_TO_SCORE = {v: k for k, v in SCORE_TO_CLASS.items()}

CLASS_LABELS = {
    "interaction": [
        "Destructive Disagreement", "Destructive Agreement",
        "Neutral/Rephrase", "Constructive Agreement", "Constructive Disagreement",
    ],
    "stance": ["Against", "Neutral", "Support"],
}

THEORY_COLS = [
    "child_len_words", "parent_len_words", "len_ratio", "child_ttr",
    "vocab_overlap", "child_excl_count", "child_quest_count",
    "child_caps_ratio", "child_neg_count", "child_intensifier_count",
    "mention_count", "child_political_count", "has_url",
    "child_first_person_ratio", "child_avg_word_len", "word_count_diff",
    "parent_caps_ratio", "sent_pos", "sent_neg", "sent_neu", "sent_compound",
]

N_EXAMPLES = 3   

# Helpers
def _load_jsonl_safe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_jsonl(str(path))


def _load_jsonl_dir(directory: Path, glob: str = "*.jsonl") -> dict[str, list[dict]]:
    """Return {stem: records} for every matching file in directory."""
    if not directory.exists():
        return {}
    return {p.stem: load_jsonl(str(p)) for p in sorted(directory.glob(glob))}


def _safe_json(obj):
    """Make obj JSON-serialisable (converts numpy types, NaN -> None)."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe_json(data), f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")


# Data loading
def load_features() -> pd.DataFrame:
    fp = FEAT_DIR / "features.parquet"
    if not fp.exists():
        print("  [WARN] features.parquet not found — stratum / text fields will be missing")
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    return df


def load_stratum_map() -> dict[str, dict]:
    """PairID -> {category, party, ParentText, ChildText, ...}"""
    mapping = {}
    for rec in _load_jsonl_safe(PAIRS_FILE):
        pid = rec.get("PairID")
        if pid:
            mapping[pid] = rec
    return mapping


def load_gold_ids() -> set[str]:
    return {r["PairID"] for r in _load_jsonl_safe(GOLD_FILE)}


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File may be mid-write (step still running) — treat as empty
        print(f"  [WARN] Could not parse {path} (may still be running) — skipping")
        return {}



# Best-model selection
def best_s15_run(task: str, metrics: dict) -> tuple[str, str] | None:
    """Return (model, feat_set) with highest macro_f1 on test for step 15."""
    task_m = metrics.get(task, {})
    best_f1 = -1.0
    best_key = None
    for model_name, feat_dict in task_m.items():
        for feat_set, m in feat_dict.items():
            f1 = m.get("macro_f1", m.get("test_macro_f1", -1))
            if f1 is None:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_key = (model_name, feat_set)
    return best_key


def best_s16_run(task: str, metrics: dict) -> str | None:
    """Return model_key with highest macro_f1 for step 16.
    Step 16 metrics.json structure: {model_key: {task: {macro_f1, ...}}}
    """
    best_f1 = -1.0
    best_key = None
    for model_key, task_dict in metrics.items():
        if not isinstance(task_dict, dict):
            continue
        m = task_dict.get(task)
        if not isinstance(m, dict):
            continue
        f1 = m.get("macro_f1", m.get("test_macro_f1", -1))
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_key = model_key
    return best_key


def all_s16_runs(task: str, metrics: dict) -> list[str]:
    """Return all model_keys with a result for this task, sorted by macro_f1 descending."""
    scored = []
    for model_key, task_dict in metrics.items():
        if not isinstance(task_dict, dict):
            continue
        m = task_dict.get(task)
        if not isinstance(m, dict):
            continue
        f1 = m.get("macro_f1", m.get("test_macro_f1", -1))
        if f1 is None:
            continue
        scored.append((f1, model_key))
    scored.sort(reverse=True)
    return [model_key for _, model_key in scored]


def best_s17_run(task: str, metrics: dict) -> str | None:
    """Return run_key string (model_label_condition_shot_mode) with highest macro_f1 for step 17.
    """
    task_m = metrics.get(task, {})
    best_f1 = -1.0
    best_key = None
    for model_label, cond_dict in task_m.items():
        if not isinstance(cond_dict, dict):
            continue
        for condition, shot_dict in cond_dict.items():
            if not isinstance(shot_dict, dict):
                continue
            for shot_mode, m in shot_dict.items():
                if not isinstance(m, dict):
                    continue
                f1 = m.get("macro_f1", m.get("test_macro_f1", -1))
                if f1 is None:
                    continue
                if f1 > best_f1:
                    best_f1 = f1
                    best_key = f"{model_label}_{condition}_{shot_mode}"
    return best_key


# Prediction loading — returns list of {PairID, y_true, y_pred}
def load_s15_preds(task: str, model_name: str, feat_set: str) -> list[dict]:
    stem = f"{task}_{model_name}_{feat_set}"
    path = S15_PRED_DIR / f"{stem}.jsonl"
    recs = _load_jsonl_safe(path)
    if task == "techniques":
        # y_true / y_pred are 14-dim binary lists
        return recs
    return recs   # integers for interaction / stance


def load_s16_preds(task: str, model_key: str) -> list[dict]:
    stem = f"{model_key}_{task}"
    path = S16_PRED_DIR / f"{stem}.jsonl"
    return _load_jsonl_safe(path)


def load_s17_preds(task: str, run_key: str) -> list[dict]:
    stem = f"{task}_{run_key}"
    path = S17_PRED_DIR / f"{stem}.jsonl"
    recs = _load_jsonl_safe(path)
    if task == "interaction":
        # Normalise float score -> class index
        out = []
        for r in recs:
            yt_raw = r.get("y_true")
            yp_raw = r.get("y_pred_score", r.get("y_pred"))
            try:
                yt = SCORE_TO_CLASS[round(float(yt_raw), 1)]
                yp = SCORE_TO_CLASS[round(float(yp_raw), 1)]
            except (KeyError, TypeError, ValueError):
                continue
            out.append({**r, "y_true": yt, "y_pred": yp})
        return out
    return recs


def preds_to_frame(recs: list[dict], task: str) -> pd.DataFrame:
    """Convert list of prediction records to DataFrame (one row per PairID)."""
    if not recs:
        return pd.DataFrame(columns=["PairID", "y_true", "y_pred"])
    rows = []
    for r in recs:
        pid = r.get("PairID")
        yt  = r.get("y_true")
        yp  = r.get("y_pred")
        if pid is None or yt is None or yp is None:
            continue
        rows.append({"PairID": pid, "y_true": yt, "y_pred": yp})
    return pd.DataFrame(rows).drop_duplicates("PairID")


# Analysis 1 — Confusion matrix 
def analysis_01_confusion_matrices(
    df_feat: pd.DataFrame,
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("\n[1/8] Confusion matrix …")
    result = {}

    for task in ("interaction", "stance"):
        result[task] = {}
        n_cls = 5 if task == "interaction" else 3
        labels = CLASS_LABELS[task]

        # Collect (family_label, preds_df) pairs
        families = []

        key15 = best_s15_run(task, s15_metrics)
        if key15:
            model15, feat15 = key15
            recs = load_s15_preds(task, model15, feat15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                families.append((f"shallow_{model15}_{feat15}", df15))

        for key16 in all_s16_runs(task, s16_metrics):
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                families.append((f"bert_{key16}", df16))

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                families.append((f"llm_{key17}", df17))

        for fam_label, df_pred in families:
            yt = df_pred["y_true"].tolist()
            yp = df_pred["y_pred"].tolist()
            cm = confusion_matrix(yt, yp, labels=list(range(n_cls))).tolist()
            macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)

            # Off-diagonal examples (true_class, pred_class) -> up to N_EXAMPLES texts
            offdiag = {}
            for true_c in range(n_cls):
                for pred_c in range(n_cls):
                    if true_c == pred_c:
                        continue
                    mask = (df_pred["y_true"] == true_c) & (df_pred["y_pred"] == pred_c)
                    wrong_pids = df_pred.loc[mask, "PairID"].tolist()[:N_EXAMPLES]
                    if not wrong_pids:
                        continue
                    examples = []
                    for pid in wrong_pids:
                        info = stratum_map.get(pid, {})
                        examples.append({
                            "PairID":    pid,
                            "ParentText": info.get("ParentText", "")[:300],
                            "ChildText":  info.get("ChildText",  "")[:300],
                            "true_label": labels[true_c] if true_c < len(labels) else str(true_c),
                            "pred_label": labels[pred_c] if pred_c < len(labels) else str(pred_c),
                        })
                    key = f"{true_c}->{pred_c}"
                    offdiag[key] = {
                        "true_label": labels[true_c] if true_c < len(labels) else str(true_c),
                        "pred_label": labels[pred_c] if pred_c < len(labels) else str(pred_c),
                        "count":    int(cm[true_c][pred_c]),
                        "examples": examples,
                    }

            result[task][fam_label] = {
                "macro_f1":          round(macro_f1, 4),
                "n_test":            len(yt),
                "labels":            labels,
                "confusion_matrix":  cm,
                "offdiag_examples":  offdiag,
            }

    return result

# Analysis 2 — Per-stratum error breakdown
def analysis_02_stratum_errors(
    df_feat: pd.DataFrame,
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[2/8] Per-stratum error breakdown …")
    result = {}

    for task in ("interaction", "stance"):
        result[task] = {}

        families = {}
        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                families[f"shallow_{key15[0]}_{key15[1]}"] = df15

        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                families[f"bert_{key16}"] = df16

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                families[f"llm_{key17}"] = df17

        for fam_label, df_pred in families.items():
            # Attach stratum info
            df_pred = df_pred.copy()
            df_pred["category"] = df_pred["PairID"].map(
                lambda pid: stratum_map.get(pid, {}).get("category", "unknown")
            )
            df_pred["party"] = df_pred["PairID"].map(
                lambda pid: stratum_map.get(pid, {}).get("party", "unknown")
            )
            df_pred["is_error"] = (df_pred["y_true"] != df_pred["y_pred"]).astype(int)

            strata_results = {}
            for (cat, party), grp in df_pred.groupby(["category", "party"]):
                strat_key = f"{cat}|{party}"
                strata_results[strat_key] = {
                    "category":  cat,
                    "party":     party,
                    "n":         len(grp),
                    "n_errors":  int(grp["is_error"].sum()),
                    "error_rate": round(grp["is_error"].mean(), 4),
                    "macro_f1":   round(
                        f1_score(grp["y_true"], grp["y_pred"],
                                 average="macro", zero_division=0), 4
                    ),
                }
            # Also overall
            strata_results["__ALL__"] = {
                "n":          len(df_pred),
                "n_errors":   int(df_pred["is_error"].sum()),
                "error_rate": round(df_pred["is_error"].mean(), 4),
            }
            result[task][fam_label] = strata_results

    return result



# Analysis 3 — Hard cases (wrong across all families)
def analysis_03_hard_cases(
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[3/8] Hard cases (wrong by all model families) …")
    result = {}

    for task in ("interaction", "stance"):
        # Collect per-family {PairID: (y_true, y_pred)}
        fam_preds: dict[str, dict[str, tuple]] = {}

        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                fam_preds["shallow"] = dict(
                    zip(df15["PairID"], zip(df15["y_true"], df15["y_pred"]))
                )

        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                fam_preds["bert"] = dict(
                    zip(df16["PairID"], zip(df16["y_true"], df16["y_pred"]))
                )

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                fam_preds["llm"] = dict(
                    zip(df17["PairID"], zip(df17["y_true"], df17["y_pred"]))
                )

        if len(fam_preds) < 2:
            result[task] = {"note": "fewer than 2 families available", "hard_cases": []}
            continue

        # Intersection of PIDs across all available families
        all_pids = set.intersection(*[set(d.keys()) for d in fam_preds.values()])
        hard = []
        for pid in all_pids:
            wrong_in = []
            y_true_val = None
            y_preds = {}
            for fam, lookup in fam_preds.items():
                yt, yp = lookup[pid]
                y_true_val = yt
                y_preds[fam] = yp
                if yt != yp:
                    wrong_in.append(fam)
            if len(wrong_in) == len(fam_preds):
                info = stratum_map.get(pid, {})
                hard.append({
                    "PairID":       pid,
                    "y_true":       int(y_true_val),
                    "y_pred_per_family": {k: int(v) for k, v in y_preds.items()},
                    "category":     info.get("category", ""),
                    "party":        info.get("party", ""),
                    "ParentText":   info.get("ParentText", "")[:300],
                    "ChildText":    info.get("ChildText",  "")[:300],
                })

        # Sort by true class for readability
        hard.sort(key=lambda x: x["y_true"])
        result[task] = {
            "families_included": list(fam_preds.keys()),
            "n_shared_pairs":    len(all_pids),
            "n_hard_cases":      len(hard),
            "hard_cases":        hard,
        }

    return result


# Analysis 4 — Model disagreement
def analysis_04_model_disagreement(
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[4/8] Model disagreement analysis …")
    result = {}

    for task in ("interaction", "stance"):
        fam_preds: dict[str, dict[str, tuple]] = {}

        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                fam_preds["shallow"] = dict(
                    zip(df15["PairID"], zip(df15["y_true"], df15["y_pred"]))
                )

        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                fam_preds["bert"] = dict(
                    zip(df16["PairID"], zip(df16["y_true"], df16["y_pred"]))
                )

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                fam_preds["llm"] = dict(
                    zip(df17["PairID"], zip(df17["y_true"], df17["y_pred"]))
                )

        if len(fam_preds) < 2:
            result[task] = {"note": "fewer than 2 families available"}
            continue

        all_pids = set.intersection(*[set(d.keys()) for d in fam_preds.values()])
        fam_names = sorted(fam_preds.keys())
        agree_count = 0
        disagree_examples = []

        pairwise_agreement: dict[str, int] = defaultdict(int)
        pairwise_total:     dict[str, int] = defaultdict(int)

        for pid in all_pids:
            preds = {fam: fam_preds[fam][pid][1] for fam in fam_names}
            yt    = fam_preds[fam_names[0]][pid][0]
            pred_vals = list(preds.values())

            # Pairwise agreement
            for i, fa in enumerate(fam_names):
                for fb in fam_names[i+1:]:
                    pair_key = f"{fa}_vs_{fb}"
                    pairwise_total[pair_key] += 1
                    if preds[fa] == preds[fb]:
                        pairwise_agreement[pair_key] += 1

            if len(set(pred_vals)) == 1:
                agree_count += 1
            else:
                if len(disagree_examples) < 20:
                    info = stratum_map.get(pid, {})
                    correct_families = [f for f in fam_names
                                        if fam_preds[f][pid][1] == yt]
                    disagree_examples.append({
                        "PairID":          pid,
                        "y_true":          int(yt),
                        "preds":           {k: int(v) for k, v in preds.items()},
                        "correct_families": correct_families,
                        "category":        info.get("category", ""),
                        "party":           info.get("party", ""),
                        "ChildText":       info.get("ChildText", "")[:200],
                    })

        n = len(all_pids)
        pairwise_rates = {
            k: round(pairwise_agreement[k] / pairwise_total[k], 4)
            for k in pairwise_total
        }
        result[task] = {
            "families":                fam_names,
            "n_shared_pairs":          n,
            "full_agreement_count":    agree_count,
            "full_agreement_rate":     round(agree_count / n, 4) if n else 0,
            "pairwise_agreement_rate": pairwise_rates,
            "disagreement_examples":   disagree_examples,
        }

    return result


# Analysis 5 — Boundary case analysis
def analysis_05_boundary_cases(
    df_feat: pd.DataFrame,
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[5/8] Boundary case analysis …")
    # Boundary = pairs whose true InteractionScore is at a class boundary:
    # like scores -0.5, 0.0, +0.5 (the ambiguous middle).
    # We compare error rates for boundary vs non-boundary pairs.

    # Map PairID -> raw InteractionScore from features
    score_map: dict[str, float] = {}
    if not df_feat.empty and "InteractionScore" in df_feat.columns:
        for _, row in df_feat.iterrows():
            pid = row.get("PairID")
            s   = row.get("InteractionScore")
            if pid and s is not None and not (isinstance(s, float) and math.isnan(s)):
                score_map[pid] = float(s)

    # Boundary classes: 1 (−0.5), 2 (0.0), 3 (+0.5) — adjacent to two others
    BOUNDARY_CLASSES = {1, 2, 3}

    result = {}
    for task in ("interaction",):   # only interaction has ordinal boundary concept
        families = {}
        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                families["shallow"] = df15

        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                families["bert"] = df16

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                families["llm"] = df17

        fam_results = {}
        for fam, df_pred in families.items():
            df_pred = df_pred.copy()
            df_pred["is_boundary"] = df_pred["y_true"].isin(BOUNDARY_CLASSES)
            df_pred["is_error"]    = df_pred["y_true"] != df_pred["y_pred"]

            boundary_grp     = df_pred[df_pred["is_boundary"]]
            non_boundary_grp = df_pred[~df_pred["is_boundary"]]

            # Adjacent-class errors: predicted ±1 class away
            df_pred["off_by_one"] = (
                df_pred["is_error"] &
                (abs(df_pred["y_pred"] - df_pred["y_true"]) == 1)
            )

            # Mann-Whitney on is_error between boundary vs non-boundary
            stat, pval = (None, None)
            if len(boundary_grp) > 0 and len(non_boundary_grp) > 0:
                try:
                    stat, pval = mannwhitneyu(
                        boundary_grp["is_error"].values,
                        non_boundary_grp["is_error"].values,
                        alternative="two-sided",
                    )
                except Exception:
                    pass

            # Examples: boundary pairs that were misclassified
            wrong_boundary = df_pred[df_pred["is_boundary"] & df_pred["is_error"]]
            examples = []
            for _, row in wrong_boundary.head(N_EXAMPLES).iterrows():
                info = stratum_map.get(row["PairID"], {})
                raw_score = score_map.get(row["PairID"])
                examples.append({
                    "PairID":     row["PairID"],
                    "y_true":     int(row["y_true"]),
                    "y_pred":     int(row["y_pred"]),
                    "raw_score":  raw_score,
                    "ChildText":  info.get("ChildText", "")[:200],
                })

            fam_results[fam] = {
                "n_boundary":         len(boundary_grp),
                "n_non_boundary":     len(non_boundary_grp),
                "boundary_error_rate":     round(
                    boundary_grp["is_error"].mean(), 4
                ) if len(boundary_grp) else None,
                "non_boundary_error_rate": round(
                    non_boundary_grp["is_error"].mean(), 4
                ) if len(non_boundary_grp) else None,
                "off_by_one_rate":    round(
                    df_pred["off_by_one"].mean(), 4
                ),
                "mannwhitney_stat":   float(stat) if stat is not None else None,
                "mannwhitney_pval":   float(pval) if pval is not None else None,
                "boundary_error_examples": examples,
            }
        result[task] = fam_results

    return result


# Analysis 6 — Technique FP / FN analysis
def analysis_06_technique_fp_fn(
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[6/8] Technique FP/FN analysis …")

    def _collect_technique_preds(task: str) -> list[dict] | None:
        """Load best available technique predictions (step 15 > 16 > 17)."""
        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            if recs:
                return recs, f"shallow_{key15[0]}_{key15[1]}"
        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            if recs:
                return recs, f"bert_{key16}"
        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            if recs:
                return recs, f"llm_{key17}"
        return None, None

    task = "techniques"
    recs, source = _collect_technique_preds(task)
    if recs is None:
        return {"note": "No technique predictions found", "source": None}

    def _to_binary_vec(val) -> list[int] | None:
        """Normalise either a 14-dim binary list or a technique-name list to a 14-dim int list."""
        if not isinstance(val, list):
            return None
        if len(val) == 0:
            return [0] * len(VALID_TECHNIQUES)
        # Name-list format (step 17): elements are strings
        if isinstance(val[0], str):
            vec = [0] * len(VALID_TECHNIQUES)
            for name in val:
                if name in VALID_TECHNIQUES:
                    vec[VALID_TECHNIQUES.index(name)] = 1
            return vec
        # Binary-vector format (steps 15/16): elements are 0/1 ints
        if len(val) == len(VALID_TECHNIQUES):
            return [int(v) for v in val]
        return None

    per_tech = {}
    for idx, tech_name in enumerate(VALID_TECHNIQUES):
        fp_examples = []
        fn_examples = []
        tp = fp = fn = tn = 0

        for r in recs:
            pid  = r.get("PairID")
            yt_v = _to_binary_vec(r.get("y_true"))
            yp_v = _to_binary_vec(r.get("y_pred"))
            if pid is None or yt_v is None or yp_v is None:
                continue
            yt_i = yt_v[idx]
            yp_i = yp_v[idx]
            if yt_i == 1 and yp_i == 1:
                tp += 1
            elif yt_i == 0 and yp_i == 1:
                fp += 1
                if len(fp_examples) < N_EXAMPLES:
                    info = stratum_map.get(pid, {})
                    fp_examples.append({
                        "PairID":    pid,
                        "ChildText": info.get("ChildText", "")[:200],
                    })
            elif yt_i == 1 and yp_i == 0:
                fn += 1
                if len(fn_examples) < N_EXAMPLES:
                    info = stratum_map.get(pid, {})
                    fn_examples.append({
                        "PairID":    pid,
                        "ChildText": info.get("ChildText", "")[:200],
                    })
            else:
                tn += 1

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_tech[tech_name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision":    round(prec, 4),
            "recall":       round(rec,  4),
            "f1":           round(f1,   4),
            "fp_examples":  fp_examples,
            "fn_examples":  fn_examples,
        }

    # Summary ranking
    ranked = sorted(per_tech.items(), key=lambda x: x[1]["f1"])

    return {
        "source":       source,
        "n_techniques": len(VALID_TECHNIQUES),
        "per_technique": per_tech,
        "ranked_by_f1_asc": [t for t, _ in ranked],
    }


# Analysis 7 — Annotation source effect
def analysis_07_annotation_source(
    gold_ids: set[str],
    stratum_map: dict,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[7/8] Annotation source effect (gold vs LLM-annotated) …")
    result = {}

    for task in ("interaction", "stance"):
        # Collect all available families
        all_preds: dict[str, pd.DataFrame] = {}

        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df15 = preds_to_frame(recs, task)
            if not df15.empty:
                all_preds[f"shallow"] = df15

        key16 = best_s16_run(task, s16_metrics)
        if key16:
            recs = load_s16_preds(task, key16)
            df16 = preds_to_frame(recs, task)
            if not df16.empty:
                all_preds["bert"] = df16

        key17 = best_s17_run(task, s17_metrics)
        if key17:
            recs = load_s17_preds(task, key17)
            df17 = preds_to_frame(recs, task)
            if not df17.empty:
                all_preds["llm"] = df17

        fam_results = {}
        for fam, df_pred in all_preds.items():
            df_pred = df_pred.copy()
            df_pred["is_gold"]  = df_pred["PairID"].isin(gold_ids)
            df_pred["is_error"] = df_pred["y_true"] != df_pred["y_pred"]

            gold_grp = df_pred[df_pred["is_gold"]]
            llm_grp  = df_pred[~df_pred["is_gold"]]

            # Chi-squared test for independence
            ct = pd.crosstab(df_pred["is_gold"], df_pred["is_error"])
            chi2, pval, dof = (None, None, None)
            if ct.shape == (2, 2):
                try:
                    chi2, pval, dof, _ = chi2_contingency(ct, correction=False)
                except Exception:
                    pass

            fam_results[fam] = {
                "gold_n":          len(gold_grp),
                "gold_error_rate": round(gold_grp["is_error"].mean(), 4) if len(gold_grp) else None,
                "llm_n":           len(llm_grp),
                "llm_error_rate":  round(llm_grp["is_error"].mean(), 4) if len(llm_grp) else None,
                "chi2":            float(chi2) if chi2 is not None else None,
                "pval":            float(pval) if pval is not None else None,
                "dof":             int(dof)    if dof  is not None else None,
                "note":            "pval < 0.05 => annotation source significantly affects error rate",
            }
        result[task] = fam_results

    return result


# Analysis 8 — Text feature correlation with errors
def analysis_08_feature_correlation(
    df_feat: pd.DataFrame,
    s15_metrics: dict,
    s16_metrics: dict,
    s17_metrics: dict,
) -> dict:
    print("[8/8] Text feature correlation with errors …")
    if df_feat.empty:
        return {"note": "features.parquet not available"}

    result = {}

    for task in ("interaction", "stance"):
        # Use the best available family's predictions
        df_pred = pd.DataFrame()
        family_used = None

        key15 = best_s15_run(task, s15_metrics)
        if key15:
            recs = load_s15_preds(task, *key15)
            df_tmp = preds_to_frame(recs, task)
            if not df_tmp.empty:
                df_pred = df_tmp
                family_used = f"shallow_{key15[0]}_{key15[1]}"

        if df_pred.empty:
            key16 = best_s16_run(task, s16_metrics)
            if key16:
                recs = load_s16_preds(task, key16)
                df_tmp = preds_to_frame(recs, task)
                if not df_tmp.empty:
                    df_pred = df_tmp
                    family_used = f"bert_{key16}"

        if df_pred.empty:
            key17 = best_s17_run(task, s17_metrics)
            if key17:
                recs = load_s17_preds(task, key17)
                df_tmp = preds_to_frame(recs, task)
                if not df_tmp.empty:
                    df_pred = df_tmp
                    family_used = f"llm_{key17}"

        if df_pred.empty:
            result[task] = {"note": "No predictions available"}
            continue

        df_pred = df_pred.copy()
        df_pred["is_error"] = (df_pred["y_true"] != df_pred["y_pred"]).astype(float)

        # Merge with features
        available_cols = [c for c in THEORY_COLS if c in df_feat.columns]
        df_merged = df_pred.merge(
            df_feat[["PairID"] + available_cols],
            on="PairID", how="inner"
        )

        if df_merged.empty or "is_error" not in df_merged.columns:
            result[task] = {"note": "Merge produced empty frame"}
            continue

        correlations = {}
        for col in available_cols:
            series = df_merged[col].dropna()
            error_series = df_merged.loc[series.index, "is_error"]
            if series.std() == 0 or len(series) < 10:
                continue
            try:
                r, pval = pointbiserialr(error_series, series)
                correlations[col] = {
                    "pearson_r": round(float(r),    4),
                    "pval":      round(float(pval), 6),
                    "abs_r":     round(abs(float(r)), 4),
                }
            except Exception:
                pass

        # Rank by |r| descending
        ranked = sorted(
            correlations.items(),
            key=lambda x: x[1]["abs_r"],
            reverse=True,
        )

        result[task] = {
            "family_used":       family_used,
            "n_pairs":           len(df_merged),
            "n_features_tested": len(correlations),
            "top_correlations":  {col: corr for col, corr in ranked[:10]},
            "all_correlations":  correlations,
        }

    return result


# Error report (summary)
def write_error_report(results: dict, path: Path):
    lines = [
        "=" * 70,
        "STEP 21 — ERROR ANALYSIS REPORT",
        "=" * 70,
        "",
    ]

    # 1. Confusion matrices
    if "confusion_matrices" in results:
        lines.append("--- 1. CONFUSION MATRICES ---")
        for task, fams in results["confusion_matrices"].items():
            lines.append(f"  Task: {task}")
            for fam, data in fams.items():
                lines.append(
                    f"    {fam:50s}  macro-F1={data.get('macro_f1', '?'):.4f}"
                    f"  n={data.get('n_test', '?')}"
                )
        lines.append("")

    # 2. Stratum errors
    if "stratum_errors" in results:
        lines.append("--- 2. STRATUM ERROR RATES ---")
        for task, fams in results["stratum_errors"].items():
            lines.append(f"  Task: {task}")
            for fam, strata in fams.items():
                lines.append(f"    {fam}")
                rows = sorted(
                    ((k, v) for k, v in strata.items() if k != "__ALL__"),
                    key=lambda x: x[1].get("error_rate", 0),
                    reverse=True,
                )
                for strat_key, sv in rows[:5]:
                    lines.append(
                        f"      {strat_key:30s}  n={sv['n']:4d}"
                        f"  err={sv['error_rate']:.3f}"
                        f"  f1={sv.get('macro_f1', float('nan')):.3f}"
                    )
        lines.append("")

    # 3. Hard cases
    if "hard_cases" in results:
        lines.append("--- 3. HARD CASES ---")
        for task, data in results["hard_cases"].items():
            n = data.get("n_hard_cases", 0)
            total = data.get("n_shared_pairs", 0)
            fams = data.get("families_included", [])
            pct = 100 * n / total if total else 0
            lines.append(
                f"  {task:12s}  hard={n}  total={total}  ({pct:.1f}%)"
                f"  families={fams}"
            )
        lines.append("")

    # 4. Model disagreement
    if "model_disagreement" in results:
        lines.append("--- 4. MODEL DISAGREEMENT ---")
        for task, data in results["model_disagreement"].items():
            if "note" in data:
                lines.append(f"  {task}: {data['note']}")
                continue
            agree = data.get("full_agreement_rate", 0)
            pw    = data.get("pairwise_agreement_rate", {})
            lines.append(f"  {task:12s}  full-agree={agree:.3f}")
            for pair_key, rate in pw.items():
                lines.append(f"    {pair_key}: {rate:.3f}")
        lines.append("")

    # 5. Boundary cases
    if "boundary_cases" in results:
        lines.append("--- 5. BOUNDARY CASE ANALYSIS ---")
        for task, fams in results["boundary_cases"].items():
            lines.append(f"  Task: {task}")
            for fam, data in fams.items():
                be = data.get("boundary_error_rate")
                ne = data.get("non_boundary_error_rate")
                p  = data.get("mannwhitney_pval")
                lines.append(
                    f"    {fam:20s}  boundary_err={be}  non_boundary_err={ne}"
                    f"  MW-p={p}"
                )
        lines.append("")

    # 6. Technique FP/FN
    if "technique_fp_fn" in results:
        lines.append("--- 6. TECHNIQUE FP/FN ---")
        tf = results["technique_fp_fn"]
        lines.append(f"  Source: {tf.get('source', '?')}")
        ranked = tf.get("ranked_by_f1_asc", [])
        pt     = tf.get("per_technique", {})
        lines.append("  Worst 5 techniques by F1:")
        for tech in ranked[:5]:
            d = pt.get(tech, {})
            lines.append(
                f"    {tech:45s}  F1={d.get('f1', 0):.3f}"
                f"  P={d.get('precision', 0):.3f}"
                f"  R={d.get('recall', 0):.3f}"
                f"  fp={d.get('fp', 0)} fn={d.get('fn', 0)}"
            )
        lines.append("")

    # 7. Annotation source
    if "annotation_source" in results:
        lines.append("--- 7. ANNOTATION SOURCE EFFECT ---")
        for task, fams in results["annotation_source"].items():
            lines.append(f"  Task: {task}")
            for fam, data in fams.items():
                ge = data.get("gold_error_rate")
                le = data.get("llm_error_rate")
                p  = data.get("pval")
                lines.append(
                    f"    {fam:20s}  gold_err={ge}  llm_err={le}"
                    f"  chi2-p={p}"
                )
        lines.append("")

    # 8. Feature correlation
    if "feature_correlation" in results:
        lines.append("--- 8. FEATURE CORRELATION WITH ERRORS ---")
        for task, data in results["feature_correlation"].items():
            if "note" in data:
                lines.append(f"  {task}: {data['note']}")
                continue
            lines.append(
                f"  Task: {task}  (family={data.get('family_used', '?')}"
                f"  n={data.get('n_pairs', '?')})"
            )
            for feat, corr in list(data.get("top_correlations", {}).items())[:5]:
                lines.append(
                    f"    {feat:35s}  r={corr['pearson_r']:+.4f}"
                    f"  p={corr['pval']:.4f}"
                )
        lines.append("")

    lines.append("=" * 70)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}")
    print(text)


# Main
def main():
    print("Loading data …")
    df_feat     = load_features()
    stratum_map = load_stratum_map()
    gold_ids    = load_gold_ids()

    # Filter features to test split only (for feature correlation analysis)
    if not df_feat.empty and "split" in df_feat.columns:
        df_feat_test = df_feat[df_feat["split"] == "test"].copy()
    else:
        df_feat_test = df_feat

    s15_metrics = load_metrics(S15_METRICS)
    s16_metrics = load_metrics(S16_METRICS)
    s17_metrics = load_metrics(S17_METRICS)

    print(f"  stratum_map: {len(stratum_map)} pairs")
    print(f"  gold_ids:    {len(gold_ids)}")
    print(f"  features:    {len(df_feat_test)} test rows")
    print(f"  s15 tasks:   {list(s15_metrics.keys())}")
    print(f"  s16 tasks:   {list(s16_metrics.keys())}")
    print(f"  s17 tasks:   {list(s17_metrics.keys())}")

    results = {}

    results["confusion_matrices"]  = analysis_01_confusion_matrices(
        df_feat_test, stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["stratum_errors"]      = analysis_02_stratum_errors(
        df_feat_test, stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["hard_cases"]          = analysis_03_hard_cases(
        stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["model_disagreement"]  = analysis_04_model_disagreement(
        stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["boundary_cases"]      = analysis_05_boundary_cases(
        df_feat_test, stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["technique_fp_fn"]     = analysis_06_technique_fp_fn(
        stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["annotation_source"]   = analysis_07_annotation_source(
        gold_ids, stratum_map, s15_metrics, s16_metrics, s17_metrics
    )
    results["feature_correlation"] = analysis_08_feature_correlation(
        df_feat_test, s15_metrics, s16_metrics, s17_metrics
    )

    # Write individual JSON files
    print("\nWriting outputs …")
    _write_json(OUT_DIR / "01_confusion_matrices.json",  results["confusion_matrices"])
    _write_json(OUT_DIR / "02_stratum_errors.json",      results["stratum_errors"])
    _write_json(OUT_DIR / "03_hard_cases.json",          results["hard_cases"])
    _write_json(OUT_DIR / "04_model_disagreement.json",  results["model_disagreement"])
    _write_json(OUT_DIR / "05_boundary_cases.json",      results["boundary_cases"])
    _write_json(OUT_DIR / "06_technique_fp_fn.json",     results["technique_fp_fn"])
    _write_json(OUT_DIR / "07_annotation_source.json",   results["annotation_source"])
    _write_json(OUT_DIR / "08_feature_correlation.json", results["feature_correlation"])
    write_error_report(results, OUT_DIR / "error_report.txt")

    print(f"\nAll outputs written to {OUT_DIR}/")

if __name__ == "__main__":
    main()
