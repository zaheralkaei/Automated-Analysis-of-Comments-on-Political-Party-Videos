"""
Thread-level 4-way classification.

Approach A — Prediction aggregation:
  Take best pair-level interaction predictions (step 15 shallow / step 16 BERT),
  average per ThreadID, apply 4-bin rule, evaluate vs ground truth (step 12).

Approach B — Thread-level feature classifier:
  Aggregate pair theory features per thread (mean, std, min, max -> 84 dims),
  train LR + RF, evaluate 4-way classification.

Further:
  - Per-topic analysis (climate vs migration)
  - Chi-squared test for topic difference
  - Confusion matrix heatmap
  - Error analysis: most confused examples
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             confusion_matrix, cohen_kappa_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.jsonl_io import load_jsonl, save_jsonl

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

FEAT_DIR         = Path("14_features")
SHALLOW_PRED_DIR = Path("15_shallow_results") / "predictions"
BERT_PRED_DIR    = Path("16_bert_results")     / "predictions"
THREAD_DIR       = Path("12_thread_scores")
FILTER_LOG       = Path("07_filtered")         / "category_report.csv"
OUT_DIR          = Path("18_thread_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_ORDER   = ["Constructive", "Slight/Neutral", "Moderately Destructive", "Highly Destructive"]
BIN_TO_INT  = {b: i for i, b in enumerate(BIN_ORDER)}
SCORE_ENUM  = [-1.0, -0.5, 0.0, 0.5, 1.0]
SCORE_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_SCORE = {v: k for k, v in SCORE_CLASS.items()}

THEORY_COLS = [
    # 15 
    "child_len_words", "parent_len_words", "len_ratio", "child_ttr",
    "vocab_overlap", "child_excl_count", "child_quest_count", "child_caps_ratio",
    "child_neg_count", "child_intensifier_count", "mention_count",
    "child_political_count", "has_url", "child_first_person_ratio", "child_avg_word_len",
    # extended sentiment 
    "sent_pos", "sent_neg", "sent_neu", "sent_compound",
    # structural
    "word_count_diff", "parent_caps_ratio",
]  # 21 features — B block


def score_to_bin(score: float) -> str:
    """Bassi et al. (2025) bin thresholds."""
    if score >= 0.25:   return "Constructive"
    if score >= -0.25:  return "Slight/Neutral"
    if score >= -0.75:  return "Moderately Destructive"
    return "Highly Destructive"


def load_video_topics() -> dict:
    """Returns {videoId: category} from 07_filtered/category_report.csv."""
    import csv
    topics = {}
    if FILTER_LOG.exists():
        with open(FILTER_LOG, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row.get("videoId", "")
                cat = row.get("category", "")
                if vid and cat and cat not in ("TOTAL", ""):
                    topics[vid] = cat
    return topics


def load_ground_truth_threads() -> dict:
    """Returns {ThreadID: {MeanScore, ThreadBin, VideoID}}."""
    gt = {}
    if not THREAD_DIR.exists():
        return gt
    for path in THREAD_DIR.glob("thread_scores_*.jsonl"):
        for r in load_jsonl(str(path)):
            gt[r["ThreadID"]] = r
    return gt


# Best model selection
def find_best_interaction_predictions() -> tuple:
    """Return (pred_file_path, model_label) for the best interaction model
    across shallow learners (step 15) and BERT fine-tuning (step 16)."""
    shallow_metrics = Path("15_shallow_results") / "metrics.json"
    bert_metrics    = Path("16_bert_results")     / "metrics.json"
    best_f1, best_path, best_label = 0.0, None, None

    if shallow_metrics.exists():
        with open(shallow_metrics, encoding="utf-8") as f:
            metrics = json.load(f)
        for model_name, feat_dict in metrics.get("interaction", {}).items():
            for feat_set, m in feat_dict.items():
                f1 = m.get("macro_f1", 0)
                if f1 > best_f1:
                    best_f1   = f1
                    best_path = SHALLOW_PRED_DIR / f"interaction_{model_name}_{feat_set}.jsonl"
                    best_label = f"{model_name}_{feat_set}"

    if bert_metrics.exists():
        with open(bert_metrics, encoding="utf-8") as f:
            bm = json.load(f)
        # step 16 metrics: {model_key: {task: {metrics}}}
        for mk, task_dict in bm.items():
            f1 = task_dict.get("interaction", {}).get("macro_f1", 0)
            if f1 > best_f1:
                best_f1   = f1
                best_path = BERT_PRED_DIR / f"{mk}_interaction.jsonl"
                best_label = f"BERT_{mk}"

    print(f"Best interaction model: {best_label}  (macro-F1={best_f1:.4f})")
    return best_path, best_label


# Approach A: Aggregation
def approach_A(gt_threads, df, video_topics):
    pred_path, model_label = find_best_interaction_predictions()
    if pred_path is None or not pred_path.exists():
        print("  No pair-level predictions found. Skipping approach A.")
        return {}, []

    # Load pair predictions and map to score
    pred_records = load_jsonl(str(pred_path))
    pair_score_map = {}
    for r in pred_records:
        y_pred = r.get("y_pred", 2)
        # y_pred is class index (0-4) for shallow; numeric score for BERT 
        if isinstance(y_pred, (int, float)):
            if y_pred in CLASS_SCORE:
                score = CLASS_SCORE[int(y_pred)]
            else:
                score = min(SCORE_ENUM, key=lambda s: abs(s - float(y_pred)))
        else:
            score = 0.0
        pair_score_map[r["PairID"]] = score

    # Build lookup maps from df upfront
    pid_to_thread = df.set_index("PairID")["ThreadID"].to_dict()
    pid_to_video  = df.set_index("PairID")["VideoID"].to_dict()

    # Aggregate per thread
    thread_scores = defaultdict(list)
    thread_video  = {}
    for pid, score in pair_score_map.items():
        tid = pid_to_thread.get(pid, pid)
        thread_scores[tid].append(score)
        if pid in pid_to_video:
            thread_video[tid] = pid_to_video[pid]

    # Thread predictions
    results = []
    y_true_all, y_pred_all = [], []
    for tid, scores in thread_scores.items():
        mean_score = np.mean(scores)
        pred_bin   = score_to_bin(mean_score)
        gt = gt_threads.get(tid, {})
        true_bin = gt.get("ThreadBin")
        if true_bin is None:
            continue
        vid = thread_video.get(tid, "")
        results.append({
            "ThreadID": tid, "VideoID": vid,
            "Topic": video_topics.get(vid, "unknown"),
            "y_true": true_bin, "y_pred": pred_bin,
            "pred_mean_score": round(mean_score, 4),
        })
        y_true_all.append(BIN_TO_INT[true_bin])
        y_pred_all.append(BIN_TO_INT[pred_bin])

    if not y_true_all:
        return {"macro_f1": 0, "accuracy": 0}, results

    labels = list(range(len(BIN_ORDER)))
    metrics = {
        "macro_f1":        round(float(f1_score(y_true_all, y_pred_all, average="macro",
                                                labels=labels, zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(y_true_all, y_pred_all)), 4),
        "macro_precision": round(float(precision_score(y_true_all, y_pred_all, average="macro",
                                                       labels=labels, zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_true_all, y_pred_all, average="macro",
                                                    labels=labels, zero_division=0)), 4),
        "kappa_quadratic": round(float(cohen_kappa_score(y_true_all, y_pred_all,
                                                          weights="quadratic")), 4),
        "model_used": model_label,
        "n_threads": len(y_true_all),
    }
    return metrics, results


# Approach B: Thread-level classifier
def approach_B(gt_threads, df, video_topics, test_thread_ids: set):
    """Train classifiers on aggregated thread features, evaluate 4-way.

    Uses the same test threads as Approach A (those corresponding to the
    pair-level test split) so both approaches are evaluated on identical
    thread sets.  All remaining threads are used for training.
    """
    # Build thread-level features from pair features
    thread_rows = defaultdict(list)
    for _, row in df.iterrows():
        tid = row.get("ThreadID", "")
        if not tid:
            continue
        feats = row[THEORY_COLS].fillna(0).values.astype(float)
        thread_rows[tid].append(feats)

    records = []
    for tid, feat_list in thread_rows.items():
        gt = gt_threads.get(tid, {})
        if not gt.get("ThreadBin"):
            continue
        mat = np.array(feat_list)
        agg = np.concatenate([mat.mean(0), mat.std(0), mat.min(0), mat.max(0)])
        records.append({
            "ThreadID": tid,
            "VideoID":  gt.get("VideoID", ""),
            "y_true":   BIN_TO_INT[gt["ThreadBin"]],
            "features": agg,
            "topic":    video_topics.get(gt.get("VideoID", ""), "unknown"),
        })

    if len(records) < 10:
        print("  Not enough threads for approach B training.")
        return {"macro_f1": 0}

    X    = np.vstack([r["features"] for r in records])
    y    = np.array([r["y_true"]   for r in records])
    tids = [r["ThreadID"] for r in records]

    # Split: test = Approach A test threads; train = everything else
    test_mask  = np.array([tid in test_thread_ids for tid in tids])
    train_mask = ~test_mask

    if test_mask.sum() == 0 or train_mask.sum() == 0:
        print("  Not enough threads for approach B evaluation.")
        return {"macro_f1": 0}

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"  Approach B split: {train_mask.sum()} train threads, "
          f"{test_mask.sum()} test threads (same as Approach A)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    classifiers = [
        ("LR",      LogisticRegression(max_iter=3000, solver="saga",
                                       class_weight="balanced",
                                       random_state=RANDOM_SEED)),
        ("RF",      RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=RANDOM_SEED, n_jobs=-1)),
        ("SVM",     LinearSVC(max_iter=3000, class_weight="balanced",
                              random_state=RANDOM_SEED)),
        ("SVM_RBF", SVC(kernel="rbf", class_weight="balanced",
                        random_state=RANDOM_SEED)),
        ("XGB",     XGBClassifier(n_estimators=300, eval_metric="mlogloss",
                                  random_state=RANDOM_SEED, n_jobs=-1)),
    ]

    results_B = {}
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        labels_b = list(range(4))
        f1   = float(f1_score(y_test, y_pred, average="macro", zero_division=0, labels=labels_b))
        acc  = float(accuracy_score(y_test, y_pred))
        pr   = float(precision_score(y_test, y_pred, average="macro", zero_division=0, labels=labels_b))
        rc   = float(recall_score(y_test, y_pred, average="macro", zero_division=0, labels=labels_b))
        kqw  = float(cohen_kappa_score(y_test, y_pred, weights="quadratic"))
        results_B[clf_name] = {
            "macro_f1":        round(f1,  4),
            "accuracy":        round(acc, 4),
            "macro_precision": round(pr,  4),
            "macro_recall":    round(rc,  4),
            "kappa_quadratic": round(kqw, 4),
        }
        print(f"  Approach B {clf_name}: macro-F1={f1:.4f}  acc={acc:.4f}  kappa={kqw:.4f}")

    return results_B


# Per-topic analysis
def per_topic_analysis(approach_A_results, video_topics):
    by_topic = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for r in approach_A_results:
        topic = r.get("Topic", "unknown")
        if r["y_true"] in BIN_TO_INT and r["y_pred"] in BIN_TO_INT:
            by_topic[topic]["y_true"].append(BIN_TO_INT[r["y_true"]])
            by_topic[topic]["y_pred"].append(BIN_TO_INT[r["y_pred"]])

    per_topic_metrics = {}
    for topic, data in by_topic.items():
        yt, yp = data["y_true"], data["y_pred"]
        if not yt:
            continue
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0, labels=list(range(4))))
        per_topic_metrics[topic] = {
            "n_threads": len(yt),
            "macro_f1": round(f1, 4),
            "bin_distribution": {
                BIN_ORDER[i]: yt.count(i) for i in range(4)
            },
        }

    # Chi-squared test on ground truth bin distributions across topics
    topics_list = [t for t in per_topic_metrics if per_topic_metrics[t]["n_threads"] > 0]
    if len(topics_list) >= 2:
        contingency = np.array([
            [per_topic_metrics[t]["bin_distribution"].get(b, 0) for b in BIN_ORDER]
            for t in topics_list
        ])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            per_topic_metrics["chi2_test"] = {"chi2": round(chi2, 4), "p_value": round(p_val, 6)}
        except Exception:
            pass

    return per_topic_metrics


# Confusion matrix plot
def plot_confusion_matrix(results, label="Approach A"):
    if not results:
        return
    y_true = [BIN_TO_INT.get(r["y_true"], 0) for r in results if r["y_true"] in BIN_TO_INT]
    y_pred = [BIN_TO_INT.get(r["y_pred"], 0) for r in results if r["y_pred"] in BIN_TO_INT]
    if not y_true:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    fig, ax = plt.subplots(figsize=(7, 5))
    short_labels = ["Constr.", "Slight", "Mod.Dest.", "High.Dest."]
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=short_labels,
                yticklabels=short_labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Thread Classification Confusion Matrix — {label}")
    plt.tight_layout()
    path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {path}")


# Error analysis
def error_analysis(results, n=5):
    """Return top N most-confused pairs per off-diagonal confusion cell."""
    errors = [r for r in results if r.get("y_true") != r.get("y_pred")]
    error_report = {}
    # Group by (true_bin, pred_bin)
    by_confusion = defaultdict(list)
    for r in errors:
        key = f"{r['y_true']} -> {r['y_pred']}"
        by_confusion[key].append(r)
    for key, cases in sorted(by_confusion.items(), key=lambda x: -len(x[1]))[:5]:
        error_report[key] = [{"ThreadID": r["ThreadID"], "pred_mean_score": r.get("pred_mean_score")}
                             for r in cases[:n]]
    return error_report


# Main
def main():
    df = pd.read_parquet(FEAT_DIR / "features.parquet")
    gt_threads   = load_ground_truth_threads()
    video_topics = load_video_topics()

    print(f"Ground truth threads loaded: {len(gt_threads)}")

    # Approach A
    print("\n--- Approach A: Prediction Aggregation ---")
    metrics_A, results_A = approach_A(gt_threads, df, video_topics)
    print(f"  Approach A: macro-F1={metrics_A.get('macro_f1', 0):.4f}"
          f"  acc={metrics_A.get('accuracy', 0):.4f}")

    # Approach B 
    test_thread_ids = {r["ThreadID"] for r in results_A}
    print(f"\n--- Approach B: Thread-level Classifier "
          f"(test_threads={len(test_thread_ids)}) ---")
    metrics_B = approach_B(gt_threads, df, video_topics, test_thread_ids)

    # Per-topic
    print("\n--- Per-topic analysis ---")
    per_topic = per_topic_analysis(results_A, video_topics)
    for topic, m in per_topic.items():
        if isinstance(m, dict) and "macro_f1" in m:
            print(f"  {topic}: n={m['n_threads']}  macro-F1={m['macro_f1']:.4f}")

    # Confusion matrix
    plot_confusion_matrix(results_A)

    # Error analysis
    errors = error_analysis(results_A)

    # Save predictions
    if results_A:
        save_jsonl(results_A, str(OUT_DIR / "thread_predictions.jsonl"))

    # Save metrics
    all_metrics = {
        "approach_A": metrics_A,
        "approach_B": metrics_B,
        "per_topic":  per_topic,
        "error_analysis": errors,
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {OUT_DIR}")

if __name__ == "__main__":
    main()
