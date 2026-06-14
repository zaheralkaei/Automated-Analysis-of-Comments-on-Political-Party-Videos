"""
Visualizations and Graphs

  A. All scraped videos (14 280)  -- channel/party dist, yearly trend, topic breakdown
  B. Annotated pairs (6 548)      -- interaction scores (proper labels), stance,
                                     techniques, split sizes
  C. Temporal analysis            -- videos per channel/topic over time (yearly),
                                     destructiveness over time per topic / per channel
  D. Model comparison             -- ablation B->B+S->B+S+T->E1/E2, shallow vs BERT
                                     vs LLM, per-task heatmap, LLM conditions
  E. Confusion matrices           -- best shallow / BERT (gbert+xlmr) / LLM, proper
                                     interaction class labels
  F. Feature importance           -- SHAP bar charts + permutation importance
  G. Thread analysis              -- bin distribution, per-topic, Approach A vs B

Outputs: 20_visualizations/<section>/<name>.png
"""
import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from dotenv import load_dotenv
load_dotenv()

# Global styling
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

DPI = 150

TOPIC_COLORS = {"migration": "#4472C4", "climate": "#70AD47",
                "unknown": "#AAAAAA", "other": "#CCCCCC"}

CHANNEL_COLORS = {
    "AfD-Fraktion Bundestag": "#009EE0",   
    "AfD TV":                 "#0060A0",   
    "DieLinke":               "#BE3075",   
    "Die Linke im Bundestag": "#8B1A4A",   
}

PARTY_MAP = {
    "AfD-Fraktion Bundestag": "AfD",
    "AfD TV":                 "AfD",
    "DieLinke":               "Die Linke",
    "Die Linke im Bundestag": "Die Linke",
}
PARTY_COLORS = {"AfD": "#009EE0", "Die Linke": "#BE3075", "Unknown": "#AAAAAA"}

# Feature names for SHAP / permutation importance charts
FEATURE_DISPLAY_NAMES = {
    "child_len_words":         "Child: word count",
    "parent_len_words":        "Parent: word count",
    "word_count_diff":         "Word count diff |P−C|",
    "len_ratio":               "Length ratio (child/parent)",
    "child_ttr":               "Child: type-token ratio",
    "vocab_overlap":           "Vocabulary overlap",
    "child_excl_count":        "Child: exclamation marks",
    "child_quest_count":       "Child: question marks",
    "child_caps_ratio":        "Child: CAPS ratio",
    "parent_caps_ratio":       "Parent: CAPS ratio",
    "child_neg_count":         "Child: negations",
    "child_intensifier_count": "Child: intensifiers",
    "mention_count":           "User mentions (@)",
    "child_political_count":   "Child: political terms",
    "has_url":                 "Has URL",
    "child_first_person_ratio":"Child: 1st-person ratio",
    "child_avg_word_len":      "Child: avg word length",
    "sent_pos":                "Sentiment: positive",
    "sent_neg":                "Sentiment: negative",
    "sent_neu":                "Sentiment: neutral",
    "sent_compound":           "Sentiment: compound (pos−neg)",
    "technique_count":         "Technique count",
    "child_stance":            "Child stance (0/1/2)",
    "parent_stance":           "Parent stance (0/1/2)",
    "same_stance":             "Same stance (parent=child)",
    "abs_stance_diff":         "Stance difference |P−C|",
}

def _feat_label(name: str) -> str:
    """Return human-readable label for a feature name."""
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " "))


LLM_DISPLAY_NAMES = {
    "gemma4_31b":  "Gemma 4 31B",
    "qwen25_7b":   "Qwen2.5 7B",
    "qwen25_72b":  "Qwen2.5 72B",
}

def _llm_label(name: str) -> str:
    """Return human-readable label for an LLM model identifier."""
    return LLM_DISPLAY_NAMES.get(name, name)


import re as _re
def _is_embedding_features(names) -> bool:
    """Return True if most names look like raw embedding dimensions (xlmr_0, bert_12, …).
    Such features have no human-interpretable meaning and their plots should be skipped."""
    if not names:
        return False
    hits = sum(1 for n in names if _re.match(r'^[a-z]+_?\d+$', str(n), _re.IGNORECASE))
    return hits / len(names) > 0.5

MODEL_COLORS = {
    "LR": "#4472C4", "RF": "#ED7D31", "SVM": "#A9D18E",
    "SVM_RBF": "#FFC000", "XGB": "#C00000",
    "gbert": "#7030A0", "xlmr": "#00B050",
    "gemma4_31b": "#FF0000", "qwen25_7b": "#FF6600", "qwen25_72b": "#C55A11",
}


# Interaction class labels  (score -> class index -> human label)
# SCORE_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
SCORE_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_TO_SCORE = {v: k for k, v in SCORE_TO_CLASS.items()}
SCORE_ENUM     = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Full labels for bar charts / legends
INT_LABELS = [
    "Destructive\nDisagreement\n(-1.0)",
    "Destructive\nAgreement\n(-0.5)",
    "Neutral /\nRephrase\n(0.0)",
    "Constructive\nAgreement\n(+0.5)",
    "Constructive\nDisagreement\n(+1.0)",
]
# Short labels for confusion-matrix tick axes
INT_SHORT = [
    "Dest.\nDisagr.",
    "Dest.\nAgreemnt",
    "Neutral /\nRephrase",
    "Constr.\nAgreemnt",
    "Constr.\nDisagr.",
]

STANCE_LABELS = ["Against", "Neutral", "Support"]

VALID_TECHNIQUES = [
    "Appeal_to_Authority", "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum", "Black-and-White_Fallacy",
    "Causal_Oversimplification", "Doubt", "Exaggeration,Minimisation",
    "Flag-Waving", "Loaded_Language", "Name_Calling,Labeling",
    "Repetition", "Slogans/Thought-terminating_Cliches",
    "Whataboutism,Straw_Men", "Appeal_to_Time",
]

# Paths
FEAT_DIR         = Path("14_features")
SHALLOW_DIR      = Path("15_shallow_results")
BERT_DIR         = Path("16_bert_results")
LLM_DIR          = Path("17_llm_results")
THREAD_SCORE_DIR = Path("12_thread_scores")
EXPL_DIR         = Path("19_explainability")
FILTER_LOG       = Path("07_filtered") / "category_report.csv"
VIDEOS_CSV_AB    = Path("data_a_b") / "01_channel_videos" / "Combined_videos.csv"
VIDEOS_CSV_CD    = Path("data_c_d") / "01_channel_videos" / "Combined_videos.csv"
OUT_DIR          = Path("20_visualizations")

for sub in ["videos", "pairs", "temporal", "models", "confusion", "features", "threads"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


# Helpers
def _save(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_jsonl(path):
    try:
        from utils.jsonl_io import load_jsonl
        return list(load_jsonl(str(path)))
    except Exception:
        return []


def _best_shallow(metrics, task):
    """Return (model_name, feat_set) with highest macro_f1 for a task."""
    best_f1, best_m, best_f = 0.0, None, None
    for mn, fd in metrics.get(task, {}).items():
        for fs, m in fd.items():
            if m.get("macro_f1", 0) > best_f1:
                best_f1, best_m, best_f = m["macro_f1"], mn, fs
    return best_m, best_f


def _build_confusion(preds, task):
    """Build (cm, labels, tick_labels) from prediction records."""
    if task == "interaction":
        yt = [int(r["y_true"]) for r in preds]
        yp = [int(r["y_pred"]) for r in preds]
        labels      = list(range(5))
        tick_labels = INT_SHORT
    elif task == "stance":
        yt = [int(r["y_true"]) for r in preds]
        yp = [int(r["y_pred"]) for r in preds]
        labels      = [0, 1, 2]
        tick_labels = STANCE_LABELS
    else:
        return None, None, None
    cm = sk_confusion_matrix(yt, yp, labels=labels)
    return cm, labels, tick_labels


def _plot_cm(cm, tick_labels, title, path, cmap="Blues", figsize=None):
    if figsize is None:
        figsize = (max(6, len(tick_labels) * 1.2), max(5, len(tick_labels) * 1.1))
    fig, ax = plt.subplots(figsize=figsize)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap=cmap,
        xticklabels=tick_labels, yticklabels=tick_labels,
        linewidths=0.5, ax=ax, cbar_kws={"label": "Row-normalised"},
        annot_kws={"size": 9},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    _save(fig, path)


def _label_bars(ax, bars, values, total=None, fontsize=9, adjust_ylim=True):
    vals = [v if v else 0 for v in values]
    if total is None:
        total = sum(vals) or 1
    ymax = max((b.get_height() for b in bars), default=1) or 1
    pad  = ymax * 0.02
    for bar, v in zip(bars, vals):
        if not v:
            continue
        pct = 100.0 * v / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            f"{int(v):,}\n({pct:.1f}% of {int(total):,})",
            ha="center", va="bottom", fontsize=fontsize,
            linespacing=1.15,
        )
    if adjust_ylim:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.25)


def _label_hbars(ax, values, total=None, fontsize=8, adjust_xlim=True):
    vals = [v if v else 0 for v in values]
    if total is None:
        total = sum(vals) or 1
    xmax = max(vals) if vals else 1
    pad  = xmax * 0.015
    for i, v in enumerate(vals):
        if not v:
            continue
        pct = 100.0 * v / total
        ax.text(v + max(pad, 1), i, f"{int(v):,}  ({pct:.1f}% of {int(total):,})",
                va="center", fontsize=fontsize)
    if adjust_xlim:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.35)


def _year_fmt(x, pos=None):
    """Yearly tick formatter."""
    try:
        return mdates.num2date(x).strftime("%Y")
    except Exception:
        return ""


def _year_n_fmt(n_per_year):
    """Yearly tick formatter that also shows N below the year."""
    def fmt(x, pos=None):
        try:
            year = mdates.num2date(x).year
            n = n_per_year.get(year, 0)
            return f"{year}\n(n={n})"
        except Exception:
            return ""
    return fmt


def _fix_yearly_axis(ax, fig):
    """Replace locator with a clean annual locator."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_year_fmt))
    fig.autofmt_xdate(rotation=45, ha="right")


# A. All scraped videos
def load_video_meta():
    """Load all video metadata (videoId, publishedAt, channelName) + category join."""
    frames = []
    for csv_path in [VIDEOS_CSV_AB, VIDEOS_CSV_CD]:
        if csv_path.exists():
            try:
                tmp = pd.read_csv(
                    csv_path,
                    usecols=lambda c: c in ["videoId", "publishedAt", "channelName", "title"],
                )
                frames.append(tmp)
            except Exception as e:
                print(f"    Warning: {csv_path}: {e}")

    video_meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Attach topic category
    if FILTER_LOG.exists() and not video_meta.empty:
        try:
            cat = pd.read_csv(
                FILTER_LOG,
                usecols=lambda c: c in ["videoId", "category"],
            )
            video_meta = video_meta.merge(cat, on="videoId", how="left")
            video_meta["category"] = video_meta["category"].fillna("other")
        except Exception as e:
            print(f"    Warning: category_report merge: {e}")

    if not video_meta.empty and "publishedAt" in video_meta.columns:
        video_meta["publishedAt"] = pd.to_datetime(
            video_meta["publishedAt"], utc=True, errors="coerce"
        )
        video_meta["year"]         = video_meta["publishedAt"].dt.year
        video_meta["year_quarter"] = video_meta["publishedAt"].dt.to_period("Q")
        video_meta["year_dt"]      = video_meta["publishedAt"].dt.to_period("Y").dt.to_timestamp()

    if "channelName" in video_meta.columns:
        video_meta["party"] = video_meta["channelName"].map(PARTY_MAP).fillna("Unknown")

    return video_meta


def plot_all_videos_overview(video_meta):
    """Section A: all scraped videos."""
    print("\n-- A. All Scraped Videos Overview")

    if video_meta.empty:
        print("  No video metadata -- skipping.")
        return

    n_total = len(video_meta)
    print(f"  Total scraped videos: {n_total}")

    # A1. Videos per channel (bar)
    chan_counts = video_meta["channelName"].value_counts()
    colors_ch = [CHANNEL_COLORS.get(c, "#999999") for c in chan_counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(chan_counts)), chan_counts.values, color=colors_ch)
    ax.set_xticks(range(len(chan_counts)))
    ax.set_xticklabels(chan_counts.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of videos")
    ax.set_title(f"All Scraped Videos per Channel  (n={n_total:,})", fontweight="bold")
    _label_bars(ax, bars, chan_counts.values, fontsize=9)
    _save(fig, OUT_DIR / "videos" / "videos_per_channel.png")

    # A2. Videos per year
    year_counts = (video_meta.dropna(subset=["year"])
                   .groupby("year").size().reset_index(name="count"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(year_counts["year"], year_counts["count"], color="#4472C4", width=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of videos")
    ax.set_title(f"Videos Uploaded per Year  (n={n_total:,})", fontweight="bold")
    ax.set_xticks(year_counts["year"])
    ax.set_xticklabels(year_counts["year"].astype(int), rotation=45, ha="right")
    for _, row in year_counts.iterrows():
        ax.text(row["year"], row["count"] + 15, str(int(row["count"])),
                ha="center", va="bottom", fontsize=7)
    _save(fig, OUT_DIR / "videos" / "videos_per_year.png")


    # A4. Topic coverage (from category_report filtered videos)
    if "category" in video_meta.columns:
        topic_counts = video_meta["category"].value_counts()
        colors_t = [TOPIC_COLORS.get(t, "#999999") for t in topic_counts.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(topic_counts.index, topic_counts.values, color=colors_t)
        ax.set_ylabel("Number of videos")
        ax.set_title("Videos by Topic Category", fontweight="bold")
        _label_bars(ax, bars, topic_counts.values, fontsize=9)
        _save(fig, OUT_DIR / "videos" / "videos_per_topic_category.png")

    # A5. Migration+Climate videos per channel
    if "category" in video_meta.columns:
        topic_sub = video_meta[video_meta["category"].isin(["migration", "climate"])]
        pivot = (topic_sub.groupby(["channelName", "category"])
                 .size().unstack(fill_value=0))
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(pivot))
        # Collect outside labels per bar-index for gap enforcement
        outside_labels = {j: [] for j in range(len(pivot))}
        for cat_name, color in [("migration", TOPIC_COLORS["migration"]),
                                  ("climate",   TOPIC_COLORS["climate"])]:
            if cat_name in pivot.columns:
                vals = pivot[cat_name].values
                seg_bars = ax.bar(range(len(pivot)), vals, bottom=bottom,
                                  label=cat_name.capitalize(), color=color)
                for j, (bar, v, b) in enumerate(zip(seg_bars, vals, bottom)):
                    if v > 0:
                        if v >= 40:   # large enough >> label inside
                            ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2,
                                    f"{v:,}", ha="center", va="center",
                                    fontsize=8, color="white", fontweight="bold")
                        else:         # too thin >> queue for outside label
                            outside_labels[j].append({
                                "y_ideal": b + v / 2,
                                "label":   f"{v:,}",
                            })
                bottom += vals
        # Place thin-segment labels above their bar to avoid overlap with
        # neighbouring bars.  Arrow goes from segment midpoint up to label.
        ax.autoscale_view()
        totals_arr = pivot.sum(axis=1).values
        for j, items in outside_labels.items():
            if not items:
                continue
            items.sort(key=lambda d: d["y_ideal"])
            bar_top = totals_arr[j]
            bar_cx  = j          # integer x position = bar centre
            n = len(items)
            for k, item in enumerate(items):
                x_offset = (k - (n - 1) / 2) * 0.25   # stagger left/right
                y_lbl = bar_top + 30 + k * 28          # data units above bar
                ax.annotate(
                    item["label"],
                    xy=(bar_cx, item["y_ideal"]),
                    xytext=(bar_cx + x_offset, y_lbl),
                    fontsize=7.5, va="bottom", ha="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="gray"),
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="gray", lw=0.5),
                )
        # Total on top of each bar — bold, slightly above any above-bar labels
        totals = pivot.sum(axis=1).values
        for j, total in enumerate(totals):
            n_outside = len(outside_labels[j])
            y_total = total + 15 + max(0, n_outside - 1) * 28 + (8 if n_outside else 0)
            ax.text(j, y_total, f"{total:,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Number of videos")
        ax.set_title("Migration & Climate Videos per Channel", fontweight="bold")
        ax.legend(loc="upper right")
        ax.margins(x=0.08)
        _save(fig, OUT_DIR / "videos" / "topic_videos_per_channel.png")

    # A6. Videos per party (AfD vs Die Linke)
    if "party" in video_meta.columns:
        party_counts = video_meta["party"].value_counts()
        colors_p = [PARTY_COLORS.get(p, "#999") for p in party_counts.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(party_counts.index, party_counts.values, color=colors_p)
        ax.set_ylabel("Number of videos")
        ax.set_title(f"All Scraped Videos per Party  (n={n_total:,})", fontweight="bold")
        _label_bars(ax, bars, party_counts.values, fontsize=10)
        _save(fig, OUT_DIR / "videos" / "videos_per_party.png")


    # A8. 2×2 facet: topic breakdown per party × channel
    if "party" in video_meta.columns and "category" in video_meta.columns:
        parties_list = sorted(video_meta["party"].dropna().unique())
        fig, axes = plt.subplots(1, len(parties_list),
                                  figsize=(6 * len(parties_list), 4), sharey=False)
        if len(parties_list) == 1:
            axes = [axes]
        for ax, party in zip(axes, parties_list):
            sub = video_meta[video_meta["party"] == party]
            tc  = sub["category"].value_counts()
            colors_t2 = [TOPIC_COLORS.get(t, "#999") for t in tc.index]
            bars = ax.bar(tc.index, tc.values, color=colors_t2)
            ax.set_title(f"{party}  (n={len(sub):,})", fontweight="bold")
            ax.set_ylabel("Videos")
            _label_bars(ax, bars, tc.values, fontsize=8)
        fig.suptitle("Topic Breakdown per Party", fontweight="bold")
        _save(fig, OUT_DIR / "videos" / "videos_topic_per_party_facet.png")


# B. Annotated pairs
def plot_annotated_pairs_overview(df):
    """Section B: all annotated comment pairs."""
    print("\n-- B. Annotated Pairs Overview")

    n_total = len(df)
    print(f"  Total annotated pairs: {n_total}")

    # colour palette for interaction classes
    INT_COLORS = ["#C00000", "#ED7D31", "#A9D18E", "#4472C4", "#7030A0"]

    # B1. Interaction score distribution (with proper class names)
    score_counts = (df["InteractionScore"]
                    .map(SCORE_TO_CLASS)
                    .value_counts().sort_index())
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(5),
                  [score_counts.get(i, 0) for i in range(5)],
                  color=INT_COLORS)
    ax.set_xticks(range(5))
    ax.set_xticklabels(INT_LABELS, fontsize=9)
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"Interaction Score Distribution  (n={n_total:,})", fontweight="bold")
    _label_bars(ax, bars, [score_counts.get(i, 0) for i in range(5)], fontsize=9)
    _save(fig, OUT_DIR / "pairs" / "interaction_score_dist.png")

    # B2. Pairs per topic
    topic_counts = df["Topic"].str.lower().replace("", "unknown").value_counts()
    colors_t = [TOPIC_COLORS.get(t, "#999") for t in topic_counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(topic_counts.index, topic_counts.values, color=colors_t)
    ax.set_ylabel("Number of pairs")
    ax.set_title(f"Comment Pairs per Topic  (n={n_total:,})", fontweight="bold")
    _label_bars(ax, bars, topic_counts.values, fontsize=10)
    _save(fig, OUT_DIR / "pairs" / "pairs_per_topic.png")

    # B3. Interaction score distribution by topic
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, (ax, topic) in enumerate(zip(axes, ["migration", "climate"])):
        sub = df[df["Topic"].str.lower() == topic]
        cls_cnts = (sub["InteractionScore"].map(SCORE_TO_CLASS)
                    .value_counts().sort_index())
        bars = ax.bar(range(5), [cls_cnts.get(i2, 0) for i2 in range(5)],
                      color=TOPIC_COLORS.get(topic, "#999"))
        ax.set_xticks(range(5))
        ax.set_xticklabels(INT_LABELS, fontsize=7)
        ax.set_title(f"Interaction Scores — {topic.capitalize()}  (n={len(sub):,})",
                     fontweight="bold")
        if i == 0:
            ax.set_ylabel("Count")
        _label_bars(ax, bars, [cls_cnts.get(i2, 0) for i2 in range(5)],
                    fontsize=8, adjust_ylim=False)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.25)
    fig.suptitle("Interaction Score Distribution by Topic", fontweight="bold")
    _save(fig, OUT_DIR / "pairs" / "interaction_dist_by_topic.png")

    # B4. Child + Parent stance distribution by topic
    stance_df = df[df["StanceLabel"].notna()].copy()
    stance_df["topic_clean"] = stance_df["Topic"].str.lower()
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.18
    topics = ["migration", "climate"]
    x = np.arange(3)
    for i, topic in enumerate(topics):
        sub = stance_df[stance_df["topic_clean"] == topic]
        counts_child = [(sub["StanceLabel"] == c).sum() for c in [0, 1, 2]]
        counts_parent = [(sub["parent_stance"] == c).sum() for c in [0, 1, 2]]
        offset = i * 2 * width
        color = TOPIC_COLORS.get(topic)
        bars_child = ax.bar(x + offset, counts_child, width,
                            label=f"{topic.capitalize()} child (n={len(sub):,})",
                            color=color)
        bars_parent = ax.bar(x + offset + width, counts_parent, width,
                             label=f"{topic.capitalize()} parent",
                             color=color, alpha=0.5, hatch="//")
        _label_bars(ax, bars_child, counts_child, total=len(sub), fontsize=7)
        for bar, v in zip(bars_parent, counts_parent):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + bar.get_height() * 0.02,
                        f"{100*v/len(sub):.0f}%",
                        ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(STANCE_LABELS, fontsize=10)
    ax.set_ylabel("Number of comments")
    ax.set_title(
        f"Child vs Parent Stance Distribution by Topic  (n={len(stance_df):,})",
        fontweight="bold"
    )
    ax.legend(fontsize=8)
    _save(fig, OUT_DIR / "pairs" / "stance_dist_by_topic.png")

    # B5. Technique frequency 
    tech_counts = defaultdict(int)
    for tech_str in df["Techniques"].dropna():
        try:
            techs = json.loads(tech_str) if isinstance(tech_str, str) else tech_str
            for t in techs:
                if t in VALID_TECHNIQUES:
                    tech_counts[t] += 1
        except Exception:
            pass
    if tech_counts:
        tech_df = (pd.DataFrame(list(tech_counts.items()),
                                columns=["Technique", "Count"])
                   .sort_values("Count"))
        # Shorten names for display
        short_names = [t.replace("_", " ").replace(",", "/") for t in tech_df["Technique"]]
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = sns.color_palette("Blues_d", len(tech_df))
        ax.barh(short_names, tech_df["Count"].values, color=colors)
        ax.set_xlabel("Count")
        ax.set_title("Propaganda / Rhetorical Technique Frequency", fontweight="bold")
        _label_hbars(ax, tech_df["Count"].values, fontsize=8)
        _save(fig, OUT_DIR / "pairs" / "technique_frequency.png")

        # B5b. Technique presence per topic
        topic_tech = {}
        for topic in ["migration", "climate"]:
            sub = df[df["Topic"].str.lower() == topic]
            tc = defaultdict(int)
            for tech_str in sub["Techniques"].dropna():
                try:
                    techs = json.loads(tech_str) if isinstance(tech_str, str) else tech_str
                    for t in techs:
                        if t in VALID_TECHNIQUES:
                            tc[t] += 1
                except Exception:
                    pass
            topic_tech[topic] = tc

        tnames = [t.replace("_", " ").replace(",", "/") for t in VALID_TECHNIQUES]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(VALID_TECHNIQUES))
        w = 0.38
        for i, (topic, tc) in enumerate(topic_tech.items()):
            vals = [tc.get(t, 0) for t in VALID_TECHNIQUES]
            b5b_bars = ax.bar(x + i * w, vals, w,
                              label=topic.capitalize(), color=TOPIC_COLORS[topic])
            _label_bars(ax, b5b_bars, vals, fontsize=6)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(tnames, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title("Technique Frequency by Topic", fontweight="bold")
        ax.legend()
        _save(fig, OUT_DIR / "pairs" / "technique_by_topic.png")

    # B6. Train / val / test split sizes
    split_counts = df["split"].value_counts().reindex(["train", "val", "test"])
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_split = ["#4472C4", "#ED7D31", "#A9D18E"]
    bars = ax.bar(split_counts.index, split_counts.values, color=colors_split)
    ax.set_ylabel("Number of pairs")
    ax.set_title("Train / Val / Test Split Sizes", fontweight="bold")
    _label_bars(ax, bars, split_counts.values, fontsize=10)
    _save(fig, OUT_DIR / "pairs" / "split_distribution.png")

    # B7. Interaction score distribution split × topic (heatmap-style)
    pivot = (df.groupby([df["Topic"].str.lower().replace("", "unknown"),
                         df["InteractionScore"].map(SCORE_TO_CLASS)])
             .size().unstack(fill_value=0))
    # Use full readable column labels (no newlines)
    col_labels = ["Dest. Disagreement", "Dest. Agreement",
                  "Neutral / Rephrase", "Constr. Agreement", "Constr. Disagreement"]
    pivot.columns = [col_labels[i] for i in pivot.columns]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Count"}, annot_kws={"size": 10})
    ax.set_title("Interaction Class × Topic  (counts)", fontweight="bold")
    ax.set_xlabel("Interaction Class")
    ax.set_ylabel("Topic")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    _save(fig, OUT_DIR / "pairs" / "interaction_topic_heatmap.png")

    # Party-based breakdowns
    if "party" not in df.columns:
        return

    INT_COLORS = ["#C00000", "#ED7D31", "#A9D18E", "#4472C4", "#7030A0"]
    parties = sorted(df["party"].dropna().unique())

    # B8. Interaction distribution per party
    fig, axes = plt.subplots(1, len(parties), figsize=(7 * len(parties), 5), sharey=True)
    if len(parties) == 1:
        axes = [axes]
    for pi, (ax, party) in enumerate(zip(axes, parties)):
        sub = df[df["party"] == party]
        cls_cnts = (sub["InteractionScore"].map(SCORE_TO_CLASS)
                    .value_counts().sort_index())
        bars = ax.bar(range(5), [cls_cnts.get(i, 0) for i in range(5)],
                      color=INT_COLORS)
        ax.set_xticks(range(5))
        ax.set_xticklabels(INT_LABELS, fontsize=7)
        ax.set_title(f"{party}  (n={len(sub):,})", fontweight="bold")
        if pi == 0:
            ax.set_ylabel("Count")
        _label_bars(ax, bars, [cls_cnts.get(i, 0) for i in range(5)],
                    fontsize=7, adjust_ylim=False)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.25)
    fig.suptitle("Interaction Score Distribution per Party", fontweight="bold")
    _save(fig, OUT_DIR / "pairs" / "interaction_dist_by_party.png")

    # B8b. Interaction distribution by party × topic
    topics_int = ["migration", "climate"]
    fig, axes = plt.subplots(len(parties), len(topics_int),
                              figsize=(7 * len(topics_int), 5 * len(parties)),
                              sharey=True)
    if len(parties) == 1:
        axes = [axes]
    for row, party in enumerate(parties):
        for col, topic in enumerate(topics_int):
            ax = axes[row][col]
            sub = df[(df["party"] == party) & (df["Topic"].str.lower() == topic)]
            cls_cnts = (sub["InteractionScore"].map(SCORE_TO_CLASS)
                        .value_counts().sort_index())
            bars = ax.bar(range(5), [cls_cnts.get(i2, 0) for i2 in range(5)],
                          color=INT_COLORS)
            ax.set_xticks(range(5))
            ax.set_xticklabels(INT_LABELS, fontsize=7)
            ax.set_title(f"{party} — {topic.capitalize()}  (n={len(sub):,})",
                         fontweight="bold", fontsize=9)
            if col == 0:
                ax.set_ylabel("Count")
            _label_bars(ax, bars, [cls_cnts.get(i2, 0) for i2 in range(5)],
                        fontsize=7, adjust_ylim=False)
    axes[0][0].set_ylim(top=axes[0][0].get_ylim()[1] * 1.25)
    fig.suptitle("Interaction Score Distribution by Party × Topic", fontweight="bold")
    plt.tight_layout()
    _save(fig, OUT_DIR / "pairs" / "interaction_dist_party_topic.png")

    # B9. Interaction × party heatmap
    pivot_party = (df.groupby([df["party"],
                               df["InteractionScore"].map(SCORE_TO_CLASS)])
                   .size().unstack(fill_value=0))
    pivot_party.columns = [INT_SHORT[i].replace("\n", " ")
                           for i in pivot_party.columns]
    fig, ax = plt.subplots(figsize=(10, max(3, len(parties) * 1.2)))
    sns.heatmap(pivot_party, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Interaction Class × Party  (counts)", fontweight="bold")
    ax.set_xlabel("Interaction Class")
    ax.set_ylabel("Party")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, va="center")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    _save(fig, OUT_DIR / "pairs" / "interaction_party_heatmap.png")

    # B10. Interaction × party × topic heatmap
    for topic in ["migration", "climate"]:
        sub = df[df["Topic"].str.lower() == topic]
        if sub.empty:
            continue
        pivot_pt = (sub.groupby([sub["party"],
                                  sub["InteractionScore"].map(SCORE_TO_CLASS)])
                    .size().unstack(fill_value=0))
        pivot_pt.columns = [INT_SHORT[i].replace("\n", " ")
                            for i in pivot_pt.columns]
        fig, ax = plt.subplots(figsize=(10, max(3, len(parties) * 1.2)))
        sns.heatmap(pivot_pt, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Count"})
        ax.set_title(f"Interaction × Party — {topic.capitalize()}  (n={len(sub):,})",
                     fontweight="bold")
        ax.set_xlabel("Interaction Class")
        ax.set_ylabel("Party")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, va="center")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        _save(fig, OUT_DIR / "pairs" / f"interaction_party_topic_{topic}_heatmap.png")

    # B11. Child + Parent stance distribution per party
    stance_df = df[df["StanceLabel"].notna()].copy()
    if not stance_df.empty:
        fig, axes = plt.subplots(1, len(parties), figsize=(7 * len(parties), 4),
                                  sharey=True)
        if len(parties) == 1:
            axes = [axes]
        bar_colors = [TOPIC_COLORS["migration"], "#AAAAAA", TOPIC_COLORS["climate"]]
        x = np.arange(3)
        width = 0.35
        for pi, (ax, party) in enumerate(zip(axes, parties)):
            sub = stance_df[stance_df["party"] == party]
            counts_child = [(sub["StanceLabel"] == c).sum() for c in [0, 1, 2]]
            counts_parent = [(sub["parent_stance"] == c).sum() for c in [0, 1, 2]]
            bars_child = ax.bar(x - width / 2, counts_child, width,
                                color=bar_colors, label="Child")
            bars_parent = ax.bar(x + width / 2, counts_parent, width,
                                 color=bar_colors, alpha=0.5, hatch="//", label="Parent")
            ax.set_xticks(x)
            ax.set_xticklabels(STANCE_LABELS, fontsize=9)
            ax.set_title(f"{party}  (n={len(sub):,})", fontweight="bold")
            if pi == 0:
                ax.set_ylabel("Count")
            _label_bars(ax, bars_child, counts_child, total=len(sub), fontsize=8, adjust_ylim=False)
            for bar, v in zip(bars_parent, counts_parent):
                if v:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + bar.get_height() * 0.02,
                            f"{100*v/len(sub):.0f}%",
                            ha="center", va="bottom", fontsize=7)
            ax.legend(fontsize=8)
        axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.25)
        fig.suptitle("Child vs Parent Stance Distribution per Party", fontweight="bold")
        _save(fig, OUT_DIR / "pairs" / "stance_dist_by_party.png")

    # B11b. Child + Parent stance distribution by party × topic (2×2 grid)
    stance_df2 = df[df["StanceLabel"].notna()].copy()
    if not stance_df2.empty and "party" in df.columns:
        topics_st = ["migration", "climate"]
        fig, axes = plt.subplots(len(parties), len(topics_st),
                                  figsize=(7 * len(topics_st), 4 * len(parties)),
                                  sharey=True)
        if len(parties) == 1:
            axes = [axes]
        bar_colors = [TOPIC_COLORS["migration"], "#AAAAAA", TOPIC_COLORS["climate"]]
        x = np.arange(3)
        width = 0.35
        for row, party in enumerate(parties):
            for col, topic in enumerate(topics_st):
                ax = axes[row][col]
                sub = stance_df2[
                    (stance_df2["party"] == party) &
                    (stance_df2["Topic"].str.lower() == topic)
                ]
                counts_child = [(sub["StanceLabel"] == c).sum() for c in [0, 1, 2]]
                counts_parent = [(sub["parent_stance"] == c).sum() for c in [0, 1, 2]]
                bars_child = ax.bar(x - width / 2, counts_child, width,
                                    color=bar_colors, label="Child")
                bars_parent = ax.bar(x + width / 2, counts_parent, width,
                                     color=bar_colors, alpha=0.5, hatch="//", label="Parent")
                ax.set_xticks(x)
                ax.set_xticklabels(STANCE_LABELS, fontsize=9)
                ax.set_title(f"{party} — {topic.capitalize()}  (n={len(sub):,})",
                             fontweight="bold", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Count")
                _label_bars(ax, bars_child, counts_child, total=len(sub),
                            fontsize=8, adjust_ylim=False)
                for bar, v in zip(bars_parent, counts_parent):
                    if v:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + bar.get_height() * 0.02,
                                f"{100*v/len(sub):.0f}%",
                                ha="center", va="bottom", fontsize=7)
                ax.legend(fontsize=8)
        axes[0][0].set_ylim(top=axes[0][0].get_ylim()[1] * 1.25)
        fig.suptitle("Child vs Parent Stance Distribution by Party × Topic", fontweight="bold")
        _save(fig, OUT_DIR / "pairs" / "stance_dist_party_topic.png")

    # B12. Technique frequency per party
    fig, axes = plt.subplots(1, len(parties), figsize=(9 * len(parties), 7),
                              sharey=True, sharex=True)
    if len(parties) == 1:
        axes = [axes]
    tnames_short = [t.replace("_", " ").replace(",", "/") for t in VALID_TECHNIQUES]
    for pi, (ax, party) in enumerate(zip(axes, parties)):
        sub = df[df["party"] == party]
        tc  = defaultdict(int)
        for tech_str in sub["Techniques"].dropna():
            try:
                techs = json.loads(tech_str) if isinstance(tech_str, str) else tech_str
                for t in techs:
                    if t in VALID_TECHNIQUES:
                        tc[t] += 1
            except Exception:
                pass
        vals = [tc.get(t, 0) for t in VALID_TECHNIQUES]
        ax.barh(tnames_short, vals, color=PARTY_COLORS.get(party, "#999"))
        _label_hbars(ax, vals, fontsize=8, adjust_xlim=False)
        ax.set_title(f"{party}  (n={len(sub):,})", fontweight="bold")
        ax.set_xlabel("Count")
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 1.35)
    fig.suptitle("Technique Frequency per Party", fontweight="bold")
    _save(fig, OUT_DIR / "pairs" / "technique_by_party.png")

    # B12b. Technique frequency by party × topic (grid: party rows × topic cols)
    topics_tech = ["migration", "climate"]
    fig, axes = plt.subplots(len(parties), len(topics_tech),
                              figsize=(10 * len(topics_tech), 7 * len(parties)),
                              sharey=True, sharex=True)
    if len(parties) == 1:
        axes = [axes]
    tnames_short2 = [t.replace("_", " ").replace(",", "/") for t in VALID_TECHNIQUES]
    last_row = len(parties) - 1
    for row, party in enumerate(parties):
        for col, topic in enumerate(topics_tech):
            ax = axes[row][col]
            sub = df[(df["party"] == party) & (df["Topic"].str.lower() == topic)]
            tc = defaultdict(int)
            for tech_str in sub["Techniques"].dropna():
                try:
                    techs = json.loads(tech_str) if isinstance(tech_str, str) else tech_str
                    for t in techs:
                        if t in VALID_TECHNIQUES:
                            tc[t] += 1
                except Exception:
                    pass
            vals = [tc.get(t, 0) for t in VALID_TECHNIQUES]
            ax.barh(tnames_short2, vals, color=PARTY_COLORS.get(party, "#999"))
            _label_hbars(ax, vals, fontsize=7, adjust_xlim=False)
            ax.set_title(f"{party} — {topic.capitalize()}  (n={len(sub):,})",
                         fontweight="bold", fontsize=9)
            if row == last_row:
                ax.set_xlabel("Count")
    axes[0][0].set_xlim(right=axes[0][0].get_xlim()[1] * 1.35)
    fig.suptitle("Technique Frequency by Party × Topic", fontweight="bold")
    plt.tight_layout()
    _save(fig, OUT_DIR / "pairs" / "technique_party_topic.png")


# C. Temporal analysis
def load_thread_scores(video_meta, df):
    """Load thread scores and join with video metadata + topic."""
    records = []
    if THREAD_SCORE_DIR.exists():
        for p in THREAD_SCORE_DIR.glob("thread_scores_*.jsonl"):
            records.extend(_load_jsonl(p))
    if not records:
        return pd.DataFrame()

    thread_df = pd.DataFrame(records)
    vid_topic = df[["VideoID", "Topic"]].drop_duplicates()

    if video_meta.empty or "year_dt" not in video_meta.columns:
        thread_df = thread_df.merge(vid_topic, on="VideoID", how="left")
        return thread_df

    meta_mini = (video_meta[["videoId", "publishedAt", "channelName",
                               "year_dt", "year_quarter"]]
                 .drop_duplicates("videoId"))
    thread_df = (thread_df
                 .merge(meta_mini, left_on="VideoID", right_on="videoId", how="left")
                 .merge(vid_topic, on="VideoID", how="left"))
    return thread_df


def plot_temporal_analysis(df, video_meta, thread_df):
    """Section C: temporal trends."""
    print("\n-- C. Temporal Analysis")

    if video_meta.empty or "year_dt" not in video_meta.columns:
        print("  No video date metadata -- skipping temporal section.")
        return

    # C1. Videos per channel over time (yearly line)
    ch_year = (video_meta.dropna(subset=["year_dt", "channelName"])
               .groupby(["year_dt", "channelName"]).size().reset_index(name="count"))
    channels = video_meta["channelName"].value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    for ch in channels:
        sub = ch_year[ch_year["channelName"] == ch].sort_values("year_dt")
        ax.plot(sub["year_dt"], sub["count"], marker="o",
                label=ch, color=CHANNEL_COLORS.get(ch, "#999"),
                linewidth=2, markersize=5)
    _fix_yearly_axis(ax, fig)
    ax.set_ylabel("Number of videos")
    ax.set_title("Videos per Channel -- Yearly", fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, OUT_DIR / "temporal" / "videos_per_channel_yearly.png")

    # C2. Videos per topic over time
    vid_topic = df[["VideoID", "Topic"]].drop_duplicates()
    vid_topic["Topic"] = vid_topic["Topic"].str.lower()
    meta_with_topic = video_meta.merge(vid_topic, left_on="videoId",
                                        right_on="VideoID", how="left")
    topic_year = (meta_with_topic.dropna(subset=["year_dt", "Topic"])
                  .groupby(["year_dt", "Topic"]).size().reset_index(name="count"))
    fig, ax = plt.subplots(figsize=(11, 4))
    for topic in ["migration", "climate"]:
        sub = topic_year[topic_year["Topic"] == topic].sort_values("year_dt")
        if not sub.empty:
            ax.plot(sub["year_dt"], sub["count"], marker="o",
                    label=topic.capitalize(), color=TOPIC_COLORS[topic], linewidth=2)
            ax.fill_between(sub["year_dt"], sub["count"],
                            alpha=0.15, color=TOPIC_COLORS[topic])
    _fix_yearly_axis(ax, fig)
    ax.set_ylabel("Number of videos")
    ax.set_title("Migration & Climate Videos -- Yearly", fontweight="bold")
    ax.legend()
    _save(fig, OUT_DIR / "temporal" / "videos_per_topic_yearly.png")

    if thread_df.empty or "year_dt" not in thread_df.columns:
        print("  No thread date data -- skipping destructiveness plots.")
        return

    thread_df = thread_df.copy()
    thread_df["destructiveness"] = -thread_df["MeanScore"]

    # C3. Destructiveness over time per topic (yearly mean)
    dest_topic = (thread_df.dropna(subset=["year_dt", "Topic"])
                  .groupby(["year_dt", "Topic"])
                  .agg(mean_dest=("destructiveness", "mean"),
                       n=("ThreadID", "count"))
                  .reset_index())
    fig, ax = plt.subplots(figsize=(11, 4))
    for topic in ["migration", "climate"]:
        sub = dest_topic[dest_topic["Topic"] == topic].sort_values("year_dt")
        if not sub.empty:
            ax.plot(sub["year_dt"], sub["mean_dest"], marker="o",
                    label=topic.capitalize(), color=TOPIC_COLORS[topic], linewidth=2)
            ax.fill_between(sub["year_dt"], sub["mean_dest"],
                            alpha=0.12, color=TOPIC_COLORS[topic])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    _fix_yearly_axis(ax, fig)
    ax.set_ylabel("Mean destructiveness\n(higher = more destructive)")
    ax.set_title("Thread Destructiveness Over Time -- By Topic", fontweight="bold")
    ax.legend()
    _save(fig, OUT_DIR / "temporal" / "destructiveness_over_time_topic.png")

    # C4. Destructiveness over time per channel
    if "channelName" in thread_df.columns:
        dest_ch = (thread_df.dropna(subset=["year_dt", "channelName"])
                   .groupby(["year_dt", "channelName"])
                   .agg(mean_dest=("destructiveness", "mean"),
                        n=("ThreadID", "count"))
                   .reset_index())
        top_channels = thread_df["channelName"].value_counts().head(4).index.tolist()
        fig, ax = plt.subplots(figsize=(12, 5))
        for ch in top_channels:
            sub = dest_ch[dest_ch["channelName"] == ch].sort_values("year_dt")
            ax.plot(sub["year_dt"], sub["mean_dest"], marker="o",
                    label=ch, color=CHANNEL_COLORS.get(ch, "#999"), linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        _fix_yearly_axis(ax, fig)
        ax.set_ylabel("Mean destructiveness")
        ax.set_title("Thread Destructiveness Over Time -- By Channel", fontweight="bold")
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        _save(fig, OUT_DIR / "temporal" / "destructiveness_over_time_channel.png")

    # C5. Thread quality composition over time (stacked area) -- per topic
    if "ThreadBin" in thread_df.columns:
        for topic in ["migration", "climate"]:
            sub = thread_df[
                thread_df["Topic"].str.lower() == topic
            ].dropna(subset=["year_dt", "ThreadBin"])
            if sub.empty:
                continue
            binned = (sub.groupby(["year_dt", "ThreadBin"])
                      .size().unstack(fill_value=0))
            binned_pct = binned.div(binned.sum(axis=1), axis=0)

            n_per_year = {y.year: n for y, n in binned.sum(axis=1).items()}

            bin_cols = [c for c in
                        ["Constructive", "Slight/Neutral",
                         "Moderately Destructive", "Highly Destructive"]
                        if c in binned_pct.columns]
            if not bin_cols:
                continue
            colors_bin = ["#70AD47", "#A9D18E", "#ED7D31", "#C00000"]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.stackplot(binned_pct.index,
                         [binned_pct[c] for c in bin_cols],
                         labels=bin_cols,
                         colors=colors_bin[:len(bin_cols)], alpha=0.85)
            _fix_yearly_axis(ax, fig)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_year_n_fmt(n_per_year)))
            ax.set_ylabel("Share of threads")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.set_title(
                f"Thread Quality Composition Over Time -- {topic.capitalize()}",
                fontweight="bold"
            )
            ax.legend(loc="upper left", fontsize=8)
            _save(fig, OUT_DIR / "temporal" /
                  f"thread_bin_share_yearly_{topic}.png")

    # C6. Thread bin share yearly by party (stacked area)
    if ("ThreadBin" in thread_df.columns and "channelName" in thread_df.columns
            and "year_dt" in thread_df.columns):
        bin_cols_all = ["Constructive", "Slight/Neutral",
                        "Moderately Destructive", "Highly Destructive"]
        colors_bin   = ["#70AD47", "#A9D18E", "#ED7D31", "#C00000"]
        parties_t    = sorted(
            thread_df["party"].dropna().unique()
        ) if "party" in thread_df.columns else []
        for party in parties_t:
            sub = thread_df[
                (thread_df["party"] == party)
            ].dropna(subset=["year_dt", "ThreadBin"])
            if sub.empty:
                continue
            binned = (sub.groupby(["year_dt", "ThreadBin"])
                      .size().unstack(fill_value=0))
            binned_pct = binned.div(binned.sum(axis=1), axis=0)

            n_per_year = {y.year: n for y, n in binned.sum(axis=1).items()}

            bin_cols = [c for c in bin_cols_all if c in binned_pct.columns]
            if not bin_cols:
                continue
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.stackplot(binned_pct.index,
                         [binned_pct[c] for c in bin_cols],
                         labels=bin_cols,
                         colors=colors_bin[:len(bin_cols)], alpha=0.85)
            _fix_yearly_axis(ax, fig)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_year_n_fmt(n_per_year)))
            ax.set_ylabel("Share of threads")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.set_title(
                f"Thread Quality Composition Over Time — {party}",
                fontweight="bold"
            )
            ax.legend(loc="upper left", fontsize=8)
            safe_name = party.replace(" ", "_").replace("/", "_")
            _save(fig, OUT_DIR / "temporal" /
                  f"thread_bin_share_yearly_party_{safe_name}.png")

# D. Model comparison
def plot_model_comparison():
    print("\n-- D. Model Comparison")

    shallow_m = _load_json(SHALLOW_DIR / "metrics.json")
    bert_m    = _load_json(BERT_DIR    / "metrics.json")
    llm_m     = _load_json(LLM_DIR     / "metrics.json")

    if not shallow_m:
        print("  No shallow metrics -- skipping.")
        return

    bassi_sets = ["B", "B+S", "B+S+T", "B+S+T+E1", "B+S+T+E2"]
    models     = ["LR", "RF", "SVM", "XGB"]
    palette    = sns.color_palette("tab10", len(models))

    # D1. Ablation B -> B+S -> B+S+T -> E1 -> E2 (interaction)
    model_vals_d1 = {}
    for i, mn in enumerate(models):
        vals = [shallow_m.get("interaction", {}).get(mn, {})
                         .get(fs, {}).get("macro_f1", None)
                for fs in bassi_sets]
        model_vals_d1[mn] = vals

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, mn in enumerate(models):
        vals = model_vals_d1[mn]
        x_valid = [j for j, v in enumerate(vals) if v is not None]
        y_valid = [v for v in vals if v is not None]
        if x_valid:
            ax.plot([bassi_sets[j] for j in x_valid], y_valid,
                    marker="o", label=mn, color=palette[i],
                    linewidth=2, markersize=7)


    all_valid_d1 = [v for vals in model_vals_d1.values() for v in vals if v is not None]
    if all_valid_d1:
        lo_d1, hi_d1 = min(all_valid_d1), max(all_valid_d1)
        y_span   = hi_d1 - lo_d1
        band_lo  = lo_d1 - y_span * 0.30   
        band_hi  = hi_d1 + y_span * 0.30   
        x_half   = 0.38   
        x_shifts = [-x_half, x_half, -x_half, x_half]

        for col_idx, fs in enumerate(bassi_sets):
            pts = [(i, mn, model_vals_d1[mn][col_idx])
                   for i, mn in enumerate(models)
                   if model_vals_d1[mn][col_idx] is not None]
            if not pts:
                continue
            pts_sorted = sorted(pts, key=lambda t: t[2])   # ascending by value
            n_pts = len(pts_sorted)
            if n_pts == 1:
                label_ys = [(band_lo + band_hi) / 2]
            else:
                step = (band_hi - band_lo) / (n_pts - 1)
                label_ys = [band_lo + k * step for k in range(n_pts)]
            for rank, (i, mn, v) in enumerate(pts_sorted):
                lx = col_idx + x_shifts[rank % len(x_shifts)]
                ly = label_ys[rank]
                ax.annotate(f"{v:.3f}",
                            xy=(col_idx, v),
                            xytext=(lx, ly),
                            xycoords=("data", "data"),
                            textcoords=("data", "data"),
                            ha="center", va="center", fontsize=8.5,
                            color=palette[i], fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                      ec=palette[i], lw=0.7, alpha=0.90),
                            arrowprops=dict(arrowstyle="-", color=palette[i],
                                            lw=0.9, shrinkA=5, shrinkB=5))
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Feature-Block Ablation — Interaction Task\n(B → B+S → B+S+T → B+S+T+E)",
                 fontweight="bold")
    ax.legend(title="Classifier", loc="upper left")
    # Set y limits to match label band so labels are always inside the axes
    if all_valid_d1:
        ax.set_ylim(band_lo - y_span * 0.05, band_hi + y_span * 0.05)
    # Widen x margins so side labels aren't clipped
    ax.margins(x=0.10)
    ax.grid(axis="y", alpha=0.4)
    _save(fig, OUT_DIR / "models" / "ablation_progression_interaction.png")

    # D2. All classifiers x all feat-sets heatmap (interaction)
    all_feat_sets = sorted({fs for fd in shallow_m.get("interaction", {}).values()
                             for fs in fd.keys()})
    all_models    = ["LR", "RF", "SVM", "SVM_RBF", "XGB"]
    hmap_data = pd.DataFrame(
        {fs: [shallow_m.get("interaction", {}).get(mn, {})
                       .get(fs, {}).get("macro_f1", np.nan)
              for mn in all_models]
         for fs in all_feat_sets},
        index=all_models,
    )
    fig, ax = plt.subplots(figsize=(max(10, len(all_feat_sets) * 0.9),
                                    max(4, len(all_models) * 0.9)))
    sns.heatmap(hmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Macro-F1"},
                annot_kws={"size": 8})
    ax.set_title("Shallow Learners -- Interaction Task -- All Feature Sets",
                 fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    _save(fig, OUT_DIR / "models" / "shallow_interaction_heatmap.png")

    # D3. Best per approach -- per task
    approach_results = {}
    for task in ["interaction", "stance", "techniques"]:
        best_mn, best_fs = _best_shallow(shallow_m, task)
        if best_mn:
            f1 = shallow_m[task][best_mn][best_fs]["macro_f1"]
            approach_results.setdefault(task, []).append(
                {"approach": f"Shallow\n({best_mn}/{best_fs})",
                 "macro_f1": f1, "color": "#4472C4"})

    for mk, task_dict in bert_m.items():
        for task, m in task_dict.items():
            f1 = m.get("macro_f1", 0)
            existing = approach_results.get(task, [])
            bert_entries = [e for e in existing if "BERT" in e["approach"]]
            if not bert_entries or f1 > bert_entries[0]["macro_f1"]:
                approach_results.setdefault(task, [])
                approach_results[task] = [e for e in approach_results[task]
                                           if "BERT" not in e["approach"]]
                approach_results[task].append(
                    {"approach": f"BERT\n({mk})", "macro_f1": f1,
                     "color": "#7030A0"})

    for task, model_dict in llm_m.items():
        best_f1_llm, best_label = 0, ""
        for ml, cond_dict in model_dict.items():
            for cond, shot_dict in cond_dict.items():
                for shot, m in shot_dict.items():
                    f1 = m.get("macro_f1", 0)
                    if f1 > best_f1_llm:
                        best_f1_llm = f1
                        shot_lbl    = "zero" if "zero" in shot else "few"
                        best_label  = f"LLM\n({_llm_label(ml)}/{shot_lbl})"
        if best_f1_llm > 0:
            approach_results.setdefault(task, []).append(
                {"approach": best_label, "macro_f1": best_f1_llm,
                 "color": "#C00000"})

    for task, entries in approach_results.items():
        if not entries:
            continue
        entries_s = sorted(entries, key=lambda e: e["macro_f1"])
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh([e["approach"] for e in entries_s],
                       [e["macro_f1"] for e in entries_s],
                       color=[e["color"] for e in entries_s])
        for bar, e in zip(bars, entries_s):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{e['macro_f1']:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, min(1.0, max(e["macro_f1"] for e in entries) + 0.1))
        ax.set_xlabel("Test macro-F1")
        ax.set_title(f"Best per Approach -- {task.capitalize()} Task",
                     fontweight="bold")
        _save(fig, OUT_DIR / "models" / f"approach_comparison_{task}.png")

    # D4. E1 vs E2 comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(models)), 0.3
    e1v = [shallow_m.get("interaction", {}).get(mn, {})
                    .get("B+S+T+E1", {}).get("macro_f1", 0) for mn in models]
    e2v = [shallow_m.get("interaction", {}).get(mn, {})
                    .get("B+S+T+E2", {}).get("macro_f1", 0) for mn in models]
    b1  = ax.bar(x - w / 2, e1v, w, label="B+S+T+E1 (gbert)",    color="#7030A0")
    b2  = ax.bar(x + w / 2, e2v, w, label="B+S+T+E2 (groberta)", color="#00B050")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Embedding Comparison: E1 (gbert) vs E2 (groberta)\nInteraction Task",
                 fontweight="bold")
    ax.legend(); ax.set_ylim(bottom=0)
    for bars_grp in (b1, b2):
        for bar in bars_grp:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    _save(fig, OUT_DIR / "models" / "e1_vs_e2_comparison.png")

    # D5. LLM conditions comparison
    llm_task = llm_m.get("interaction", {})
    if llm_task:
        cond_short = {
            "full_instructions":                   "B\n(text only)",
            "full_instructions+stance":            "B+S\n(+stance)",
            "full_instructions+stance+techniques": "B+S+T\n(+tech)",
        }
        llm_models  = list(llm_task.keys())
        conditions  = list(cond_short.keys())
        fig, ax     = plt.subplots(figsize=(10, 5))
        x           = np.arange(len(conditions))
        w           = 0.25
        pal_llm     = ["#C00000", "#FF6600", "#C55A11"]
        for i, ml in enumerate(llm_models):
            vals = []
            for cond in conditions:
                best = max(
                    (llm_task.get(ml, {}).get(cond, {})
                              .get(s, {}).get("macro_f1", 0)
                     for s in ["zero_shot", "few_shot"]),
                    default=0,
                )
                vals.append(best)
            d5_bars = ax.bar(x + i * w, vals, w, label=_llm_label(ml),
                             color=pal_llm[i % len(pal_llm)])
            for bar, v in zip(d5_bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x + w)
        ax.set_xticklabels([cond_short[c] for c in conditions], fontsize=11)
        ax.set_ylabel("Test macro-F1 (best of zero/few-shot)")
        ax.set_title("LLM Interaction Task — Context Condition Comparison",
                     fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=len(llm_models), frameon=True)
        plt.subplots_adjust(bottom=0.28)
        _save(fig, OUT_DIR / "models" / "llm_conditions_interaction.png")

    # D6. Best per approach x task (heatmap)
    all_tasks = ["interaction", "stance", "techniques"]
    rows = []
    for task in all_tasks:
        for mn in all_models:
            for fs, m in shallow_m.get(task, {}).get(mn, {}).items():
                rows.append({"approach": "Shallow", "task": task,
                              "model": mn, "feat_set": fs,
                              "macro_f1": m.get("macro_f1", np.nan)})
        for mk, td in bert_m.items():
            m = td.get(task, {})
            if m:
                rows.append({"approach": "BERT", "task": task,
                              "model": mk, "feat_set": "fine-tuned",
                              "macro_f1": m.get("macro_f1", np.nan)})
        for ml, cond_dict in llm_m.get(task, {}).items():
            for cond, shot_dict in cond_dict.items():
                for shot, m in shot_dict.items():
                    rows.append({"approach": "LLM", "task": task,
                                  "model": _llm_label(ml), "feat_set": shot,
                                  "macro_f1": m.get("macro_f1", np.nan)})


    # D7. Zero-shot vs few-shot — all tasks
    cond_short_d7 = {
        "full_instructions":                   "B",
        "full_instructions+stance":            "B+S",
        "full_instructions+stance+techniques": "B+S+T",
    }
    for task_d7 in ["interaction", "stance", "techniques"]:
        llm_task_d7 = llm_m.get(task_d7, {})
        if not llm_task_d7:
            continue
        shot_rows_d7 = []
        for ml, cond_dict in llm_task_d7.items():
            for cond, shot_dict in cond_dict.items():
                for shot, m in shot_dict.items():
                    shot_rows_d7.append({
                        "model":    _llm_label(ml),
                        "condition": cond_short_d7.get(cond, cond),
                        "shot":     shot.replace("_", " "),
                        "macro_f1": m.get("macro_f1", 0),
                    })
        if not shot_rows_d7:
            continue
        shot_df_d7 = pd.DataFrame(shot_rows_d7)
        pivot_d7 = shot_df_d7.pivot_table(
            index=["model", "condition"], columns="shot",
            values="macro_f1", aggfunc="first",
        )
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot_d7) * 0.45)))
        sns.heatmap(pivot_d7, annot=True, fmt=".3f", cmap="RdYlGn",
                    linewidths=0.5, ax=ax, vmin=0,
                    cbar_kws={"label": "Macro-F1"})
        ax.set_title(
            f"LLM: Zero-shot vs Few-shot — {task_d7.capitalize()} Task",
            fontweight="bold"
        )
        ax.set_xlabel("")
        _save(fig, OUT_DIR / "models" /
              f"llm_zeroshot_vs_fewshot_{task_d7}.png")


# E. Confusion matrices
def plot_confusion_matrices():
    print("\n-- E. Confusion Matrices")

    shallow_m = _load_json(SHALLOW_DIR / "metrics.json")
    pred_dir  = SHALLOW_DIR / "predictions"

    for task in ["interaction", "stance"]:
        best_mn, best_fs = _best_shallow(shallow_m, task)
        if best_mn is None:
            continue
        preds = _load_jsonl(pred_dir / f"{task}_{best_mn}_{best_fs}.jsonl")
        if preds:
            cm, _, tick_labels = _build_confusion(preds, task)
            if cm is not None:
                _plot_cm(
                    cm, tick_labels,
                    f"Shallow Learner -- {task.capitalize()}\n({best_mn} / {best_fs})",
                    OUT_DIR / "confusion" / f"confusion_shallow_{task}.png",
                )

    # Best LLM confusion (interaction)
    llm_pred_dir = LLM_DIR / "predictions"
    if llm_pred_dir.exists():
        from sklearn.metrics import f1_score
        best_f1_llm, best_key = 0, None
        best_yt, best_yp = [], []
        for pred_file in sorted(llm_pred_dir.glob("interaction_*.jsonl")):
            records = _load_jsonl(pred_file)
            if not records:
                continue
            try:
                yt = [SCORE_TO_CLASS[min(SCORE_ENUM,
                      key=lambda k: abs(k - float(r["y_true"])))]
                      for r in records]
                yp = [SCORE_TO_CLASS[min(SCORE_ENUM,
                      key=lambda k: abs(k - float(r.get("y_pred_score", 0))))]
                      for r in records]
                f1 = f1_score(yt, yp, average="macro", zero_division=0)
                if f1 > best_f1_llm:
                    best_f1_llm = f1
                    best_key    = pred_file.stem
                    best_yt, best_yp = yt, yp
            except Exception:
                pass
        if best_key:
            cm = sk_confusion_matrix(best_yt, best_yp, labels=list(range(5)))
            _plot_cm(
                cm, INT_SHORT,
                f"LLM -- Interaction  ({best_key}  F1={best_f1_llm:.3f})",
                OUT_DIR / "confusion" / "confusion_llm_interaction.png",
                cmap="Reds",
            )

        # Best LLM confusion — stance
        best_f1_s, best_key_s, best_yt_s, best_yp_s = 0, None, [], []
        for pred_file in sorted(llm_pred_dir.glob("stance_*.jsonl")):
            records = _load_jsonl(pred_file)
            if not records:
                continue
            try:
                yt = [int(r["y_true"]) for r in records]
                yp = [int(r["y_pred"]) for r in records]
                f1 = f1_score(yt, yp, average="macro", zero_division=0)
                if f1 > best_f1_s:
                    best_f1_s, best_key_s = f1, pred_file.stem
                    best_yt_s, best_yp_s  = yt, yp
            except Exception:
                pass
        if best_key_s:
            cm = sk_confusion_matrix(best_yt_s, best_yp_s, labels=[0, 1, 2])
            _plot_cm(
                cm, STANCE_LABELS,
                f"LLM -- Stance  ({best_key_s}  F1={best_f1_s:.3f})",
                OUT_DIR / "confusion" / "confusion_llm_stance.png",
                cmap="Reds",
            )


    # Side-by-side: best shallow for both tasks
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plotted = 0
    for ax, task in zip(axes, ["interaction", "stance"]):
        best_mn, best_fs = _best_shallow(shallow_m, task)
        if best_mn is None:
            continue
        preds = _load_jsonl(pred_dir / f"{task}_{best_mn}_{best_fs}.jsonl")
        if not preds:
            continue
        cm, _, tick_labels = _build_confusion(preds, task)
        if cm is None:
            continue
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Row-norm."},
                    annot_kws={"size": 9})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_title(f"{task.capitalize()} -- {best_mn}/{best_fs}", fontweight="bold")
        plotted += 1
    if plotted:
        fig.suptitle("Best Shallow Learner -- Confusion Matrices", fontweight="bold")
        _save(fig, OUT_DIR / "confusion" / "confusion_shallow_side_by_side.png")
    else:
        plt.close(fig)

    # BERT side-by-side (gbert vs xlmr) for interaction
    for task in ["interaction", "stance"]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, mk in zip(axes, ["gbert", "xlmr"]):
            bert_preds = _load_jsonl(BERT_DIR / "predictions" / f"{mk}_{task}.jsonl")
            if not bert_preds:
                continue
            cm, _, tick_labels = _build_confusion(bert_preds, task)
            if cm is None:
                continue
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
            sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Purples",
                        xticklabels=tick_labels, yticklabels=tick_labels,
                        linewidths=0.5, ax=ax, cbar_kws={"label": "Row-norm."},
                        annot_kws={"size": 9})
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            ax.set_title(f"BERT ({mk}) -- {task.capitalize()}", fontweight="bold")
        fig.suptitle(f"BERT Models -- {task.capitalize()} Confusion Matrices",
                     fontweight="bold")
        _save(fig, OUT_DIR / "confusion" / f"confusion_bert_sidebyside_{task}.png")


# F. Feature importance
def plot_feature_importance():
    print("\n-- F. Feature Importance")

    shap_dir  = EXPL_DIR / "shap_shallow"
    perm_file = EXPL_DIR / "permutation_importance.json"
    perm_data = _load_json(perm_file)

    # F1. SHAP mean |SHAP| -- top-20 per task/model
    if shap_dir.exists():
        for shap_json in sorted(shap_dir.glob("*_mean_shap.json")):
            stem  = shap_json.stem
            parts = stem.split("_")
            task  = parts[0]
            model = parts[1] if len(parts) > 1 else "?"
            data  = _load_json(shap_json)
            if not data:
                continue
            items = sorted(data.items(), key=lambda x: -x[1])[:20]
            names_raw, vals = zip(*items)
            if _is_embedding_features(names_raw):
                print(f"  Skipping SHAP {stem}: raw embedding dimensions, not interpretable.")
                continue
            names = [_feat_label(n) for n in names_raw]
            fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.38)))
            colors  = sns.color_palette("Blues_d", len(names))
            shap_bars = ax.barh(list(reversed(names)), list(reversed(vals)),
                                color=list(reversed(colors)))
            for bar, v in zip(shap_bars, list(reversed(vals))):
                ax.text(bar.get_width() + max(list(reversed(vals))) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=7)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance — {task} / {model}",
                         fontweight="bold")
            _save(fig, OUT_DIR / "features" / f"shap_{task}_{model}.png")

    # F2. Permutation importance -- top-20 per task/model
    for task, model_dict in perm_data.items():
        for model_name, feat_list in model_dict.items():
            if not feat_list:
                continue
            top       = sorted(feat_list, key=lambda x: -x["importance"])[:20]
            raw_names = [f["feature"] for f in reversed(top)]
            if _is_embedding_features(raw_names):
                print(f"  Skipping perm importance {task}/{model_name}: raw embedding dimensions.")
                continue
            names  = [_feat_label(n) for n in raw_names]
            vals   = [f["importance"] for f in reversed(top)]
            stds   = [f["std"]        for f in reversed(top)]
            fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.38)))
            colors  = sns.color_palette("Oranges_d", len(names))
            perm_bars = ax.barh(names, vals, xerr=stds, color=colors, capsize=3)
            # Compute axis span for a relative margin
            val_range = max(abs(v) + s for v, s in zip(vals, stds)) or 1e-9
            margin = val_range * 0.04
            for bar, v, s in zip(perm_bars, vals, stds):
                if v >= 0:
                    x_pos, ha = v + s + margin, "left"
                else:
                    x_pos, ha = v - s - margin, "right"
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", ha=ha, fontsize=7)
            # Expand x-limits to ensure labels are not clipped
            all_extents = [v + s + margin * 4 for v, s in zip(vals, stds)] + \
                          [v - s - margin * 4 for v, s in zip(vals, stds)]
            ax.set_xlim(min(all_extents + [ax.get_xlim()[0]]),
                        max(all_extents + [ax.get_xlim()[1]]))
            ax.set_xlabel("Permutation importance (delta macro-F1)")
            ax.set_title(f"Permutation Importance — {task} / {model_name}",
                         fontweight="bold")
            _save(fig, OUT_DIR / "features" /
                  f"perm_importance_{task}_{model_name}.png")

    # F3. Block-level contribution (B / B+S / B+S+T) for interaction
    shallow_m  = _load_json(SHALLOW_DIR / "metrics.json")
    block_sets = ["B", "B+S", "B+S+T"]
    fig, ax    = plt.subplots(figsize=(8, 5))
    x, w       = np.arange(len(block_sets)), 0.2
    for i, mn in enumerate(["LR", "RF", "SVM", "XGB"]):
        vals = [shallow_m.get("interaction", {}).get(mn, {})
                         .get(fs, {}).get("macro_f1", 0)
                for fs in block_sets]
        f3_bars = ax.bar(x + i * w, vals, w, label=mn,
                         color=sns.color_palette("tab10", 4)[i])
        for bar, v in zip(f3_bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(["B\n(linguistic only)", "B+S\n(+ stance)",
                         "B+S+T\n(+ techniques)"])
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Feature Block Contribution -- Interaction Task", fontweight="bold")
    ax.legend(title="Classifier", ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    plt.subplots_adjust(bottom=0.30)
    _save(fig, OUT_DIR / "features" / "block_contribution_interaction.png")

    # F4. Copy beeswarm PNGs from step 19 into features output folder
    beeswarm_src = EXPL_DIR / "shap_shallow"
    if beeswarm_src.exists():
        for bee_png in sorted(beeswarm_src.glob("*_beeswarm.png")):
            dest = OUT_DIR / "features" / f"beeswarm_{bee_png.name}"
            shutil.copy(bee_png, dest)
            print(f"  Copied beeswarm -> {dest}")
    else:
        print("  [WARN] shap_shallow beeswarm PNGs not found (run step 19 first)")

    # F5. BERT token attribution bar charts (signed mean, coloured by direction)
    bert_attr_dir = EXPL_DIR / "shap_bert"
    SPECIAL  = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
    MIN_OCC  = 3   # minimum occurrences for a token to be plotted
    MIN_LEN  = 4   # minimum character length (filters wordpiece fragments)
    TOP_N    = 25  # tokens to show

    # Axis label and subtitle — same for all tasks:
    # attributions are computed w.r.t. the predicted class (target=pred_class),
    # so positive = token reinforces the model's prediction, negative = undermines it.
    ATTR_XLABEL  = "Mean signed LIG attribution (relative to predicted class)\n← undermines prediction / reinforces prediction →"
    ATTR_SUBTITLE = "Green = token reinforces model prediction, Red = token undermines prediction"

    def _clean_token(tok: str, model: str):
        """Return (clean_text, is_word_start) for a raw tokenizer token.

        gbert  uses WordPiece: continuation pieces are prefixed with ##.
        xlmr   uses SentencePiece: word-start pieces are prefixed with ▁.
        Returns None if the token is a continuation piece (not a word start).
        """
        if tok in SPECIAL:
            return None
        if model == "gbert":
            if tok.startswith("##"):
                return None          
            return tok               
        else:                        # xlmr / sentencepiece
            if tok.startswith("▁"):
                return tok[1:]       

            return None

    if bert_attr_dir.exists():
        for attr_json in sorted(bert_attr_dir.glob("*_attributions.json")):
            stem  = attr_json.stem                          # e.g. gbert_interaction_attributions
            parts = stem.split("_")                         # ['gbert','interaction','attributions']
            model_name = parts[0]                           # gbert / xlmr
            task_name  = parts[1] if len(parts) > 1 else "?"

            try:
                with open(attr_json, encoding="utf-8") as fh:
                    attr_data = json.load(fh)
            except Exception as exc:
                print(f"  [WARN] Could not load {attr_json}: {exc}")
                continue

            # Aggregate: token -> list of signed attribution values
            token_vals: dict[str, list[float]] = defaultdict(list)
            for pair_id, rec in attr_data.items():
                tokens       = rec.get("tokens", [])
                attributions = rec.get("attributions", [])
                if len(tokens) != len(attributions):
                    continue
                for tok, attr in zip(tokens, attributions):
                    clean = _clean_token(tok, model_name)
                    if not clean or len(clean) < MIN_LEN:
                        continue
                    token_vals[clean].append(float(attr))

            # Filter by minimum occurrences and compute mean signed attribution
            agg = {
                tok: (sum(vals) / len(vals), len(vals))
                for tok, vals in token_vals.items()
                if len(vals) >= MIN_OCC
            }
            if not agg:
                print(f"  [WARN] No tokens with n>={MIN_OCC} in {attr_json.name}")
                continue

            # Sort by |mean|, take top N
            ranked = sorted(agg.items(), key=lambda x: abs(x[1][0]), reverse=True)[:TOP_N]
            ranked = list(reversed(ranked))   # lowest |mean| at top of horizontal bar chart

            names  = [f"{tok}  (n={cnt})" for tok, (mean, cnt) in ranked]
            means  = [mean for _, (mean, _) in ranked]
            colors = ["#2ca02c" if m >= 0 else "#d62728" for m in means]

            fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.38)))
            bars = ax.barh(names, means, color=colors)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            val_range = max(abs(m) for m in means) or 1e-9
            margin    = val_range * 0.03
            for bar, m in zip(bars, means):
                ha  = "left"  if m >= 0 else "right"
                x   = m + margin if m >= 0 else m - margin
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        f"{m:+.3f}", va="center", ha=ha, fontsize=7)
            ax.set_xlabel(ATTR_XLABEL, fontsize=9)
            ax.set_title(
                f"BERT Token Attributions (Layer IG) — {model_name} / {task_name}\n"
                f"{ATTR_SUBTITLE}",
                fontweight="bold", fontsize=9,
            )
            _save(fig, OUT_DIR / "features" / f"bert_attr_{model_name}_{task_name}.png")
    else:
        print("  [WARN] shap_bert directory not found (run step 19 first)")

    # F6. SHAP per-class directionality: constructive vs destructive direction
    # Uses per_class_mean_shap from signed_shap.json.
    # Interaction: diff = SHAP(CD+1.0) − SHAP(DD-1.0)  → positive = constructive
    # Stance:      diff = SHAP(Support) − SHAP(Against) → positive = support
    DIRECTIONALITY_PAIRS = {
        "interaction": ("CD(+1.0)", "DD(-1.0)",
                        "Constructive Disagr.", "Destructive Disagr.",
                        "← destructive / constructive →"),
        "stance":      ("Support",  "Against",
                        "Support",  "Against",
                        "← against / support →"),
    }
    signed_dir = EXPL_DIR / "shap_shallow"
    if signed_dir.exists():
        for signed_json in sorted(signed_dir.glob("*_signed_shap.json")):
            stem  = signed_json.stem          # e.g. interaction_XGB_signed_shap
            parts = stem.split("_")
            task  = parts[0]
            model = parts[1] if len(parts) > 1 else "?"
            if task not in DIRECTIONALITY_PAIRS:
                continue                       # skip techniques (no ordinal direction)
            pos_cls, neg_cls, pos_label, neg_label, xlabel_dir = DIRECTIONALITY_PAIRS[task]
            data = _load_json(signed_json)
            if not data:
                continue
            per_cls = data.get("per_class_mean_shap", {})
            pos_shap = per_cls.get(pos_cls, {})
            neg_shap = per_cls.get(neg_cls, {})
            if not pos_shap or not neg_shap:
                continue
            all_feats = set(pos_shap) | set(neg_shap)
            if _is_embedding_features(list(all_feats)):
                continue
            diffs = {
                f: pos_shap.get(f, 0.0) - neg_shap.get(f, 0.0)
                for f in all_feats
            }
            ranked = sorted(diffs.items(), key=lambda x: x[1])
            # Keep top-N at each extreme for a balanced view
            N_SIDE = 10
            bottom = ranked[:N_SIDE]
            top    = ranked[-N_SIDE:]
            display = bottom + top
            names  = [_feat_label(f) for f, _ in display]
            vals   = [v for _, v in display]
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
            fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4)))
            bars = ax.barh(names, vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            vmax = max(abs(v) for v in vals) or 1e-9
            margin = vmax * 0.03
            for bar, v in zip(bars, vals):
                ha = "left" if v >= 0 else "right"
                x  = v + margin if v >= 0 else v - margin
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        f"{v:+.4f}", va="center", ha=ha, fontsize=7)
            ax.set_xlabel(
                f"SHAP({pos_label}) − SHAP({neg_label})\n{xlabel_dir}",
                fontsize=9)
            ax.set_title(
                f"SHAP Directionality — {task} / {model}\n"
                f"Green = pushes toward {pos_label}, "
                f"Red = pushes toward {neg_label}",
                fontweight="bold", fontsize=9)
            _save(fig, OUT_DIR / "features" /
                  f"shap_direction_{task}_{model}.png")

    # F7. BERT directionality: class-stratified mean attribution
    # Split pairs into destructive-predicted (classes 0,1) vs constructive-predicted
    # (classes 3,4) and compute differential mean signed attribution per token.
    # For interaction: diff = mean_attr(constructive preds) − mean_attr(destructive preds)
    # For stance:      diff = mean_attr(Support preds)      − mean_attr(Against preds)
    BERT_DIR_GROUPS = {
        "interaction": ({3, 4}, {0, 1},
                        "Constructive (CA/CD)", "Destructive (DD/DA)",
                        "← destructive-typical / constructive-typical →"),
        "stance":      ({2},    {0},
                        "Support", "Against",
                        "← against-typical / support-typical →"),
    }
    if bert_attr_dir.exists():
        for attr_json in sorted(bert_attr_dir.glob("*_attributions.json")):
            stem  = attr_json.stem
            parts = stem.split("_")
            model_name = parts[0]
            task_name  = parts[1] if len(parts) > 1 else "?"
            if task_name not in BERT_DIR_GROUPS:
                continue          # skip techniques
            pos_set, neg_set, pos_lbl, neg_lbl, xlabel_lbl = BERT_DIR_GROUPS[task_name]
            try:
                with open(attr_json, encoding="utf-8") as fh:
                    attr_data = json.load(fh)
            except Exception as exc:
                print(f"  [WARN] Could not load {attr_json}: {exc}")
                continue

            # Aggregate per-group token attributions
            pos_vals: dict[str, list[float]] = defaultdict(list)
            neg_vals: dict[str, list[float]] = defaultdict(list)
            for pair_id, rec in attr_data.items():
                tokens       = rec.get("tokens", [])
                attributions = rec.get("attributions", [])
                pred_class   = rec.get("predicted_class")
                if len(tokens) != len(attributions) or pred_class is None:
                    continue
                if pred_class in pos_set:
                    target_dict = pos_vals
                elif pred_class in neg_set:
                    target_dict = neg_vals
                else:
                    continue      # neutral class — skip for directionality
                for tok, attr in zip(tokens, attributions):
                    clean = _clean_token(tok, model_name)
                    if not clean or len(clean) < MIN_LEN:
                        continue
                    target_dict[clean].append(float(attr))

            # Compute differential: require min occ in BOTH groups
            diffs = {}
            for tok in set(pos_vals) | set(neg_vals):
                pv = pos_vals.get(tok, [])
                nv = neg_vals.get(tok, [])
                if len(pv) < MIN_OCC or len(nv) < MIN_OCC:
                    continue
                diffs[tok] = (sum(pv) / len(pv)) - (sum(nv) / len(nv))

            if not diffs:
                print(f"  [WARN] No tokens with n>={MIN_OCC} in both groups "
                      f"for {attr_json.name}")
                continue

            ranked = sorted(diffs.items(), key=lambda x: x[1])
            N_SIDE = 12
            display = ranked[:N_SIDE] + ranked[-N_SIDE:]
            names  = [f"{tok}" for tok, _ in display]
            vals   = [v for _, v in display]
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]

            fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4)))
            bars = ax.barh(names, vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            vmax   = max(abs(v) for v in vals) or 1e-9
            margin = vmax * 0.03
            for bar, v in zip(bars, vals):
                ha = "left" if v >= 0 else "right"
                x  = v + margin if v >= 0 else v - margin
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        f"{v:+.3f}", va="center", ha=ha, fontsize=7)
            ax.set_xlabel(
                f"Mean attr (predicted {pos_lbl}) − Mean attr (predicted {neg_lbl})\n"
                f"{xlabel_lbl}",
                fontsize=9)
            ax.set_title(
                f"BERT Token Directionality (LIG) — {model_name} / {task_name}\n"
                f"Green = more associated with {pos_lbl} predictions, "
                f"Red = more associated with {neg_lbl} predictions",
                fontweight="bold", fontsize=9)
            _save(fig, OUT_DIR / "features" /
                  f"bert_direction_{model_name}_{task_name}.png")


# G. Thread analysis
def plot_thread_analysis(thread_df):
    print("\n-- G. Thread Analysis")

    thread_results_dir = Path("18_thread_results")
    metrics_data = _load_json(thread_results_dir / "metrics.json")

    # G1. Thread bin distribution
    if not thread_df.empty and "ThreadBin" in thread_df.columns:
        bin_order_clean = ["Constructive", "Slight/Neutral",
                           "Moderately Destructive", "Highly Destructive"]
        bin_counts = thread_df["ThreadBin"].value_counts()
        ordered    = [bin_counts.get(b, 0) for b in bin_order_clean]
        colors_bin = ["#70AD47", "#A9D18E", "#ED7D31", "#C00000"]
        fig, ax    = plt.subplots(figsize=(8, 4))
        bars = ax.bar(["Constructive", "Slight/\nNeutral",
                       "Mod.\nDestructive", "Highly\nDestructive"],
                      ordered, color=colors_bin)
        ax.set_ylabel("Number of threads")
        ax.set_title("Ground Truth Thread Quality Distribution", fontweight="bold")
        _label_bars(ax, bars, ordered, fontsize=9)
        _save(fig, OUT_DIR / "threads" / "thread_bin_distribution.png")

    # G2. Per-topic thread distribution
    if not thread_df.empty and "Topic" in thread_df.columns and "ThreadBin" in thread_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ti, (ax, topic) in enumerate(zip(axes, ["migration", "climate"])):
            sub = thread_df[thread_df["Topic"].str.lower() == topic]
            if sub.empty:
                continue
            counts = sub["ThreadBin"].value_counts()
            ordered_c = [counts.get(b, 0) for b in
                         ["Constructive", "Slight/Neutral",
                          "Moderately Destructive", "Highly Destructive"]]
            short_labels = ["Construct.", "Slight/Neutral",
                            "Mod.Dest.", "Highly\nDest."]
            g2_bars = ax.bar(short_labels, ordered_c,
                             color=["#70AD47", "#A9D18E", "#ED7D31", "#C00000"])
            _label_bars(ax, g2_bars, ordered_c, fontsize=9, adjust_ylim=False)
            ax.set_title(f"{topic.capitalize()} (n={len(sub):,})", fontweight="bold")
            if ti == 0:
                ax.set_ylabel("Number of threads")
        axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.25)
        fig.suptitle("Thread Quality Distribution by Topic", fontweight="bold")
        _save(fig, OUT_DIR / "threads" / "thread_dist_by_topic.png")

    # G2b. Thread bin distribution by party
    if (not thread_df.empty and "ThreadBin" in thread_df.columns
            and "party" in thread_df.columns):
        bin_order_clean = ["Constructive", "Slight/Neutral",
                           "Moderately Destructive", "Highly Destructive"]
        short_labels    = ["Construct.", "Slight/\nNeutral",
                           "Mod.\nDest.", "Highly\nDest."]
        colors_bin      = ["#70AD47", "#A9D18E", "#ED7D31", "#C00000"]
        parties_t       = sorted(thread_df["party"].dropna().unique())
        fig, axes = plt.subplots(1, len(parties_t),
                                  figsize=(6 * len(parties_t), 4), sharey=True)
        if len(parties_t) == 1:
            axes = [axes]
        for pi, (ax, party) in enumerate(zip(axes, parties_t)):
            sub = thread_df[thread_df["party"] == party]
            counts = sub["ThreadBin"].value_counts()
            ordered = [counts.get(b, 0) for b in bin_order_clean]
            g2b_bars = ax.bar(short_labels, ordered, color=colors_bin)
            ax.set_title(f"{party}  (n={len(sub):,})", fontweight="bold")
            if pi == 0:
                ax.set_ylabel("Number of threads")
            _label_bars(ax, g2b_bars, ordered, fontsize=9, adjust_ylim=False)
        axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.25)
        fig.suptitle("Thread Quality Distribution by Party", fontweight="bold")
        _save(fig, OUT_DIR / "threads" / "thread_dist_by_party.png")

    # G2c. Thread bin distribution by party × topic (grid: party rows × topic cols)
    if (not thread_df.empty and "ThreadBin" in thread_df.columns
            and "party" in thread_df.columns and "Topic" in thread_df.columns):
        bin_order_pt  = ["Constructive", "Slight/Neutral",
                         "Moderately Destructive", "Highly Destructive"]
        short_labels_pt = ["Construct.", "Slight/\nNeutral",
                            "Mod.\nDest.", "Highly\nDest."]
        colors_bin_pt = ["#70AD47", "#A9D18E", "#ED7D31", "#C00000"]
        parties_pt    = sorted(thread_df["party"].dropna().unique())
        topics_pt     = ["migration", "climate"]
        fig, axes = plt.subplots(len(parties_pt), len(topics_pt),
                                  figsize=(6 * len(topics_pt), 4 * len(parties_pt)),
                                  sharey=True)
        if len(parties_pt) == 1:
            axes = [axes]
        for row, party in enumerate(parties_pt):
            for col, topic in enumerate(topics_pt):
                ax = axes[row][col]
                sub = thread_df[
                    (thread_df["party"] == party) &
                    (thread_df["Topic"].str.lower() == topic)
                ]
                counts = sub["ThreadBin"].value_counts()
                ordered = [counts.get(b, 0) for b in bin_order_pt]
                g2c_bars = ax.bar(short_labels_pt, ordered, color=colors_bin_pt)
                ax.set_title(f"{party} — {topic.capitalize()}  (n={len(sub):,})",
                             fontweight="bold", fontsize=9)
                if col == 0:
                    ax.set_ylabel("Number of threads")
                _label_bars(ax, g2c_bars, ordered, fontsize=9, adjust_ylim=False)
        axes[0][0].set_ylim(top=axes[0][0].get_ylim()[1] * 1.25)
        fig.suptitle("Thread Quality Distribution by Party × Topic", fontweight="bold")
        plt.tight_layout()
        _save(fig, OUT_DIR / "threads" / "thread_dist_party_topic.png")

    # G3. Approach A vs B
    a_m = metrics_data.get("approach_A", {})
    b_m = metrics_data.get("approach_B", {})

    if a_m:
        metric_keys = ["macro_f1", "accuracy", "macro_precision",
                       "macro_recall", "kappa_quadratic"]
        fig, ax = plt.subplots(figsize=(9, 5))
        x, w    = np.arange(len(metric_keys)), 0.3
        a_vals  = [a_m.get(k, 0) for k in metric_keys]
        g3_a = ax.bar(x - w / 2, a_vals, w, label="Approach A (Aggregation)",
                      color="#4472C4")
        for bar, v in zip(g3_a, a_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # Resolve approach B metrics regardless of flat vs per-classifier structure
        b_flat = None
        if isinstance(b_m, dict) and b_m:
            first_val = next(iter(b_m.values()))
            if isinstance(first_val, dict):
                # per-classifier dict: pick the classifier with best macro_f1
                best_clf = max(b_m, key=lambda c: b_m[c].get("macro_f1", 0))
                b_flat   = b_m[best_clf]
                b_label  = f"Approach B — best clf ({best_clf})"
            else:
                b_flat  = b_m
                b_label = "Approach B (Classifier)"
        if b_flat:
            b_vals = [b_flat.get(k, 0) for k in metric_keys]
            g3_b = ax.bar(x + w / 2, b_vals, w, label=b_label, color="#ED7D31")
            for bar, v in zip(g3_b, b_vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["Macro-F1", "Accuracy", "Precision",
                             "Recall", "Kappa-Q"], rotation=15)
        ax.set_ylabel("Score")
        ax.set_title("Thread Classification -- Approach A vs B", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        plt.subplots_adjust(bottom=0.28)
        _save(fig, OUT_DIR / "threads" / "approach_a_vs_b.png")

    # G4. Approach B -- per-classifier
    if isinstance(b_m, dict) and any(isinstance(v, dict) for v in b_m.values()):
        clf_names  = list(b_m.keys())
        f1_vals    = [b_m[c].get("macro_f1", 0)        for c in clf_names]
        kappa_vals = [b_m[c].get("kappa_quadratic", 0) for c in clf_names]
        fig, ax    = plt.subplots(figsize=(8, 5))
        x, w       = np.arange(len(clf_names)), 0.35
        g4_f1  = ax.bar(x - w / 2, f1_vals,    w, label="Macro-F1",    color="#4472C4")
        g4_kap = ax.bar(x + w / 2, kappa_vals, w, label="Kappa-Quadr.", color="#ED7D31")
        for bar, v in zip(g4_f1, f1_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, v in zip(g4_kap, kappa_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(clf_names)
        ax.set_ylabel("Score")
        ax.set_title("Approach B -- Per-Classifier Performance", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        plt.subplots_adjust(bottom=0.22)
        _save(fig, OUT_DIR / "threads" / "approach_b_classifiers.png")


# Main
def main():
    print("=" * 60)
    print("Step 20 -- Visualizations and Graphs")
    print("=" * 60)

    print("\nLoading features.parquet ...")
    df = pd.read_parquet(FEAT_DIR / "features.parquet")
    df["Topic"] = df["Topic"].fillna("").str.strip()


    empty_mask = df["Topic"] == ""
    if empty_mask.any() and FILTER_LOG.exists():
        try:
            cat_lookup = (pd.read_csv(FILTER_LOG, usecols=["videoId", "category"],
                                      encoding="utf-8")
                          .rename(columns={"category": "_cat"}))
            df = df.merge(cat_lookup, left_on="VideoID", right_on="videoId", how="left")
            df.loc[empty_mask & df["_cat"].notna(), "Topic"] = \
                df.loc[empty_mask & df["_cat"].notna(), "_cat"]
            df.drop(columns=["videoId", "_cat"], errors="ignore", inplace=True)
            still_empty = (df["Topic"] == "").sum()
            if empty_mask.sum() > still_empty:
                print(f"  Topic backfilled for {empty_mask.sum() - still_empty} pairs "
                      f"(re-run step 14 to fix permanently; {still_empty} still empty).")
        except Exception as e:
            print(f"  Warning: topic backfill failed: {e}")

    print(f"  {len(df):,} pairs loaded.")

    print("Loading video metadata ...")
    video_meta = load_video_meta()
    print(f"  {len(video_meta):,} video records.")

    print("Loading thread scores ...")
    thread_df = load_thread_scores(video_meta, df)
    print(f"  {len(thread_df):,} thread records.")

    # Attach channel and party to pairs dataframe AND thread_df
    if not video_meta.empty and "videoId" in video_meta.columns:
        party_lookup = (video_meta[["videoId", "channelName", "party"]]
                        .drop_duplicates("videoId"))
        df = df.merge(party_lookup, left_on="VideoID", right_on="videoId", how="left")
        df["party"]       = df["party"].fillna("Unknown")
        df["channelName"] = df["channelName"].fillna("Unknown")
        df.drop(columns=["videoId"], errors="ignore", inplace=True)
        # Add party to thread_df (channelName already present from load_thread_scores)
        if not thread_df.empty and "channelName" in thread_df.columns:
            thread_df["party"] = thread_df["channelName"].map(PARTY_MAP).fillna("Unknown")

    # Run all sections
    plot_all_videos_overview(video_meta)
    plot_annotated_pairs_overview(df)
    plot_temporal_analysis(df, video_meta, thread_df)
    plot_model_comparison()
    plot_confusion_matrices()
    plot_feature_importance()
    plot_thread_analysis(thread_df)

    print(f"\n{'='*60}")
    total = sum(1 for _ in OUT_DIR.rglob("*.png"))
    print(f"All visualizations saved to -> {OUT_DIR}/")
    print(f"Total plots generated: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
