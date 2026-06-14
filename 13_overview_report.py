"""
Data overview report

  1. Collection overview: videos / comments / threads / pairs by party x topic
  2. Sampling overview
  3. Annotation overview: gold standard vs LLM annotated, by party x topic
  4. Label distributions by party x topic:
       - Stance (Against / Neutral / Support)
       - Interaction quality (pair level)
       - Thread score bins
  5. Propaganda technique frequencies by topic

Reads:
  07_filtered/category_report.csv
  08_pairs/statistics_report.csv
  08_pairs/sample_summary.csv
  08_pairs/sampled_pairs.jsonl
  09_manual_annotations/gold_standard.jsonl
  10_llm_annotations/llm_annotated_subset.jsonl
  11_llm_annotations/llm_annotated_rest.jsonl
  12_thread_scores/thread_scores_*.jsonl

Outputs:
  13_overview/overview_report.txt
  13_overview/overview_report.json
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from utils.jsonl_io import load_jsonl

# Paths
CATEGORY_CSV   = Path("./07_filtered/category_report.csv")
STATS_CSV      = Path("./08_pairs/statistics_report.csv")
SAMPLE_CSV     = Path("./08_pairs/sample_summary.csv")
SAMPLED_PAIRS  = Path("./08_pairs/sampled_pairs.jsonl")
GOLD_FILE      = Path("./09_manual_annotations/gold_standard.jsonl")
LLM_SUBSET     = Path("./10_llm_annotations/llm_annotated_subset.jsonl")
LLM_REST       = Path("./11_llm_annotations/llm_annotated_rest.jsonl")
THREAD_SCORE_DIR = Path("./12_thread_scores")

OUT_DIR = Path("./13_overview")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_JSON = OUT_DIR / "overview_report.json"
REPORT_TXT  = OUT_DIR / "overview_report.txt"

TOPICS   = ("climate", "migration")
PARTIES  = ("AfD", "Linke")
STRATA   = [(t, p) for t in TOPICS for p in PARTIES]

BIN_ORDER = ["Constructive", "Slight/Neutral", "Moderately Destructive", "Highly Destructive"]
VALID_TECHNIQUES = [
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
    'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
    'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
    'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
    'Repetition', 'Slogans/Thought-terminating_Cliches',
    'Whataboutism,Straw_Men', 'Appeal_to_Time',
]


# Helpers
def score_to_bin(score) -> str:
    """Bassi et al. (2025) bin thresholds."""
    s = float(score) if score is not None else 0.0
    if s >= 0.25:   return "Constructive"
    if s >= -0.25:  return "Slight/Neutral"
    if s >= -0.75:  return "Moderately Destructive"
    return "Highly Destructive"


def score_to_5class(score) -> str:
    s = float(score) if score is not None else 0.0
    if s >= 1.0:  return "Constructive Disagreement"
    if s >= 0.5:  return "Constructive Agreement"
    if s >= 0.0:  return "Neutral/Rephrase"
    if s >= -0.5: return "Destructive Agreement"
    return "Destructive Disagreement"


CLASS5_ORDER = [
    "Constructive Disagreement",
    "Constructive Agreement",
    "Neutral/Rephrase",
    "Destructive Agreement",
    "Destructive Disagreement",
]


def pct(n, total):
    return round(100 * n / total, 1) if total else 0.0


def bar(value, width=20):
    filled = int(round(value / 100 * width))
    return "#" * filled + "-" * (width - filled)


def stratum_key(topic, party):
    return f"{topic}_{party}"


# Loaders
def load_video_map() -> dict:
    """Returns {videoId: {topic, party, channel}}."""
    if not CATEGORY_CSV.exists():
        return {}
    afd_channels = {"AfD TV", "AfD-Fraktion Bundestag"}
    result = {}
    csv.field_size_limit(10 ** 7)
    with open(CATEGORY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row.get("category", "other")
            if cat not in TOPICS:
                continue
            channel = row.get("channelName", "")
            party = "AfD" if channel in afd_channels else "Linke"
            result[row["videoId"]] = {
                "topic":   cat,
                "party":   party,
                "channel": channel,
            }
    return result


def load_collection_stats() -> list:
    """Load full corpus statistics from step 08 statistics_report.csv."""
    if not STATS_CSV.exists():
        return []
    csv.field_size_limit(10 ** 7)
    with open(STATS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sample_summary() -> list:
    """Load stratified sample summary from step 08."""
    if not SAMPLE_CSV.exists():
        return []
    with open(SAMPLE_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all_annotated() -> list:
    """Merge gold + LLM subset + LLM rest. Gold takes precedence for shared PairIDs."""
    records = {}
    for path in (LLM_REST, LLM_SUBSET, GOLD_FILE):
        if path.exists():
            for r in load_jsonl(str(path)):
                records[r["PairID"]] = r
    return list(records.values())


def load_thread_scores(video_map: dict) -> list:
    """Load all thread score records, enriched with topic/party from video_map."""
    records = []
    if not THREAD_SCORE_DIR.exists():
        return records
    for path in sorted(THREAD_SCORE_DIR.glob("thread_scores_*.jsonl")):
        video_id = path.stem.replace("thread_scores_", "")
        meta = video_map.get(video_id, {})
        for r in load_jsonl(str(path)):
            r["topic"] = meta.get("topic", "unknown")
            r["party"] = meta.get("party", "unknown")
            records.append(r)
    return records


# Statistics
def annotation_breakdown(pairs: list, video_map: dict) -> dict:
    """
    Returns counts of gold / llm-annotated / failed pairs broken down
    by stratum (topic x party).
    """
    gold_ids = set()
    if GOLD_FILE.exists():
        gold_ids = {r["PairID"] for r in load_jsonl(str(GOLD_FILE))}

    breakdown = defaultdict(lambda: {"gold": 0, "llm": 0, "failed": 0, "total": 0})
    for p in pairs:
        vid  = p.get("VideoID", "")
        meta = video_map.get(vid, {})
        key  = stratum_key(meta.get("topic", "unknown"), meta.get("party", "unknown"))
        breakdown[key]["total"] += 1
        if p["PairID"] in gold_ids:
            breakdown[key]["gold"] += 1
        else:
            breakdown[key]["llm"] += 1
        if p.get("AnnotationFailed") or p.get("InteractionScore") is None:
            breakdown[key]["failed"] += 1
    return dict(breakdown)


def label_distributions(pairs: list, video_map: dict) -> dict:
    """
    Returns stance and interaction distributions broken down by stratum.
    Includes 4-bin, 5-class, agreement/disagreement split, 3-bin summary,
    and per-stratum + overall technique totals.
    """
    stance_by_stratum      = defaultdict(Counter)
    interaction_by_stratum = defaultdict(Counter)   # 4-bin
    class5_by_stratum      = defaultdict(Counter)   # 5-class
    technique_by_stratum   = defaultdict(Counter)
    pair_count_by_stratum  = defaultdict(int)

    for p in pairs:
        vid  = p.get("VideoID", "")
        meta = video_map.get(vid, {})
        key  = stratum_key(meta.get("topic", "unknown"), meta.get("party", "unknown"))
        pair_count_by_stratum[key] += 1

        sl = p.get("StanceLabel")
        if sl is not None:
            stance_by_stratum[key][int(sl)] += 1

        score = p.get("InteractionScore")
        if score is not None:
            interaction_by_stratum[key][score_to_bin(score)] += 1
            class5_by_stratum[key][score_to_5class(score)] += 1

        for t in (p.get("Techniques") or []):
            if t in VALID_TECHNIQUES:
                technique_by_stratum[key][t] += 1

    # Overall technique totals across all strata
    overall_techniques = Counter()
    for ctr in technique_by_stratum.values():
        overall_techniques.update(ctr)

    result = {}
    for key in pair_count_by_stratum:
        total_s  = sum(stance_by_stratum[key].values()) or 1
        total_i  = sum(interaction_by_stratum[key].values()) or 1
        total_p  = pair_count_by_stratum[key] or 1
        c5       = class5_by_stratum[key]

        # 5-class counts
        cd  = c5.get("Constructive Disagreement", 0)
        ca  = c5.get("Constructive Agreement",    0)
        n   = c5.get("Neutral/Rephrase",          0)
        da  = c5.get("Destructive Agreement",     0)
        dd  = c5.get("Destructive Disagreement",  0)

        # 3-bin summary: Constructive=CA+CD, Neutral=N, Destructive=DA+DD
        constructive = cd + ca
        neutral_3    = n
        destructive  = da + dd

        # Agreement/Disagreement split: Agreement=CA+DA, Disagreement=CD+DD, Neutral=N
        agreement    = ca + da
        disagreement = cd + dd

        result[key] = {
            "pair_count": pair_count_by_stratum[key],
            "stance": {
                name: {"count": stance_by_stratum[key].get(i, 0),
                       "pct": pct(stance_by_stratum[key].get(i, 0), total_s)}
                for i, name in [(0, "Against"), (1, "Neutral"), (2, "Support")]
            },
            "interaction_4bin": {
                b: {"count": interaction_by_stratum[key].get(b, 0),
                    "pct": pct(interaction_by_stratum[key].get(b, 0), total_i)}
                for b in BIN_ORDER
            },
            # keep old key for backward compat
            "interaction": {
                b: {"count": interaction_by_stratum[key].get(b, 0),
                    "pct": pct(interaction_by_stratum[key].get(b, 0), total_i)}
                for b in BIN_ORDER
            },
            "interaction_5class": {
                c: {"count": c5.get(c, 0),
                    "pct": pct(c5.get(c, 0), total_i)}
                for c in CLASS5_ORDER
            },
            "interaction_3bin": {
                "Constructive":   {"count": constructive, "pct": pct(constructive, total_i)},
                "Neutral/Rephrase": {"count": neutral_3,  "pct": pct(neutral_3,    total_i)},
                "Destructive":    {"count": destructive,  "pct": pct(destructive,  total_i)},
            },
            "agreement_split": {
                "Agreement":    {"count": agreement,    "pct": pct(agreement,    total_i)},
                "Neutral":      {"count": n,            "pct": pct(n,            total_i)},
                "Disagreement": {"count": disagreement, "pct": pct(disagreement, total_i)},
            },
            "techniques": {
                t: {"count": technique_by_stratum[key].get(t, 0),
                    "pct": pct(technique_by_stratum[key].get(t, 0), total_p)}
                for t in VALID_TECHNIQUES
            },
        }

    # Attach overall technique summary as a special key
    result["__overall__"] = {
        "techniques_total": {
            t: overall_techniques.get(t, 0) for t in VALID_TECHNIQUES
        },
        "techniques_ranked": sorted(
            VALID_TECHNIQUES,
            key=lambda t: overall_techniques.get(t, 0),
            reverse=True,
        ),
    }
    return result


def thread_distributions(threads: list) -> dict:
    """Returns thread bin distributions broken down by stratum."""
    bins_by_stratum = defaultdict(list)
    for t in threads:
        key = stratum_key(t.get("topic", "unknown"), t.get("party", "unknown"))
        bins_by_stratum[key].append(t.get("ThreadBin", "Slight/Neutral"))

    result = {}
    for key, bins in bins_by_stratum.items():
        total = len(bins)
        result[key] = {
            "total_threads": total,
            "bins": {
                b: {"count": bins.count(b), "pct": pct(bins.count(b), total)}
                for b in BIN_ORDER
            },
        }
    return result


# Text report builder
def build_txt(report: dict) -> str:
    lines = []

    def header(title, width=62):
        lines.append("\n" + "=" * width)
        lines.append(title)
        lines.append("=" * width)

    def subheader(title):
        lines.append(f"\n-- {title} " + "-" * max(1, 55 - len(title)))

    lines += [
        "=" * 62,
        "DATA OVERVIEW REPORT",
        f"Generated: {report['metadata']['generated_at']}",
        "=" * 62,
    ]

    # 1. Collection overview
    header("1. FULL CORPUS COLLECTION (Steps 1-8)")
    lines.append(f"  {'Category':<12} {'Party':<8} {'Videos':>7} {'Comments':>10}"
                 f" {'Threads':>9} {'Pairs':>8}")
    lines.append("  " + "-" * 56)
    for row in report["collection"]:
        if row["category"] in ("TOTAL", ""):
            continue
        if row["channel"] == "SUBTOTAL":
            lines.append(f"  {row['category']:<12} {row['party']:<8} {int(row['videos']):>7}"
                         f" {int(row['comments']):>10} {int(row['threads']):>9}"
                         f" {int(row['pairs']):>8}  <- subtotal")
        else:
            lines.append(f"  {row['category']:<12} {row['party']:<8} {int(row['videos']):>7}"
                         f" {int(row['comments']):>10} {int(row['threads']):>9}"
                         f" {int(row['pairs']):>8}")
    # Grand total
    for row in report["collection"]:
        if row["category"] == "TOTAL":
            lines.append("  " + "-" * 56)
            lines.append(f"  {'TOTAL':<21} {int(row['videos']):>7}"
                         f" {int(row['comments']):>10} {int(row['threads']):>9}"
                         f" {int(row['pairs']):>8}")

    # 2. Sampling overview
    header("2. STRATIFIED SAMPLE (Step 8)")
    lines.append(f"  {'Stratum':<22} {'Available':>10} {'Sampled':>8}"
                 f" {'Comments':>10} {'Pairs':>8}")
    lines.append("  " + "-" * 60)
    total_sampled_pairs = 0
    for row in report["sample"]:
        lines.append(f"  {row['stratum']:<22} {int(row['available']):>10}"
                     f" {int(row['sampled']):>8} {int(row['total_comments']):>10}"
                     f" {int(row['total_pairs']):>8}")
        total_sampled_pairs += int(row["total_pairs"])
    lines.append("  " + "-" * 60)
    lines.append(f"  {'TOTAL':<22} {total_sampled_pairs:>47}")

    # 3. Annotation overview
    header("3. ANNOTATION OVERVIEW")
    cov = report["coverage"]
    lines += [
        f"  Total sampled pairs:   {cov['total_pairs']:>8}   100%",
        f"  Gold annotated:        {cov['gold_annotated']:>8}   {cov['gold_pct']}%",
        f"  LLM annotated:         {cov['llm_annotated']:>8}   {cov['llm_pct']}%",
        f"  Failed annotations:    {cov['failed']:>8}",
        "",
        f"  {'Stratum':<22} {'Total':>8} {'Gold':>8} {'LLM':>8} {'Failed':>8}",
        "  " + "-" * 56,
    ]
    for key in [stratum_key(t, p) for t, p in STRATA]:
        d = report["annotation_breakdown"].get(key, {})
        lines.append(f"  {key:<22} {d.get('total',0):>8} {d.get('gold',0):>8}"
                     f" {d.get('llm',0):>8} {d.get('failed',0):>8}")

    # 4. Label distributions by stratum
    header("4. LABEL DISTRIBUTIONS BY PARTY x TOPIC")
    for key in [stratum_key(t, p) for t, p in STRATA]:
        dist = report["label_distributions"].get(key)
        if not dist:
            continue
        subheader(key.upper())
        lines.append(f"  Pairs: {dist['pair_count']}")

        lines.append("  Stance:")
        for name, d in dist["stance"].items():
            lines.append(f"    {name:<10} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

        lines.append("  Interaction quality (4-bin):")
        for b, d in dist["interaction_4bin"].items():
            lines.append(f"    {b:<28} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

        lines.append("  Interaction quality (5-class):")
        for c, d in dist["interaction_5class"].items():
            lines.append(f"    {c:<32} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

        lines.append("  Interaction quality (3-bin summary):")
        for b, d in dist["interaction_3bin"].items():
            lines.append(f"    {b:<28} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

        lines.append("  Agreement/Disagreement split:")
        for b, d in dist["agreement_split"].items():
            lines.append(f"    {b:<28} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

        lines.append("  Techniques (% of pairs containing technique):")
        sorted_techs = sorted(dist["techniques"].items(), key=lambda x: x[1]["count"], reverse=True)
        for t, d in sorted_techs:
            if d["count"] > 0:
                lines.append(f"    {t:<42} {d['count']:>6}  {d['pct']:>5.1f}%")

    # Overall technique totals
    header("4b. OVERALL TECHNIQUE TOTALS (all strata combined)")
    ov = report["label_distributions"].get("__overall__", {})
    totals = ov.get("techniques_total", {})
    ranked = ov.get("techniques_ranked", VALID_TECHNIQUES)
    grand_total_pairs = sum(
        report["label_distributions"][k]["pair_count"]
        for k in report["label_distributions"]
        if k != "__overall__"
    )
    lines.append(f"  {'Technique':<42} {'Count':>8}  {'% pairs':>7}")
    lines.append("  " + "-" * 62)
    for t in ranked:
        n_t = totals.get(t, 0)
        if n_t > 0:
            lines.append(f"  {t:<42} {n_t:>8}  {pct(n_t, grand_total_pairs):>6.1f}%")

    # 5. Thread score distributions
    header("5. THREAD SCORE DISTRIBUTIONS BY PARTY x TOPIC")
    for key in [stratum_key(t, p) for t, p in STRATA]:
        td = report["thread_distributions"].get(key)
        if not td:
            continue
        subheader(key.upper())
        lines.append(f"  Total threads: {td['total_threads']}")
        for b, d in td["bins"].items():
            lines.append(f"    {b:<28} {d['count']:>6}  {d['pct']:>5.1f}%  {bar(d['pct'])}")

    lines += ["", "=" * 62, "END OF REPORT", "=" * 62]
    return "\n".join(lines)


# Main

def main():
    print("Loading data...")
    video_map       = load_video_map()
    collection_rows = load_collection_stats()
    sample_rows     = load_sample_summary()
    all_pairs       = load_all_annotated()
    thread_records  = load_thread_scores(video_map)

    print(f"Videos mapped:       {len(video_map)}")
    print(f"Annotated pairs:     {len(all_pairs)}")
    print(f"Thread records:      {len(thread_records)}")

    # Coverage summary
    gold_ids = set()
    if GOLD_FILE.exists():
        gold_ids = {r["PairID"] for r in load_jsonl(str(GOLD_FILE))}
    failed  = sum(1 for p in all_pairs if p.get("AnnotationFailed") or p.get("InteractionScore") is None)
    total   = len(all_pairs)
    llm_n   = sum(1 for p in all_pairs if p["PairID"] not in gold_ids)
    coverage = {
        "total_pairs":    total,
        "gold_annotated": len(gold_ids),
        "llm_annotated":  llm_n,
        "failed":         failed,
        "gold_pct":       pct(len(gold_ids), total),
        "llm_pct":        pct(llm_n, total),
    }

    ann_breakdown = annotation_breakdown(all_pairs, video_map)
    label_dist    = label_distributions(all_pairs, video_map)
    thread_dist   = thread_distributions(thread_records)

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "collection":            collection_rows,
        "sample":                sample_rows,
        "coverage":              coverage,
        "annotation_breakdown":  ann_breakdown,
        "label_distributions":   label_dist,
        "thread_distributions":  thread_dist,
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved -> {REPORT_JSON}")

    txt = build_txt(report)
    REPORT_TXT.write_text(txt, encoding="utf-8")
    print(f"Text report saved -> {REPORT_TXT}")
    print("\n" + txt)


if __name__ == "__main__":
    main()
