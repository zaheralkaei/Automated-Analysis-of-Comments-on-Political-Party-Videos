"""
Aggregate pair-level interaction scores to thread level.

For each ThreadID: mean(InteractionScore) -> 4 bins (Bassi et al. 2025 thresholds):
  mean >= 0.25        -> Constructive
  -0.25 <= mean < 0.25 -> Slight/Neutral
  -0.75 <= mean < -0.25 -> Moderately Destructive
  mean < -0.75        -> Highly Destructive

Reads:
  09_manual_annotations/gold_standard.jsonl
  10_llm_annotations/llm_annotated_subset.jsonl
  11_llm_annotations/llm_annotated_rest.jsonl

Outputs:
  12_thread_scores/thread_scores_<VideoID>.jsonl  (one per video)
  12_thread_scores/summary.csv
"""
import csv
from collections import defaultdict
from pathlib import Path
from utils.jsonl_io import load_jsonl, save_jsonl

GOLD_FILE   = Path("./09_manual_annotations/gold_standard.jsonl")
LLM_SUBSET  = Path("./10_llm_annotations/llm_annotated_subset.jsonl")
LLM_REST    = Path("./11_llm_annotations/llm_annotated_rest.jsonl")

OUT_DIR = Path("./12_thread_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT_DIR / "summary.csv"

SUMMARY_FIELDS = ["VideoID", "ThreadBin", "Count", "Percentage"]
BIN_ORDER = ["Constructive", "Slight/Neutral", "Moderately Destructive", "Highly Destructive"]


def score_to_bin(mean_score: float) -> str:
    """Bassi et al. (2025) bin thresholds."""
    if mean_score >= 0.25:   return "Constructive"
    if mean_score >= -0.25:  return "Slight/Neutral"
    if mean_score >= -0.75:  return "Moderately Destructive"
    return "Highly Destructive"


def load_all_pairs() -> list:
    """Merge gold + LLM subset + LLM rest. Gold takes precedence for shared PairIDs."""
    records = {}
    for path in (LLM_REST, LLM_SUBSET, GOLD_FILE):
        if path.exists():
            for r in load_jsonl(str(path)):
                records[r["PairID"]] = r  
    pairs = list(records.values())
    print(f"Total pairs loaded: {len(pairs)}")
    return pairs


def aggregate_threads(pairs: list) -> list:
    """Group pairs by (VideoID, ThreadID) and compute mean score."""
    threads = defaultdict(lambda: {"scores": [], "video_id": ""})
    for pair in pairs:
        score = pair.get("InteractionScore")
        if score is None:
            continue
        key = (pair["VideoID"], pair.get("ThreadID", pair["PairID"]))
        threads[key]["scores"].append(float(score))
        threads[key]["video_id"] = pair["VideoID"]

    records = []
    for (video_id, thread_id), data in threads.items():
        mean = sum(data["scores"]) / len(data["scores"])
        records.append({
            "ThreadID":  thread_id,
            "VideoID":   video_id,
            "PairCount": len(data["scores"]),
            "MeanScore": round(mean, 4),
            "ThreadBin": score_to_bin(mean),
        })
    return records


def write_summary(all_records: list):
    by_video = defaultdict(list)
    for r in all_records:
        by_video[r["VideoID"]].append(r["ThreadBin"])
    by_video["__ALL__"] = [r["ThreadBin"] for r in all_records]

    rows = []
    for video_id, bins in sorted(by_video.items()):
        total = len(bins)
        for bin_name in BIN_ORDER:
            count = bins.count(bin_name)
            rows.append({
                "VideoID":    video_id,
                "ThreadBin":  bin_name,
                "Count":      count,
                "Percentage": round(100 * count / total, 1) if total else 0,
            })

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved -> {SUMMARY_FILE}")


def main():
    pairs = load_all_pairs()
    all_records = aggregate_threads(pairs)
    print(f"Threads aggregated: {len(all_records)}")

    # Save per-video files
    by_video = defaultdict(list)
    for r in all_records:
        by_video[r["VideoID"]].append(r)

    for video_id, records in by_video.items():
        out_path = OUT_DIR / f"thread_scores_{video_id}.jsonl"
        save_jsonl(records, str(out_path))

    write_summary(all_records)

    # Print bin distribution
    bins = [r["ThreadBin"] for r in all_records]
    total = len(bins)
    print("\nOverall thread distribution:")
    for b in BIN_ORDER:
        n = bins.count(b)
        print(f"  {b}: {n} ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
