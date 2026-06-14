# Pair structure and mismatched-target concept adapted from:
# Bassi et al. (2024) & Bassi et al. (2025) — Old but Gold (pair schema, incorrect_parent flag)
# Stratified sampling is original to this work.

import os
import csv
import json
import re
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# paths
FILTERED_DIR  = Path("./07_filtered")
CATEGORY_CSV  = FILTERED_DIR / "category_report.csv"
OUT_DIR       = Path("./08_pairs")
OUT_DIR.mkdir(exist_ok=True)

CATEGORIES    = ("climate", "migration")
TARGET_THREADS = 2000            # number of total threads to sample
RANDOM_SEED    = int(os.getenv("RANDOM_SEED", 42))

# helpers
def party_of(channel_name: str) -> str:
    afd_channels = {"AfD TV", "AfD-Fraktion Bundestag"}
    return "AfD" if channel_name in afd_channels else "Linke"


def load_jsonl(path: Path) -> list:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def save_jsonl(records: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_mismatched(child_text: str, parent_author: str) -> bool:
    """True if child @mentions someone other than the parent (invisible target)."""
    parent_handle = parent_author.lstrip("@")
    mentions = re.findall(r"@(\w+)", child_text)
    if not mentions:
        return False
    return all(m != parent_handle for m in mentions)


def build_pairs_from_thread(comments: list) -> list:
    """Extract all parent->child pairs from a thread."""
    index = {c["CommentID"]: c for c in comments}
    pairs = []
    for child in comments:
        parent_id = child.get("Response to")
        if not parent_id:
            continue
        parent = index.get(parent_id)
        if not parent or child["CommentID"] == parent_id:
            continue
        mismatched = is_mismatched(child["CommentText"], parent["AuthorName"])
        pairs.append({
            "PairID":             f"{child['VideoID']}_{child['CommentID']}",
            "ThreadID":           child.get("ThreadID", ""),
            "VideoID":            child["VideoID"],
            "ParentCommentID":    parent["CommentID"],
            "ParentText":         parent["CommentText"],
            "ParentAuthor":       parent["AuthorName"],
            "ParentLevel":        parent.get("Level", 0),
            "ParentTimestamp":    parent.get("Timestamp", ""),
            "ChildCommentID":     child["CommentID"],
            "ChildText":          child["CommentText"],
            "ChildAuthor":        child["AuthorName"],
            "ChildLevel":         child.get("Level", 0),
            "ChildTimestamp":     child.get("Timestamp", ""),
            "mismatched_target":  mismatched,
        })
    return pairs


# main
def main():
    random.seed(RANDOM_SEED)

    # load video and channel mapping from step 07
    csv.field_size_limit(10 ** 7)
    with open(CATEGORY_CSV, encoding="utf-8") as f:
        cat_map = {r["videoId"]: r for r in csv.DictReader(f)}

    # buckets: {(category, party): list of full threads}
    # each thread = {"meta": {...}, "comments": [...sorted by level/timestamp]}
    buckets: dict[tuple, list] = defaultdict(list)

    # stats: {(category, party, channel): {videos, comments, threads, pairs}}
    stats: dict[tuple, dict] = defaultdict(
        lambda: {"videos": 0, "comments": 0, "threads": 0, "pairs": 0}
    )

    print("Loading comments and grouping into threads...")
    for cat in CATEGORIES:
        folder = FILTERED_DIR / cat
        files  = sorted(folder.glob("Precomments_*.jsonl"))
        print(f"\n  {cat}: {len(files)} files")

        for fpath in files:
            vid_id  = fpath.stem.replace("Precomments_", "")
            meta    = cat_map.get(vid_id, {})
            channel = meta.get("channelName", "unknown")
            party   = party_of(channel)
            key     = (cat, party, channel)

            comments = load_jsonl(fpath)
            if not comments:
                continue

            # group comments by ThreadID
            thread_map: dict[str, list] = defaultdict(list)
            for c in comments:
                tid = c.get("ThreadID", c["CommentID"])
                thread_map[tid].append(c)

            for tid, thread_comments in thread_map.items():
                # sort: level 0 first, then by timestamp
                thread_comments.sort(key=lambda c: (c.get("Level", 0), c.get("Timestamp", "")))
                all_pairs   = build_pairs_from_thread(thread_comments)
                clean_pairs = [p for p in all_pairs if not p["mismatched_target"]]
                buckets[(cat, party)].append({
                    "thread_id":   tid,
                    "video_id":    vid_id,
                    "category":    cat,
                    "party":       party,
                    "channel":     channel,
                    "n_comments":  len(thread_comments),
                    "n_pairs":     len(clean_pairs),    # only clean pairs count
                    "n_pairs_all": len(all_pairs),      # total incl. mismatched
                    "max_level":   max(c.get("Level", 0) for c in thread_comments),
                    "comments":    thread_comments,
                })
                stats[key]["threads"] += 1
                stats[key]["pairs"]   += len(clean_pairs)

            stats[key]["videos"]   += 1
            stats[key]["comments"] += len(comments)

    # statistics report
    print("\n\n-- DATA STATISTICS --")
    print(f"{'Category':<12} {'Party':<8} {'Channel':<35} {'Videos':>7} {'Comments':>10} {'Threads':>9} {'Pairs':>8}")
    print("-" * 94)

    stat_rows = []
    cat_party_totals: dict[tuple, dict] = defaultdict(
        lambda: {"videos": 0, "comments": 0, "threads": 0, "pairs": 0}
    )
    grand = {"videos": 0, "comments": 0, "threads": 0, "pairs": 0}

    for (cat, party, channel), s in sorted(stats.items()):
        print(f"{cat:<12} {party:<8} {channel:<35} {s['videos']:>7} {s['comments']:>10} {s['threads']:>9} {s['pairs']:>8}")
        stat_rows.append({"category": cat, "party": party, "channel": channel, **s})
        for k in s:
            cat_party_totals[(cat, party)][k] += s[k]
            grand[k] += s[k]

    print("-" * 94)
    for (cat, party), s in sorted(cat_party_totals.items()):
        print(f"{cat:<12} {party:<8} {'SUBTOTAL':<35} {s['videos']:>7} {s['comments']:>10} {s['threads']:>9} {s['pairs']:>8}")
        stat_rows.append({"category": cat, "party": party, "channel": "SUBTOTAL", **s})

    print("-" * 94)
    print(f"{'GRAND TOTAL':<21} {'':<35} {grand['videos']:>7} {grand['comments']:>10} {grand['threads']:>9} {grand['pairs']:>8}")
    stat_rows.append({"category": "TOTAL", "party": "", "channel": "", **grand})

    stats_csv = OUT_DIR / "statistics_report.csv"
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category","party","channel","videos","comments","threads","pairs"])
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f"\nStatistics saved -> {stats_csv}")

    # Load gold pair child-comment IDs so their threads are always included.
    GOLD_FILE = Path("./09_manual_annotations/gold_standard.jsonl")
    gold_child_ids: set[str] = set()
    if GOLD_FILE.exists():
        for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                gold_child_ids.add(rec["ChildCommentID"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"\nGold lock: {len(gold_child_ids)} child comment IDs loaded from {GOLD_FILE.name}")

    # stratified sampling of threads
    strata = [
        ("climate",   "AfD"),
        ("climate",   "Linke"),
        ("migration", "AfD"),
        ("migration", "Linke"),
    ]
    per_stratum = TARGET_THREADS // len(strata)   # 500 each

    print(f"\n\n== STRATIFIED SAMPLING ({TARGET_THREADS} threads, {per_stratum} per stratum) ==")
    sampled_threads = []
    sample_rows = []

    for cat, party in strata:
        pool = [t for t in buckets[(cat, party)] if t["n_pairs"] >= 1]

        # Separate threads that contain at least one gold pair (must always be included)
        # from the rest (filled randomly up to per_stratum).
        gold_thread_ids = set()
        gold_threads    = []
        for t in pool:
            if any(c["CommentID"] in gold_child_ids for c in t["comments"]):
                gold_thread_ids.add(id(t))
                gold_threads.append(t)
        other_threads = [t for t in pool if id(t) not in gold_thread_ids]

        random.shuffle(other_threads)
        n_fill  = max(0, per_stratum - len(gold_threads))
        chosen  = gold_threads + other_threads[:n_fill]
        sampled_threads.extend(chosen)
        sample_rows.append({
            "stratum":   f"{cat}_{party}",
            "available": len(pool),
            "requested": per_stratum,
            "sampled":   len(chosen),
            "total_comments": sum(t["n_comments"] for t in chosen),
            "total_pairs":    sum(t["n_pairs"]    for t in chosen),
        })
        print(f"  {cat} x {party:<6}: {len(pool):>6} threads available -> {len(chosen)} sampled "
              f"(gold-locked: {len(gold_threads)}, random-fill: {len(other_threads[:n_fill])}) "
              f"| {sum(t['n_comments'] for t in chosen)} comments, "
              f"{sum(t['n_pairs'] for t in chosen)} pairs)")

    total_comments = sum(r["total_comments"] for r in sample_rows)
    total_pairs    = sum(r["total_pairs"]    for r in sample_rows)
    print(f"\n  Total sampled: {len(sampled_threads)} threads | {total_comments} comments | {total_pairs} pairs")

    # save full threads
    threads_path = OUT_DIR / "sampled_threads.jsonl"
    save_jsonl(sampled_threads, threads_path)
    print(f"\n  Full threads saved -> {threads_path}")

    # save flat pairs  
    all_pairs = []
    for thread in sampled_threads:
        pairs = build_pairs_from_thread(thread["comments"])
        for p in pairs:
            p["category"] = thread["category"]
            p["party"]    = thread["party"]
            p["channel"]  = thread["channel"]
        all_pairs.extend(p for p in pairs if not p["mismatched_target"])

    pairs_path = OUT_DIR / "sampled_pairs.jsonl"
    save_jsonl(all_pairs, pairs_path)
    print(f"  Flat pairs saved  -> {pairs_path}  ({len(all_pairs)} pairs)")

    # save sample summary
    summary_path = OUT_DIR / "sample_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stratum","available","requested","sampled","total_comments","total_pairs"])
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"  Sample summary to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
