import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

#paths
DATASETS = {
    "a_b": Path(os.getenv("BASE_DATA_DIR_AB", "./data_a_b")),
    "c_d": Path(os.getenv("BASE_DATA_DIR_CD", "./data_c_d")),
}

BERTOPIC_CSV = Path("./06_bertopic_output/video_topics.csv")
OUT_DIR      = Path("./07_filtered")

#topic mapped to category
MIGRATION_TOPICS = {1, 9, 11, 33, 35}
CLIMATE_TOPICS   = {3, 4, 36}

#output files
CATEGORY_CSV = OUT_DIR / "category_report.csv"
BALANCE_CSV  = OUT_DIR / "balance_report.csv"

CATEGORY_FIELDS = [
    "videoId", "dataset", "channelName", "title", "publishedAt",
    "category", "primary_topic_id", "primary_topic_keywords",
]
BALANCE_FIELDS = [
    "category", "dataset", "channelName", "video_count",
]


def assign_category(primary_topic_id: int) -> str:
    if primary_topic_id in MIGRATION_TOPICS:
        return "migration"
    if primary_topic_id in CLIMATE_TOPICS:
        return "climate"
    return "other"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "migration").mkdir(exist_ok=True)
    (OUT_DIR / "climate").mkdir(exist_ok=True)

    #load video_topics.csv from step 06 
    csv.field_size_limit(10 ** 7)
    with open(BERTOPIC_CSV, encoding="utf-8") as f:
        video_rows = list(csv.DictReader(f))
    print(f"Loaded {len(video_rows)} video topic rows")

    #assign categories 
    category_rows = []
    for row in video_rows:
        try:
            primary = int(row["primary_topic_id"])
        except (ValueError, KeyError):
            primary = -1
        cat = assign_category(primary)
        category_rows.append({
            "videoId":                row["videoId"],
            "dataset":                row["dataset"],
            "channelName":            row["channelName"],
            "title":                  row["title"],
            "publishedAt":            row["publishedAt"],
            "category":               cat,
            "primary_topic_id":       primary,
            "primary_topic_keywords": row.get("primary_topic_keywords", ""),
        })

    # save category report  
    with open(CATEGORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CATEGORY_FIELDS)
        writer.writeheader()
        writer.writerows(category_rows)
    print(f"Category report saved -> {CATEGORY_CSV}")

    #balance report
    counts: dict[tuple, int] = defaultdict(int)
    for row in category_rows:
        if row["category"] != "other":
            key = (row["category"], row["dataset"], row["channelName"])
            counts[key] += 1

    balance_rows = [
        {"category": cat, "dataset": ds, "channelName": ch, "video_count": n}
        for (cat, ds, ch), n in sorted(counts.items())
    ]

    with open(BALANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BALANCE_FIELDS)
        writer.writeheader()
        writer.writerows(balance_rows)
    print(f"Balance report saved -> {BALANCE_CSV}")

    #  print balance 
    print("\n-- Category balance --")
    print(f"{'Category':<12} {'Dataset':<8} {'Channel':<40} {'Videos':>7}")
    print("-" * 72)
    cat_totals: dict[str, int] = defaultdict(int)
    for r in balance_rows:
        print(f"{r['category']:<12} {r['dataset']:<8} {r['channelName']:<40} {r['video_count']:>7}")
        cat_totals[r["category"]] += r["video_count"]
    print("-" * 72)
    for cat, total in sorted(cat_totals.items()):
        print(f"{cat:<12} {'TOTAL':<8} {'':<40} {total:>7}")

    # copy comment files
    print("\n-- Copying comment files --")
    copied   = defaultdict(int)
    missing  = defaultdict(int)

    for row in category_rows:
        cat = row["category"]
        if cat == "other":
            continue

        ds_dir   = DATASETS.get(row["dataset"])
        if ds_dir is None:
            continue
        src = ds_dir / "05_relations" / f"Precomments_{row['videoId']}.jsonl"
        dst = OUT_DIR / cat / src.name

        if dst.exists():
            copied[cat] += 1
            continue
        if src.exists():
            shutil.copy2(src, dst)
            copied[cat] += 1
        else:
            missing[cat] += 1

    for cat in ("migration", "climate"):
        print(f"  {cat}: {copied[cat]} files copied, {missing[cat]} source files not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
