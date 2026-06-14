"""
11_dedup_annotations.py  —  Deduplicate step 11 LLM annotation output.

11_llm_annotator_rest.py resumes by skipping pairs where ALL three annotation
fields (InteractionScore, StanceLabel, ParentStanceLabel) are non-None.
If only one field failed (e.g. StanceLabel=None), the pair is retried and
appended again on every re-run, producing duplicate rows.

This script collapses duplicates to one row per PairID using the rule:
  1) Prefer AnnotationFailed=False over AnnotationFailed=True.
  2) Among records with the same status, keep the last one (most recent run).

Run from the project root:
    python 11_dedup_annotations.py

A timestamped backup of the original file is written before overwriting.
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_FILE = Path("./11_llm_annotations/llm_annotated_rest.jsonl")


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dedup(records: list[dict]) -> list[dict]:
    """
    Keep one record per PairID.
    Priority: AnnotationFailed=False > AnnotationFailed=True.
    Tie-break: last occurrence (latest run) wins.
    """
    best: dict[str, dict] = {}
    for r in records:
        pid = r["PairID"]
        if pid not in best:
            best[pid] = r
        else:
            existing_ok = not best[pid].get("AnnotationFailed", True)
            new_ok      = not r.get("AnnotationFailed", True)
            if new_ok and not existing_ok:
                best[pid] = r          # upgrade failed to success
            elif new_ok == existing_ok:
                best[pid] = r          # same quality: prefer latest
    return list(best.values())


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not OUT_FILE.exists():
        print(f"Output file not found: {OUT_FILE}")
        sys.exit(1)

    records = load_records(OUT_FILE)
    n_before = len(records)
    deduped  = dedup(records)
    n_after  = len(deduped)
    removed  = n_before - n_after

    if removed == 0:
        print(f"No duplicates found ({n_before} records). Nothing to do.")
        return

    # Backup
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    back = OUT_FILE.with_suffix(f".bak_{ts}.jsonl")
    shutil.copy2(OUT_FILE, back)
    print(f"Backup → {back}")

    # Write deduplicated file
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    clean   = sum(1 for r in deduped if not r.get("AnnotationFailed"))
    failed  = sum(1 for r in deduped if     r.get("AnnotationFailed"))
    print(f"Before : {n_before:>6} rows")
    print(f"After  : {n_after:>6} unique PairIDs  ({removed} duplicate rows removed)")
    print(f"  Clean (AnnotationFailed=False) : {clean}")
    print(f"  Failed (best-effort kept)      : {failed}")
    print(f"Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
