import os
import re
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from utils.jsonl_io import load_jsonl, save_jsonl

load_dotenv()

BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR"))
IN_DIR = BASE_DATA_DIR / "02_raw_scraped"
OUT_DIR = BASE_DATA_DIR / "03_anonymized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_FILE = OUT_DIR / "username_map.json"  


def build_global_mapping(comment_files: list) -> dict:
    """Collect all unique AuthorNames across all files, assign sequential pseudonyms."""
    all_names = set()
    for path in comment_files:
        for record in load_jsonl(str(path)):
            name = record.get("AuthorName", "")
            if name:
                all_names.add(name)

    mapping = {}
    for i, name in enumerate(sorted(all_names), start=1):
        mapping[name] = f"User_{i:05d}"
    print(f"Built mapping for {len(mapping)} unique usernames.")
    return mapping


def replace_mentions_in_text(text: str, mapping: dict) -> str:
    """Replace @RealName with @Pseudonym for all known names."""
    def replacer(match):
        handle = match.group(1)
        return "@" + mapping.get("@" + handle, handle)
    return re.sub(r"@([\w][\w\-\.]*)", replacer, text)


def anonymize_comment(record: dict, mapping: dict) -> dict:
    record = dict(record)
    real_name = record.get("AuthorName", "")
    record["AuthorName"] = mapping.get(real_name, real_name)
    record["CommentText"] = replace_mentions_in_text(record.get("CommentText", ""), mapping)
    return record


def load_mapping(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_mapping(mapping: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Username map saved at {path}")


def main():
    comment_files = sorted(IN_DIR.glob("comments_*.jsonl"))
    transcript_files = sorted(IN_DIR.glob("transcripts_*.jsonl"))

    if not comment_files:
        print("No comment files found in", IN_DIR)
        return

    # Load or build mapping
    if MAP_FILE.exists():
        print("Existing username map found — loading.")
        mapping = load_mapping(MAP_FILE)
    else:
        mapping = build_global_mapping(comment_files)
        save_mapping(mapping, MAP_FILE)

    # Anonymize comment files
    for path in comment_files:
        out_path = OUT_DIR / path.name
        if out_path.exists():
            print(f"Already anonymized: {path.name}")
            continue
        records = load_jsonl(str(path))
        anon = [anonymize_comment(r, mapping) for r in records]
        save_jsonl(anon, str(out_path))
        print(f"Anonymized {len(anon)} comments at {out_path.name}")

    # Copy transcript files
    for path in transcript_files:
        out_path = OUT_DIR / path.name
        if not out_path.exists():
            shutil.copy2(path, out_path)
            print(f"Copied transcript -> {out_path.name}")

    print("\nAnonymization complete.")


if __name__ == "__main__":
    main()
