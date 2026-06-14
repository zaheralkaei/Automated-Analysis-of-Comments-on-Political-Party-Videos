# Based on https://github.com/BassiDavide/YouTube_Hybrid_Interactions_Analysis
# by D. Bassi, M. J. Maggini, R. Vieira and M. Pereira-Fariña,
# "A Pipeline for the Analysis of User Interactions in YouTube Comments:
# A Hybridization of LLMs and Rule-Based Methods," 2024 11th International
# Conference on Social Networks Analysis, Management and Security (SNAMS),
# Gran Canaria, Spain, 2024, pp. 146-153, doi: 10.1109/SNAMS64316.2024.10883781.

import os
from pathlib import Path
from dotenv import load_dotenv
from utils.jsonl_io import load_jsonl, save_jsonl

load_dotenv()

BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR"))
IN_DIR = BASE_DATA_DIR / "03_anonymized"
OUT_DIR = BASE_DATA_DIR / "04_username_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_authors(comments: list) -> set:
    return {c["AuthorName"] for c in comments}


def fix_usernames(comments: list, author_set: set) -> list:
    """Verbatim from DEF_Username_Iter.py lines 27-55."""
    for comment in comments:
        text = comment["CommentText"]
        search_idx = 0
        while "@@" in text[search_idx:]:
            start_idx = text.find("@@", search_idx)
            end_idx = start_idx + 2
            while end_idx < len(text) and (text[end_idx].isalnum() or text[end_idx] in {"_", "-"}):
                end_idx += 1

            full_username = text[start_idx + 2:end_idx]
            username = "@" + full_username

            valid_username_found = False
            for i in range(len(username), 0, -1):
                if username[:i] in author_set:
                    valid_username = username[:i]
                    remaining_text = text[start_idx + 1 + i:end_idx] + text[end_idx:]
                    text = text[:start_idx + 1] + valid_username + " " + remaining_text
                    search_idx = start_idx + len(valid_username) + 2
                    valid_username_found = True
                    break

            if not valid_username_found:
                search_idx = end_idx

        comment["CommentText"] = text
    return comments


def process_files(input_dir: Path, output_dir: Path):
    for path in sorted(input_dir.glob("comments_*.jsonl")):
        out_path = output_dir / ("Pre" + path.name)
        if out_path.exists():
            print(f"Already processed: {path.name}")
            continue
        comments = load_jsonl(str(path))
        author_set = extract_authors(comments)
        fixed = fix_usernames(comments, author_set)
        save_jsonl(fixed, str(out_path))
        print(f"Fixed: {path.name} -> {out_path.name}")


def main():
    process_files(IN_DIR, OUT_DIR)
    print("Username fixing complete.")


if __name__ == "__main__":
    main()
