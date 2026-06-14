# Based on https://github.com/BassiDavide/YouTube_Hybrid_Interactions_Analysis
# by D. Bassi, M. J. Maggini, R. Vieira and M. Pereira-Fariña,
# "A Pipeline for the Analysis of User Interactions in YouTube Comments:
# A Hybridization of LLMs and Rule-Based Methods," 2024 11th International
# Conference on Social Networks Analysis, Management and Security (SNAMS),
# Gran Canaria, Spain, 2024, pp. 146-153, doi: 10.1109/SNAMS64316.2024.10883781.

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DATA_DIR = Path(os.getenv("BASE_DATA_DIR"))
IN_DIR = BASE_DATA_DIR / "04_username_fixed"
OUT_DIR = BASE_DATA_DIR / "05_relations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_jsonl_comments(input_path: Path, output_path: Path):
    """Verbatim from DEF_Relation_Iter.py lines 6-47."""
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        last_comment_by_author = {}
        levels = {}

        for line in infile:
            comment = json.loads(line)

            author = comment["AuthorName"]
            comment_id = comment["CommentID"]
            parent_id = comment["ParentCommentID"]
            is_reply = comment["IsReply"] == "True"
            comment_text = comment["CommentText"]

            last_comment_by_author["@" + author] = comment_id

            comment["Response to"] = None
            comment["Level"] = 0

            if is_reply:
                comment["Response to"] = parent_id
                parent_level = levels.get(parent_id, 0)
                response_level = parent_level + 1

                mentions = re.findall(r"@(\w+)", comment_text)
                if mentions:
                    last_mention = "@" + mentions[-1]
                    referenced_id = last_comment_by_author.get(last_mention, None)
                    if referenced_id and referenced_id in levels:
                        comment["Response to"] = referenced_id
                        response_level = levels[referenced_id] + 1

                comment["Level"] = response_level

            levels[comment_id] = comment["Level"]
            json.dump(comment, outfile, ensure_ascii=False)
            outfile.write("\n")


def process_directory(input_dir: Path, output_dir: Path):
    """Verbatim from DEF_Relation_Iter.py lines 49-60."""
    for path in sorted(input_dir.glob("*.jsonl")):
        out_path = output_dir / path.name
        if out_path.exists():
            print(f"Already processed: {path.name}")
            continue
        process_jsonl_comments(path, out_path)
        print(f"Relations built: {path.name}")


def main():
    process_directory(IN_DIR, OUT_DIR)
    print("Relation building complete.")


if __name__ == "__main__":
    main()
