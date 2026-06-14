# Based on https://github.com/BassiDavide/YouTube_Hybrid_Interactions_Analysis
# by D. Bassi, M. J. Maggini, R. Vieira and M. Pereira-Fariña,
# "A Pipeline for the Analysis of User Interactions in YouTube Comments:
# A Hybridization of LLMs and Rule-Based Methods," 2024 11th International
# Conference on Social Networks Analysis, Management and Security (SNAMS),
# Gran Canaria, Spain, 2024, pp. 146-153, doi: 10.1109/SNAMS64316.2024.10883781.

import json


def load_jsonl(file_path: str) -> list:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Truncated line from interrupted write — skip and warn
                print(f"  [WARN] load_jsonl: skipping corrupt line {lineno} in {file_path}")
    return records


def save_jsonl(records: list, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] iter_jsonl: skipping corrupt line {lineno} in {file_path}")
