"""
Phase 2: Annotate all pairs not in the gold standard with the LLM.
Reads  : 08_pairs/sampled_pairs.jsonl
         09_manual_annotations/gold_standard.jsonl  (to exclude gold IDs)
Outputs: 11_llm_annotations/llm_annotated_rest.jsonl
Resumable: skips PairIDs already present in the output file.
"""
import os
import time
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
import anthropic
from utils.jsonl_io import load_jsonl, append_jsonl

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

PAIRS_FILE = Path("./08_pairs/sampled_pairs.jsonl")
GOLD_FILE  = Path("./09_manual_annotations/gold_standard.jsonl")
OUT_DIR    = Path("./11_llm_annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REST_OUT   = OUT_DIR / "llm_annotated_rest.jsonl"

PROMPT_VERSION = "v1.0"
CALL_DELAY = 0.5  # seconds between API calls

VALID_TECHNIQUES = {
    'Appeal_to_Authority', 'Appeal_to_fear-prejudice',
    'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy',
    'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation',
    'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling',
    'Repetition', 'Slogans/Thought-terminating_Cliches',
    'Whataboutism,Straw_Men', 'Appeal_to_Time',
}

INTERACTION_TOOL = {
    "name": "classify_interaction",
    "description": "Classify the interaction quality of a comment pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "enum": [1.0, 0.5, 0.0, -0.5, -1.0],
                "description": "+1=Constructive Disagreement, +0.5=Constructive Agreement, 0=Neutral/Rephrase, -0.5=Destructive Agreement, -1=Destructive Disagreement",
            },
            "label": {
                "type": "string",
                "enum": [
                    "Constructive Disagreement", "Constructive Agreement",
                    "Neutral/Rephrase", "Destructive Agreement", "Destructive Disagreement",
                ],
            },
            "reasoning": {"type": "string"},
        },
        "required": ["score", "label", "reasoning"],
    },
}


# Prompt loaders
def _load(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("#"):
        return ""
    return text

def load_prompts() -> dict:
    return {
        "stance_migration":        _load(Path("prompts/stance_migration.txt")),
        "stance_climate":          _load(Path("prompts/stance_climate.txt")),
        "parent_stance_migration": _load(Path("prompts/parent_stance_migration.txt")),
        "parent_stance_climate":   _load(Path("prompts/parent_stance_climate.txt")),
        "techniques":              _load(Path("prompts/techniques.txt")),
        "interaction":             _load(Path("prompts/interaction.txt")),
    }

def load_video_topics() -> dict:
    mapping = {}
    if not PAIRS_FILE.exists():
        return mapping
    for line in PAIRS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            p = __import__("json").loads(line)
            mapping[p["VideoID"]] = p.get("category", "migration")
        except Exception:
            pass
    return mapping


# Annotation functions
def annotate_stance(client, parent_text: str, child_text: str, system_prompt: str,
                    target: str = "child"):
    if not system_prompt:
        return None
    # For parent stance: classify only the parent comment (child is provided for context)
    if target == "parent":
        user_msg = f"Comment to classify: {parent_text}\nContext (response): {child_text}"
    else:
        user_msg = f"Parent Comment: {parent_text}\nResponse Comment: {child_text}"
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=10,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            digit = resp.content[0].text.strip()
            if digit in ("0", "1", "2"):
                return int(digit)
        except Exception as e:
            print(f"    stance error [{target}] (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None  # flagged as AnnotationFailed in annotate_pair


def annotate_techniques(client, child_text: str, system_prompt: str) -> list:
    if not system_prompt:
        return []
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": child_text}],
            )
            text = resp.content[0].text.strip()
            if "no propaganda detected" in text.lower():
                return []
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if l in VALID_TECHNIQUES]
        except Exception as e:
            print(f"    techniques error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None  # flagged as AnnotationFailed in annotate_pair


def annotate_interaction(client, parent_text: str, child_text: str, system_prompt: str) -> dict:
    if not system_prompt:
        return {"score": None, "label": None, "reasoning": None}
    user_msg = f"PARENT COMMENT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}"
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=512,
                system=system_prompt,
                tools=[INTERACTION_TOOL],
                tool_choice={"type": "tool", "name": "classify_interaction"},
                messages=[{"role": "user", "content": user_msg}],
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return block.input
        except Exception as e:
            print(f"    interaction error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return {"score": None, "label": None, "reasoning": None}  # flagged as AnnotationFailed


def annotate_pair(client, pair: dict, prompts: dict, video_topics: dict) -> dict:
    topic = video_topics.get(pair["VideoID"], pair.get("category", "migration"))
    stance_prompt        = (prompts["stance_climate"]
                            if topic == "climate" else prompts["stance_migration"])
    parent_stance_prompt = (prompts["parent_stance_climate"]
                            if topic == "climate" else prompts["parent_stance_migration"])

    child_stance  = annotate_stance(client, pair["ParentText"], pair["ChildText"],
                                    stance_prompt, target="child")
    time.sleep(CALL_DELAY)
    parent_stance = annotate_stance(client, pair["ParentText"], pair["ChildText"],
                                    parent_stance_prompt, target="parent")
    time.sleep(CALL_DELAY)
    techs    = annotate_techniques(client, pair["ChildText"], prompts["techniques"])
    time.sleep(CALL_DELAY)
    interact = annotate_interaction(client, pair["ParentText"], pair["ChildText"],
                                    prompts["interaction"])
    time.sleep(CALL_DELAY)

    failed = ((child_stance is None) or (parent_stance is None) or
              (techs is None) or (interact.get("score") is None))
    return {
        **pair,
        "StanceLabel":          child_stance,
        "ParentStanceLabel":    parent_stance,
        "Techniques":           techs if techs is not None else [],
        "InteractionScore":     interact.get("score"),
        "InteractionLabel":     interact.get("label"),
        "InteractionReasoning": interact.get("reasoning"),
        "Topic":                topic,
        "PromptVersion":        PROMPT_VERSION,
        "ModelUsed":            CLAUDE_MODEL,
        "AnnotatedAt":          datetime.now(timezone.utc).isoformat(),
        "AnnotationFailed":     failed,
    }


# Main
def main():
    if not PAIRS_FILE.exists():
        print("sampled_pairs.jsonl not found — run step 8 first.")
        return

    client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url="https://api.anthropic.com")
    prompts      = load_prompts()
    video_topics = load_video_topics()

    all_pairs = load_jsonl(str(PAIRS_FILE))
    gold_ids  = set()
    if GOLD_FILE.exists():
        gold_ids = {r["PairID"] for r in load_jsonl(str(GOLD_FILE))}

    # Resume: skip only pairs that are fully annotated 
    done = set()
    if REST_OUT.exists():
        for r in load_jsonl(str(REST_OUT)):
            if (r.get("StanceLabel") is not None
                    and r.get("ParentStanceLabel") is not None
                    and r.get("InteractionScore") is not None):
                done.add(r["PairID"])

    rest    = [p for p in all_pairs if p["PairID"] not in gold_ids]
    pending = [p for p in rest if p["PairID"] not in done]

    print(f"Total pairs:      {len(all_pairs)}")
    print(f"Gold (excluded):  {len(gold_ids)}")
    print(f"Remaining:        {len(rest)}")
    print(f"Already done:     {len(done)}")
    print(f"To annotate:      {len(pending)}")

    for i, pair in enumerate(pending, 1):
        result = annotate_pair(client, pair, prompts, video_topics)
        append_jsonl(result, str(REST_OUT))
        if i % 50 == 0 or i == len(pending):
            ok = "OK" if result.get("InteractionScore") is not None else "PARTIAL"
            print(f"  [{i}/{len(pending)}] {pair['PairID']} — {ok}")

    print(f"\nDone. Annotations saved in {REST_OUT}")


if __name__ == "__main__":
    main()
