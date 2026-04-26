from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.resources.prompts import VERIGRAPH_PROMPT_SFT


DEFAULT_GROUND_TRUTH_KEYS = (
    "ground_truth",
    "reference_answer",
    "reference",
    "answer",
    "answers",
    "gold_answer",
    "gold",
    "label",
    "final_claims",
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_value(payload: Any, candidate_keys: Iterable[str]) -> Optional[Any]:
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for value in payload.values():
            found = find_value(value, candidate_keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_value(item, candidate_keys)
            if found not in (None, ""):
                return found
    return None


def build_working_dir(item: Dict[str, Any], context_root_dir: Optional[str]) -> Optional[str]:
    if item.get("working_dir"):
        return str(item["working_dir"])
    if item.get("context_dir"):
        return str(item["context_dir"])
    if context_root_dir and item.get("dir_id") is not None:
        return os.path.join(context_root_dir, str(item["dir_id"]))
    return None


def _normalize_ground_truth_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        normalized_items = [_normalize_ground_truth_value(item) for item in value]
        filtered_items = [item for item in normalized_items if item not in (None, "", [], {})]
        if not filtered_items:
            return None
        return filtered_items
    if isinstance(value, dict):
        normalized_dict = {
            str(key): normalized_value
            for key, val in value.items()
            if (normalized_value := _normalize_ground_truth_value(val)) not in (None, "", [], {})
        }
        if not normalized_dict:
            return None
        return normalized_dict
    return str(value)


def normalize_ground_truth(value: Any) -> Optional[str]:
    normalized_value = _normalize_ground_truth_value(value)
    if normalized_value is None:
        return None
    if isinstance(normalized_value, str):
        return normalized_value
    return json.dumps(normalized_value, ensure_ascii=False, indent=2)


def normalize_files(files: Any) -> list[str]:
    if files in (None, ""):
        return []
    if isinstance(files, list):
        return [str(item) for item in files if str(item).strip()]
    return [str(files)]


def is_research_dir_id(dir_id: Any) -> bool:
    return "research" in str(dir_id or "").lower()


def build_verigraph_messages(question: str, files: Optional[list[str]] = None) -> list[dict[str, str]]:
    file_text = ""
    if files:
        file_text = "\n(Attach files: " + ",".join(str(file_name) for file_name in files if str(file_name).strip()) + ")"
    return [
        {"role": "system", "content": VERIGRAPH_PROMPT_SFT},
        {"role": "user", "content": str(question or "") + file_text},
    ]


def convert_item(item: Dict[str, Any], index: int, context_root_dir: Optional[str]) -> Dict[str, Any]:
    question = str(item.get("question", "")).strip()
    files = normalize_files(item.get("files", []))
    working_dir = build_working_dir(item, context_root_dir)
    raw_ground_truth = find_value(item, DEFAULT_GROUND_TRUTH_KEYS)
    ground_truth = normalize_ground_truth(raw_ground_truth)
    raw_dir_id = item.get("dir_id")
    dir_id = None if raw_dir_id in (None, "") else str(raw_dir_id)
    judge_prompt_type = "research" if is_research_dir_id(dir_id) else "qa"

    return {
        "prompt": build_verigraph_messages(question, files),
        "data_source": "verigraph",
        "ability": "verigraph",
        "reward_model": {
            "style": "llm_judge",
            "ground_truth": ground_truth,
        },
        "working_dir": working_dir,
        "extra_info": {
            "index": index,
            "id": str(item.get("id", index)),
            "question": question,
            "files": files,
            "working_dir": working_dir,
            "context_dir": working_dir,
            "dir_id": dir_id,
            "judge_prompt_type": judge_prompt_type,
            "has_ground_truth": ground_truth not in (None, "", [], {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_root_dir", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_missing_ground_truth",
        action="store_true",
        help="Keep rows with empty ground_truth (not recommended for RL training).",
    )
    args = parser.parse_args()

    raw_items = load_json(args.input_json)
    if not isinstance(raw_items, list):
        raise TypeError("input_json must contain a JSON list")

    context_root_dir = os.path.expanduser(args.context_root_dir) if args.context_root_dir else None
    rows = []
    dropped_missing_ground_truth = 0
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        row = convert_item(item, index, context_root_dir)
        if not args.keep_missing_ground_truth and not row["extra_info"]["has_ground_truth"]:
            dropped_missing_ground_truth += 1
            continue
        rows.append(row)

    if len(rows) < 2:
        raise ValueError(
            "Need at least 2 samples after filtering to build train/test splits. "
            f"Got {len(rows)}. "
            "If you intentionally want to keep samples without ground truth, "
            "pass --keep_missing_ground_truth."
        )

    random.Random(args.seed).shuffle(rows)
    split_idx = int(len(rows) * args.train_ratio)
    split_idx = max(1, min(len(rows) - 1, split_idx))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    print(
        f"train={len(train_rows)} test={len(test_rows)} "
        f"dropped_missing_ground_truth={dropped_missing_ground_truth} "
        f"context_root_dir={context_root_dir} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
