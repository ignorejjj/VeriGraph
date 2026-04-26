import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.resources.prompts import VERIGRAPH_PROMPT_SFT
from recipe.verigraph.preprocess_verigraph_rl import convert_item


def test_convert_item_uses_multiturn_verigraph_prompt_and_research_judge():
    row = convert_item(
        {
            "id": "sample-1",
            "question": "Summarize the key findings.",
            "files": ["report.csv"],
            "answers": ["claim a", "claim b"],
            "dir_id": "research_case_001",
        },
        index=3,
        context_root_dir="/tmp/verigraph-context",
    )

    assert row["prompt"] == [
        {"role": "system", "content": VERIGRAPH_PROMPT_SFT},
        {"role": "user", "content": "Summarize the key findings.\n(Attach files: report.csv)"},
    ]
    assert row["reward_model"]["ground_truth"] == json.dumps(["claim a", "claim b"], ensure_ascii=False, indent=2)
    assert row["working_dir"] == str(Path("/tmp/verigraph-context") / "research_case_001")
    assert row["extra_info"]["judge_prompt_type"] == "research"
    assert row["extra_info"]["has_ground_truth"] is True
