import numpy as np
import pytest
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

pytest.importorskip("ray")
TensorDict = pytest.importorskip("tensordict").TensorDict

from verl import DataProto
from verl.workers.reward_manager.verigraph import (
    VeriGraphRewardManager,
    _extract_judge_response_text,
)


class DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(int(token)) for token in token_ids)


def make_batch():
    batch = TensorDict(
        {
            "prompts": torch.tensor([[11, 12], [21, 22]], dtype=torch.long),
            "responses": torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.long),
        },
        batch_size=[2],
    )
    non_tensor_batch = {
        "reward_model": np.array(
            [
                {"ground_truth": "ref-a"},
                {"ground_truth": None},
            ],
            dtype=object,
        ),
        "extra_info": np.array(
            [
                {"question": "qa", "dir_id": "sample_research_case", "judge_prompt_type": "research"},
                {"question": "qb", "dir_id": "sample_qa_case", "judge_prompt_type": "qa"},
            ],
            dtype=object,
        ),
        "trajectory_info": np.array(
            [
                {
                    "tool_stats": {"total_calls": 2, "successful_calls": 1, "failed_calls": 1, "process_reward": 0.5},
                    "final_claims": ["claim-a"],
                    "submitted_answer_present": True,
                    "valid_trajectory": 1,
                },
                {
                    "tool_stats": {"total_calls": 0, "successful_calls": 0, "failed_calls": 0, "process_reward": 0.0},
                    "final_claims": [],
                    "submitted_answer_present": False,
                    "valid_trajectory": 0,
                },
            ],
            dtype=object,
        ),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def test_verigraph_reward_manager_mixes_process_and_final_scores():
    manager = VeriGraphRewardManager(
        tokenizer=DummyTokenizer(),
        config={
            "judge_model": "dummy-judge",
            "process_weight": 0.5,
            "final_weight": 0.5,
        },
    )
    captured_payloads = []

    def _mock_score(payloads):
        captured_payloads.extend(payloads)
        return [(0.8, "ok") for _ in payloads]

    manager._score_final_claims_batch = _mock_score
    data = make_batch()

    result = manager(data, return_dict=True)
    rewards = result["reward_tensor"].sum(dim=-1).tolist()

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["system_prompt"] == manager.research_judge_prompt
    assert abs(rewards[0] - 0.65) < 1e-6
    assert abs(rewards[1] - 0.0) < 1e-6
    assert result["reward_extra_info"]["reward/process"] == [0.5, 0.0]
    assert result["reward_extra_info"]["reward/final_judge"] == [0.8, 0.0]
    assert result["reward_extra_info"]["judge/reason"] == ["ok", "missing_ground_truth"]
    assert result["reward_extra_info"]["judge/prompt_type"] == ["research", "qa"]


def test_extract_judge_response_text_accepts_plain_string():
    content = _extract_judge_response_text('{"score": 0.7, "reason": "ok"}')
    assert content == '{"score": 0.7, "reason": "ok"}'


def test_extract_judge_response_text_accepts_dict_response():
    content = _extract_judge_response_text(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"score": 0.9, "reason": "great"}',
                    }
                }
            ]
        }
    )
    assert content == '{"score": 0.9, "reason": "great"}'
