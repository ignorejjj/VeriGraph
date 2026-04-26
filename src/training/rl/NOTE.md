# VeriGraph RL Integration Notes

This subtree is a fork of [verl](https://github.com/volcengine/verl) with a
minimal multi-turn RL pipeline for `VeriGraphAgent`.

The main additions on top of upstream verl are:

| Path | Purpose |
| --- | --- |
| `recipe/verigraph/preprocess_verigraph_rl.py` | Convert VeriGraph training samples into the parquet format consumed by `RLHFDataset`. |
| `verl/utils/verigraph_rl.py` | Helpers shared by rollout and reward: prompt builder, code-block extractor, persistent Python runtime, observation formatter, final-claim extractor, submit-answer detector. |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | Adds a `_is_verigraph_mode` branch that parses `<code_interpreter>` blocks, drives `CodeExecutor`, injects `<tool_response>` as the next `user` turn, and stops as soon as `submit_answer()` fires. |
| `verl/workers/rollout/schemas.py` | `AsyncRolloutRequest` carries `working_dir` and `trajectory_info`. |
| `verl/trainer/ppo/ray_trainer.py` | Allows `tool_config_path=null` when `verigraph.enable=True` so no extra tool schema is injected into the prompt. |
| `verl/workers/reward_manager/verigraph.py` | Reward manager that combines a process reward (tool-call success rate) with a final reward from an LLM judge over the submitted final claims. |
| `recipe/dapo/main_dapo.py` | Threads `reward_model.reward_kwargs` into the new reward manager. |
| `run_rl.sh` | Launcher wired up for the VeriGraph data + judge endpoint. |

## Running

1. Set the environment variables required by `run_rl.sh`:

   ```bash
   export MODEL_PATH=/path/to/sft_checkpoint
   export CKPTS_DIR=/path/to/save/rl_checkpoints
   export RAW_JSON=/path/to/data/training/rl_data.json
   export CONTEXT_ROOT_DIR=/path/to/data/training/context
   export VERIGRAPH_JUDGE_MODEL=gpt-4o-mini
   export VERIGRAPH_JUDGE_API_BASE=https://api.openai.com/v1
   export VERIGRAPH_JUDGE_API_KEY=sk-...
   # Optional: export WANDB_API_KEY=...
   ```

2. Launch:

   ```bash
   bash run_rl.sh
   ```

The script preprocesses `RAW_JSON` into parquet, brings up a local Ray
cluster, and runs `recipe.dapo.main_dapo` with the VeriGraph reward
manager and the multi-turn rollout.

## Tests

CPU-only smoke tests live under `tests/`:

- `tests/utils/test_verigraph_rl_on_cpu.py`
- `tests/utils/test_dapo_workspace_verigraph_on_cpu.py`
- `tests/utils/test_preprocess_verigraph_rl_on_cpu.py`
- `tests/workers/reward_manager/test_verigraph_reward_manager_on_cpu.py`

Run them with `pytest tests/`.
