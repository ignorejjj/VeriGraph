#!/usr/bin/env bash

set -xeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_rl.sh [flags]

Required flags (or matching env vars):
  --model_path PATH          Path to the SFT checkpoint to start from. (env: MODEL_PATH)
  --ckpts_dir  PATH          Directory to save RL checkpoints into.    (env: CKPTS_DIR)
  --raw_json   PATH          rl_data.json with the prompt list.        (env: RAW_JSON)
  --context_root_dir PATH    Root containing per-item context dirs.    (env: CONTEXT_ROOT_DIR)
  --judge_model NAME         Judge model id for the reward.            (env: VERIGRAPH_JUDGE_MODEL)
  --judge_api_base URL       OpenAI-compatible base URL for judge.     (env: VERIGRAPH_JUDGE_API_BASE)
  --judge_api_key KEY        API key for the judge endpoint.           (env: VERIGRAPH_JUDGE_API_KEY)

Optional flags:
  --root_dir PATH            Where to drop artifacts (default: ./runs).
  --project_name NAME        wandb / logger project (default: verigraph_rl).
  --exp_name NAME            wandb / logger run name (default: verigraph_dapo).
  --rl_data_dir PATH         Pre-built parquet dir; skips preprocess if set.
  --skip_preprocess          Same as setting RUN_PREPROCESS=false.
  -h, --help                 Print this message and exit.
EOF
}

# Parse flags first; fall back to env vars / defaults afterwards so existing
# env-var workflows keep working.
_CLI_MODEL_PATH=""
_CLI_CKPTS_DIR=""
_CLI_RAW_JSON=""
_CLI_CONTEXT_ROOT_DIR=""
_CLI_JUDGE_MODEL=""
_CLI_JUDGE_API_BASE=""
_CLI_JUDGE_API_KEY=""
_CLI_ROOT_DIR=""
_CLI_PROJECT_NAME=""
_CLI_EXP_NAME=""
_CLI_RL_DATA_DIR=""
_CLI_RUN_PREPROCESS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)        _CLI_MODEL_PATH="$2"; shift 2 ;;
    --ckpts_dir)         _CLI_CKPTS_DIR="$2"; shift 2 ;;
    --raw_json)          _CLI_RAW_JSON="$2"; shift 2 ;;
    --context_root_dir)  _CLI_CONTEXT_ROOT_DIR="$2"; shift 2 ;;
    --judge_model)       _CLI_JUDGE_MODEL="$2"; shift 2 ;;
    --judge_api_base)    _CLI_JUDGE_API_BASE="$2"; shift 2 ;;
    --judge_api_key)     _CLI_JUDGE_API_KEY="$2"; shift 2 ;;
    --root_dir)          _CLI_ROOT_DIR="$2"; shift 2 ;;
    --project_name)      _CLI_PROJECT_NAME="$2"; shift 2 ;;
    --exp_name)          _CLI_EXP_NAME="$2"; shift 2 ;;
    --rl_data_dir)       _CLI_RL_DATA_DIR="$2"; shift 2 ;;
    --skip_preprocess)   _CLI_RUN_PREPROCESS="false"; shift ;;
    -h|--help)           usage; exit 0 ;;
    *)                   echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
ARTIFACT_ROOT_DEFAULT="${REPO_ROOT}/runs"

# Keep artifacts configurable, but always launch code from the repo root so
# Hydra can resolve `file://verl/trainer/config` reliably.
ROOT_DIR="${_CLI_ROOT_DIR:-${ROOT_DIR:-${ARTIFACT_ROOT_DEFAULT}}}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export RAY_DEDUP_LOGS=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export VLLM_USE_V1=0
# Optional: export WANDB_API_KEY=... before running to enable wandb logging.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS="${#GPU_IDS[@]}"

PYTHON_BIN="python"

PROJECT_NAME="${_CLI_PROJECT_NAME:-${PROJECT_NAME:-verigraph_rl}}"
EXP_NAME="${_CLI_EXP_NAME:-${EXP_NAME:-verigraph_dapo}}"
LOGGER_BACKENDS="${LOGGER_BACKENDS:-['console','wandb']}"

MODEL_PATH="${_CLI_MODEL_PATH:-${MODEL_PATH:-}}"
CKPTS_DIR="${_CLI_CKPTS_DIR:-${CKPTS_DIR:-}}"

RUN_PREPROCESS="${_CLI_RUN_PREPROCESS:-${RUN_PREPROCESS:-true}}"
RAW_JSON="${_CLI_RAW_JSON:-${RAW_JSON:-}}"
CONTEXT_ROOT_DIR="${_CLI_CONTEXT_ROOT_DIR:-${CONTEXT_ROOT_DIR:-}}"
RL_DATA_DIR="${_CLI_RL_DATA_DIR:-${RL_DATA_DIR:-${ROOT_DIR}/data/verigraph_rl}}"
TRAIN_FILE="${RL_DATA_DIR}/train.parquet"
TEST_FILE="${RL_DATA_DIR}/test.parquet"

# Reward judge endpoint. Pass via CLI flags or set env vars before running.
VERIGRAPH_JUDGE_MODEL="${_CLI_JUDGE_MODEL:-${VERIGRAPH_JUDGE_MODEL:-}}"
VERIGRAPH_JUDGE_API_BASE="${_CLI_JUDGE_API_BASE:-${VERIGRAPH_JUDGE_API_BASE:-}}"
VERIGRAPH_JUDGE_API_KEY="${_CLI_JUDGE_API_KEY:-${VERIGRAPH_JUDGE_API_KEY:-}}"
export VERIGRAPH_JUDGE_MODEL
export VERIGRAPH_JUDGE_API_BASE
export VERIGRAPH_JUDGE_API_KEY

ROLLOUT_DATA_DIR="${ROOT_DIR}/outputs/${EXP_NAME}/rollout"
VALIDATION_DATA_DIR="${ROOT_DIR}/outputs/${EXP_NAME}/validation"

MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=32768
MAX_MODEL_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# Long-context defaults:
# - keep GRPO group size at 4
# - shrink prompt batch sizes aggressively
# - use all visible GPUs for Ulysses SP by default
ROLLOUT_N=4
TRAIN_PROMPT_BSZ=16
GEN_PROMPT_BSZ=4
TRAIN_PROMPT_MINI_BSZ=2
SP_SIZE="${N_GPUS}"
GEN_TP="${N_GPUS}"
ROLLOUT_GPU_MEMORY_UTILIZATION=0.2

# These knobs mainly change micro-batching / KV-cache reservation,
# which reduces peak memory without changing the RL objective.
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=4096
ACTOR_PPO_MICRO_BSZ_PER_GPU=1
ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU=3072
ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU=1
ROLLOUT_MAX_NUM_BATCHED_TOKENS=10000
ACTOR_SHUFFLE=true


MAX_TURNS=32
TEMPERATURE=1.0
TOP_P=0.95

ENABLE_FILTER_GROUPS=true
FILTER_GROUPS_METRIC="seq_final_reward"
# MAX_NUM_GEN_BATCHES=8
MAX_NUM_GEN_BATCHES=0


CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
ACTOR_LR=1e-6
ACTOR_LR_WARMUP_STEPS=10
ACTOR_WEIGHT_DECAY=0.1

ENABLE_OVERLONG_BUFFER=true
OVERLONG_BUFFER_LEN=2048
OVERLONG_PENALTY_FACTOR=1.0
PROCESS_REWARD_WEIGHT=0.2
FINAL_REWARD_WEIGHT=1
INFER_REWARD_WEIGHT=0.2

VAL_BATCH_SIZE=4
VAL_BEFORE_TRAIN=false
VAL_ONLY=false
TEST_FREQ=10
SAVE_FREQ=10
MAX_ACTOR_CKPT_TO_KEEP=3
TOTAL_EPOCHS=1

require_non_empty() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "ERROR: ${name} is empty" >&2
    exit 1
  fi
}

require_non_empty "MODEL_PATH" "${MODEL_PATH}"
require_non_empty "RAW_JSON" "${RAW_JSON}"
require_non_empty "CONTEXT_ROOT_DIR" "${CONTEXT_ROOT_DIR}"
require_non_empty "VERIGRAPH_JUDGE_MODEL" "${VERIGRAPH_JUDGE_MODEL}"
require_non_empty "VERIGRAPH_JUDGE_API_BASE" "${VERIGRAPH_JUDGE_API_BASE}"
require_non_empty "VERIGRAPH_JUDGE_API_KEY" "${VERIGRAPH_JUDGE_API_KEY}"

if [[ "${RUN_PREPROCESS}" == "true" ]]; then
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m recipe.verigraph.preprocess_verigraph_rl \
    --input_json "${RAW_JSON}" \
    --context_root_dir "${CONTEXT_ROOT_DIR}" \
    --output_dir "${RL_DATA_DIR}"
fi

ray stop
sleep 5
ray start --head --node-ip-address 0.0.0.0 --num-gpus "${N_GPUS}" --ray-debugger-external --port 6378

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m recipe.dapo.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation=left \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_turns="${MAX_TURNS}" \
  algorithm.adv_estimator=grpo \
  algorithm.filter_groups.enable="${ENABLE_FILTER_GROUPS}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_MINI_BSZ}" \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}" \
  actor_rollout_ref.actor.optim.weight_decay="${ACTOR_WEIGHT_DECAY}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.actor.shuffle="${ACTOR_SHUFFLE}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
actor_rollout_ref.model.use_liger=True \
actor_rollout_ref.model.enable_activation_offload=True \
actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LENGTH}" \
    actor_rollout_ref.actor.mu_peak=0 \
  actor_rollout_ref.actor.mu_valley=0 \
  +actor_rollout_ref.rollout.repetition_penalty=1.1 \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  reward_model.reward_manager=verigraph \
  reward_model.overlong_buffer.enable="${ENABLE_OVERLONG_BUFFER}" \
  reward_model.overlong_buffer.len="${OVERLONG_BUFFER_LEN}" \
  reward_model.overlong_buffer.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  +reward_model.reward_kwargs.process_weight="${PROCESS_REWARD_WEIGHT}" \
  +reward_model.reward_kwargs.final_weight="${FINAL_REWARD_WEIGHT}" \
  +reward_model.reward_kwargs.infer_weight="${INFER_REWARD_WEIGHT}" \
  +reward_model.reward_kwargs.judge_model="${VERIGRAPH_JUDGE_MODEL}" \
  +reward_model.reward_kwargs.judge_api_base="${VERIGRAPH_JUDGE_API_BASE}" \
  +reward_model.reward_kwargs.judge_api_key="${VERIGRAPH_JUDGE_API_KEY}" \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.val_only="${VAL_ONLY}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
  actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False \
  +actor_rollout_ref.rollout.multi_turn.verigraph.enable=True \
  +actor_rollout_ref.rollout.multi_turn.verigraph.max_tool_output_chars=1200 \
  +actor_rollout_ref.rollout.multi_turn.verigraph.tool_execution_timeout_seconds=120 \
  +actor_rollout_ref.rollout.multi_turn.verigraph.per_turn_max_new_tokens=10000 \
  +actor_rollout_ref.rollout.multi_turn.verigraph.trajectory_timeout_seconds=1800 \
  max_turns=32 \
  do_execute=False \
  "$@" 2>&1 | tee "${EXP_NAME}.log"
