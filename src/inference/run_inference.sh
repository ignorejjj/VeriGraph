#!/usr/bin/env bash
# Inference entrypoint for VeriGraph.
#
# Before launching, serve the policy model at $VERIGRAPH_API_URL using any
# OpenAI-compatible runtime (sglang / vLLM / TGI). The convenience launcher
# at ../serving/host_model.sh wraps a typical sglang invocation.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET_NAME="${DATASET_NAME:-tablebench}"
AGENT_TYPE="${AGENT_TYPE:-verigraph}"
SAVE_NOTE="${SAVE_NOTE:-default}"
MAX_WORKERS="${MAX_WORKERS:-16}"

EXTRA_ARGS=()
if [[ "${MULTI_TURN:-true}" == "true" ]]; then
  EXTRA_ARGS+=("--multi_turn")
fi
if [[ "${KEEP_HISTORY_CLAIMS:-true}" == "true" ]]; then
  EXTRA_ARGS+=("--keep_history_claims")
fi

python run_inference.py \
    --dataset_name "${DATASET_NAME}" \
    --agent_type "${AGENT_TYPE}" \
    --max_workers "${MAX_WORKERS}" \
    --save_note "${SAVE_NOTE}" \
    "${EXTRA_ARGS[@]}"
