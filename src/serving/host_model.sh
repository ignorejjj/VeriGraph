#!/usr/bin/env bash
# Convenience launcher for an OpenAI-compatible inference server.
#
# Required env vars:
#   MODEL_PATH        Local path or hub id of the model to serve.
# Optional env vars:
#   PORT              HTTP port (default: 8000)
#   TP_SIZE           Tensor-parallel size (default: 4)
#   MAX_LEN           Max context length (default: 131072)
#   MEM_FRACTION      sglang static memory fraction (default: 0.8)
#   ROPE_OVERRIDE     If set, applied as --json-model-override-args.
#
# Example:
#   MODEL_PATH=/path/to/checkpoint \
#   ROPE_OVERRIDE='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
#   bash serving/host_model.sh

set -e

: "${MODEL_PATH:?Set MODEL_PATH to the model directory or hub id.}"

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_LEN="${MAX_LEN:-131072}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"

EXTRA=()
if [[ -n "${ROPE_OVERRIDE:-}" ]]; then
  EXTRA+=(--json-model-override-args "${ROPE_OVERRIDE}")
fi

SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --context-length "${MAX_LEN}" \
  "${EXTRA[@]}"
