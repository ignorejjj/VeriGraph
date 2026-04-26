#!/usr/bin/env bash
# Judge-based evaluation entrypoint.
#
# Before running, serve the judge model at $JUDGE_API_URL with an
# OpenAI-compatible runtime (see ../serving/host_model.sh).

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Each inference run produces one directory under $OUTPUT_ROOT named
# "<dataset>-<agent_type>-<model_suffix>[-<save_note>]".
# Pass the directories to score in OUTPUT_DIRS, separated by spaces.
if [[ ${#OUTPUT_DIRS[@]} -eq 0 ]]; then
  echo "Set OUTPUT_DIRS=(\"./outputs/tablebench-verigraph-...\" ...) before running." >&2
  exit 1
fi

for output_dir in "${OUTPUT_DIRS[@]}"; do
  python judge.py --output_dir "${output_dir}"
done
