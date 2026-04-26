#!/usr/bin/env bash
set -e

cd /root/data/jjj/VerifyReport/src

python run_evaluation.py \
    --dataset_name tablebench \
    --agent_type verigraph \
    --max_workers 16 \
    --multi_turn \
    --save_note test \
    --keep_history_claims
