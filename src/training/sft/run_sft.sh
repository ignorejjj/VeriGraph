#!/usr/bin/env bash
# Supervised fine-tuning entrypoint (uses ms-swift).
#
# Required env vars:
#   BASE_MODEL        Path or hub id of the model to fine-tune.
#   SFT_DATA          Path to the SFT JSON dataset
#                     (see data/training/sft_data.json for the schema).
#   OUTPUT_DIR        Directory to write checkpoints to.
# Optional env vars:
#   NPROC_PER_NODE    Number of GPUs per node (default: 8)
#   CUDA_VISIBLE_DEVICES (default: 0,1,2,3,4,5,6,7)
#   NUM_EPOCHS        (default: 3)
#   MAX_LENGTH        (default: 40000)
#   LEARNING_RATE     (default: 1e-5)
#   GRAD_ACCUM        Gradient accumulation steps (default: 16)
#   CACHE_ROOT        Directory used for HF / Triton / Inductor caches.

set -e

: "${BASE_MODEL:?Set BASE_MODEL.}"
: "${SFT_DATA:?Set SFT_DATA to the path of sft_data.json.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR.}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-40000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

if [[ -n "${CACHE_ROOT:-}" ]]; then
  export CACHE_ROOT
  export HF_HOME="${CACHE_ROOT}/huggingface"
  export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
  export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
  export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/hub"
  export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
  export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
  export XDG_CACHE_HOME="${CACHE_ROOT}"
  mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
           "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
           "${TORCHINDUCTOR_CACHE_DIR}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
    --model "${BASE_MODEL}" \
    --train_type full \
    --dataset "${SFT_DATA}" \
    --torch_dtype bfloat16 \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate "${LEARNING_RATE}" \
    --target_modules all-linear \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_shuffle false \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --dataset_num_proc 16 \
    --train_dataloader_shuffle false
