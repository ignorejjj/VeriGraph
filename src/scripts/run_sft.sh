nproc_per_node=8

export CACHE_ROOT=/root/data/yz/.cache
export HF_HOME=$CACHE_ROOT/huggingface
export HF_DATASETS_CACHE=$CACHE_ROOT/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$CACHE_ROOT/huggingface/hub
export TRANSFORMERS_CACHE=$CACHE_ROOT/huggingface/hub
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export XDG_CACHE_HOME=$CACHE_ROOT
export TMPDIR=/root/data/yz/tmp
export TEMP=/root/data/yz/tmp
export TMP=/root/data/yz/tmp
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"


# NPROC_PER_NODE=$nproc_per_node \
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# swift sft \
#     --model /root/data/jjj/VerifyReport/model_checkpoints/sft_v3_multiturn_0404/v2-20260404-174422/checkpoint-1911 \
#     --train_type "full" \
#     --dataset /root/data/yz/workspace/data_postprocess/final_data_v1_after_v3/sft_data.json \
#     --torch_dtype bfloat16 \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --learning_rate 1e-5 \
#     --target_modules all-linear \
#     --gradient_accumulation_steps 16 \
#     --eval_steps 100 \
#     --save_steps 100 \
#     --save_total_limit 2 \
#     --logging_steps 5 \
#     --dataset_shuffle false \
#     --max_length 40000 \
#     --output_dir /root/data/jjj/VerifyReport/model_checkpoints/sft_v1_continued_v3 \
#     --warmup_ratio 0.05 \
#     --dataloader_num_workers 16 \
#     --model_author swift \
#     --model_name swift-robot \
#     --deepspeed zero3 \
#     --attn_impl flash_attn \
#     --use_liger_kernel true \
#     --dataset_num_proc 16 \
#     --train_dataloader_shuffle false \
#     --resume_from_checkpoint /root/data/jjj/VerifyReport/model_checkpoints/sft_v1_continued_v3/v1-20260425-125738/checkpoint-600

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /root/data/jjj/VerifyReport/model_checkpoints/sft_v3_multiturn_0404/v2-20260404-174422/checkpoint-1911 \
    --train_type full \
    --dataset /root/data/yz/workspace/data_postprocess/complete_only_research/sft_data.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_shuffle false \
    --max_length 40000 \
    --output_dir /root/data/jjj/VerifyReport/model_checkpoints/sft_0425_continue_only_research \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --model_author swift \
    --model_name swift-robot \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --use_liger_kernel true \
    --dataset_num_proc 16 \
    --train_dataloader_shuffle false

        # --dataset '/root/data/yz/workspace/data_postprocess/final_data/final_merged_sft_data_v2_with-failed.json' \
    # --dataset /root/data/yz/workspace/Verify_data/ablation/final_no_context_split \
    # --dataset /root/data/yz/workspace/Verify_data/ablation/final_no_trail_split \



