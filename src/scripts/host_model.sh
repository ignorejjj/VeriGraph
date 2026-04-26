# vllm serve /root/data/jjj/models/qwq-32b \
#   --dtype auto \
#   --tensor-parallel-size 4 \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 32768

SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /root/data/jjj/models/qwq-32b --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 80000 
# SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /root/data/jjj/models/qwen3.5-35b-a3b --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 131072
# SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /public/huggingface-models/Qwen/Qwen3.5-27B --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 131072


# SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /root/data/jjj/VerifyReport/model_checkpoints/sft_v3_multiturn_0404/v2-20260404-174422/checkpoint-1911 --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 131072 --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

# SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /root/data/jjj/VerifyReport/model_checkpoints/sft_v3_no_atom_0419/v0-20260419-014942/checkpoint-206 --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 131072 --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

# SGLANG_USE_MODELSCOPE=false python -m sglang.launch_server --model-path /root/data/jjj/VerifyReport/model_checkpoints/sft_v1_continued_v3/v0-20260424-200135/checkpoint-461 --port 8000 --tp-size 4 --mem-fraction-static 0.8 --context-length 131072 --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'


# python -m verl.model_merger merge \
#   --backend fsdp \
#   --local_dir /root/data/jjj/VerifyReport/model_checkpoints/rl_v2_filtered_not_valid/global_step_20/actor \
#   --target_dir /root/data/jjj/VerifyReport/model_checkpoints/rl_v2_filtered_not_valid/global_step_20/actor/huggingface


# /root/data/jjj/models/qwen3.5-35b-a3b

# vllm serve /root/data/jjj/VerifyReport/model_checkpoints/sft_v1_0310/v0-20260310-153744/checkpoint-356 \
#   --dtype auto \
#   --tensor-parallel-size 4 \
#   --gpu-memory-utilization 0.9 \
#   --max-model-len 60000 


# qwen3-32b
# vllm serve /root/data/jjj/models/gpt-oss-20b \
#   --dtype auto \
#   --tensor-parallel-size 4 \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 130000 \
#   --tool-call-parser openai \
#   --enable-auto-tool-choice
