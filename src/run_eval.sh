cd /root/data/jjj/VerifyReport/src

prefix='verigraph-checkpoint-1911-sft-v3-0404-multiturn-keep-history'
# prefix='verigraph-sft-v2-0327-multiturn'
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/tablebench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/infiagent_dabbench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/dsbench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/dabstep_research-${prefix}"
python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/qrdata-${prefix}"

# python run_evaluation.py --dataset_name tablebench --agent_type 'verigraph' --num_samples 1 --debug

prefix='verigraph-checkpoint-1911-sft-v3-0404-multiturn-no-history'
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/tablebench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/infiagent_dabbench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/dsbench-${prefix}"
# python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/dabstep_research-${prefix}"
python eval_results.py --output_dir "/root/data/jjj/VerifyReport/outputs/qrdata-${prefix}"