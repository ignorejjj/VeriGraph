<div align="center">

# VeriGraph: Grounding Agentic Reasoning in Executable Evidence Graphs

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green">
  <img alt="Status" src="https://img.shields.io/badge/code-anonymous--review-lightgrey">
</p>

*A traceable neuro-symbolic agent that builds an executable evidence DAG instead of a linear chain of thought, and is trained with a graph-aware composite reward.*

</div>

---

## Overview

LLM agents on data-intensive tasks usually emit a single text trajectory that entangles deterministic computation with free-form natural-language deduction, which makes their numerical claims hard to reproduce and their qualitative judgments hard to audit.

**VeriGraph** reformulates the agent's objective: instead of producing an unstructured chain of thought, the agent incrementally constructs a **heterogeneous evidence DAG** that links raw data, interpreter variables, computed results, and natural-language claims. Under this formulation, deciding whether an answer is verifiable reduces to a graph-reachability check from raw data nodes to terminal claim nodes.

<div align="center">
  <img src="assets/intro_figure.png" alt="VeriGraph vs. linear reasoning" width="92%">
  <br>
  <sub><b>Figure 1.</b> Comparison between VeriGraph and linear thought–action–observation paradigms.</sub>
</div>

## Key Features

- **Heterogeneous evidence DAG.** *Data nodes* preserve executable provenance over interpreter variables and computed values; *claim nodes* expose semantic derivations among natural-language facts.
- **Three expansion primitives.** `bind` (grounding), `infer` (derivational), and automatic dependency tracing during code execution (computational), embedded directly in the agent's code action space.
- **Graph-aware policy optimization.** A composite reward combining outcome correctness, computational integrity, and derivational coherence, built on top of a DAPO-style trainer.
- **Traceability score.** A graph-based metric that jointly measures (i) connectivity from terminal claims back to raw data sources and (ii) local logical soundness of each derivational edge.
- **End-to-end reproducibility.** Inference, LLM-judge evaluation, supervised fine-tuning, and RL share a single agent core and a unified data layout; everything runs against any OpenAI-compatible endpoint.

## Architecture

<div align="center">
  <img src="assets/main_figure.png" alt="VeriGraph framework overview" width="96%">
  <br>
  <sub><b>Figure 2.</b> The agent iteratively generates code that simultaneously performs computation and extends a heterogeneous evidence DAG; the policy is optimized via a graph-aware composite reward.</sub>
</div>

The codebase covers the three pieces needed to reproduce the paper:

| Stage | Module | Description |
| --- | --- | --- |
| **Inference** | [`src/agents/`](src/agents/), [`src/inference/`](src/inference/) | The `VerigraphAgent` runs in a persistent Python sandbox and exposes `bind` / `infer` / `submit_answer` primitives that incrementally extend the evidence graph. |
| **Evaluation** | [`src/evaluation/`](src/evaluation/) | LLM-judge scoring for short-answer QA (TableBench, InfiAgent-DABBench, DSBench, QRData) and a two-axis (Content / Format) rubric for the report-generation benchmark (DAB-Step Research). |
| **Training** | [`src/training/sft/`](src/training/sft/), [`src/training/rl/`](src/training/rl/) | Cold-start supervised fine-tuning (ms-swift) and a graph-aware RL pipeline built on a [verl](https://github.com/volcengine/verl) fork (DAPO + composite reward). |

## Repository Layout

```text
.
├── assets/                            # README figures
├── data/
│   ├── eval/tablebench/               # Sample eval set (questions.json + per-item context/)
│   └── training/                      # Sample SFT and RL data + per-item working dirs
└── src/
    ├── agents/
    │   ├── codeact.py                 # CodeAct baseline
    │   ├── verigraph.py               # VerigraphAgent (ours)
    │   ├── core/                      # Persistent Python interpreter, OpenAI-compatible client
    │   └── resources/                 # Prompts and the in-sandbox claim-graph runtime
    ├── inference/                     # Run an agent over a benchmark
    ├── evaluation/                    # LLM-judge scoring and summary
    ├── serving/                       # OpenAI-compatible serving helpers (sglang/vLLM)
    └── training/
        ├── sft/                       # Trajectory-distillation cold start (ms-swift)
        └── rl/                        # RL pipeline (verl fork + VeriGraph recipe and reward)
```

## Installation

Python 3.10+ and CUDA 12.x are recommended.

```bash
# Core dependencies for inference and evaluation
pip install openai transformers numpy pandas tqdm json-repair

# Optional: serving / training stacks (install the ones you need)
pip install sglang vllm                              # serving runtimes
pip install "ms-swift[all]" deepspeed                # SFT
pip install -r src/training/rl/requirements.txt      # RL
```

## Quick Start

### 1. Serve a policy model

VeriGraph reads and writes through any OpenAI-compatible endpoint. The convenience launcher wraps a typical sglang invocation:

```bash
MODEL_PATH=Qwen/QwQ-32B \
PORT=8000 TP_SIZE=4 MAX_LEN=131072 \
bash src/serving/host_model.sh
```

For RoPE-scaled long-context inference (e.g. when serving a checkpoint fine-tuned at 32k but inferring at 128k), pass `ROPE_OVERRIDE`:

```bash
ROPE_OVERRIDE='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
MODEL_PATH=/path/to/checkpoint \
bash src/serving/host_model.sh
```

### 2. Run inference

`run_inference.py` drives the agent over a benchmark and writes one JSON file per question into the output directory.

```bash
export VERIGRAPH_MODEL_PATH=Qwen/QwQ-32B
export VERIGRAPH_API_URL=http://localhost:8000/v1
export VERIGRAPH_API_KEY=EMPTY
export VERIGRAPH_OUTPUT_ROOT=$(pwd)/outputs

DATASET_NAME=tablebench AGENT_TYPE=verigraph SAVE_NOTE=run1 \
bash src/inference/run_inference.sh
```

Common flags (all settable from the environment, see [`src/inference/run_inference.sh`](src/inference/run_inference.sh)):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATASET_NAME` | `tablebench` | One of `tablebench`, `infiagent_dabbench`, `dsbench`, `qrdata`, `dabstep_research`. |
| `AGENT_TYPE` | `verigraph` | `verigraph` (ours) or `codeact` (baseline). |
| `MULTI_TURN` | `true` | Use the multi-turn chat template instead of single-turn raw text. |
| `KEEP_HISTORY_CLAIMS` | `true` | Preserve full claim summaries in the compressed history. |
| `MAX_WORKERS` | `16` | Per-machine concurrency. |
| `SAVE_NOTE` | `default` | Suffix appended to the output directory name. |

Each output directory is named `<dataset>-<agent_type>-<model_suffix>[-<save_note>]`.

### 3. Score the predictions

```bash
# Serve a judge model (any OpenAI-compatible chat endpoint works)
MODEL_PATH=Qwen/QwQ-32B PORT=8000 bash src/serving/host_model.sh

# Run the judge over one or more output directories
OUTPUT_DIRS=(
  "$(pwd)/outputs/tablebench-verigraph-QwQ-32B-run1"
)
JUDGE_API_URL=http://localhost:8000/v1 \
JUDGE_MODEL=Qwen/QwQ-32B \
bash src/evaluation/run_judge.sh
```

The script writes `_final_judge_results.json` (per-item judgments) and `_final_judge_summary.json` (per-question-type and global aggregates) into each scored directory. Short-answer datasets are graded with an exact-match / LLM-judge composite; the `dabstep_research` dataset uses the two-axis Content / Format report rubric.

## Data Format

A small ready-to-use sample of TableBench is shipped under [`data/eval/tablebench/`](data/eval/tablebench/) so that the inference and judge pipelines can be exercised end-to-end without external downloads:

- `questions.json` — list of items with `id`, `question`, `answer`, `qtype`, `qsubtype`, and `files` (file names available inside the per-item context directory).
- `context/<id>/` — working directory copied into the agent's sandbox at inference time. The agent's `os.getcwd()` will be this folder.

To plug in a new benchmark, mirror the same layout and register the path in [`src/inference/run_inference.py`](src/inference/run_inference.py):

```python
DATASET_PATHS["my_benchmark"] = "/abs/path/to/my_benchmark"
```

[`data/training/sft_data.json`](data/training/sft_data.json) and [`data/training/rl_data.json`](data/training/rl_data.json) show the schemas consumed by the SFT and RL launchers, respectively.

## Training

### Supervised fine-tuning (cold start)

The cold-start trajectory-distillation stage uses [ms-swift](https://github.com/modelscope/ms-swift) for full-parameter multi-turn fine-tuning:

```bash
BASE_MODEL=Qwen/Qwen2.5-32B \
SFT_DATA=$(pwd)/data/training/sft_data.json \
OUTPUT_DIR=$(pwd)/checkpoints/verigraph_sft \
NPROC_PER_NODE=8 \
bash src/training/sft/run_sft.sh
```

### Reinforcement learning (graph-aware policy optimization)

The RL stage lives under [`src/training/rl/`](src/training/rl/) and is described in detail in [`src/training/rl/NOTE.md`](src/training/rl/NOTE.md). At a high level it reuses the DAPO trainer and adds:

- a `VerigraphAgent`-aligned multi-turn rollout that parses `<code_interpreter>` blocks, executes them in a persistent `CodeExecutor`, returns observations as the next `user` turn, and stops on `submit_answer()`;
- a composite reward (`outcome` + `process` + `infer`) implemented in `verl/workers/reward_manager/verigraph.py`, with the outcome and inference terms backed by an LLM judge.

Launching RL:

```bash
cd src/training/rl
export MODEL_PATH=/path/to/sft_checkpoint
export CKPTS_DIR=/path/to/save/rl_checkpoints
export RAW_JSON=$(pwd)/../../../data/training/rl_data.json
export CONTEXT_ROOT_DIR=$(pwd)/../../../data/training/context
export VERIGRAPH_JUDGE_MODEL=gpt-4o-mini
export VERIGRAPH_JUDGE_API_BASE=https://api.openai.com/v1
export VERIGRAPH_JUDGE_API_KEY=sk-...
# Optional: export WANDB_API_KEY=...

bash run_rl.sh
```

## Configuration Reference

| Component | Variable | Notes |
| --- | --- | --- |
| Inference | `VERIGRAPH_MODEL_PATH` / `VERIGRAPH_MODEL_NAME` | Tokenizer / model id used in the request. |
| Inference | `VERIGRAPH_API_URL` / `VERIGRAPH_API_KEY` | OpenAI-compatible endpoint serving the policy model. |
| Inference | `VERIGRAPH_OUTPUT_ROOT` / `VERIGRAPH_DATA_ROOT` | Override the default `outputs/` and `data/eval/` roots. |
| Inference | `VERIGRAPH_PER_ITEM_TIMEOUT` | Per-question wall-clock budget in seconds (default `2000`). |
| Inference | `VERIGRAPH_MAX_CONCURRENCY` | Cap on outstanding LLM requests across workers. |
| Sandbox | `CODE_EXECUTION_TIMEOUT_SECONDS` | Hard timeout for a single code-cell execution (default `120`). |
| Judging | `JUDGE_MODEL`, `JUDGE_API_URL`, `JUDGE_API_KEY`, `JUDGE_MAX_WORKERS` | Defaults for [`src/evaluation/judge.py`](src/evaluation/judge.py). |
| RL | `MODEL_PATH`, `CKPTS_DIR`, `RAW_JSON`, `CONTEXT_ROOT_DIR`, `VERIGRAPH_JUDGE_*` | See [`src/training/rl/run_rl.sh`](src/training/rl/run_rl.sh). |
| RL | `WANDB_API_KEY` | Optional, only needed if `wandb` logging is enabled. |

## Citation

Anonymous submission to NeurIPS 2026. A citation will be added after the review period.

```bibtex
@inproceedings{verigraph2026,
  title     = {VeriGraph: Grounding Agentic Reasoning in Executable Evidence Graphs},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

The RL pipeline under [`src/training/rl/`](src/training/rl/) is built on top of [verl](https://github.com/volcengine/verl) (Apache 2.0). The inference agent reuses the high-level structure of CodeAct-style agents.

