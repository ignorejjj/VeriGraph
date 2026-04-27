import argparse
import asyncio
import json
import os
import time
import sys
import numpy as np
import traceback
from typing import Any, Dict, Optional

# Make `agents` importable when launched as a script.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agents.codeact import CodeActAgent
from agents.verigraph import VerigraphAgent

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else []


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data", "eval")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs")

# Populated from CLI in main(); kept module-level so downstream helpers and
# DATASET_PATHS resolution can read them.
MODEL_PATH = "Qwen/QwQ-32B"
MODEL_NAME = MODEL_PATH
API_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
DATA_ROOT = DEFAULT_DATA_ROOT
MAX_VLLM_CONCURRENCY = 128
PER_ITEM_TIMEOUT = 2000


def _dataset_paths(data_root: str) -> Dict[str, str]:
    return {
        "tablebench": os.path.join(data_root, "tablebench"),
        "dabstep_research": os.path.join(data_root, "dabstep_research"),
        "infiagent_dabbench": os.path.join(data_root, "infiagent_dabbench"),
        "dsbench": os.path.join(data_root, "dsbench"),
        "qrdata": os.path.join(data_root, "qrdata"),
    }


DATASET_PATHS = _dataset_paths(DATA_ROOT)

LLM_CONFIG: Dict[str, Any] = {
    "model_name": MODEL_NAME,
    "api_key": API_KEY,
    "api_url": API_URL,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 32768,
    "timeout": 600,
    "extra_body": {
        "top_k": 20,
        "include_stop_str_in_output": True,
        "repetition_penalty": 1.1,
    },
}

AGENT_CONFIG = {
    "codeact": {"max_turn": 30},
    "verigraph": {
        "max_turn": 32, 
        'max_keep_history_turns': 32,
    },
}

_TOKENIZER = None



def console_write(message: str):
    if hasattr(tqdm, "write"):
        tqdm.write(message)
        return
    print(message, file=sys.stderr, flush=True)


def log_case_exception(agent_type: str, item: Dict[str, Any], exc: Exception, traceback_text: str):
    case_id = item.get("id", "<unknown>")
    data_dir = item.get("data_dir", "")
    console_write(f"[ERROR] case={case_id} agent={agent_type} data_dir={data_dir}")
    console_write(f"{type(exc).__name__}: {exc}")
    console_write(traceback_text.rstrip())

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _TOKENIZER


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        if hasattr(obj, 'tolist'):
            return obj.tolist()
            
        return super(NpEncoder, self).default(obj)

def write_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, cls=NpEncoder)


def load_benchmark_data(dataset_dir: str):
    with open(os.path.join(dataset_dir, "questions.json"), "r", encoding="utf-8") as file:
        questions = json.load(file)
    for item in questions:
        item["data_dir"] = os.path.join(dataset_dir, "context", item["id"])
    return questions


def build_agent(agent_type: str, working_dir: str, multi_turn=False, keep_history_claims=False):
    agent_config = dict(AGENT_CONFIG[agent_type])
    agent_config["working_dir"] = working_dir
    agent_config["tokenizer"] = get_tokenizer()
    agent_config['multi_turn'] = multi_turn
    agent_config['keep_history_claims'] = keep_history_claims
    if agent_type == "codeact":
        return CodeActAgent(LLM_CONFIG, agent_config)
    return VerigraphAgent(LLM_CONFIG, agent_config)


async def run_one(
    agent_type: str,
    item: Dict[str, Any],
    output_dir: str,
    gen_report: bool = False,
    multi_turn: bool = False,
    keep_history_claims: bool = False,
    vllm_semaphore: Optional[asyncio.Semaphore] = None,
):
    output_path = os.path.join(output_dir, f"{item['id']}.json")
    if os.path.exists(output_path):
        output =  await asyncio.to_thread(load_json, output_path)
        if not output.get('prediction', '') == '':
            return output

    start_time = time.time()
    agent = None
    should_save = True
    try:
        agent = build_agent(agent_type, item["data_dir"], multi_turn, keep_history_claims)

        async def _run_agent():
            if 'veri' in agent_type:
                if multi_turn:
                    agent_coro = agent.arun_multiturn(item["question"], item.get("files", []), gen_report)
                else:
                    agent_coro = agent.arun(item["question"], item.get("files", []), gen_report)
            else:
                if multi_turn:
                    agent_coro = agent.arun_multiturn(item["question"], item.get("files", []))
                else:
                    agent_coro = agent.arun(item["question"], item.get("files", []))
            if PER_ITEM_TIMEOUT and PER_ITEM_TIMEOUT > 0:
                return await asyncio.wait_for(agent_coro, timeout=float(PER_ITEM_TIMEOUT))
            return await agent_coro

        if vllm_semaphore is None:
            result = await _run_agent()
        else:
            async with vllm_semaphore:
                result = await _run_agent()
    except asyncio.TimeoutError as exc:
        should_save = False
        traceback_text = traceback.format_exc()
        log_case_exception(agent_type, item, exc, traceback_text)
        result = {
            "question": item.get("question"),
            "prediction": "ERROR",
            "termination_reason": "timeout",
            "exception": f"per_item_timeout={PER_ITEM_TIMEOUT}s exceeded",
            "exception_type": type(exc).__name__,
            "traceback": traceback_text,
            "running_messages": [],
            "turn_history": [],
        }
    except Exception as exc:
        should_save = False
        traceback_text = traceback.format_exc()
        log_case_exception(agent_type, item, exc, traceback_text)
        result = {
            "question": item.get("question"),
            "prediction": "ERROR",
            "termination_reason": "exception",
            "exception": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback_text,
            "running_messages": [],
            "turn_history": [],
        }
    finally:
        if agent is not None:
            try:
                await agent.aclose()
            except Exception:
                pass

    result.setdefault("running_messages", [])
    result.setdefault("turn_history", result.get("execution_history", []))
    result.setdefault("execution_history", result.get("turn_history", []))
    result["idx"] = item["id"]
    result["question_item"] = item
    result["agent_type"] = "VerigraphAgent" if agent_type == "verigraph" else "CodeActAgent"
    result["model_name"] = MODEL_NAME
    result["total_time_seconds"] = time.time() - start_time
    if should_save:
        await asyncio.to_thread(write_json, output_path, result)
    return result


async def _run_with_semaphore(semaphore: asyncio.Semaphore, coro):
    async with semaphore:
        return await coro


async def run_experiment(
    agent_type: str,
    dataset_name: str,
    save_note: str = "",
    num_samples: Optional[int] = None,
    multi_turn=False,
    keep_history_claims=False,
    debug_mode: bool = False,
    max_concurrency: int = 8,
):
    dataset_dir = DATASET_PATHS[dataset_name]
    model_suffix = MODEL_NAME.split("/")[-1]
    suffix = f"-{save_note}" if save_note else ""
    output_dir = os.path.join(OUTPUT_ROOT, f"{dataset_name}-{agent_type}-{model_suffix}{suffix}")
    items = load_benchmark_data(dataset_dir)
    if num_samples is not None:
        items = items[:num_samples]
    if debug_mode:
        items = items[:1]

    if 'research' in dataset_name:
        gen_report = True
    else:
        gen_report = False

    await asyncio.to_thread(get_tokenizer)
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency or 1)))
    effective_vllm_limit = MAX_VLLM_CONCURRENCY
    vllm_semaphore = (
        asyncio.Semaphore(effective_vllm_limit) if effective_vllm_limit is not None else None
    )
    if effective_vllm_limit is not None or PER_ITEM_TIMEOUT is not None:
        console_write(
            f"[CONFIG] workers={max(1, int(max_concurrency or 1))} "
            f"vllm_limit={effective_vllm_limit} per_item_timeout={PER_ITEM_TIMEOUT}"
        )
    tasks = [
        asyncio.create_task(
            _run_with_semaphore(
                semaphore,
                run_one(agent_type, item, output_dir, gen_report, multi_turn, keep_history_claims, vllm_semaphore),
            )
        )
        for item in items
    ]

    results = []
    iterator = asyncio.as_completed(tasks)
    if hasattr(tqdm, "__call__"):
        iterator = tqdm(iterator, total=len(tasks), desc="Processing")
    for task in iterator:
        results.append(await task)
    
    error_count = sum(1 for result in results if result.get("termination_reason") == "exception")
    if error_count:
        console_write(f"[SUMMARY] completed={len(results)} errors={error_count} output_dir={output_dir}")
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="verigraph", choices=["codeact", "verigraph"])
    parser.add_argument("--dataset_name", type=str, default="tablebench",
                        help="Benchmark to evaluate. Choices are inferred from --data_root.")
    parser.add_argument("--save_note", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--multi_turn", action='store_true', default=False)
    parser.add_argument("--keep_history_claims", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max_concurrency", "--max_workers", dest="max_concurrency", type=int, default=32)

    parser.add_argument("--model_path", type=str, default="Qwen/QwQ-32B",
                        help="Local path or hub id of the policy model (used as the tokenizer source).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model id sent in the OpenAI-compatible request. Defaults to --model_path.")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1",
                        help="OpenAI-compatible chat-completions endpoint.")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Root directory containing per-benchmark eval data.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="Root directory under which per-run output folders are created.")
    parser.add_argument("--per_item_timeout", type=int, default=2000,
                        help="Per-question wall-clock budget in seconds.")
    return parser.parse_args()


def _apply_runtime_args(args):
    global MODEL_PATH, MODEL_NAME, API_URL, API_KEY, OUTPUT_ROOT, DATA_ROOT
    global PER_ITEM_TIMEOUT, DATASET_PATHS, LLM_CONFIG

    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name or args.model_path
    API_URL = args.api_url
    API_KEY = args.api_key
    OUTPUT_ROOT = args.output_root
    DATA_ROOT = args.data_root
    PER_ITEM_TIMEOUT = args.per_item_timeout

    DATASET_PATHS = _dataset_paths(DATA_ROOT)
    LLM_CONFIG.update({"model_name": MODEL_NAME, "api_url": API_URL, "api_key": API_KEY})

    if args.dataset_name not in DATASET_PATHS:
        raise SystemExit(
            f"Unknown --dataset_name '{args.dataset_name}'. "
            f"Available: {sorted(DATASET_PATHS)}"
        )


if __name__ == "__main__":
    args = parse_args()
    _apply_runtime_args(args)
    asyncio.run(
        run_experiment(
            agent_type=args.agent_type,
            dataset_name=args.dataset_name,
            save_note=args.save_note,
            num_samples=args.num_samples,
            multi_turn=args.multi_turn,
            keep_history_claims=args.keep_history_claims,
            debug_mode=args.debug,
            max_concurrency=args.max_concurrency,
        )
    )
