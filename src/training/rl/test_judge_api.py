#!/usr/bin/env python3
"""Smoke test for the VeriGraph reward-judge endpoint.

Mirrors the call shape used by ``VeriGraphRewardManager._AsyncJudgeClient``,
so any breakage at the network/auth layer surfaces here first.

Usage:
    export VERIGRAPH_JUDGE_MODEL="gpt-4o-mini"
    export VERIGRAPH_JUDGE_API_BASE="https://api.openai.com/v1"
    export VERIGRAPH_JUDGE_API_KEY="sk-..."
    python test_judge_api.py
"""

import asyncio
import json
import os

from openai import AsyncOpenAI


# ---- Helpers (kept identical to the runtime versions) ----

def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return "".join(_coerce_text(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            return _coerce_text(value.get("text"))
        if "content" in value:
            return _coerce_text(value.get("content"))
    return str(value)


def _extract_judge_response_text(response):
    if isinstance(response, (str, bytes)):
        return _coerce_text(response)
    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            return _coerce_text(getattr(message, "content", None))
    return _coerce_text(getattr(response, "content", None))


def _parse_judge_result(text):
    raw = str(text or "").strip()
    if not raw:
        return 0.0, "empty_response"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0, "json_parse_error"
    score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
    reason = str(payload.get("reason", "")).strip()
    return score, reason


SYSTEM_PROMPT = """You are a strict grader for a data-analysis QA task.
Score how well the candidate final claims answer the question according to the reference answer.

Return JSON only:
{"score": <float between 0 and 1>, "reason": "<short reason>"}
"""

USER_PROMPT = """Reference answer:
The average revenue growth rate is 15% year-over-year.

Candidate final claims:
The company experienced a 15% annual revenue growth on average.
"""


async def main():
    model = os.environ.get("VERIGRAPH_JUDGE_MODEL", "").strip()
    api_base = os.environ.get("VERIGRAPH_JUDGE_API_BASE", "").strip()
    api_key = os.environ.get("VERIGRAPH_JUDGE_API_KEY", "EMPTY").strip()

    print(f"model:    {model}")
    print(f"api_base: {api_base}")
    print(f"api_key:  {api_key[:8]}...", flush=True)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=120,
    )

    print("\nCalling judge...", flush=True)
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        )
        text = _extract_judge_response_text(response)
        score, reason = _parse_judge_result(text)
        print(f"raw response: {text}")
        print(f"score:        {score}")
        print(f"reason:       {reason}")
    except Exception as e:
        print(f"failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
