from __future__ import annotations

import ast
import asyncio
import copy
import datetime
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import threading
import time
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


EMPTY_GRAPH = {"graph": {"claims": [], "edges": []}, "final_claim_ids": []}


def _executor_error_result(message: str) -> Dict[str, Any]:
    return {"stdout": "", "stderr": message, "error": True}


def _build_executor_globals() -> Dict[str, Any]:
    return {
        "__builtins__": __builtins__,
        "asyncio": asyncio,
        "datetime": datetime,
        "json": json,
        "math": math,
        "np": np,
        "os": os,
        "pd": pd,
        "random": random,
        "re": re,
        "sys": sys,
    }


def _render_init_code(init_code: str, working_dir: str) -> str:
    return init_code.replace("<<SYSTEM_DIR_PLACEHOLDER>>", working_dir.replace("\\", "/"))


def _format_syntax_error(exc: SyntaxError) -> str:
    lineno = int(getattr(exc, "lineno", 0) or 0)
    text = str(getattr(exc, "text", "") or "").rstrip("\n")
    offset = getattr(exc, "offset", None)
    message = getattr(exc, "msg", None) or str(exc)

    lines = ["Traceback (most recent call last):"]
    location = '  File "<string>"'
    if lineno > 0:
        location += f", line {lineno}"
    lines.append(location)

    if text:
        lines.append(f"    {text}")
        if isinstance(offset, int) and offset > 0:
            lines.append("    " + (" " * (offset - 1)) + "^")

    lines.append(f"{type(exc).__name__}: {message}")
    return "\n".join(lines)


def _format_user_traceback(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        return _format_syntax_error(exc)

    extracted = traceback.extract_tb(exc.__traceback__)
    user_frames = [frame for frame in extracted if frame.filename == "<string>"]
    if not user_frames:
        return f"{type(exc).__name__}: {exc}"

    lines = ["Traceback (most recent call last):"]
    for frame in user_frames:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
        frame_line = str(frame.line or "").strip()
        if frame_line:
            lines.append(f"    {frame_line}")

    lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _append_formatted_exception(stderr_capture: io.StringIO, exc: BaseException) -> None:
    if stderr_capture.tell() > 0:
        existing = stderr_capture.getvalue()
        if existing and not existing.endswith("\n"):
            stderr_capture.write("\n")
    stderr_capture.write(_format_user_traceback(exc))


def _execute_ast_block(block: str, executor_globals: Dict[str, Any]) -> None:
    parsed = ast.parse(block)
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        if len(parsed.body) > 1:
            head = ast.Module(body=parsed.body[:-1], type_ignores=[])
            exec(compile(head, "<string>", "exec"), executor_globals)
        tail = ast.Expression(parsed.body[-1].value)
        result = eval(compile(tail, "<string>", "eval"), executor_globals)
        if result is not None:
            print(repr(result))
        return
    exec(compile(parsed, "<string>", "exec"), executor_globals)


def _consume_async_main(executor_globals: Dict[str, Any]) -> Optional[Any]:
    async_main = executor_globals.get("async_main")
    if asyncio.iscoroutinefunction(async_main):
        return executor_globals.pop("async_main", None)
    return None


def _execute_code(executor_globals: Dict[str, Any], code: str) -> Dict[str, Any]:
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    has_error = False

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            _execute_ast_block(code, executor_globals)
    except Exception as exc:
        has_error = True
        _append_formatted_exception(stderr_capture, exc)

    async_main = _consume_async_main(executor_globals)
    if not has_error and async_main is not None:
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                asyncio.run(async_main())
        except Exception as exc:
            has_error = True
            _append_formatted_exception(stderr_capture, exc)

    return {
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "error": has_error,
    }


def _export_graph(executor_globals: Dict[str, Any]) -> Dict[str, Any]:
    graph_mgr = executor_globals.get("_graph_mgr")
    if graph_mgr is None:
        return copy.deepcopy(EMPTY_GRAPH)

    claims = []
    edges = []
    for claim in getattr(graph_mgr, "claims", {}).values():
        item = {
            "id": claim.id,
            "content": claim.content,
            "type": claim.type,
            "template": getattr(claim, "template", ""),
            "premise_ids": list(getattr(claim, "premise_ids", [])),
            "reasoning": getattr(claim, "reasoning", ""),
            "final_node": getattr(claim, "final_node", False),
        }
        claims.append(item)
        for premise_id in item["premise_ids"]:
            edges.append({"source": premise_id, "target": item["id"]})

    final_claim_ids = list(getattr(graph_mgr, "submitted_claim_ids", []) or [])
    if not final_claim_ids:
        final_claim_ids = [item["id"] for item in claims if item.get("final_node")]
    return {"graph": {"claims": claims, "edges": edges}, "final_claim_ids": final_claim_ids}


def _format_graph_context(executor_globals: Dict[str, Any]) -> str:
    graph_mgr = executor_globals.get("_graph_mgr")
    if graph_mgr is None or not hasattr(graph_mgr, "get_state_summary"):
        return "No claims established yet."
    try:
        return str(graph_mgr.get_state_summary(executor_globals))
    except Exception:
        return "No claims established yet."


def _format_variable_info(executor_globals: Dict[str, Any]) -> str:
    hidden = {"__builtins__", "os", "pd", "np", "json", "math", "re", "random", "datetime", "asyncio", "sys", "data_dir"}
    lines = []
    for name, value in executor_globals.items():
        if name.startswith("_") or name in hidden:
            continue
        if isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        if isinstance(value, pd.DataFrame):
            lines.append(f"- {name}: DataFrame {value.shape}")
        elif isinstance(value, np.ndarray):
            lines.append(f"- {name}: ndarray {value.shape}")
        elif isinstance(value, (list, tuple, set)):
            lines.append(f"- {name}: {type(value).__name__} len={len(value)}")
        elif isinstance(value, (int, float, bool, str)):
            text = str(value)
            if len(text) > 100:
                text = text[:100] + "..."
            lines.append(f"- {name}: {type(value).__name__} = {text}")
    return "\n".join(lines) if lines else "No variables defined yet."


def _is_submission_finished(executor_globals: Dict[str, Any]) -> bool:
    graph_mgr = executor_globals.get("_graph_mgr")
    if graph_mgr is None:
        return False
    return bool(getattr(graph_mgr, "finished", False))


def _build_snapshot(executor_globals: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "variable_info": _format_variable_info(executor_globals),
        "graph_context": _format_graph_context(executor_globals),
        "graph": _export_graph(executor_globals),
        "submission_finished": _is_submission_finished(executor_globals),
    }


def _code_executor_worker(conn: Connection, working_dir: str, init_code: str) -> None:
    executor_globals = _build_executor_globals()
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    if init_code:
        _execute_code(executor_globals, _render_init_code(init_code, working_dir))

    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                return

            op = str(request.get("op") or "")
            if op == "close":
                return
            if op == "snapshot":
                conn.send({"ok": True, **_build_snapshot(executor_globals)})
                continue
            if op == "execute":
                call_result = _execute_code(executor_globals, str(request.get("code") or ""))
                conn.send({"ok": True, "call_result": call_result, **_build_snapshot(executor_globals)})
                continue
            conn.send({"ok": False, "error": f"Unsupported operation: {op}"})
    finally:
        conn.close()


class CodeExecutor:
    def __init__(self, working_dir: str, init_code: str = "", execution_timeout_seconds: Optional[float] = None):
        self.working_dir = os.path.abspath(working_dir)
        self.init_code = init_code or ""
        self.execution_timeout_seconds = execution_timeout_seconds
        self.startup_timeout_seconds = max(30.0, float(execution_timeout_seconds or 0))
        self._closed = False
        self._broken = False
        self._process: Optional[mp.Process] = None
        self._conn: Optional[Connection] = None
        self._request_lock = threading.Lock()
        self._variable_info = "No variables defined yet."
        self._graph_context = "No claims established yet."
        self._graph = copy.deepcopy(EMPTY_GRAPH)
        self._submission_finished = False

    def _start_worker(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_code_executor_worker,
            args=(child_conn, self.working_dir, self.init_code),
            name="code-executor",
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn
        response = self._communicate({"op": "snapshot"}, timeout=self.startup_timeout_seconds)
        if response is not None:
            self._update_snapshot(response)

    def _ensure_available(self) -> bool:
        if self._closed or self._broken:
            return False
        return self._process is not None and self._process.is_alive() and self._conn is not None

    def _terminate_worker(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1)
            self._process = None

    def _mark_broken(self) -> None:
        self._broken = True
        self._terminate_worker()

    def _recv_response(self, timeout: Optional[float]) -> Optional[Dict[str, Any]]:
        if self._conn is None or self._process is None:
            return None

        if timeout is None or timeout <= 0:
            while True:
                if self._conn.poll(0.1):
                    return self._conn.recv()
                if not self._process.is_alive():
                    return None

        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Code execution timed out after {timeout} seconds.")
            if self._conn.poll(min(0.1, remaining)):
                return self._conn.recv()
            if not self._process.is_alive():
                return None

    def _communicate(self, request: Dict[str, Any], timeout: Optional[float]) -> Optional[Dict[str, Any]]:
        if not self._ensure_available():
            return None
        if self._conn is None:
            return None

        try:
            self._conn.send(request)
            return self._recv_response(timeout)
        except TimeoutError:
            self._mark_broken()
            raise
        except Exception:
            self._mark_broken()
            return None

    def _update_snapshot(self, response: Dict[str, Any]) -> None:
        self._variable_info = str(response.get("variable_info") or "No variables defined yet.")
        self._graph_context = str(response.get("graph_context") or "No claims established yet.")
        graph = response.get("graph")
        if isinstance(graph, dict):
            self._graph = copy.deepcopy(graph)
        else:
            self._graph = copy.deepcopy(EMPTY_GRAPH)
        self._submission_finished = bool(response.get("submission_finished", False))

    def execute(self, code: str) -> Dict[str, Any]:
        if self._closed:
            return _executor_error_result("CodeExecutor is already closed.")
        if self._broken:
            return _executor_error_result("CodeExecutor is unavailable because the worker process has stopped.")

        with self._request_lock:
            if self._process is None:
                try:
                    self._start_worker()
                except TimeoutError as exc:
                    self._mark_broken()
                    return _executor_error_result(str(exc))
                except Exception:
                    self._mark_broken()
                    return _executor_error_result("CodeExecutor worker process failed to start.")
            try:
                response = self._communicate({"op": "execute", "code": code}, timeout=self.execution_timeout_seconds)
            except TimeoutError as exc:
                return _executor_error_result(str(exc))

        if response is None:
            return _executor_error_result("CodeExecutor worker process stopped unexpectedly.")
        if not response.get("ok", False):
            return _executor_error_result(str(response.get("error") or "CodeExecutor worker request failed."))

        self._update_snapshot(response)
        call_result = response.get("call_result")
        if isinstance(call_result, dict):
            return dict(call_result)
        return _executor_error_result("CodeExecutor worker returned an invalid response.")

    async def aexecute(self, code: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.execute, code)

    def get_variable(self, name: str) -> Any:
        if name == "_graph":
            return self.export_graph()
        return None

    def get_variable_info(self) -> str:
        return self._variable_info

    def get_graph_context_with_vars(self) -> str:
        return self._graph_context

    def export_graph(self) -> Dict[str, Any]:
        return copy.deepcopy(self._graph)

    def is_submission_finished(self) -> bool:
        return self._submission_finished

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # If a model-generated snippet is stuck (e.g., infinite loop) and no timeout was
        # enforced for some reason, execute() can be holding the lock indefinitely.
        # We must be able to close without deadlocking the main pipeline.
        acquired = self._request_lock.acquire(timeout=1.0)
        if not acquired:
            self._terminate_worker()
            return
        try:
            if self._conn is not None and self._process is not None and self._process.is_alive():
                try:
                    self._conn.send({"op": "close"})
                except Exception:
                    pass
            self._terminate_worker()
        finally:
            try:
                self._request_lock.release()
            except Exception:
                pass

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
