from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from agents.core.code_executor import CodeExecutor
from agents.core.llm_client import LLMClient
from agents.resources.init_workspaces import BASE_INIT_CODE
from agents.resources.prompts import CODEACT_PROMPT


class CodeActAgent:
    def __init__(self, llm_config: Dict[str, Any], agent_config: Dict[str, Any]):
        self.llm = LLMClient(llm_config)
        self.model_name = llm_config["model_name"]
        self.working_dir = agent_config.get("working_dir", ".")
        self.max_turn = int(agent_config.get("max_turn", 30))
        self.max_tool_output_chars = int(agent_config.get("max_tool_output_chars", 1200))
        self.max_prompt_chars = int(agent_config.get("max_prompt_chars", 120000))
        self.max_keep_history_turns = int(agent_config.get("max_keep_history_turns", 100))

        # Safety: executing arbitrary model-generated Python must have a hard timeout;
        # otherwise a single infinite loop can stall the whole SFT annotation run.
        raw_timeout = agent_config.get("execution_timeout_seconds", os.getenv("CODE_EXECUTION_TIMEOUT_SECONDS", "120"))
        try:
            execution_timeout_seconds = float(raw_timeout) if raw_timeout is not None else 120.0
        except (TypeError, ValueError):
            execution_timeout_seconds = 120.0
        if execution_timeout_seconds <= 0:
            execution_timeout_seconds = None
        self.execution_timeout_seconds = execution_timeout_seconds

        self.stop_words = agent_config.get("stop_words", ["</code_interpreter>", "</code_result>", "</answer>"])
        self.code_result_tags = agent_config.get("code_result_tags", ("<code_result>", "</code_result>"))
        self.code_call_tags = agent_config.get("code_call_tags", ("<code_interpreter>", "</code_interpreter>"))
        self.messages: List[Dict[str, Any]] = []
        self.turn_history: List[Dict[str, Any]] = []
        self.code_executor = None

        self.tokenizer = agent_config.get("tokenizer") or self._load_tokenizer(agent_config.get("tokenizer_name"))
        self.prompt = agent_config.get("prompt", CODEACT_PROMPT)
        self.init_code = agent_config.get("init_code", BASE_INIT_CODE)
        self.prepare_code_executor()

    def _load_tokenizer(self, tokenizer_name: Optional[str]):
        if tokenizer_name is False:
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_name or self.model_name)

    def prepare_code_executor(self):
        self.code_executor = CodeExecutor(
            self.working_dir,
            self.init_code,
            execution_timeout_seconds=self.execution_timeout_seconds,
        )

    def build_init_prompt(self, question: str, files: Optional[List[str]] = None, return_messages=False):
        file_text = ""
        if files:
            file_text = "\n(Attach files: " + ",".join(str(file_name) for file_name in files if str(file_name).strip()) + ")"
        question = str(question or "") + file_text
        
        self.messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": question},
        ]
        if return_messages:
            return self.messages
        else:
            return self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)

    def format_code_result(self, call_result: Dict[str, Any]):
        stdout = str(call_result.get("stdout") or "").strip() or "No output"
        stderr = str(call_result.get("stderr") or "").strip()
        environment = self.code_executor.get_variable_info()[: self.max_tool_output_chars]
        lines = [
            "Execution Result:",
            f'Execution Status: {"Failed" if call_result.get("error") else "Success"}',
            f"Output:\n```{stdout[: self.max_tool_output_chars]}```",
        ]
        if call_result.get("error"):
            lines.append(f"Traceback/Error:\n```{stderr[: self.max_tool_output_chars]}```")
        lines.extend(["", "Current Environment Space:", environment or "No variables defined yet."])
        return "\n".join(lines)

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
    ):
        return self.llm.generate(prompt=prompt, messages=messages, stop=stop)

    async def agenerate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
    ):
        return await self.llm.agenerate(prompt=prompt, messages=messages, stop=stop)

    def extract_between(self, text: str, start_marker: str, end_marker: str):
        matches = re.findall(
            re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
            str(text or ""),
            flags=re.DOTALL,
        )
        if not matches:
            return None
        content = matches[-1].strip()
        fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
        return fenced.group(1).strip() if fenced else content

    def should_stop_after_code(self, call_result: Dict[str, Any]):
        return False

    def _compress_history(self, messages: List[Dict[str, Any]], multi_turn=False) -> List[Dict[str, Any]]:
        keep_turn = self.max_keep_history_turns
        start_tag, end_tag = self.code_result_tags
        init_msgs, solve_msgs = messages[:self.init_msg_count], messages[self.init_msg_count:]
        for msg in solve_msgs[::-1]:
            target_key = 'user' if multi_turn else 'tool'
            if msg["role"] == target_key:
                if keep_turn <= 0:
                    msg["content"] = f"\n\n{start_tag}[omitted tool result]{end_tag}\n\n"
                else:
                    # 保持原始content不变
                    keep_turn -= 1
        messages = init_msgs + solve_msgs
        return messages

    async def arun(self, question: str, files: Optional[List[str]] = None):
        prompt_head = self.build_init_prompt(question, files)
        self.init_msg_count = len(self.messages)

        prediction = ""
        termination = "max_turn"
        current_turn = 0

        code_call_tag, code_call_end_tag = self.code_call_tags
        code_result_tag, code_result_end_tag = self.code_result_tags

        while current_turn < self.max_turn:
            current_turn += 1

            current_prompt = prompt_head + "".join(msg["content"] for msg in self.messages[self.init_msg_count:])
            response = await self.agenerate(current_prompt, stop=self.stop_words)
            if '<code_interpreter>' in response and '</code_interpreter>' not in response:
                response += '</code_interpreter>'
            code_snippet = self.extract_between(response, code_call_tag, code_call_end_tag)

            assistant_msg = {"role": "assistant", "content": response.strip()}
            self.messages.append(assistant_msg)

            if code_snippet and current_turn < self.max_turn:
                call_result = await self.code_executor.aexecute(code_snippet.strip())
                code_result = self.format_code_result(call_result)
            elif code_snippet and current_turn >= self.max_turn:
                call_result = {
                    "stdout": "",
                    "stderr": "Max turns reached! You have no more turns to call tools. Please submit your final answer.",
                    "error": True,
                }
                code_result = self.format_code_result(call_result)
            else:
                prediction = response
                termination = "answer"
                break

            tool_msg = {
                "role": "tool",
                "extracted_code": f"\n\n{code_call_tag}{code_snippet}{code_call_end_tag}\n\n",
                "content": f"\n\n{code_result_tag}{code_result}{code_result_end_tag}\n\n",
            }
            self.messages.append(tool_msg)
            self.turn_history.append(
                {
                    "turn": current_turn,
                    "code": code_snippet.strip() if code_snippet else None,
                    "result": code_result if code_snippet else None,
                }
            )

            self.messages = self._compress_history(self.messages)

            if code_snippet and self.should_stop_after_code(call_result):
                prediction = response
                termination = "answer"
                break

        final_seq = prompt_head + "".join(msg["content"] for msg in self.messages[self.init_msg_count:])

        return {
            "question": question,
            "prediction": prediction,
            "termination_reason": termination,
            "steps_taken": current_turn,
            "final_seq": final_seq,
            "running_messages": self.messages,
            "turn_history": self.turn_history,
        }

    async def arun_multiturn(self, question: str, files: Optional[List[str]] = None):
        self.build_init_prompt(question, files, return_messages=True)
        self.init_msg_count = len(self.messages)

        prediction = ""
        termination = "max_turn"
        current_turn = 0

        code_call_tag, code_call_end_tag = self.code_call_tags
        code_result_tag, code_result_end_tag = self.code_result_tags

        while current_turn < self.max_turn:
            current_turn += 1

            response = await self.agenerate(messages=self.messages, stop=self.stop_words)
            if '<code_interpreter>' in response and '</code_interpreter>' not in response:
                response += '</code_interpreter>'
            code_snippet = self.extract_between(response, code_call_tag, code_call_end_tag)

            assistant_msg = {"role": "assistant", "content": response.strip()}
            self.messages.append(assistant_msg)

            if code_snippet and current_turn < self.max_turn:
                call_result = await self.code_executor.aexecute(code_snippet.strip())
                code_result = self.format_code_result(call_result)
            elif code_snippet and current_turn >= self.max_turn:
                call_result = {
                    "stdout": "",
                    "stderr": "Max turns reached! You have no more turns to call tools. Please submit your final answer.",
                    "error": True,
                }
                code_result = self.format_code_result(call_result)
            else:
                prediction = response
                termination = "answer"
                break

            tool_msg = {
                "role": "tool",
                "extracted_code": f"{code_call_tag}{code_snippet}{code_call_end_tag}",
                "code_result": f"{code_result_tag}\n{code_result}{code_result_end_tag}",
            }
            self.messages.append({"role": "user", "content": tool_msg['code_result']})
            self.turn_history.append(
                {
                    "turn": current_turn,
                    "code": code_snippet.strip() if code_snippet else None,
                    "result": code_result if code_snippet else None,
                }
            )

            self.messages = self._compress_history(self.messages, multi_turn=True)

            if code_snippet and self.should_stop_after_code(call_result):
                prediction = response
                termination = "answer"
                break

        return {
            "question": question,
            "prediction": prediction,
            "termination_reason": termination,
            "steps_taken": current_turn,
            "final_seq": '',
            "running_messages": self.messages,
            "turn_history": self.turn_history,
        }

    def close(self):
        self.llm.close()
        if self.code_executor is not None:
            self.code_executor.close()

    async def aclose(self):
        await self.llm.aclose()
        if self.code_executor is not None:
            await self.code_executor.aclose()

    def run(self, question: str, files: Optional[List[str]] = None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(question, files))
        raise RuntimeError("CodeActAgent.run() cannot be used inside an active event loop; use `await agent.arun(...)`.")
