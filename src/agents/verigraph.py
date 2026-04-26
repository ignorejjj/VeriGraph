from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from agents.codeact import CodeActAgent
from agents.resources.init_workspaces import VERIGRAPH_INIT_CODE
from agents.resources.prompts import REPORT_PROMPT, VERIGRAPH_PROMPT, VERIGRAPH_PROMPT_SFT


class VerigraphAgent(CodeActAgent):
    def __init__(self, llm_config: Dict[str, Any], agent_config: Dict[str, Any]):
        multi_turn = agent_config.get("multi_turn", False)
        self.multi_turn = multi_turn
        self.keep_history_claims = agent_config.get("keep_history_claims", True)
        if multi_turn:
            default_prompt = VERIGRAPH_PROMPT_SFT
        else:
            default_prompt = VERIGRAPH_PROMPT
        agent_config.setdefault("prompt", default_prompt)
        agent_config.setdefault("init_code", VERIGRAPH_INIT_CODE)
        agent_config.setdefault("stop_words", ["</code_interpreter>", "</tool_response>"])
        agent_config.setdefault("code_result_tags", ("<tool_response>", "</tool_response>"))
        agent_config.setdefault("max_keep_history_turns", 5)
        super().__init__(llm_config, agent_config)

    def format_code_result(self, call_result: Dict[str, Any]):
        stdout = str(call_result.get("stdout") or "").strip() or "No output"
        stderr = str(call_result.get("stderr") or "").strip()
        environment = self.code_executor.get_variable_info()[: self.max_tool_output_chars]
        claims = self.code_executor.get_graph_context_with_vars()[: self.max_tool_output_chars]
        lines = [
            "Execution Result:",
            f'Execution Status: {"Failed" if call_result.get("error") else "Success"}',
            f"Output:\n```{stdout[-self.max_tool_output_chars:]}```",
        ]
        if call_result.get("error"):
            lines.append(f"Traceback/Error:\n```{stderr[: self.max_tool_output_chars]}```")
        lines.extend(
            [
                "",
                "Current Environment Space:",
                environment or "No variables defined yet.",
                "",
                "Current claims:",
                claims or "No claims established yet.",
            ]
        )
        return "\n".join(lines)

    def _extract_final_claims(self, exported_graph: Dict[str, Any]) -> List[str]:
        final_ids = [str(item) for item in exported_graph.get("final_claim_ids", []) if item]
        claim_map = {
            str(item.get("id")): str(item.get("content") or "").strip()
            for item in exported_graph.get("graph", {}).get("claims", [])
            if isinstance(item, dict) and item.get("id")
        }
        return [claim_map[claim_id] for claim_id in final_ids if claim_map.get(claim_id)]

    def _extract_tool_result_body(self, content: str) -> str:
        start_tag, end_tag = self.code_result_tags
        text = str(content or "")
        start_idx = text.find(start_tag)
        end_idx = text.rfind(end_tag)
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return text.strip()
        start_idx += len(start_tag)
        return text[start_idx:end_idx].strip()

    def _wrap_tool_result_body(self, body: str, multi_turn: bool = False) -> str:
        start_tag, end_tag = self.code_result_tags
        wrapped = f"{start_tag}{body}{end_tag}"
        return wrapped if multi_turn else f"\n\n{wrapped}\n\n"

    def _compress_history(self, messages: List[Dict[str, Any]], multi_turn: bool = False) -> List[Dict[str, Any]]:
        remain_turn = self.max_keep_history_turns
        init_msgs, solve_msgs = messages[:self.init_msg_count], messages[self.init_msg_count:]
        target_role = "user" if multi_turn else "tool"

        for msg in solve_msgs[::-1]:
            if msg["role"] != target_role:
                continue

            if remain_turn <= 0:
                # Drop the entire tool observation.
                msg["content"] = self._wrap_tool_result_body("[omitted tool result]", multi_turn=multi_turn)
            else:
                # For all but the most recent tool response, strip claim info to save context.
                if remain_turn < self.max_keep_history_turns and not self.keep_history_claims:
                    # Keep only the execution summary for older tool turns.
                    body = self._extract_tool_result_body(msg.get("content", ""))
                    body = body.split("Current Environment Space:")[0].rstrip()
                    msg["content"] = self._wrap_tool_result_body(body, multi_turn=multi_turn)
                remain_turn -= 1

        return init_msgs + solve_msgs

    def should_stop_after_code(self, call_result: Dict[str, Any]):
        return self.code_executor.is_submission_finished()

    def export_graph(self, question: str):
        exported_graph = self.code_executor.export_graph()
        return {
            "question": question,
            "graph": exported_graph.get("graph", {"claims": [], "edges": []}),
            "final_claim_ids": exported_graph.get("final_claim_ids", []),
        }

    def _fill_final_claims_from_graph_if_needed(
        self,
        exported_graph: Dict[str, Any],
        *,
        gen_report: bool = False,
    ) -> Dict[str, Any]:
        if not gen_report or self.code_executor.is_submission_finished():
            return exported_graph

        graph = exported_graph.get("graph", {})
        claims = graph.get("claims", []) if isinstance(graph, dict) else []
        if not claims:
            return exported_graph

        exported_graph["final_claim_ids"] = [str(item.get("id")) for item in claims if isinstance(item, dict) and item.get("id")]
        return exported_graph

    def get_report_output(self, question: str, exported_graph: Dict[str, Any]):
        final_claims = self._extract_final_claims(exported_graph)
        if not final_claims:
            return "", "", ""
        if len(final_claims) == 1:
            return "", final_claims[0], final_claims[0]

        prompt = (
            REPORT_PROMPT
            + "\n\nQuestion:\n"
            + str(question or "")
            + "\n\nFinal claims:\n"
            + json.dumps(final_claims, ensure_ascii=False, indent=2)
        )
        try:
            report = self.generate(messages=[{"role": "user", "content": prompt}]).strip()
        except Exception:
            report = ""

        report = re.sub(r"^<answer>|</answer>$", "", report).strip()
        if not report:
            report = "\n".join(f"- {claim}" for claim in final_claims)
        return prompt, report, "\n".join(final_claims)

    async def aget_report_output(self, question: str, final_claims: list):
        if not final_claims or len(final_claims) == 1:
            return None, None

        prompt = (
            REPORT_PROMPT
            + "\n\nQuestion:\n"
            + str(question or "")
            + "\n\nFinal claims:\n"
            + json.dumps(final_claims, ensure_ascii=False, indent=2)
        )
        try:
            report = (await self.agenerate(prompt)).strip()
        except Exception:
            report = ""

        report = re.sub(r"^<answer>|</answer>$", "", report).strip()
        if not report:
            report = "\n".join(f"- {claim}" for claim in final_claims)

        return prompt, report

    async def _finalize_run_output(self, question: str, running_output: Dict[str, Any], gen_report: bool = True):
        exported_graph = self.export_graph(question)
        # exported_graph = self._fill_final_claims_from_graph_if_needed(exported_graph, gen_report=gen_report)
        final_claims = self._extract_final_claims(exported_graph)
        short_answer = "\n".join(final_claims) if final_claims else ""

        if gen_report:
            report_input, report_output = await self.aget_report_output(question, final_claims)
        else:
            report_input, report_output = None, None

        if gen_report and report_output:
            running_output["prediction"] = report_output
            running_output["termination_reason"] = "answer"
        else:
            running_output["prediction"] = short_answer

        running_output["short_answer"] = short_answer
        running_output["executor_results"] = exported_graph
        running_output["final_claims"] = final_claims
        running_output["report_input"] = report_input
        running_output["report_output"] = report_output
        return running_output

    async def arun(self, question: str, files: Optional[List[str]] = None, gen_report: bool = True):
        running_output = await super().arun(question, files)
        return await self._finalize_run_output(question, running_output, gen_report=gen_report)

    async def arun_multiturn(self, question: str, files: Optional[List[str]] = None, gen_report: bool = True):
        running_output = await super().arun_multiturn(question, files)
        return await self._finalize_run_output(question, running_output, gen_report=gen_report)

    def run(self, question: str, files: Optional[List[str]] = None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(question, files))
        raise RuntimeError("VerigraphAgent.run() cannot be used inside an active event loop; use `await agent.arun(...)`.")
