from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the repo `src/` directory to sys.path so we can reuse the VeriGraph
# executor/runtime defined under `src/agents/`.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.core.code_executor import CodeExecutor
from agents.resources.init_workspaces import VERIGRAPH_INIT_CODE
from agents.resources.prompts import VERIGRAPH_PROMPT, VERIGRAPH_PROMPT_SFT

CODE_CALL_TAGS = ("<code_interpreter>", "</code_interpreter>")
CODE_RESULT_TAGS = ("<tool_response>", "</tool_response>")


def build_verigraph_messages(question: str, files: Optional[List[str]] = None) -> List[Dict[str, str]]:
    file_text = ""
    if files:
        file_text = "\n(Attach files: " + ",".join(str(file_name) for file_name in files if str(file_name).strip()) + ")"
    return [
        {"role": "system", "content": VERIGRAPH_PROMPT_SFT},
        {"role": "user", "content": str(question or "") + file_text},
    ]


def extract_last_code_block(
    text: str,
    start_marker: str = CODE_CALL_TAGS[0],
    end_marker: str = CODE_CALL_TAGS[1],
) -> Optional[str]:
    text = str(text or "")

    def _strip_fenced_code(content: str) -> str:
        content = content.strip()
        fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        open_fence = re.match(r"```(?:python)?\s*(.*)", content, flags=re.DOTALL)
        if open_fence:
            content = open_fence.group(1).strip()
            if content.endswith("```"):
                content = content[:-3].rstrip()
        return content

    matches = re.findall(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), text, flags=re.DOTALL)
    if matches:
        return _strip_fenced_code(matches[-1])

    # If the closing tag is missing, fall back to the last opening tag.
    start_idx = text.rfind(start_marker)
    if start_idx < 0:
        return None
    content = text[start_idx + len(start_marker) :]
    return _strip_fenced_code(content)


def extract_final_claims(exported_graph: Dict[str, Any]) -> List[str]:
    final_ids = [str(item) for item in exported_graph.get("final_claim_ids", []) if item]
    claim_map = {
        str(item.get("id")): str(item.get("content") or "").strip()
        for item in exported_graph.get("graph", {}).get("claims", [])
        if isinstance(item, dict) and item.get("id")
    }
    return [claim_map[claim_id] for claim_id in final_ids if claim_map.get(claim_id)]


def is_submission_finished(code_executor: CodeExecutor) -> bool:
    return code_executor.is_submission_finished()


def format_tool_response(call_result: Dict[str, Any], code_executor: CodeExecutor, max_tool_output_chars: int = 1200) -> str:
    stdout = str(call_result.get("stdout") or "").strip() or "No output"
    stderr = str(call_result.get("stderr") or "").strip()
    environment = code_executor.get_variable_info()[:max_tool_output_chars]
    claims = code_executor.get_graph_context_with_vars()[:max_tool_output_chars]
    lines = [
        "Execution Result:",
        f'Execution Status: {"Failed" if call_result.get("error") else "Success"}',
        f"Output:\n```{stdout[-max_tool_output_chars:]}```",
    ]
    if call_result.get("error"):
        lines.append(f"Traceback/Error:\n```{stderr[:max_tool_output_chars]}```")
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


@dataclass
class VeriGraphStepResult:
    call_result: Dict[str, Any]
    observation: str
    submitted_answer_present: bool
    final_claims: List[str]


def build_verigraph_step_result(
    code_executor: CodeExecutor,
    call_result: Dict[str, Any],
    max_tool_output_chars: int = 1200,
) -> VeriGraphStepResult:
    observation = format_tool_response(call_result, code_executor, max_tool_output_chars=max_tool_output_chars)
    exported_graph = code_executor.export_graph()
    final_claims = extract_final_claims(exported_graph)
    submitted_answer_present = is_submission_finished(code_executor)
    return VeriGraphStepResult(
        call_result=dict(call_result),
        observation=observation,
        submitted_answer_present=submitted_answer_present,
        final_claims=final_claims,
    )


class VeriGraphTrajectoryRuntime:
    def __init__(
        self,
        working_dir: str,
        init_code: str = VERIGRAPH_INIT_CODE,
        max_tool_output_chars: int = 1200,
        execution_timeout_seconds: Optional[float] = 120.0,
    ) -> None:
        self.code_executor = CodeExecutor(
            working_dir=working_dir,
            init_code=init_code,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        self.max_tool_output_chars = int(max_tool_output_chars)

    async def aexecute(self, code: str) -> VeriGraphStepResult:
        call_result = await self.code_executor.aexecute(code)
        return build_verigraph_step_result(
            self.code_executor,
            call_result,
            max_tool_output_chars=self.max_tool_output_chars,
        )

    def close(self) -> None:
        self.code_executor.close()

    async def aclose(self) -> None:
        await self.code_executor.aclose()

