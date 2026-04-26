import asyncio
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from verl.utils.verigraph_rl import VeriGraphTrajectoryRuntime, extract_last_code_block


def test_extract_last_code_block_uses_code_interpreter_tags():
    text = "<think>x</think><code_interpreter>\nprint(1)\n</code_interpreter>"
    assert extract_last_code_block(text) == "print(1)"


def test_verigraph_runtime_detects_submission_and_final_claims():
    with tempfile.TemporaryDirectory() as workdir:
        runtime = VeriGraphTrajectoryRuntime(workdir)
        try:
            result = asyncio.run(
                runtime.aexecute(
                    """
c1 = bind("The mean is {x}", x=1.234)
submit_answer(c1)
"""
                )
            )
        finally:
            asyncio.run(runtime.aclose())

    assert result.submitted_answer_present is True
    assert result.final_claims == ["The mean is 1.234"]


if __name__ == "__main__":
    test_extract_last_code_block_uses_code_interpreter_tags()
    test_verigraph_runtime_detects_submission_and_final_claims()
    print("verigraph CPU smoke tests passed")
