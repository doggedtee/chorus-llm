import subprocess
import time
from typing import Optional
from pydantic import BaseModel


class CodeExecutionResult(BaseModel):
    code: str
    stdout: str
    stderr: str
    exit_code: int
    latency_ms: float
    failure_mode: Optional[str] = None     # none, timeout, malformed


def execute_code(code: str, timeout: float = 10.0) -> CodeExecutionResult:
    """
    Executes a Python code snippet in a subprocess sandbox.

    Failure contracts:
    - timeout:   code takes longer than timeout seconds
    - malformed: code is blank or not a string
    """
    start = time.time()

    # malformed input
    if not code or not isinstance(code, str):
        return CodeExecutionResult(
            code=str(code),
            stdout="",
            stderr="Invalid input: code must be a non-empty string.",
            exit_code=1,
            latency_ms=0.0,
            failure_mode="malformed",
        )

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        latency = (time.time() - start) * 1000
        print(f"[code_executor] exit_code={result.returncode} latency={latency:.0f}ms")

        return CodeExecutionResult(
            code=code,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            exit_code=result.returncode,
            latency_ms=latency,
            failure_mode="none",
        )

    except subprocess.TimeoutExpired:
        latency = (time.time() - start) * 1000
        return CodeExecutionResult(
            code=code,
            stdout="",
            stderr=f"Code execution timed out after {timeout} seconds.",
            exit_code=1,
            latency_ms=latency,
            failure_mode="timeout",
        )

    except Exception as e:
        latency = (time.time() - start) * 1000
        return CodeExecutionResult(
            code=code,
            stdout="",
            stderr=str(e),
            exit_code=1,
            latency_ms=latency,
            failure_mode="malformed",
        )