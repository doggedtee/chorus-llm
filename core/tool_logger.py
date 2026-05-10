import time
from typing import Any, Optional
from sqlalchemy.orm import Session
from db.models import ToolCallLog


class ToolLogger:
    def __init__(self, db: Session, job_id: str):
        self.db = db
        self.job_id = job_id

    def log(
        self,
        agent_id: str,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        latency_ms: float,
        accepted: bool,
        attempt: int = 1,
        rejection_reason: Optional[str] = None,
        failure_mode: Optional[str] = "none",
    ):
        """Insert one tool call record into the database."""
        record = ToolCallLog(
            job_id=self.job_id,
            agent_id=agent_id,
            tool_name=tool_name,
            attempt=attempt,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            accepted=accepted,
            rejection_reason=rejection_reason,
            failure_mode=failure_mode,
        )
        self.db.add(record)
        self.db.commit()

        status = "accepted" if accepted else f"rejected ({rejection_reason})"
        print(
            f"[tool_logger] job={self.job_id} agent={agent_id} "
            f"tool={tool_name} attempt={attempt} {status} latency={latency_ms:.0f}ms"
        )

    def log_with_retry(
        self,
        agent_id: str,
        tool_name: str,
        call_fn,
        input_data: dict,
        accept_fn,
        max_retries: int = 2,
    ):
        """
        Calls a tool up to max_retries+1 times.
        Each attempt is logged separately.
        accept_fn(result) → (bool, reason, modified_input_or_None)
        """
        current_input = input_data

        for attempt in range(1, max_retries + 2):
            start = time.time()
            result = call_fn(**current_input)
            latency = (time.time() - start) * 1000

            accepted, reason, next_input = accept_fn(result, attempt)

            self.log(
                agent_id=agent_id,
                tool_name=tool_name,
                input_data=current_input,
                output_data=result.model_dump() if hasattr(result, "model_dump") else result,
                latency_ms=latency,
                accepted=accepted,
                attempt=attempt,
                rejection_reason=reason if not accepted else None,
                failure_mode=getattr(result, "failure_mode", "none"),
            )

            if accepted:
                return result

            # no more retries
            if attempt == max_retries + 1:
                print(f"[tool_logger] max retries reached for {tool_name}")
                return result

            # update input for next retry
            if next_input:
                current_input = next_input

        return result