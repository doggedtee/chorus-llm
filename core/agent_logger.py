import hashlib
import json
import time
from typing import Any, Optional
from db.database import SessionLocal
from db.models import ExecutionEvent
from core.streaming import emit


def _hash(data: Any) -> str:
    """Stable SHA256 hash of any JSON-serializable data."""
    try:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()[:16]


class AgentLogger:
    """
    Wraps an agent run to record:
    - ExecutionEvent rows in SQLite (input/output hashes, latency, tokens)
    - SSE events to the user (agent_start, agent_end, policy_violation)
    """

    def __init__(self, job_id: str, agent_id: str):
        self.job_id = job_id
        self.agent_id = agent_id
        self.start_time: Optional[float] = None
        self.input_data: Any = None
        self._sequence: Optional[int] = None

    async def start(self, input_data: Any):
        self.start_time = time.time()
        self.input_data = input_data
        await emit(self.job_id, "agent_start", agent=self.agent_id, input_hash=_hash(input_data))

    async def end(
        self,
        output_data: Any,
        token_count: int = 0,
        policy_violation: bool = False,
        violation_detail: Optional[str] = None,
    ):
        latency_ms = (time.time() - (self.start_time or time.time())) * 1000

        db = SessionLocal()
        try:
            sequence = (
                db.query(ExecutionEvent)
                .filter(ExecutionEvent.job_id == self.job_id)
                .count()
            ) + 1

            record = ExecutionEvent(
                job_id=self.job_id,
                sequence=sequence,
                agent_id=self.agent_id,
                event_type="agent_end",
                input_hash=_hash(self.input_data),
                output_hash=_hash(output_data),
                input_data=self.input_data if isinstance(self.input_data, (dict, list)) else {"data": str(self.input_data)[:500]},
                output_data=output_data if isinstance(output_data, (dict, list)) else {"data": str(output_data)[:500]},
                latency_ms=latency_ms,
                token_count=token_count,
                policy_violation=policy_violation,
                violation_detail=violation_detail,
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

        await emit(
            self.job_id,
            "agent_end",
            agent=self.agent_id,
            latency_ms=round(latency_ms, 1),
            tokens=token_count,
            output_hash=_hash(output_data),
            policy_violation=policy_violation,
        )

        if policy_violation:
            await emit(
                self.job_id,
                "policy_violation",
                agent=self.agent_id,
                detail=violation_detail or "budget exceeded",
            )


async def log_routing(job_id: str, from_agent: str, to_agent: str, reason: str):
    """Log a routing decision both to DB and SSE."""
    db = SessionLocal()
    try:
        sequence = (
            db.query(ExecutionEvent).filter(ExecutionEvent.job_id == job_id).count()
        ) + 1

        record = ExecutionEvent(
            job_id=job_id,
            sequence=sequence,
            agent_id="orchestrator",
            event_type="routing",
            input_data={"from": from_agent, "to": to_agent},
            output_data={"reason": reason},
            input_hash=_hash({"from": from_agent}),
            output_hash=_hash({"to": to_agent}),
            latency_ms=0.0,
            token_count=0,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    await emit(job_id, "routing", from_agent=from_agent, to_agent=to_agent, reason=reason)


async def emit_budget(job_id: str, summary: dict):
    """Emit current budget status for SSE clients."""
    await emit(job_id, "budget_update", budget=summary)