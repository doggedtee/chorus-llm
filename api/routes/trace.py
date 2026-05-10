from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import Job, ExecutionEvent, ToolCallLog

router = APIRouter()


@router.get("/trace/{job_id}")
def get_trace(job_id: str, db: Session = Depends(get_db)):
    """
    Returns the full execution trace for a completed job.
    Reconstructs the exact sequence of agent decisions, tool calls, and handoffs.
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "message": f"No job with id {job_id}", "job_id": job_id},
        )

    events = (
        db.query(ExecutionEvent)
        .filter(ExecutionEvent.job_id == job_id)
        .order_by(ExecutionEvent.sequence)
        .all()
    )

    tool_calls = (
        db.query(ToolCallLog)
        .filter(ToolCallLog.job_id == job_id)
        .order_by(ToolCallLog.timestamp)
        .all()
    )

    return {
        "job_id": job_id,
        "query": job.query,
        "status": job.status,
        "final_answer": job.final_answer,
        "provenance_map": job.provenance_map,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "execution_events": [
            {
                "sequence": e.sequence,
                "agent_id": e.agent_id,
                "event_type": e.event_type,
                "input_data": e.input_data,
                "output_data": e.output_data,
                "latency_ms": e.latency_ms,
                "token_count": e.token_count,
                "policy_violation": e.policy_violation,
                "violation_detail": e.violation_detail,
                "timestamp": e.timestamp,
            }
            for e in events
        ],
        "tool_calls": [
            {
                "tool_name": t.tool_name,
                "agent_id": t.agent_id,
                "attempt": t.attempt,
                "input_data": t.input_data,
                "output_data": t.output_data,
                "latency_ms": t.latency_ms,
                "accepted": t.accepted,
                "rejection_reason": t.rejection_reason,
                "failure_mode": t.failure_mode,
                "timestamp": t.timestamp,
            }
            for t in tool_calls
        ],
    }