import asyncio
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from db.database import get_db, SessionLocal
from db.models import Job
from core.context import SharedContext
from core.streaming import event_stream, get_queue, emit, emit_done
from agents.orchestrator import run_pipeline_async

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., description="The research question to answer", examples=["What is climate change?"])


async def _run_job(job_id: str, query: str):
    """Run the full agent pipeline and stream events to the SSE queue."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "running"
        db.commit()

        await emit(job_id, "pipeline_start", message="Pipeline started", query=query)

        context = SharedContext(job_id=job_id, original_query=query)
        result = await run_pipeline_async(context)

        job.status = "completed"
        job.final_answer = result.final_answer
        job.provenance_map = [p.model_dump() for p in result.provenance_map]
        job.completed_at = datetime.utcnow()
        db.commit()

        await emit(job_id, "pipeline_done", final_answer=result.final_answer)

    except Exception as e:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            db.commit()
        await emit(job_id, "error", message=str(e), error_code="PIPELINE_FAILURE", job_id=job_id)

    finally:
        await emit_done(job_id)
        db.close()


@router.post("/query")
async def submit_query(body: QueryRequest, db: Session = Depends(get_db)):
    """
    Submit a research query.
    Returns a streaming SSE response with real-time agent activity.
    """
    query = body.query.strip()

    if not query:
        return {
            "error": "EMPTY_QUERY",
            "message": "Query cannot be empty",
            "job_id": None,
        }

    job = Job(query=query, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    job_id = job.id

    # pre-create queue, then kick off pipeline as a parallel task
    # (using BackgroundTasks would deadlock with StreamingResponse —
    #  background tasks only run AFTER the response body finishes)
    get_queue(job_id)
    asyncio.create_task(_run_job(job_id, query))

    return StreamingResponse(
        event_stream(job_id, get_queue(job_id)),
        media_type="text/event-stream",
        headers={
            "X-Job-ID": job_id,
            "Cache-Control": "no-cache",
        },
    )