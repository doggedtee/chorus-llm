import asyncio
import json
from typing import AsyncGenerator
from datetime import datetime


# global registry of queues — one per active job
_queues: dict[str, asyncio.Queue] = {}


def get_queue(job_id: str) -> asyncio.Queue:
    if job_id not in _queues:
        _queues[job_id] = asyncio.Queue()
    return _queues[job_id]


def remove_queue(job_id: str):
    _queues.pop(job_id, None)


def make_event(event_type: str, job_id: str, **kwargs) -> dict:
    """Build a structured SSE event dict."""
    return {
        "type": event_type,
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs,
    }


async def emit(job_id: str, event_type: str, **kwargs):
    """Push an event into the job's queue."""
    queue = get_queue(job_id)
    event = make_event(event_type, job_id, **kwargs)
    await queue.put(event)


async def emit_done(job_id: str):
    """Signal that the pipeline is finished."""
    queue = get_queue(job_id)
    await queue.put(None)


async def event_stream(job_id: str, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Read events from queue and yield them as SSE-formatted strings."""
    try:
        while True:
            event = await queue.get()

            if event is None:
                yield "event: done\ndata: {}\n\n"
                break

            yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
    finally:
        remove_queue(job_id)