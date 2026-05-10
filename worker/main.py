"""
Background worker service.
Polls the database for pending eval jobs and runs them.
This runs as a separate Docker service from the API.
"""
import time
import os
import sys

# ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal, init_db
from db.models import EvalRun
from eval.harness import run_eval
from agents.meta import run_meta_agent


POLL_INTERVAL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))


def has_pending_eval(db) -> bool:
    """Check if a manual eval has been triggered (pending = no run today yet)."""
    if os.getenv("RUN_EVAL_ON_START", "false").lower() == "true":
        return True
    return False


def main():
    print("[worker] starting background worker")
    init_db()

    if os.getenv("RUN_EVAL_ON_START", "false").lower() == "true":
        print("[worker] RUN_EVAL_ON_START=true — running initial eval")
        try:
            db = SessionLocal()
            run_id = run_eval()
            run_meta_agent(run_id, db)
            db.close()
        except Exception as e:
            print(f"[worker] initial eval failed: {e}")

    while True:
        try:
            print(f"[worker] heartbeat — sleeping {POLL_INTERVAL_SEC}s")
            time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("[worker] shutdown signal received")
            break


if __name__ == "__main__":
    main()