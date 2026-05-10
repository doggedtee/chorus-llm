from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import EvalRun, EvalResult, PromptRewrite

router = APIRouter()


@router.post("/reeval")
def trigger_reeval(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Triggers a targeted re-eval on previously failed cases using the latest approved prompts.
    """
    # get latest eval run
    latest_run = db.query(EvalRun).order_by(EvalRun.triggered_at.desc()).first()

    if not latest_run:
        return {"error": "NO_EVAL_RUN", "message": "No eval runs found. Run /eval first.", "job_id": None}

    # get failed cases from latest run (score below 0.6)
    failed = (
        db.query(EvalResult)
        .filter(EvalResult.run_id == latest_run.id, EvalResult.total_score < 0.6)
        .all()
    )

    if not failed:
        return {"message": "No failed cases found in latest eval run.", "failed_count": 0}

    # get latest approved rewrite
    approved_rewrite = (
        db.query(PromptRewrite)
        .filter(PromptRewrite.status == "approved")
        .order_by(PromptRewrite.reviewed_at.desc())
        .first()
    )

    if not approved_rewrite:
        return {"error": "NO_APPROVED_REWRITE", "message": "No approved rewrites found. Approve a rewrite first.", "job_id": None}

    failed_case_ids = [r.case_id for r in failed]

    # run re-eval in background
    from eval.harness import run_reeval
    background_tasks.add_task(run_reeval, failed_case_ids, approved_rewrite.id)

    return {
        "message": f"Re-eval started on {len(failed_case_ids)} failed cases using rewrite {approved_rewrite.id}",
        "failed_cases": failed_case_ids,
        "rewrite_id": approved_rewrite.id,
        "agent_id": approved_rewrite.agent_id,
    }