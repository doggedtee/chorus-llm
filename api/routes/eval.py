from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import EvalRun, EvalResult

router = APIRouter()


@router.get("/eval/latest")
def get_latest_eval(db: Session = Depends(get_db)):
    """
    Returns the latest eval run summary broken down by category and scoring dimension.
    """
    run = db.query(EvalRun).order_by(EvalRun.triggered_at.desc()).first()

    if not run:
        return {"error": "NO_EVAL_RUN", "message": "No eval runs found yet", "job_id": None}

    results = db.query(EvalResult).filter(EvalResult.run_id == run.id).all()

    return {
        "run_id": run.id,
        "triggered_at": run.triggered_at,
        "completed_at": run.completed_at,
        "total_cases": run.total_cases,
        "passed": run.passed,
        "overall_score": run.overall_score,
        "category_scores": run.category_scores,
        "dimension_scores": run.dimension_scores,
        "results": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "query": r.query,
                "total_score": r.total_score,
                "scores": {
                    "correctness": r.score_correctness,
                    "citation": r.score_citation,
                    "contradiction": r.score_contradiction,
                    "tool_efficiency": r.score_tool_efficiency,
                    "budget_compliance": r.score_budget_compliance,
                    "critique_agreement": r.score_critique_agreement,
                },
                "justification": r.justification,
            }
            for r in results
        ],
    }