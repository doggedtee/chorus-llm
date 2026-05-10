from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from db.database import get_db
from db.models import PromptRewrite

router = APIRouter()


@router.post("/approve/{rewrite_id}")
def approve_rewrite(
    rewrite_id: str,
    body: dict,
    db: Session = Depends(get_db),
):
    """
    Submit a human approval or rejection for a pending prompt rewrite.
    Body: {"decision": "approved" | "rejected"}
    """
    rewrite = db.query(PromptRewrite).filter(PromptRewrite.id == rewrite_id).first()

    if not rewrite:
        raise HTTPException(
            status_code=404,
            detail={"error": "REWRITE_NOT_FOUND", "message": f"No rewrite with id {rewrite_id}", "job_id": None},
        )

    if rewrite.status != "pending":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ALREADY_REVIEWED",
                "message": f"Rewrite already {rewrite.status}",
                "job_id": None,
            },
        )

    decision = body.get("decision", "").lower()
    if decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_DECISION", "message": "Decision must be 'approved' or 'rejected'", "job_id": None},
        )

    rewrite.status = decision
    rewrite.reviewed_at = datetime.utcnow()
    db.commit()

    return {
        "rewrite_id": rewrite_id,
        "agent_id": rewrite.agent_id,
        "decision": decision,
        "reviewed_at": rewrite.reviewed_at,
        "message": f"Rewrite {decision}. Use POST /reeval to test it on failed cases." if decision == "approved" else "Rewrite rejected.",
    }