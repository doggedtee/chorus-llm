from datetime import datetime
from db.database import SessionLocal
from db.models import EvalRun, EvalResult, Job, RewriteDelta, ToolCallLog, ExecutionEvent
from core.context import SharedContext
from agents.orchestrator import run_pipeline
from agents.decomposition import SYSTEM_PROMPT as DECOMP_PROMPT
from agents.retrieval import SYSTEM_PROMPT as RETRIEVAL_PROMPT
from agents.critique import SYSTEM_PROMPT as CRITIQUE_PROMPT
from agents.synthesis import SYSTEM_PROMPT as SYNTH_PROMPT
from eval.test_cases import ALL_CASES, TestCase
from eval.scoring import score_case


def run_single_case(case: TestCase, db) -> tuple[SharedContext, dict]:
    """Run one test case through the full pipeline and score it."""
    context = SharedContext(
        job_id=f"eval_{case.case_id}_{datetime.utcnow().strftime('%H%M%S')}",
        original_query=case.query,
    )

    # create a job row for traceability
    job = Job(id=context.job_id, query=case.query, status="running")
    db.add(job)
    db.commit()

    try:
        result = run_pipeline(context)

        job.status = "completed"
        job.final_answer = result.final_answer
        job.provenance_map = [p.model_dump() for p in result.provenance_map]
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        result = context
        job.status = "failed"
        db.commit()
        print(f"[harness] case {case.case_id} failed: {e}")

    score = score_case(case, result)
    return result, score


def run_eval() -> str:
    """Run all 15 test cases and store results. Returns the eval run ID."""
    db = SessionLocal()

    run = EvalRun(
        triggered_at=datetime.utcnow(),
        total_cases=len(ALL_CASES),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    print(f"[harness] starting eval run {run.id} with {len(ALL_CASES)} cases")

    all_scores = []
    category_totals = {"baseline": [], "ambiguous": [], "adversarial": []}
    dimension_totals = {
        "correctness": [], "citation": [], "contradiction": [],
        "tool_efficiency": [], "budget_compliance": [], "critique_agreement": [],
    }

    for case in ALL_CASES:
        print(f"[harness] running case {case.case_id} ({case.category})")
        context, score = run_single_case(case, db)

        result_row = EvalResult(
            run_id=run.id,
            job_id=context.job_id,
            case_id=case.case_id,
            category=case.category,
            query=case.query,
            expected_answer=case.expected_answer,
            actual_answer=context.final_answer,
            score_correctness=score.correctness.score,
            score_citation=score.citation.score,
            score_contradiction=score.contradiction.score,
            score_tool_efficiency=score.tool_efficiency.score,
            score_budget_compliance=score.budget_compliance.score,
            score_critique_agreement=score.critique_agreement.score,
            total_score=score.total,
            justification={
                "correctness": score.correctness.justification,
                "citation": score.citation.justification,
                "contradiction": score.contradiction.justification,
                "tool_efficiency": score.tool_efficiency.justification,
                "budget_compliance": score.budget_compliance.justification,
                "critique_agreement": score.critique_agreement.justification,
            },
            exact_prompts={
                "decomposition": DECOMP_PROMPT,
                "retrieval": RETRIEVAL_PROMPT,
                "critique": CRITIQUE_PROMPT,
                "synthesis": SYNTH_PROMPT,
            },
            exact_tool_calls=[
                {
                    "tool": t.tool_name,
                    "agent": t.agent_id,
                    "attempt": t.attempt,
                    "input": t.input_data,
                    "output": t.output_data,
                    "accepted": t.accepted,
                    "failure_mode": t.failure_mode,
                }
                for t in db.query(ToolCallLog).filter(ToolCallLog.job_id == context.job_id).all()
            ],
        )
        db.add(result_row)
        db.commit()

        all_scores.append(score.total)
        category_totals[case.category].append(score.total)
        dimension_totals["correctness"].append(score.correctness.score)
        dimension_totals["citation"].append(score.citation.score)
        dimension_totals["contradiction"].append(score.contradiction.score)
        dimension_totals["tool_efficiency"].append(score.tool_efficiency.score)
        dimension_totals["budget_compliance"].append(score.budget_compliance.score)
        dimension_totals["critique_agreement"].append(score.critique_agreement.score)

    # compute averages
    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0

    overall = avg(all_scores)
    passed = sum(1 for s in all_scores if s >= 0.6)

    run.completed_at = datetime.utcnow()
    run.overall_score = overall
    run.passed = passed
    run.category_scores = {k: avg(v) for k, v in category_totals.items()}
    run.dimension_scores = {k: avg(v) for k, v in dimension_totals.items()}
    db.commit()

    print(f"[harness] eval complete. overall={overall} passed={passed}/{len(ALL_CASES)}")
    db.close()
    return run.id


def run_reeval(failed_case_ids: list[str], rewrite_id: str):
    """Re-run only the failed cases after a prompt rewrite is approved."""
    db = SessionLocal()

    failed_cases = [c for c in ALL_CASES if c.case_id in failed_case_ids]

    print(f"[harness] re-eval on {len(failed_cases)} failed cases with rewrite {rewrite_id}")

    for case in failed_cases:
        before_result = (
            db.query(EvalResult)
            .filter(EvalResult.case_id == case.case_id)
            .order_by(EvalResult.id.desc())
            .first()
        )
        before_score = before_result.total_score if before_result else 0.0

        context, score = run_single_case(case, db)

        delta = RewriteDelta(
            rewrite_id=rewrite_id,
            case_id=case.case_id,
            before_score=before_score,
            after_score=score.total,
            delta=round(score.total - before_score, 2),
        )
        db.add(delta)
        db.commit()

        print(f"[harness] {case.case_id}: {before_score} → {score.total} (delta={delta.delta})")

    db.close()