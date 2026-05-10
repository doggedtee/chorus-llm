from pydantic import BaseModel
from typing import Optional
from core.context import SharedContext
from eval.test_cases import TestCase


class DimensionScore(BaseModel):
    score: float            # 0.0 to 1.0
    justification: str      # written reason


class EvalScore(BaseModel):
    case_id: str
    correctness: DimensionScore
    citation: DimensionScore
    contradiction: DimensionScore
    tool_efficiency: DimensionScore
    budget_compliance: DimensionScore
    critique_agreement: DimensionScore
    total: float


def score_correctness(case: TestCase, context: SharedContext) -> DimensionScore:
    """Did the answer contain the expected keywords or match the expected answer?"""
    answer = (context.final_answer or "").lower()

    if not answer:
        return DimensionScore(score=0.0, justification="No final answer was produced.")

    if case.expected_keywords:
        matched = [kw for kw in case.expected_keywords if kw.lower() in answer]
        score = len(matched) / len(case.expected_keywords)
        justification = (
            f"Matched {len(matched)}/{len(case.expected_keywords)} expected keywords: "
            f"{matched}."
        )
    else:
        # ambiguous cases — check if answer is non-empty and on-topic
        score = 0.7 if len(answer) > 100 else 0.3
        justification = "No expected keywords defined (ambiguous case). Scored on answer length and presence."

    return DimensionScore(score=round(score, 2), justification=justification)


def score_citation(case: TestCase, context: SharedContext) -> DimensionScore:
    """Did the answer cite chunks properly? Is the provenance map populated?"""
    if not context.provenance_map:
        return DimensionScore(score=0.0, justification="No provenance map was produced.")

    if not context.retrieved_chunks:
        return DimensionScore(score=0.0, justification="No chunks were retrieved.")

    chunk_ids = {chunk.chunk_id for chunk in context.retrieved_chunks}
    cited_ids = {entry.source_chunk_id for entry in context.provenance_map if entry.source_chunk_id}

    if not chunk_ids:
        return DimensionScore(score=0.0, justification="No chunk IDs available to verify citations.")

    coverage = len(cited_ids & chunk_ids) / len(chunk_ids)
    score = round(coverage, 2)
    justification = (
        f"{len(cited_ids)} chunk(s) cited out of {len(chunk_ids)} retrieved. "
        f"Citation coverage: {score:.0%}."
    )

    return DimensionScore(score=score, justification=justification)


def score_contradiction(case: TestCase, context: SharedContext) -> DimensionScore:
    """Were flagged contradictions resolved in the final answer?"""
    flagged = [c for c in context.critiqued_claims if c.flagged]

    if not flagged:
        return DimensionScore(score=1.0, justification="No contradictions flagged by critique agent.")

    final_answer = (context.final_answer or "").lower()
    resolved = [
        c for c in flagged
        if c.text.lower()[:40] not in final_answer
    ]

    score = round(len(resolved) / len(flagged), 2)
    justification = (
        f"{len(flagged)} claim(s) flagged by critique. "
        f"{len(resolved)} resolved (removed or rephrased) in final answer. "
        f"Resolution rate: {score:.0%}."
    )

    return DimensionScore(score=score, justification=justification)


def score_tool_efficiency(case: TestCase, context: SharedContext) -> DimensionScore:
    """Did agents avoid unnecessary tool calls? Penalize excessive retries."""
    tool_results = context.tool_results
    routing_log = context.routing_log

    # count total tool calls from routing log
    tool_events = [e for e in routing_log if "tool" in e.get("from", "").lower()]
    total_calls = len(tool_events)

    # ideal: 2-4 tool calls for a normal query
    if total_calls == 0:
        return DimensionScore(score=0.5, justification="No tool calls recorded. Cannot assess efficiency.")
    elif total_calls <= 4:
        score = 1.0
        justification = f"{total_calls} tool call(s) — within efficient range (≤4)."
    elif total_calls <= 7:
        score = 0.7
        justification = f"{total_calls} tool calls — slightly above ideal range."
    else:
        score = 0.3
        justification = f"{total_calls} tool calls — excessive, suggests unnecessary retries or redundant calls."

    return DimensionScore(score=score, justification=justification)


def score_budget_compliance(case: TestCase, context: SharedContext) -> DimensionScore:
    """Did all agents stay within their token budgets?"""
    violations = context.budget_violations

    if not violations:
        return DimensionScore(score=1.0, justification="No budget violations recorded. All agents complied.")

    justification = (
        f"{len(violations)} budget violation(s): {violations}. "
        f"These agents exceeded their token budgets."
    )
    score = max(0.0, 1.0 - (len(violations) * 0.25))

    return DimensionScore(score=round(score, 2), justification=justification)


def score_critique_agreement(case: TestCase, context: SharedContext) -> DimensionScore:
    """Does the final answer agree with the critique agent's accepted claims?"""
    accepted = [c for c in context.critiqued_claims if not c.flagged]

    if not accepted:
        return DimensionScore(score=0.5, justification="No accepted claims from critique agent to compare.")

    final_answer = (context.final_answer or "").lower()
    agreed = [c for c in accepted if any(word in final_answer for word in c.text.lower().split()[:5])]

    score = round(len(agreed) / len(accepted), 2)
    justification = (
        f"{len(agreed)}/{len(accepted)} accepted claims reflected in final answer. "
        f"Agreement rate: {score:.0%}."
    )

    return DimensionScore(score=score, justification=justification)


def score_case(case: TestCase, context: SharedContext) -> EvalScore:
    """Run all 6 scoring dimensions for one test case."""
    correctness      = score_correctness(case, context)
    citation         = score_citation(case, context)
    contradiction    = score_contradiction(case, context)
    tool_efficiency  = score_tool_efficiency(case, context)
    budget_compliance = score_budget_compliance(case, context)
    critique_agreement = score_critique_agreement(case, context)

    # equal weights for all 6 dimensions
    total = round(
        (correctness.score + citation.score + contradiction.score +
         tool_efficiency.score + budget_compliance.score + critique_agreement.score) / 6,
        2,
    )

    return EvalScore(
        case_id=case.case_id,
        correctness=correctness,
        citation=citation,
        contradiction=contradiction,
        tool_efficiency=tool_efficiency,
        budget_compliance=budget_compliance,
        critique_agreement=critique_agreement,
        total=total,
    )