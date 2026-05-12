import os
import json
import anthropic
from pydantic import BaseModel
from typing import Optional
from core.context import SharedContext
from eval.test_cases import TestCase

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class DimensionScore(BaseModel):
    score: float            # 0.0 to 1.0
    justification: str      # written reason


class EvalScore(BaseModel):
    case_id: str
    correctness: DimensionScore
    citation: DimensionScore
    critique_quality: DimensionScore
    total: float


def _llm_score(prompt: str) -> DimensionScore:
    """Call Claude to score a dimension. Returns a neutral score on failure."""
    try:
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text.strip())
        return DimensionScore(score=float(data["score"]), justification=data["justification"])
    except Exception as e:
        return DimensionScore(score=0.5, justification=f"LLM scoring failed: {e}")


def score_correctness(case: TestCase, context: SharedContext) -> DimensionScore:
    answer = context.final_answer or ""
    if not answer:
        return DimensionScore(score=0.0, justification="No final answer was produced.")

    expected_section = ""
    if case.expected_answer:
        expected_section += f"Expected answer: {case.expected_answer}\n"
    if case.expected_keywords:
        expected_section += f"Expected keywords: {', '.join(case.expected_keywords)}\n"

    prompt = (
        f"Score how correct this answer is for the given question.\n\n"
        f"Question: {case.query}\n"
        f"{expected_section}"
        f"Actual answer: {answer}\n\n"
        f"Return JSON only: {{\"score\": <0.0 to 1.0>, \"justification\": \"<one sentence>\"}}"
    )
    return _llm_score(prompt)


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


def score_critique_quality(context: SharedContext) -> DimensionScore:
    flagged = [c for c in context.critiqued_claims if c.flagged]
    accepted = [c for c in context.critiqued_claims if not c.flagged]

    if not flagged and not accepted:
        return DimensionScore(score=0.5, justification="No claims from critique agent to evaluate.")

    final_answer = context.final_answer or ""
    if not final_answer:
        return DimensionScore(score=0.0, justification="No final answer was produced.")

    flagged_text = "\n".join(f"- {c.text}" for c in flagged) if flagged else "None"
    accepted_text = "\n".join(f"- {c.text}" for c in accepted) if accepted else "None"

    prompt = (
        f"The critique agent reviewed claims and produced two lists.\n\n"
        f"Flagged (bad — should be removed or fixed):\n{flagged_text}\n\n"
        f"Approved (good — should appear in the final answer):\n{accepted_text}\n\n"
        f"Final answer: {final_answer}\n\n"
        f"Score 0.0 to 1.0 how well the final answer handled both lists: "
        f"bad claims removed and good claims kept. 1.0 means perfect on both.\n\n"
        f"Return JSON only: {{\"score\": <0.0 to 1.0>, \"justification\": \"<one sentence>\"}}"
    )
    return _llm_score(prompt)


def score_case(case: TestCase, context: SharedContext) -> EvalScore:
    correctness     = score_correctness(case, context)
    citation        = score_citation(case, context)
    critique_quality = score_critique_quality(context)

    total = round(
        (correctness.score + citation.score + critique_quality.score) / 3,
        2,
    )

    return EvalScore(
        case_id=case.case_id,
        correctness=correctness,
        citation=citation,
        critique_quality=critique_quality,
        total=total,
    )
