import time
from typing import Optional
from pydantic import BaseModel
from core.context import SharedContext


class ReflectionResult(BaseModel):
    agent_outputs_reviewed: list[str]       # which agents were reviewed
    contradictions: list[str]               # list of contradiction descriptions
    summary: str                            # overall summary of findings
    has_contradictions: bool
    latency_ms: float
    failure_mode: Optional[str] = None      # none, empty, malformed


def self_reflect(context: SharedContext) -> ReflectionResult:
    """
    Re-reads all previous agent outputs in the session and identifies contradictions.

    Failure contracts:
    - empty:    no agent outputs exist yet to reflect on
    - malformed: context is None or wrong type
    """
    start = time.time()

    # malformed input
    if not context or not isinstance(context, SharedContext):
        return ReflectionResult(
            agent_outputs_reviewed=[],
            contradictions=[],
            summary="Invalid context provided.",
            has_contradictions=False,
            latency_ms=0.0,
            failure_mode="malformed",
        )

    # empty — nothing to reflect on yet
    if not context.agent_outputs:
        return ReflectionResult(
            agent_outputs_reviewed=[],
            contradictions=[],
            summary="No agent outputs available to reflect on yet.",
            has_contradictions=False,
            latency_ms=0.0,
            failure_mode="empty",
        )

    contradictions = []
    reviewed = list(context.agent_outputs.keys())

    # collect all output texts
    outputs = {
        agent_id: output.output_text
        for agent_id, output in context.agent_outputs.items()
    }

    # check critique agent flags against synthesis output
    if "critique" in outputs and "synthesis" in outputs:
        flagged_claims = [
            claim for claim in context.critiqued_claims if claim.flagged
        ]
        if flagged_claims:
            for claim in flagged_claims:
                if claim.text in outputs.get("synthesis", ""):
                    contradictions.append(
                        f"Synthesis includes flagged claim: '{claim.text[:80]}...' "
                        f"— Critique reason: {claim.flag_reason}"
                    )

    # check if decomposition sub-tasks are reflected in retrieval output
    if "decomposition" in outputs and "retrieval" in outputs:
        pending_tasks = [t for t in context.sub_tasks if t.status != "completed"]
        if pending_tasks:
            contradictions.append(
                f"{len(pending_tasks)} sub-task(s) from decomposition were not addressed: "
                + ", ".join(t.description[:40] for t in pending_tasks)
            )

    has_contradictions = len(contradictions) > 0

    if has_contradictions:
        summary = f"Found {len(contradictions)} contradiction(s) across {len(reviewed)} agent outputs."
    else:
        summary = f"No contradictions found across {len(reviewed)} agent outputs."

    latency = (time.time() - start) * 1000
    print(f"[self_reflection] reviewed={reviewed} contradictions={len(contradictions)} latency={latency:.0f}ms")

    return ReflectionResult(
        agent_outputs_reviewed=reviewed,
        contradictions=contradictions,
        summary=summary,
        has_contradictions=has_contradictions,
        latency_ms=latency,
        failure_mode="none",
    )