import difflib
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session
from db.models import EvalRun, EvalResult, PromptRewrite
from agents.decomposition import SYSTEM_PROMPT as DECOMP_PROMPT
from agents.retrieval import SYSTEM_PROMPT as RETRIEVAL_PROMPT
from agents.critique import SYSTEM_PROMPT as CRITIQUE_PROMPT
from agents.synthesis import SYSTEM_PROMPT as SYNTH_PROMPT

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a meta agent. You analyze failed eval cases and propose improvements to agent prompts.

Your job:
1. Read the failed test cases and their scores per dimension
2. Identify which dimension scored lowest overall
3. Find the agent responsible for that dimension
4. Propose a rewritten system prompt for that agent that would fix the failures
5. Be specific — explain exactly what was wrong and how the rewrite fixes it

Return your response in this format:
AGENT: <agent_id>
DIMENSION: <dimension_name>
JUSTIFICATION: <why this prompt needs changing>
NEW_PROMPT:
<the full rewritten system prompt>"""

AGENT_PROMPTS = {
    "decomposition": DECOMP_PROMPT,
    "retrieval":     RETRIEVAL_PROMPT,
    "critique":      CRITIQUE_PROMPT,
    "synthesis":     SYNTH_PROMPT,
}


def _find_worst_dimension(results: list[EvalResult]) -> tuple[str, str]:
    """Find the dimension with the lowest average score and its responsible agent."""
    dimension_map = {
        "correctness":    "retrieval",
        "citation":       "retrieval",
        "critique_quality": "critique",
    }

    dim_scores = {
        "correctness":    [r.score_correctness for r in results],
        "citation":       [r.score_citation for r in results],
        "critique_quality": [r.score_critique_quality for r in results],
    }

    averages = {
        dim: sum(scores) / len(scores)
        for dim, scores in dim_scores.items()
        if scores
    }

    worst_dim = min(averages, key=averages.get)
    responsible_agent = dimension_map.get(worst_dim, "retrieval")

    return worst_dim, responsible_agent


def _make_diff(original: str, proposed: str) -> str:
    """Generate a readable diff between two prompts."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile="original_prompt",
        tofile="proposed_prompt",
    )
    return "".join(diff)


def run_meta_agent(eval_run_id: str, db: Session):
    """
    Reads failed cases from the latest eval run.
    Identifies the worst-performing prompt.
    Proposes a rewrite and stores it — does NOT apply automatically.
    """
    results = (
        db.query(EvalResult)
        .filter(EvalResult.run_id == eval_run_id, EvalResult.total_score < 0.6)
        .all()
    )

    if not results:
        print("[meta] no failed cases found, skipping rewrite proposal")
        return

    worst_dim, agent_id = _find_worst_dimension(results)
    original_prompt = AGENT_PROMPTS.get(agent_id, "")

    # summarize failures for the meta agent
    failures_text = "\n".join(
        f"- Case {r.case_id} ({r.category}): total={r.total_score} | "
        f"{worst_dim}={getattr(r, f'score_{worst_dim}', 'N/A')} | "
        f"justification: {(r.justification or {}).get(worst_dim, 'none')}"
        for r in results
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Worst performing dimension: {worst_dim}\n"
            f"Responsible agent: {agent_id}\n\n"
            f"Current prompt for {agent_id}:\n{original_prompt}\n\n"
            f"Failed cases:\n{failures_text}\n\n"
            f"Propose a rewritten prompt that would fix these failures."
        )),
    ])

    raw = response.content.strip()

    # parse the structured response
    proposed_prompt = original_prompt
    justification = "No justification provided."

    if "NEW_PROMPT:" in raw:
        parts = raw.split("NEW_PROMPT:")
        header = parts[0]
        proposed_prompt = parts[1].strip()

        if "JUSTIFICATION:" in header:
            justification = header.split("JUSTIFICATION:")[-1].split("DIMENSION:")[0].strip()

    diff = _make_diff(original_prompt, proposed_prompt)

    rewrite = PromptRewrite(
        eval_run_id=eval_run_id,
        agent_id=agent_id,
        dimension=worst_dim,
        original_prompt=original_prompt,
        proposed_prompt=proposed_prompt,
        diff=diff,
        justification=justification,
        status="pending",
    )
    db.add(rewrite)
    db.commit()

    print(
        f"[meta] proposed rewrite for {agent_id} on dimension '{worst_dim}'. "
        f"Rewrite ID: {rewrite.id} — awaiting human approval."
    )
    return rewrite.id