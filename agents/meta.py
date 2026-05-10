import difflib
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session
from db.models import EvalRun, EvalResult, PromptRewrite

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

# current prompts for each agent — meta agent can propose rewrites for these
AGENT_PROMPTS = {
    "decomposition": """You are a decomposition agent. Your job is to break a research query into clear sub-tasks.

Rules:
- Break the query into 2-4 sub-tasks
- Each sub-task must have a type: research, compute, lookup, or summarize
- If a sub-task depends on results from another, list those task_ids in dependencies
- Dependent tasks must not run before their dependencies complete
- Return ONLY valid JSON, no extra text""",

    "retrieval": """You are a retrieval agent. You receive research sub-tasks and retrieved chunks of information.

Your job:
1. Reason across ALL provided chunks (minimum 2 chunks required)
2. For each part of your answer, cite exactly which chunk it came from using [chunk_id]
3. Do NOT answer from memory — only use the provided chunks
4. If chunks are insufficient, say so explicitly""",

    "critique": """You are a critique agent. You review research outputs claim by claim.

Your job:
- Extract individual claims from the text
- Assign a confidence score (0.0 to 1.0) to each claim
- Flag claims that are unsupported, exaggerated, or contradict the source chunks
- Be specific — flag the exact span of text, not the whole answer""",

    "synthesis": """You are a synthesis agent. Your job is to produce a final answer by merging all agent outputs.

Rules:
- Use the retrieval agent output as your main source
- If the critique agent flagged a claim, do NOT include it unless you can rephrase it accurately
- Every sentence in your final answer must have a source
- After your answer, include a PROVENANCE section""",
}


def _find_worst_dimension(results: list[EvalResult]) -> tuple[str, str]:
    """Find the dimension with the lowest average score and its responsible agent."""
    dimension_map = {
        "correctness":       "retrieval",
        "citation":          "retrieval",
        "contradiction":     "synthesis",
        "tool_efficiency":   "orchestrator",
        "budget_compliance": "orchestrator",
        "critique_agreement": "critique",
    }

    dim_scores = {
        "correctness":        [r.score_correctness for r in results],
        "citation":           [r.score_citation for r in results],
        "contradiction":      [r.score_contradiction for r in results],
        "tool_efficiency":    [r.score_tool_efficiency for r in results],
        "budget_compliance":  [r.score_budget_compliance for r in results],
        "critique_agreement": [r.score_critique_agreement for r in results],
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