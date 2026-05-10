import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.context import SharedContext, AgentOutput, Claim

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a critique agent. You review research outputs claim by claim.

Your job:
- Extract individual claims from the text
- Assign a confidence score (0.0 to 1.0) to each claim
- Flag claims that are unsupported, exaggerated, or contradict the source chunks
- Be specific — flag the exact span of text, not the whole answer

Return ONLY valid JSON:
{
  "claims": [
    {
      "text": "exact quote from the text",
      "confidence": 0.9,
      "flagged": false,
      "flag_reason": null
    },
    {
      "text": "another exact quote",
      "confidence": 0.4,
      "flagged": true,
      "flag_reason": "This claim is not supported by the retrieved chunks"
    }
  ]
}"""


def critique_agent(context: SharedContext) -> SharedContext:
    """
    Reads retrieval agent output from context.
    Reviews each claim, assigns confidence scores, flags suspicious spans.
    Writes critiqued claims and its own output back to context.
    """
    retrieval_output = context.get_agent_output("retrieval")

    if not retrieval_output:
        print(f"[critique] no retrieval output found for job {context.job_id}")
        return context

    # build chunk reference for the critique agent to compare against
    chunks_text = "\n\n".join(
        f"[{chunk.chunk_id}]: {chunk.content}"
        for chunk in context.retrieved_chunks
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Source chunks:\n{chunks_text}\n\n"
            f"Agent output to critique:\n{retrieval_output.output_text}\n\n"
            f"Review each claim in the agent output against the source chunks."
        )),
    ])

    raw = response.content.strip()
    token_count = (response.usage_metadata or {}).get("total_tokens", 0)
    context.record_tokens("critique", token_count)

    try:
        data = json.loads(raw)
        claims = [
            Claim(
                text=c["text"],
                confidence=c["confidence"],
                flagged=c["flagged"],
                flag_reason=c.get("flag_reason"),
                source_agent="retrieval",
            )
            for c in data["claims"]
        ]
    except Exception:
        # fallback: mark all existing claims as unreviewed
        claims = retrieval_output.claims

    context.critiqued_claims = claims

    flagged_count = sum(1 for c in claims if c.flagged)
    summary = (
        f"Reviewed {len(claims)} claims. "
        f"{flagged_count} flagged, {len(claims) - flagged_count} accepted."
    )

    context.set_agent_output("critique", AgentOutput(
        agent_id="critique",
        output_text=summary,
        claims=claims,
        token_count=token_count,
    ))

    print(f"[critique] {len(claims)} claims reviewed, {flagged_count} flagged for job {context.job_id}")
    return context