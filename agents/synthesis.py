import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.context import SharedContext, AgentOutput, ProvenanceEntry
from core.streaming import emit

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a synthesis agent. Your job is to produce a final answer by merging all agent outputs.

Rules:
- Use the retrieval agent output as your main source
- If the critique agent flagged a claim, do NOT include it unless you can rephrase it accurately
- Every sentence in your final answer must have a source
- After your answer, include a PROVENANCE section mapping each sentence to its source

Output format:
ANSWER:
<your final answer here, sentence by sentence>

PROVENANCE:
[
  {"sentence": "first sentence of answer", "source_agent": "retrieval", "source_chunk_id": "chunk_id_here"}
]"""


async def synthesis_agent(context: SharedContext) -> SharedContext:
    """
    Streams the final answer token-by-token via SSE.
    Reads all agent outputs and critiqued claims from context.
    Resolves contradictions flagged by critique agent.
    """
    retrieval_output = context.get_agent_output("retrieval")

    if not retrieval_output:
        context.final_answer = "Insufficient information to generate an answer."
        return context

    flagged = [c for c in context.critiqued_claims if c.flagged]
    flagged_text = "\n".join(
        f"- \"{c.text}\" → {c.flag_reason}"
        for c in flagged
    ) if flagged else "None"

    accepted = [c for c in context.critiqued_claims if not c.flagged]
    accepted_text = "\n".join(f"- \"{c.text}\"" for c in accepted) if accepted else "All claims accepted"

    chunks_text = "\n".join(
        f"[{chunk.chunk_id}]: {chunk.content}"
        for chunk in context.retrieved_chunks
    )

    user_message = (
        f"Original query: {context.original_query}\n\n"
        f"Retrieval agent output:\n{retrieval_output.output_text}\n\n"
        f"Accepted claims:\n{accepted_text}\n\n"
        f"Flagged claims (do NOT include these as-is):\n{flagged_text}\n\n"
        f"Source chunks for reference:\n{chunks_text}\n\n"
        f"Write the final answer with provenance map."
    )

    # stream tokens to the client in real time
    full_text = ""
    token_count = 0

    async for chunk in llm.astream([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]):
        if chunk.content:
            full_text += chunk.content
            await emit(
                context.job_id,
                "token",
                agent="synthesis",
                text=chunk.content,
            )

        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            token_count = chunk.usage_metadata.get("total_tokens", token_count)

    context.record_tokens("synthesis", token_count)

    # split ANSWER and PROVENANCE
    final_answer = full_text.strip()
    provenance_map = []

    try:
        if "PROVENANCE:" in full_text:
            parts = full_text.split("PROVENANCE:")
            answer_part = parts[0].replace("ANSWER:", "").strip()
            provenance_part = parts[1].strip()

            final_answer = answer_part
            entries = json.loads(provenance_part)
            provenance_map = [ProvenanceEntry(**e) for e in entries]
    except Exception:
        pass

    context.final_answer = final_answer
    context.provenance_map = provenance_map

    context.set_agent_output("synthesis", AgentOutput(
        agent_id="synthesis",
        output_text=final_answer,
        token_count=token_count,
    ))

    print(f"[synthesis] final answer streamed, {len(provenance_map)} provenance entries for job {context.job_id}")
    return context