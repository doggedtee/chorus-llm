from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.context import SharedContext, AgentOutput

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a compression agent. You summarize conversational text to save space.

Rules:
- NEVER compress or alter structured data (JSON, scores, citations, chunk IDs, URLs)
- NEVER remove numbers, scores, or citations
- Only compress conversational/filler text — make it shorter while keeping all facts
- Return the compressed version only, no explanation"""


def compression_agent(context: SharedContext, target_agent_id: str) -> SharedContext:
    """
    Called when an agent is near its token budget limit.
    Compresses that agent's previous output to free up space.
    Structured data (claims, chunks, scores) is never touched.
    """
    agent_output = context.get_agent_output(target_agent_id)

    if not agent_output:
        print(f"[compression] no output found for {target_agent_id}, skipping")
        return context

    original_text = agent_output.output_text
    original_length = len(original_text)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Compress this text. Keep all facts, scores, citations, and chunk IDs intact:\n\n"
            f"{original_text}"
        )),
    ])

    compressed_text = response.content.strip()
    token_count = (response.usage_metadata or {}).get("total_tokens", 0)
    context.record_tokens("compression", token_count)

    # update the agent output with compressed text
    agent_output.output_text = compressed_text
    context.set_agent_output(target_agent_id, agent_output)

    compression_ratio = len(compressed_text) / original_length if original_length > 0 else 1.0

    context.set_agent_output("compression", AgentOutput(
        agent_id="compression",
        output_text=f"Compressed {target_agent_id} output. Ratio: {compression_ratio:.2f}",
        token_count=token_count,
    ))

    print(
        f"[compression] compressed {target_agent_id} output "
        f"{original_length} → {len(compressed_text)} chars "
        f"(ratio={compression_ratio:.2f}) for job {context.job_id}"
    )
    return context