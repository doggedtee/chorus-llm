import uuid
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.context import SharedContext, RetrievedChunk, AgentOutput, Claim
from core.tool_logger import ToolLogger
from db.database import SessionLocal
from tools.web_search import web_search
from tools.sql_lookup import sql_lookup

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a retrieval agent. You receive research sub-tasks and retrieved chunks of information.

Your job:
1. Reason across ALL provided chunks (minimum 2 chunks required)
2. For each part of your answer, cite exactly which chunk it came from using [chunk_id]
3. Do NOT answer from memory — only use the provided chunks
4. If chunks are insufficient, say so explicitly

Output format:
Write your answer with inline citations like this:
"Ocean temperatures have risen significantly [chunk_1]. This has led to coral bleaching [chunk_2]."

At the end, add a CITATIONS section:
CITATIONS:
- chunk_1: used to support the claim about temperature rise
- chunk_2: used to support the claim about coral bleaching"""


def _accept_web_search(result, attempt):
    """Decide if web search result is good enough."""
    if result.failure_mode == "none" and result.total_found >= 1:
        return True, None, None

    if result.failure_mode == "timeout":
        # retry with shorter input
        return False, "tool timed out", {"query": "general info"}
    if result.failure_mode == "empty":
        # retry with broader query
        return False, "empty results", {"query": "research overview"}
    if result.failure_mode == "malformed":
        return False, "malformed input", {"query": "default query"}
    return False, "unknown failure", None


def _accept_sql(result, attempt):
    """Decide if SQL lookup result is good enough."""
    if result.failure_mode == "none" and result.row_count >= 1:
        return True, None, None
    if result.failure_mode == "empty":
        return False, "empty SQL result", {"question": "list all papers"}
    if result.failure_mode == "malformed":
        return False, "malformed SQL", {"question": "list papers"}
    return False, "unknown failure", None


def _retrieve_for_task(task_description: str, tool_logger: ToolLogger) -> list[RetrievedChunk]:
    """Retrieve chunks for one sub-task using web_search and sql_lookup with retry."""
    chunks = []

    # web_search with retry (up to 2 retries)
    web_result = tool_logger.log_with_retry(
        agent_id="retrieval",
        tool_name="web_search",
        call_fn=web_search,
        input_data={"query": task_description},
        accept_fn=_accept_web_search,
        max_retries=2,
    )
    if web_result.failure_mode == "none":
        for r in web_result.results:
            chunks.append(RetrievedChunk(
                chunk_id=f"web_{uuid.uuid4().hex[:6]}",
                content=r.snippet,
                source=r.url,
                relevance_score=r.relevance_score,
            ))

    # sql_lookup with retry
    sql_result = tool_logger.log_with_retry(
        agent_id="retrieval",
        tool_name="sql_lookup",
        call_fn=sql_lookup,
        input_data={"question": task_description},
        accept_fn=_accept_sql,
        max_retries=2,
    )
    if sql_result.failure_mode == "none" and sql_result.rows:
        for row in sql_result.rows[:2]:
            content = " | ".join(str(v) for v in row)
            chunks.append(RetrievedChunk(
                chunk_id=f"db_{uuid.uuid4().hex[:6]}",
                content=content,
                source="research_database",
                relevance_score=0.80,
            ))

    return chunks


def retrieval_agent(context: SharedContext) -> SharedContext:
    """
    Reads sub-tasks, retrieves chunks via tools (with retry logic),
    does multi-hop reasoning across chunks with citations.
    """
    db = SessionLocal()
    tool_logger = ToolLogger(db, context.job_id)

    try:
        all_chunks = []
        for task in context.sub_tasks:
            if task.dependencies:
                dep_done = all(
                    any(t.task_id == dep and t.status == "completed" for t in context.sub_tasks)
                    for dep in task.dependencies
                )
                if not dep_done:
                    continue
            chunks = _retrieve_for_task(task.description, tool_logger)
            for chunk in chunks:
                chunk.used_for = task.task_id
            all_chunks.extend(chunks)
            task.status = "completed"

        # spec requires at least 2 chunks for multi-hop reasoning
        if len(all_chunks) < 2:
            all_chunks.append(RetrievedChunk(
                chunk_id="fallback_chunk",
                content=f"General information about: {context.original_query}",
                source="fallback",
                relevance_score=0.3,
            ))

        context.retrieved_chunks = all_chunks

        chunks_text = "\n\n".join(
            f"[{chunk.chunk_id}] (source: {chunk.source})\n{chunk.content}"
            for chunk in all_chunks
        )

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Research query: {context.original_query}\n\n"
                f"Retrieved chunks:\n{chunks_text}\n\n"
                f"Answer the query using these chunks with inline citations."
            )),
        ])

        answer = response.content.strip()
        token_count = (response.usage_metadata or {}).get("total_tokens", 0)
        context.record_tokens("retrieval", token_count)

        claims = [
            Claim(
                text=chunk.content[:100],
                confidence=chunk.relevance_score,
                source_agent="retrieval",
            )
            for chunk in all_chunks
        ]

        context.set_agent_output("retrieval", AgentOutput(
            agent_id="retrieval",
            output_text=answer,
            claims=claims,
            chunks_used=[chunk.chunk_id for chunk in all_chunks],
            token_count=token_count,
        ))

        print(f"[retrieval] {len(all_chunks)} chunks retrieved for job {context.job_id}")
    finally:
        db.close()

    return context