import asyncio
from langgraph.graph import StateGraph, END
from core.context import SharedContext
from core.budget_manager import BudgetManager
from core.agent_logger import AgentLogger, log_routing, emit_budget
from agents.decomposition import decomposition_agent
from agents.retrieval import retrieval_agent
from agents.critique import critique_agent
from agents.synthesis import synthesis_agent
from agents.compression import compression_agent


# ── async wrappers — handle logging, budget checks BEFORE running ────────────

async def run_decomposition(context: SharedContext) -> SharedContext:
    await log_routing(context.job_id, "orchestrator", "decomposition", "query needs to be broken into sub-tasks")
    context.add_routing_decision("orchestrator", "decomposition", "breaking query into sub-tasks")

    # check budget BEFORE running, compress if near limit
    budget = BudgetManager(context)
    if budget.needs_compression("decomposition"):
        context = compression_agent(context, "decomposition")

    logger = AgentLogger(context.job_id, "decomposition")
    await logger.start(input_data={"query": context.original_query})

    # run sync agent in a worker thread so we don't block the event loop
    context = await asyncio.to_thread(decomposition_agent, context)

    tokens = context.token_usage.get("decomposition", 0)
    violated = "decomposition" in context.budget_violations

    await logger.end(
        output_data={"sub_tasks": [t.model_dump() for t in context.sub_tasks]},
        token_count=tokens,
        policy_violation=violated,
        violation_detail=f"{tokens}/{budget.get_budget('decomposition')} tokens" if violated else None,
    )
    await emit_budget(context.job_id, budget.summary())
    return context


async def run_retrieval(context: SharedContext) -> SharedContext:
    await log_routing(context.job_id, "orchestrator", "retrieval", "sub-tasks ready, retrieving information")
    context.add_routing_decision("orchestrator", "retrieval", "retrieving information for sub-tasks")

    budget = BudgetManager(context)
    if budget.needs_compression("retrieval"):
        context = compression_agent(context, "retrieval")

    logger = AgentLogger(context.job_id, "retrieval")
    await logger.start(input_data={"sub_tasks_count": len(context.sub_tasks)})

    context = await asyncio.to_thread(retrieval_agent, context)

    tokens = context.token_usage.get("retrieval", 0)
    violated = "retrieval" in context.budget_violations

    await logger.end(
        output_data={
            "chunks_count": len(context.retrieved_chunks),
            "answer_preview": (context.get_agent_output("retrieval").output_text or "")[:200] if context.get_agent_output("retrieval") else "",
        },
        token_count=tokens,
        policy_violation=violated,
        violation_detail=f"{tokens}/{budget.get_budget('retrieval')} tokens" if violated else None,
    )
    await emit_budget(context.job_id, budget.summary())
    return context


async def run_critique(context: SharedContext) -> SharedContext:
    await log_routing(context.job_id, "orchestrator", "critique", "retrieval done, reviewing claims")
    context.add_routing_decision("orchestrator", "critique", "reviewing retrieval claims")

    budget = BudgetManager(context)
    if budget.needs_compression("critique"):
        context = compression_agent(context, "critique")

    logger = AgentLogger(context.job_id, "critique")
    await logger.start(input_data={"claims_to_review": len(context.retrieved_chunks)})

    context = await asyncio.to_thread(critique_agent, context)

    tokens = context.token_usage.get("critique", 0)
    violated = "critique" in context.budget_violations
    flagged_count = sum(1 for c in context.critiqued_claims if c.flagged)

    await logger.end(
        output_data={
            "claims_reviewed": len(context.critiqued_claims),
            "flagged_count": flagged_count,
        },
        token_count=tokens,
        policy_violation=violated,
        violation_detail=f"{tokens}/{budget.get_budget('critique')} tokens" if violated else None,
    )
    await emit_budget(context.job_id, budget.summary())
    return context


async def run_synthesis(context: SharedContext) -> SharedContext:
    await log_routing(context.job_id, "orchestrator", "synthesis", "critique done, synthesizing final answer")
    context.add_routing_decision("orchestrator", "synthesis", "synthesizing final answer")

    budget = BudgetManager(context)
    if budget.needs_compression("synthesis"):
        context = compression_agent(context, "synthesis")

    logger = AgentLogger(context.job_id, "synthesis")
    await logger.start(input_data={"flagged_claims": sum(1 for c in context.critiqued_claims if c.flagged)})

    # synthesis agent supports token streaming
    context = await synthesis_agent(context)

    tokens = context.token_usage.get("synthesis", 0)
    violated = "synthesis" in context.budget_violations

    await logger.end(
        output_data={
            "final_answer_preview": (context.final_answer or "")[:200],
            "provenance_entries": len(context.provenance_map),
        },
        token_count=tokens,
        policy_violation=violated,
        violation_detail=f"{tokens}/{budget.get_budget('synthesis')} tokens" if violated else None,
    )
    await emit_budget(context.job_id, budget.summary())
    return context


# ── routing logic ────────────────────────────────────────────────────────────

def route(context: SharedContext) -> str:
    if not context.sub_tasks:
        return "decomposition"
    if not context.get_agent_output("retrieval"):
        return "retrieval"
    if not context.get_agent_output("critique"):
        return "critique"
    if not context.final_answer:
        return "synthesis"
    return END


# ── build the graph ──────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(SharedContext)
    graph.add_node("decomposition", run_decomposition)
    graph.add_node("retrieval", run_retrieval)
    graph.add_node("critique", run_critique)
    graph.add_node("synthesis", run_synthesis)

    graph.add_conditional_edges("decomposition", route)
    graph.add_conditional_edges("retrieval", route)
    graph.add_conditional_edges("critique", route)
    graph.add_conditional_edges("synthesis", route)

    graph.set_entry_point("decomposition")
    return graph.compile()


pipeline = build_graph()


async def run_pipeline_async(context: SharedContext) -> SharedContext:
    """Async pipeline run — used by the API SSE endpoint."""
    print(f"[orchestrator] starting async pipeline for job {context.job_id}")
    result = await pipeline.ainvoke(context)
    print(f"[orchestrator] pipeline complete for job {context.job_id}")
    if isinstance(result, dict):
        result = SharedContext(**result)
    return result


def run_pipeline(context: SharedContext) -> SharedContext:
    """Sync pipeline run — used by the eval harness."""
    return asyncio.run(run_pipeline_async(context))