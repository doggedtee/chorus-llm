import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from core.context import SharedContext, SubTask

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

SYSTEM_PROMPT = """You are a decomposition agent. Your job is to break a research query into clear sub-tasks.

Rules:
- Break the query into 2-4 sub-tasks
- Each sub-task must have a type: research or summarize
- Use research for tasks that need information from the web
- Use summarize for tasks that reason over or combine results from other tasks
- If a sub-task depends on results from another, list those task_ids in dependencies
- Dependent tasks must not run before their dependencies complete
- Return ONLY valid JSON, no extra text

Output format:
{
  "sub_tasks": [
    {
      "task_id": "task_1",
      "description": "...",
      "task_type": "research",
      "dependencies": []
    },
    {
      "task_id": "task_2",
      "description": "...",
      "task_type": "summarize",
      "dependencies": ["task_1"]
    }
  ]
}"""


def decomposition_agent(context: SharedContext) -> SharedContext:
    """
    Reads the original query from context.
    Breaks it into typed sub-tasks with a dependency graph.
    Writes sub-tasks back to context.
    """
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Break this research query into sub-tasks:\n\n{context.original_query}"),
    ])

    raw = response.content.strip()

    # record token usage
    token_count = (response.usage_metadata or {}).get("total_tokens", 0)
    context.record_tokens("decomposition", token_count)

    try:
        data = json.loads(raw)
        sub_tasks = [SubTask(**task) for task in data["sub_tasks"]]
    except Exception:
        # fallback: treat whole query as one task if JSON parsing fails
        sub_tasks = [
            SubTask(
                task_id="task_1",
                description=context.original_query,
                task_type="research",
                dependencies=[],
            )
        ]

    context.sub_tasks = sub_tasks

    print(f"[decomposition] {len(sub_tasks)} sub-tasks created for job {context.job_id}")
    return context