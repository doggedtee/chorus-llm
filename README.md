# Chorus-LLM — Multi-Agent LLM Orchestration & Evaluation System

A containerized, production-grade multi-agent research assistant with dynamic
routing, self-improving prompt loop, adversarial evaluation, and real-time
SSE streaming. Built with FastAPI + LangGraph + Claude.

---

## Quick Start (under 5 minutes)

### Prerequisites
- Docker + Docker Compose
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tavily API key](https://app.tavily.com/) (free tier: 1,000 queries/month)

### Run it

```bash
git clone <this-repo>
cd chorus-llm

cp .env.example .env
# open .env and paste your ANTHROPIC_API_KEY and TAVILY_API_KEY

docker compose up
```

That's it. Three services start:

| Service | URL | What it is |
|---------|-----|-----------|
| API + Swagger UI | http://localhost:8000/docs | All 5 endpoints, fully documented |
| Health check    | http://localhost:8000/health | Liveness probe |
| DB browser      | http://localhost:8080 | `sqlite-web` — browse all tables |

### First test

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the effects of climate change on ocean life?"}'
```

You'll see SSE events stream back in real time — agent activity, tool calls,
and finally the synthesis answer token by token.

---

## Architecture

```
                        ┌──────────────────────────┐
        USER ─────────► │  FastAPI  (api/main.py)  │
                        │  POST /query (SSE)       │
                        │  GET  /trace/{job_id}    │
                        │  GET  /eval/latest       │
                        │  POST /approve/{id}      │
                        │  POST /reeval            │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────────┐
                       │  LangGraph Orchestrator      │
                       │  (agents/orchestrator.py)    │
                       │  dynamic conditional routing │
                       └─┬──────┬───────┬───────┬─────┘
                         │      │       │       │
                         ▼      ▼       ▼       ▼
                   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
                   │decompose│ │retrieval│ │critique │ │synthesis │
                   └─────────┘ └────┬────┘ └─────────┘ └──────────┘
                                    │ tool calls (research tasks only)
                                    ▼
                   ┌──────────────────────────────────┐
                   │  Tools (with retry up to 2x)     │
                   │  • web_search   (Tavily)         │
                   │  • self_reflection (session log) │
                   └──────────────────────────────────┘

   ┌───────────────────────┐
   │ SharedContext         │
   │ (core/context.py)     │
   │ - sub_tasks           │
   │ - retrieved_chunks    │
   │ - critiqued_claims    │
   │ - provenance_map      │
   │ - routing_log         │
   │ - token_usage         │
   └───────────────────────┘

   ┌─────────────────────────── EVAL LOOP ─────────────────────────────┐
   │                                                                   │
   │  test_cases.py  →  harness.py  →  scoring.py (3 dimensions)       │
   │       │                                                           │
   │       ▼                                                           │
   │  meta.py reads failures → proposes prompt rewrite                 │
   │       │                                                           │
   │       ▼                                                           │
   │  POST /approve  →  POST /reeval  →  RewriteDelta logged           │
   │                                                                   │
   └───────────────────────────────────────────────────────────────────┘
```

---

## Agents & Decision Boundaries

| Agent | What it owns | What it does NOT do |
|-------|------------|--------------------|
| **Orchestrator** | Routing logic, ExecutionEvent logging | Does not call tools or generate user-facing content |
| **Decomposition** | Splits query into `research`/`summarize` sub-tasks with dependency graph | Does not retrieve information or answer the query |
| **Retrieval** | Generates search queries, calls `web_search` for research tasks, gathers chunks, multi-hop reasoning, citations | Does not judge correctness or merge across sub-tasks |
| **Critique** | Per-claim confidence scores, flags specific spans against source chunks | Does not rewrite the answer — only flags |
| **Synthesis** | Merges accepted claims, resolves contradictions, builds provenance map, streams tokens | Does not retrieve new information |
| **Meta** | Reads failures, identifies worst dimension, proposes prompt rewrite + diff | Never auto-applies rewrites — requires human approval |

**No agent calls another directly.** All inter-agent communication flows through
the `SharedContext` whiteboard. The orchestrator mediates every handoff and
logs the routing decision with justification.

---

## Tool Failure Contracts

Every tool returns one of: `none` (success), `timeout`, `empty`, `malformed`.
The retrieval agent uses `tool_logger.log_with_retry` which retries up to **2
times per tool**, with each attempt logged separately:

| Failure mode | Retry strategy |
|--------------|---------------|
| `timeout` | Retry with reduced/general query |
| `empty` | Retry with broader query |
| `malformed` | Retry with default query |
| `none` | Accept |

Fallback logic lives **in code** (not embedded in prompts) — see
[`agents/retrieval.py`](agents/retrieval.py).

---

## Scoring Dimensions

Every test case is scored across **3 dimensions** using Claude as a judge,
with a written justification for each:

| Dimension | What it measures |
|-----------|-----------------|
| `correctness` | Did the answer address the question accurately? |
| `citation` | What % of retrieved chunks were cited in the provenance map? |
| `critique_quality` | Did the final answer drop bad claims and keep approved ones? |

**LLM-as-judge:** `correctness` and `critique_quality` are scored by
`claude-haiku-4-5` with a structured JSON prompt. `citation` is rule-based
(chunk ID coverage). All scoring logic is in [`eval/scoring.py`](eval/scoring.py).

---

## Self-Improving Prompt Loop

```
eval run completes
    │
    ▼
meta agent reads failures (score < 0.6)
    │
    ▼
identifies worst dimension across all failures
    │
    ▼
maps dimension → responsible agent
    │
    ▼
proposes new system prompt (with diff + justification)
    │
    ▼
stored as PromptRewrite (status="pending")  ← NEVER auto-applied
    │
    ▼
human: POST /approve/{id} {"decision": "approved"}
    │
    ▼
POST /reeval re-runs only failed cases
    │
    ▼
RewriteDelta logged: before_score, after_score, delta
```

**What it does:** Identifies the lowest-scoring dimension, proposes a targeted
prompt rewrite, exposes a human-in-the-loop approval gate, measures the delta.

**What it does NOT do:** It does **not** auto-apply rewrites, does not run
unsupervised, does not learn across sessions. By design — every change is
auditable.

---

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/query` | Submit query, returns SSE stream of agent activity |
| GET | `/trace/{job_id}` | Full execution trace (events + tool calls) |
| GET | `/eval/latest` | Latest eval summary by category & dimension |
| POST | `/approve/{rewrite_id}` | Approve or reject a pending prompt rewrite |
| POST | `/reeval` | Re-run failed cases using latest approved prompt |

All errors include `error` (machine code), `message` (human readable), and
`job_id` if applicable.

Full Swagger docs: http://localhost:8000/docs

---

## Database Schema (SQLite)

| Table | Purpose |
|-------|--------|
| `jobs` | Every submitted query, final answer, provenance |
| `execution_events` | Per-agent: input/output hash, latency, tokens |
| `tool_call_logs` | Every tool call with attempt #, accepted, failure_mode |
| `eval_runs` | Eval run summaries (overall + category + dimension scores) |
| `eval_results` | Per-case scores, justifications, exact prompts, exact tool calls |
| `prompt_rewrites` | Meta-agent proposals with diff, justification, status |
| `rewrite_deltas` | Performance change after a rewrite is approved |

Browse at http://localhost:8080 — all tables queryable via SQL.

---

## Project Layout

```
chorus-llm/
├── docker-compose.yml          ← 3 services: api + worker + logs
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md
├── AI_COLLABORATION.md         ← AI tool attestation
│
├── api/                        ← FastAPI app + 5 routes
├── agents/                     ← 6 agents (incl. orchestrator + meta)
├── tools/                      ← 2 tools with failure contracts
├── core/                       ← context, logging, streaming
├── db/                         ← SQLAlchemy models + connection
├── eval/                       ← test cases, scoring, harness
└── worker/                     ← background eval worker
```

---

## Known Limitations

1. **Token streaming is partial.** Only `synthesis` streams token-by-token.
   Other agents emit `agent_start` / `agent_end` events but their bodies are
   not streamed.

2. **Decomposition dependency graph is not enforced strictly.** The retrieval
   agent honors dependencies sequentially within a single pass, but does not
   re-run dependent tasks if their dependencies change mid-run.

3. **Adversarial robustness is partial.** Our 5 adversarial cases test prompt
   injection, wrong premises, and contradiction resolution — but a determined
   attacker can likely still find weaknesses.

4. **Eval determinism.** `temperature=0` on all LLM calls keeps output stable,
   but Claude is not 100% deterministic. Small variation is possible.

5. **Meta agent rewrite quality.** It identifies the worst dimension correctly
   but the proposed rewrite quality depends on Claude's judgment. Human approval
   gate exists for exactly this reason.

---

## What I'd Build Next

- **Real vector store** (Chroma/pgvector) for retrieval — augment web search
  with persistent memory across sessions.
- **Trace viewer UI** — currently traces are JSON via `/trace/{id}`.
  A timeline view would make debugging dramatically faster.
- **Multi-turn conversations** — currently each query is a fresh job.
- **Cost dashboard** — token usage is tracked but not translated to $.
- **A/B prompt testing** — run a rewrite against the accepted prompt on the
  same cases simultaneously, not sequentially.
- **Eval-driven CI** — block PRs that regress overall eval score >5%.

---

## Running an Eval

```bash
# inside the api container
docker compose exec api python -c "from eval.harness import run_eval; run_eval()"

# or set RUN_EVAL_ON_START=true in .env to run on worker boot
```

Then:

```bash
curl http://localhost:8000/eval/latest
```

---

## License

MIT.
