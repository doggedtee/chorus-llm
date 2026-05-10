# AI Collaboration Attestation

This document discloses where and how AI assistants were used to build this
project, per the assessment's attestation requirement.

## Tools Used

- **Claude (Anthropic)** — used as a pair-programming assistant via Claude Code
  CLI for code authoring, design discussion, and debugging.

## How AI Was Used

| Area | Role of AI |
|------|-----------|
| Project architecture | Brainstorming partner — proposed the LangGraph + FastAPI + SQLite stack and the file structure. Final architectural choices were mine. |
| Code authoring | AI authored most file scaffolding under my direction. I reviewed, edited, and accepted/rejected each file. I asked clarifying questions about every concept I did not understand (Pydantic vs TypedDict, asyncio.Queue, SSE, LangGraph state, Enum semantics, etc.) before approving any code. |
| Prompt engineering | System prompts for each agent (decomposition, retrieval, critique, synthesis, compression, meta) were drafted by AI and refined through iteration. |
| Failure contracts | Tool failure modes (timeout/empty/malformed) were enumerated together — I confirmed the design before implementation. |
| Eval test cases | The 15 test cases (5 baseline + 5 ambiguous + 5 adversarial) were drafted by AI, then I reviewed for relevance to the spec's failure modes (prompt injection, wrong premises, contradiction). |
| Scoring logic | All 6 scoring dimensions were designed in conversation. The decision to include written justification strings (not just numeric scores) came from the spec — implementation followed. |
| Documentation | This README and inline comments were drafted with AI and then reviewed for accuracy. |

## How AI Was NOT Used

- The model providers, API keys, and runtime decisions are mine.
- I made every approve/reject decision on AI-suggested edits.
- I do not claim deep familiarity with every line of LangGraph internals — I
  understand the concepts at the level required to maintain and extend this
  system, and I asked questions until I did.

## Verification Steps I Took

- Walked through the full request flow file by file, asked questions on every
  concept I did not already understand.
- Verified each spec requirement (4 tools, 6 dimensions, 5 endpoints,
  failure contracts, retry logic, provenance map, self-improvement loop) is
  implemented in actual code, not just in prompts or comments.
- Tested SSE streaming end-to-end (decomposition → retrieval → critique →
  synthesis token streaming).
- Confirmed `docker compose up` boots all 3 services without manual steps.

## Why This Disclosure Matters

The spec stated: *"Document where and how — your final report will surface
AI-collaboration signals."* I would rather be honest up front than have the
collaboration pattern be surfaced as a gotcha during review. The skill being
evaluated is whether I can use a powerful tool well — understanding what it
produces, catching mistakes, and making sound architectural decisions — not
whether I can pretend to type every character myself.