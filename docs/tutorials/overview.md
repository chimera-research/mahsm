---
title: Tutorials Overview
---

# Tutorials

Use these end-to-end, copy‑pasteable guides to build with mahsm fast. Each tutorial contains runnable snippets you can adapt to your app without adding bloat to the core library.

## Core tutorials

- Hello Graph — Minimal DSPy node + LangGraph state, with tracing  
  See: [hello-graph.md](hello-graph.md)
- Codegen + Critique Loop — Iterative refinement with conditional routing  
  See: [codegen-critique.md](codegen-critique.md)
- Observability with Langfuse — Callbacks + @observe for custom spans  
  See: [langfuse-observability.md](langfuse-observability.md)
- Testing & Evaluation — Run experiments with EvalProtocol  
  See: [evaluation-testing.md](evaluation-testing.md)

## Phase 2 tutorials

- Retrieval + Tools (RAG) — In‑memory retriever tool + answerer  
  See: [rag-tools.md](rag-tools.md)
- Multi‑Agent Orchestration — Plan → research → synthesize  
  See: [multi-agent.md](multi-agent.md)
- Human‑in‑the‑Loop (HITL) — Approval gate with pause/resume  
  See: [hitl.md](hitl.md)

## Next wave tutorials

- Deterministic Testing (CI) — Stubs and zero‑variance configs  
  See: [deterministic-testing.md](deterministic-testing.md)
- Parameter Sweeps & Comparisons — EvalProtocol grids  
  See: [parameter-sweeps.md](parameter-sweeps.md)
- Cost & Latency Budgeting — Budgets and fail‑fast routes  
  See: [cost-latency-budgeting.md](cost-latency-budgeting.md)
- Streaming UX — Drive responsive UIs with graph.stream  
  See: [streaming.md](streaming.md)
- Safety & Moderation Guards — Pre/post moderation + redaction  
  See: [moderation-guards.md](moderation-guards.md)
- Best‑of‑N Judge — Aggregate multiple runs with a judge  
  See: [best-of-n-judge.md](best-of-n-judge.md)
- Policy Router Quickstart — Route by policies/constraints  
  See: [policy-router-quickstart.md](policy-router-quickstart.md)
- Special Edges Quickstart — Using special edges effectively  
  See: [special-edges-quickstart.md](special-edges-quickstart.md)

## Tutorials Index

| Tutorial | Summary | Tags |
| --- | --- | --- |
| Hello Graph | Minimal DSPy node + LangGraph state with tracing | basics, orchestration |
| Codegen + Critique Loop | Generate → critique → optional refine | orchestration, loops |
| Observability with Langfuse | Callbacks and custom spans | observability, langfuse |
| Testing & Evaluation | EP + pytest harness | testing, eval |
| Retrieval + Tools (RAG) | In‑memory retriever tool + answerer | rag, tools |
| Multi‑Agent Orchestration | Plan → research → synthesize | orchestration, multi‑agent |
| Human‑in‑the‑Loop (HITL) | Approval gate with pause/resume | hitl, routing |
| Deterministic Testing (CI) | Stubs and zero‑variance configs | testing, ci |
| Parameter Sweeps & Comparisons | EvalProtocol grids | testing, eval |
| Cost & Latency Budgeting | Budgets and fail‑fast routes | budgets, performance |
| Streaming UX | Stream events to UI | streaming, ux |
| Safety & Moderation Guards | Pre/post moderation + redaction | safety, moderation |
| Best‑of‑N Judge | Aggregate multiple runs with a judge | eval, quality |
| Policy Router Quickstart | Route by policies/constraints | routing, governance |
| Special Edges Quickstart | Using special edges effectively | orchestration |

## Tutorial design principles

- Small, composable building blocks
- End-to-end runnable code (no hidden glue)
- Copies cleanly into your own codebase
- Explains concepts inline; links deeper references

## Roadmap (next wave)

- Retrieval + Tools: RAG node with tool invocation and state updates
- Multi‑Agent Orchestration: Coordinator + workers with message passing
- Human‑in‑the‑Loop (HITL): Review/approve gates using conditional edges
- Deterministic Testing: Mock LMs and golden files for CI
- Parameter Sweeps: Compare models/temps via EvalProtocol rollout processor

## Contributing a tutorial

1. Create a new markdown file in `docs/tutorials/`
2. Include: prerequisites, code, how to run, and links
3. Prefer imports via `import mahsm as ma`
4. Reference external docs as sources at the end

## Sources

1. DSPy documentation: https://dspy.ai/ [1]
2. LangGraph (Python) docs: https://docs.langchain.com/oss/python/langgraph/overview [2]
3. Langfuse Python SDK & decorators: https://langfuse.com/docs/sdk/python/decorators [3]
4. Eval Protocol (EP): https://github.com/eval-protocol [4]
