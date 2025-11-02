---
title: Hello Graph
---

# Hello Graph

Build the smallest end‑to‑end mahsm app: a single DSPy node wrapped into a LangGraph workflow, traced with Langfuse.

## Prerequisites

- `pip install mahsm dspy-ai langgraph langfuse`
- `OPENAI_API_KEY` exported (or swap to your LM provider)

## Code

```python
import os
import dspy
import mahsm as ma
from typing import TypedDict

# 1) Configure the LM (use any dspy-supported provider)
dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))

# 2) Initialize tracing (optional but recommended)
lf_handler = ma.tracing.init()  # reads LANGFUSE_* env vars if set

# 3) Define shared state
class State(TypedDict):
    question: str
    answer: str

# 4) Define a DSPy module and expose it as a node
@ma.dspy_node
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        return self.cot(question=question)

# 5) Build the graph
workflow = ma.graph.StateGraph(State)
workflow.add_node("qa", QA())
workflow.add_edge(ma.START, "qa")
workflow.add_edge("qa", ma.END)
graph = workflow.compile()

# 6) Run it (pass Langfuse handler if available)
result = graph.invoke(
    {"question": "What is DSPy in one sentence?"},
    config={"callbacks": [lf_handler]} if lf_handler else None,
)

print(result["answer"])  # Traced automatically if Langfuse is configured
```

## Notes

- `@ma.dspy_node` bridges DSPy modules to LangGraph nodes automatically by introspecting `forward()` and mapping state fields.
- Returning values from DSPy predictors updates the shared state.
- Pass the Langfuse `CallbackHandler` in `config` for end‑to‑end graph traces.

## See also

- [Multi-Agent Orchestration](multi-agent.md)
- [Observability with Langfuse](langfuse-observability.md)
- [Testing & Evaluation](evaluation-testing.md)

## Sources

1. DSPy docs: https://dspy.ai/ [1]
2. LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview [2]
3. Langfuse Python SDK & callbacks: https://langfuse.com/docs/observability/sdk/python/overview [3]
