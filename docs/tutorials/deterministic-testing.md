---
title: Deterministic Testing (CI)
---

# Deterministic Testing (CI)

Make graph tests reproducible by stubbing nodes and/or zero‑variance LMs.

## Approach A — Stub nodes (recommended)

```python
from typing import TypedDict
import mahsm as ma
import dspy

class State(TypedDict):
    question: str
    answer: str

# Production node
@ma.dspy_node
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    def forward(self, question: str):
        return self.qa(question=question)

# Test stub (pure function via @dspy_node pattern)
@ma.dspy_node
class QAStub(dspy.Module):
    def forward(self, question: str):
        return {"answer": "Python"}

def build_graph(node):
    wf = ma.graph.StateGraph(State)
    wf.add_node("qa", node)
    wf.add_edge(ma.START, "qa")
    wf.add_edge("qa", ma.END)
    return wf.compile()

# In app: graph = build_graph(QA())
# In tests: use stub for stable output
graph = build_graph(QAStub())
res = graph.invoke({"question": "Name a programming language"})
assert res["answer"] == "Python"
```

## Approach B — Low‑variance LM + seed

```python
import os, dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, seed=7)
)
```

This reduces variance but is not perfectly deterministic across providers/releases. Prefer stubs for CI.

## Notes

- Structure your builder so nodes are injectable (see `build_graph(node)` pattern).
- For deeper graphs, stub only leaf nodes that hit external LMs.
- Combine with golden files for snapshot testing.

## See also

- [Testing & Evaluation](evaluation-testing.md)
- [Parameter Sweeps & Comparisons](parameter-sweeps.md)

## Sources

1. DSPy docs: https://dspy.ai/ [1]
2. LangGraph basics: https://docs.langchain.com/oss/python/langgraph/overview [2]
