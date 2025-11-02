---
title: Cost & Latency Budgeting
---

# Cost & Latency Budgeting

Guard your workflow with simple, explicit budgets (steps/time) and fail‑fast routes.

## Pattern

```python
import time
from typing import TypedDict
import dspy
import mahsm as ma

class State(TypedDict):
    query: str
    answer: str
    steps: int
    started_at: float
    max_steps: int
    max_seconds: float

@ma.dspy_node
class Think(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("query -> answer")
    def forward(self, query: str):
        return self.cot(query=query)

def route(state: State):
    # Step budget
    if state.get("steps", 0) >= state.get("max_steps", 5):
        return "budget_exceeded"
    # Time budget
    if (time.time() - state.get("started_at", time.time())) > state.get("max_seconds", 10.0):
        return "budget_exceeded"
    return "think"

@ma.dspy_node
class Increment(dspy.Module):
    def forward(self, steps: int):
        return {"steps": steps + 1}

@ma.dspy_node
class BudgetExceeded(dspy.Module):
    def forward(self, query: str, steps: int):
        return {"answer": f"Aborted after {steps} steps: budget exceeded."}

wf = ma.graph.StateGraph(State)
wf.add_node("route", lambda s: {})  # no-op placeholder; routing function is separate
wf.add_node("think", Think())
wf.add_node("inc", Increment())
wf.add_node("budget_exceeded", BudgetExceeded())
wf.add_edge(ma.START, "route")
wf.add_conditional_edges("route", route, {"think": "think", "budget_exceeded": "budget_exceeded"})
wf.add_edge("think", "inc")
wf.add_edge("inc", "route")
wf.add_edge("budget_exceeded", ma.END)
wf.add_edge("think", ma.END)  # if one-shot is acceptable
graph = wf.compile()

state = graph.invoke({
    "query": "Explain DSPy succinctly",
    "steps": 0,
    "started_at": time.time(),
    "max_steps": 1,
    "max_seconds": 5.0,
})
print(state["answer"])
```

## Notes

- For precise cost tracking, sync Langfuse usage metrics into your own counters, or wrap LM calls with custom accounting.
- Make budgets visible in your state to inspect in traces.

## Sources

1. LangGraph routing: https://docs.langchain.com/oss/python/langgraph/overview [1]
