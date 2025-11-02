---
title: Human-in-the-Loop (HITL)
---

# Human-in-the-Loop (HITL)

Insert a human approval gate into a mahsm graph. This pattern pauses until an external approval flag is provided.

## Prerequisites

```bash
pip install mahsm dspy-ai langgraph langfuse
```

## Code

```python
import os
from typing import TypedDict, Optional
import dspy
import mahsm as ma

dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))
lf = ma.tracing.init()

class State(TypedDict):
    prompt: str
    draft: str
    approved: Optional[bool]
    published: Optional[str]

@ma.dspy_node
class DraftNode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("prompt -> draft")
    def forward(self, prompt: str):
        return self.cot(prompt=prompt)

def route_after_draft(state: State):
    # If approved flag already provided, proceed; else await review
    return "publish" if state.get("approved") else "await_review"

@ma.dspy_node
class AwaitReview(dspy.Module):
    def forward(self, draft: str):
        # No changes; acts as a pause point
        return {"draft": draft}

@ma.dspy_node
class PublishNode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("draft -> published")
    def forward(self, draft: str):
        return self.predict(draft=draft)

# Graph
wf = ma.graph.StateGraph(State)
wf.add_node("draft", DraftNode())
wf.add_node("await_review", AwaitReview())
wf.add_node("publish", PublishNode())
wf.add_edge(ma.START, "draft")
wf.add_conditional_edges("draft", route_after_draft, {"publish": "publish", "await_review": "await_review"})
wf.add_edge("publish", ma.END)
graph = wf.compile()

# 1) Run without approval to produce a draft and pause
state = graph.invoke({"prompt": "Write a short product blurb."}, config={"callbacks": [lf]} if lf else None)
print("Draft:\n", state["draft"])
print("Approved?", state.get("approved"))

# 2) Simulate human approval: set flag and run again to publish
state["approved"] = True
state = graph.invoke(state, config={"callbacks": [lf]} if lf else None)
print("\nPublished:\n", state.get("published"))
```

## Notes

- The "await_review" node acts as a pause point; you can store `state` externally, collect human input, then re‑invoke.
- In a service, persist the state between invocations and set `approved` based on UI input.
- You can add additional gates (legal, safety) by chaining more review nodes.

## See also

- [Safety & Moderation Guards](moderation-guards.md)
- [Multi-Agent Orchestration](multi-agent.md)
- [Cost & Latency Budgeting](cost-latency-budgeting.md)

## Sources

1. LangGraph state & routing: https://docs.langchain.com/oss/python/langgraph/overview [1]
2. DSPy modules: https://dspy.ai/ [2]
3. Langfuse callbacks: https://langfuse.com/integrations/frameworks/langchain [3]
