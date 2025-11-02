---
title: Safety & Moderation Guards
---

# Safety & Moderation Guards

Add pre/post moderation checks with conditional routing to keep outputs within policy.

## Example

```python
import os
import re
from typing import TypedDict
import dspy
import mahsm as ma

dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))

class State(TypedDict):
    prompt: str
    draft: str
    safe: bool

@ma.dspy_node
class Draft(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("prompt -> draft")
    def forward(self, prompt: str):
        return self.cot(prompt=prompt)

def moderate_text(text: str) -> bool:
    banned = re.compile(r"\b(violence|hate|exploit)\b", re.I)
    return not banned.search(text or "")

@ma.dspy_node
class Moderation(dspy.Module):
    def forward(self, draft: str):
        return {"safe": moderate_text(draft)}

def route(state: State):
    return ma.END if state.get("safe") else "redact"

@ma.dspy_node
class Redact(dspy.Module):
    def forward(self, draft: str):
        return {"draft": "[REDACTED]"}

wf = ma.graph.StateGraph(State)
wf.add_node("draft", Draft())
wf.add_node("moderate", Moderation())
wf.add_node("redact", Redact())
wf.add_edge(ma.START, "draft")
wf.add_edge("draft", "moderate")
wf.add_conditional_edges("moderate", route, {ma.END: ma.END, "redact": "redact"})
wf.add_edge("redact", ma.END)
graph = wf.compile()

state = graph.invoke({"prompt": "Write a short blurb about LangGraph."})
print(state["draft"])
```

## Notes

- Replace `moderate_text` with your provider’s moderation API.
- Use pre‑moderation (on inputs) and post‑moderation (on outputs) as needed.

## Sources

1. LangGraph routing: https://docs.langchain.com/oss/python/langgraph/overview [1]
