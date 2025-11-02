---
title: Streaming UX
---

# Streaming UX

Stream intermediate events from your graph to drive a responsive UI.

## Example

```python
import os
import dspy
import mahsm as ma
from typing import TypedDict

dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))
lf = ma.tracing.init()

class State(TypedDict):
    prompt: str
    draft: str

@ma.dspy_node
class Draft(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("prompt -> draft")
    def forward(self, prompt: str):
        return self.cot(prompt=prompt)

wf = ma.graph.StateGraph(State)
wf.add_node("draft", Draft())
wf.add_edge(ma.START, "draft")
wf.add_edge("draft", ma.END)
graph = wf.compile()

for event in graph.stream({"prompt": "Write a haiku about LangGraph"}, config={"callbacks": [lf]} if lf else None):
    # event can be (node_name, state_update) depending on LangGraph version
    print(event)
```

Common pattern: accumulate partial text on the frontend; when END is reached, finalize display.

## Notes

- Exact event schema depends on LangGraph version. Use `print(event)` to inspect and adapt.
- Combine with Langfuse to monitor latency of streamed steps.

## Sources

1. LangGraph streaming: https://docs.langchain.com/oss/python/langgraph/overview [1]
