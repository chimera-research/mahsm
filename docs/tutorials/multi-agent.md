---
title: Multi-Agent Orchestration
---

# Multi-Agent Orchestration

Coordinator and worker agents in a simple pipeline: plan → research → synthesize.

## Prerequisites

```bash
pip install mahsm dspy-ai langgraph langfuse
```

## Code

```python
import os
from typing import TypedDict
import dspy
import mahsm as ma

dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))
lf = ma.tracing.init()

class State(TypedDict):
    topic: str
    plan: str
    findings: str
    report: str

@ma.dspy_node
class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.Predict("topic -> plan")
    def forward(self, topic: str):
        return self.plan(topic=topic)

@ma.dspy_node
class Researcher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought("plan -> findings")
    def forward(self, plan: str):
        return self.research(plan=plan)

@ma.dspy_node
class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synth = dspy.ChainOfThought("topic, findings -> report")
    def forward(self, topic: str, findings: str):
        return self.synth(topic=topic, findings=findings)

# Graph
wf = ma.graph.StateGraph(State)
wf.add_node("plan", Planner())
wf.add_node("research", Researcher())
wf.add_node("synthesize", Synthesizer())
wf.add_edge(ma.START, "plan")
wf.add_edge("plan", "research")
wf.add_edge("research", "synthesize")
wf.add_edge("synthesize", ma.END)
graph = wf.compile()

res = graph.invoke({"topic": "Benefits of LangGraph with DSPy"}, config={"callbacks": [lf]} if lf else None)
print("Plan:\n", res["plan"])
print("\nFindings:\n", res["findings"])
print("\nReport:\n", res["report"])
```

## Notes

- Model roles are separated but share state via the graph.
- Swap the Planner/Researcher prompts or attach tools as needed.
- Extend by adding a Reviewer node with conditional routing.

## Sources

1. DSPy docs: https://dspy.ai/ [1]
2. LangGraph basics: https://docs.langchain.com/oss/python/langgraph/overview [2]
3. Langfuse callbacks: https://langfuse.com/integrations/frameworks/langchain [3]
