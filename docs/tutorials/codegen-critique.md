---
title: Codegen + Critique Loop
---

# Codegen + Critique Loop

An end‑to‑end pattern that generates code from a spec, critiques it, and optionally refines until quality is met.

## Prerequisites

- `pip install mahsm dspy-ai langgraph langfuse`
- `OPENAI_API_KEY` exported

## Code

```python
import os
from typing import TypedDict
import dspy
import mahsm as ma

# Configure LM and tracing
dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))
lf = ma.tracing.init()

# Shared state
class State(TypedDict):
    spec: str
    generated_code: str
    critique: str
    should_refine: bool

# Nodes
@ma.dspy_node
class CodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("spec -> generated_code")

    def forward(self, spec: str):
        return self.gen(spec=spec)

@ma.dspy_node
class CodeCritic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.crit = dspy.Predict("generated_code -> critique, should_refine")

    def forward(self, generated_code: str):
        return self.crit(generated_code=generated_code)

# Router
def route(state: State):
    return "refine" if state.get("should_refine") else ma.END

# Optional refiner that takes critique into account
@ma.dspy_node
class Refiner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought("generated_code, critique -> generated_code")

    def forward(self, generated_code: str, critique: str):
        return self.refine(generated_code=generated_code, critique=critique)

# Graph
workflow = ma.graph.StateGraph(State)
workflow.add_node("generate", CodeGenerator())
workflow.add_node("critic", CodeCritic())
workflow.add_node("refine", Refiner())

workflow.add_edge(ma.START, "generate")
workflow.add_edge("generate", "critic")
workflow.add_conditional_edges("critic", route, {"refine": "refine", ma.END: ma.END})
workflow.add_edge("refine", "critic")

graph = workflow.compile()

# Run
inputs = {
    "spec": "Write a Python function `add(a, b)` with type hints and a docstring."
}
result = graph.invoke(inputs, config={"callbacks": [lf]} if lf else None)

print("\n=== Code ===\n", result["generated_code"])
print("\n=== Critique ===\n", result.get("critique"))
```

## Tips

- Keep the state minimal and typed; only store what subsequent nodes need.
- Use `Predict` when you want exact field names in outputs; use `ChainOfThought` to elicit reasoning.
- For bounded loops, add a `steps: int` counter in state and stop after N iterations.

## See also

- [Multi-Agent Orchestration](multi-agent.md)
- [Safety & Moderation Guards](moderation-guards.md)
- [Deterministic Testing (CI)](deterministic-testing.md)

## Sources

1. DSPy docs: https://dspy.ai/ [1]
2. LangGraph conditional edges: https://docs.langchain.com/oss/python/langgraph/conditional-edges [2]
3. Langfuse callbacks: https://langfuse.com/integrations/frameworks/langchain [3]
