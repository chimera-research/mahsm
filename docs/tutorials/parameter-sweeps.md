---
title: Parameter Sweeps & Comparisons
---

# Parameter Sweeps & Comparisons

Compare models/temperatures on the same evaluation with EvalProtocol.

## Example

```python
import os
import dspy
import mahsm as ma
from typing import TypedDict
from eval_protocol import evaluation_test, EvaluationRow

dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))

class State(TypedDict):
    question: str
    answer: str

@ma.dspy_node
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
    def forward(self, question: str):
        return self.qa(question=question)

wf = ma.graph.StateGraph(State)
wf.add_node("qa", QA())
wf.add_edge(ma.START, "qa")
wf.add_edge("qa", ma.END)
GRAPH = wf.compile()

DATASET = [
    {"question": "Name a popular programming language."},
    {"question": "What is LangGraph used for?"},
]

@evaluation_test(
    name="qa-sweep",
    completion_params=[
        {"model": "openai/gpt-5-mini", "temperature": 0.0},
        {"model": "openai/gpt-5-mini", "temperature": 0.7},
        {"model": "openai/gpt-5", "temperature": 0.0},
    ],
)
async def test_quality(row: EvaluationRow):
    res = GRAPH.invoke({"question": row.inputs["question"]})
    ok = len(res.get("answer", "")) > 0
    return {"passed": ok, "answer": res.get("answer", "")}

def pytest_generate_tests(metafunc):
    if "row" in metafunc.fixturenames:
        rows = [EvaluationRow(inputs=ex) for ex in DATASET]
        metafunc.parametrize("row", rows)
```

Run: `pytest -q`

## Tips

- Add more params (e.g., max_tokens) to the grid as needed.
- Export results to Langfuse for trend analysis.

## Sources

1. Eval Protocol: https://github.com/eval-protocol [1]
2. DSPy docs: https://dspy.ai/ [2]
