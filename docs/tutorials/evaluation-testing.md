---
title: Testing & Evaluation with EvalProtocol
---

# Testing & Evaluation with EvalProtocol

Run systematic experiments over your mahsm graphs using Eval Protocol (EP) and pytest.

## Install

```bash
pip install eval-protocol pytest
```

## Minimal example

```python
import os
import dspy
import mahsm as ma
from typing import TypedDict

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

# --- EvalProtocol harness ---
harness = ma.testing.PytestHarness(GRAPH)

from eval_protocol import evaluation_test, EvaluationRow

@evaluation_test(
    name="qa_contains_python",
    completion_params=[{"model": "openai/gpt-5-mini"}, {"model": "openai/gpt-5"}],
)
async def test_contains_python(row: EvaluationRow):
    """Row has a 'question' and we check if 'Python' appears in the answer."""
    res = GRAPH.invoke({"question": row.inputs["question"]})
    ok = "Python" in res.get("answer", "")
    return {"passed": ok, "answer": res.get("answer", "")}

# Utility to yield rows (could also come from Langfuse)
DATASET = [{"question": "Name a popular programming language."}]

def pytest_generate_tests(metafunc):
    if "row" in metafunc.fixturenames:
        rows = [EvaluationRow(inputs=ex) for ex in DATASET]
        metafunc.parametrize("row", rows)
```

Run with:

```bash
pytest -q
```

## Using Langfuse as a data source

```python
harness = ma.testing.PytestHarness(GRAPH)
harness.from_langfuse(project_id="my-project", dataset="eval-qa")
# harness.data_loaders now yields EvaluationRow records from Langfuse
```

## Tips

- Keep evaluation functions pure and fast; long‑running I/O will slow your suite.
- Use `completion_params` to compare models/temps in one run.
- Persist results in Langfuse to analyze regressions across pushes.

## See also

- [Deterministic Testing (CI)](deterministic-testing.md)
- [Parameter Sweeps & Comparisons](parameter-sweeps.md)
- [Observability with Langfuse](langfuse-observability.md)

## Sources

1. Eval Protocol: https://github.com/eval-protocol [1]
2. Langfuse adapter concept: https://langfuse.com/docs/observability/overview [2]
3. pytest docs: https://docs.pytest.org/ [3]
