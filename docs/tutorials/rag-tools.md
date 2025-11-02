---
title: Retrieval + Tools (RAG)
---

# Retrieval + Tools (RAG)

Minimal RAG with an in‑memory retriever “tool” and a DSPy answerer. Swap the retriever with your own implementation.

## Prerequisites

```bash
pip install mahsm dspy-ai langgraph langfuse
```

## Code

```python
import os
from typing import TypedDict, List
import dspy
import mahsm as ma
from mahsm.tracing import observe

# Configure
dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY")))
lf = ma.tracing.init()

# In-memory corpus (replace with your vector DB / retriever API)
CORPUS = [
    {"id": "doc1", "text": "DSPy programs structure prompt engineering into code."},
    {"id": "doc2", "text": "LangGraph builds stateful agent workflows with nodes and edges."},
    {"id": "doc3", "text": "Langfuse provides observability for LLM apps via traces and spans."},
]

def simple_retrieve(query: str, k: int = 2) -> List[dict]:
    # Toy retriever: naive keyword match by overlap count
    qs = set(query.lower().split())
    scored = [
        (len(qs.intersection(set(d["text"].lower().split()))), d) for d in CORPUS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

class State(TypedDict):
    question: str
    context: List[dict]
    answer: str
    citations: List[str]

@observe(name="retrieve", type="tool")
def retrieve_tool(question: str) -> list[dict]:
    return simple_retrieve(question)

@ma.dspy_node
class RetrieveNode(dspy.Module):
    def forward(self, question: str):
        ctx = retrieve_tool(question)
        return {"context": ctx}

@ma.dspy_node
class AnswerNode(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question, context -> answer, citations")

    def forward(self, question: str, context: list[dict]):
        ctx_text = "\n".join(f"[{d['id']}] {d['text']}" for d in context)
        out = self.predict(question=question, context=ctx_text)
        # Ensure citations is a list of IDs if model returned string
        citations = out.citations if isinstance(out.citations, list) else [str(out.citations)]
        return {"answer": out.answer, "citations": citations}

# Build graph
wf = ma.graph.StateGraph(State)
wf.add_node("retrieve", RetrieveNode())
wf.add_node("answer", AnswerNode())
wf.add_edge(ma.START, "retrieve")
wf.add_edge("retrieve", "answer")
wf.add_edge("answer", ma.END)
graph = wf.compile()

# Run
res = graph.invoke(
    {"question": "How do DSPy and LangGraph relate?"},
    config={"callbacks": [lf]} if lf else None,
)
print("Answer:\n", res["answer"])
print("Citations:", res.get("citations"))
```

## Notes

- Replace `simple_retrieve` with your vector DB or API client; keep the node interface the same.
- Store structured context in state, but convert to text for LLM inputs.
- Use `@observe` to trace tool latency and errors in Langfuse.

## See also

- [Multi-Agent Orchestration](multi-agent.md)
- [Safety & Moderation Guards](moderation-guards.md)
- [Cost & Latency Budgeting](cost-latency-budgeting.md)

## Sources

1. DSPy docs: https://dspy.ai/ [1]
2. LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview [2]
3. Langfuse decorators: https://langfuse.com/docs/sdk/python/decorators [3]
```
