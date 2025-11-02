---
title: Observability with Langfuse
---

# Observability with Langfuse

Instrument your mahsm apps end‑to‑end using Langfuse callbacks and custom spans.

## Prerequisites

Set environment variables:

```bash
export LANGFUSE_PUBLIC_KEY=pk_lf_...
export LANGFUSE_SECRET_KEY=sk_lf_...
# Optional region override
# export LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Graph‑level tracing

```python
import mahsm as ma

# Initialize once
lf = ma.tracing.init()  # returns Langfuse CallbackHandler or None

# ... build your graph as usual
graph = workflow.compile()

# Pass callbacks during execution
result = graph.invoke(inputs, config={"callbacks": [lf]} if lf else None)
```

## Custom spans with @observe

Use `@ma.tracing.observe` to create spans for non‑LLM code or to group steps.

```python
from mahsm.tracing import observe

@observe(name="fetch_context", type="tool")
def fetch_context(query: str):
    # your I/O or CPU work here
    return {"facts": ["..."]}

@observe(name="post_process", type="task")
def post_process(text: str):
    return text.strip()
```

These spans appear nested under the active trace when called inside a graph run invoked with the Langfuse callback handler.

## Tips

- If you don't see traces, verify API keys and check the correct `LANGFUSE_HOST` for your region.
- To reduce noise, only decorate meaningful blocks with `@observe`.
- See advanced SDK features like masking and sampling in Langfuse docs.

## See also

- [Streaming UX](streaming.md)
- [Testing & Evaluation](evaluation-testing.md)
- [Cost & Latency Budgeting](cost-latency-budgeting.md)

## Sources

1. Langfuse Python SDK decorators: https://langfuse.com/docs/sdk/python/decorators [1]
2. LangChain/LangGraph integration: https://langfuse.com/integrations/frameworks/langchain [2]
3. Advanced usage (masking, sampling): https://langfuse.com/docs/observability/sdk/python/advanced-usage [3]
