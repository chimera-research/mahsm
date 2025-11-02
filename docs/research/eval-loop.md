 # Closing the Loop: LangFuse + EvalProtocol

 This note explains how special edges emit spans to LangFuse and how to run end-to-end evaluations via EvalProtocol.

 ## Instrumentation

 - Edges `vmap`, `reduce_edge`, and `parallel` are wrapped with `ma.tracing.observe`, which maps to LangFuse's `@observe` decorator when available.
 - Call `ma.tracing.init()` once to set up LangFuse client, DSPy instrumentation, and obtain a LangGraph callback handler.

 Example:

 ```python
 import mahsm as ma

 handler = ma.tracing.init()  # reads LANGFUSE_* from env; returns LangGraph callback or None
 graph = workflow.compile()
 result = graph.invoke(state, config={"callbacks": [handler] if handler else []})
 ```

The spans include names like `edges.vmap`, `edges.reduce`, `edges.parallel`. Inputs/outputs are captured by the decorator if your LangFuse setup enables it.

Optional telemetry in-state:

- You can enable lightweight per-edge metadata by setting `telemetry=True` when constructing edges. Metadata is returned under `_edges_meta` by default (configurable via `telemetry_key`).
- Example:

```python
edge_map = ma.edges.vmap(..., telemetry=True)            # adds fanout, rps, burst, etc.
edge_reduce = ma.edges.reduce_edge(..., telemetry=True)  # adds reducer name and input length
edge_par = ma.edges.parallel(..., telemetry=True)        # adds scheduler and branch count
```

This makes it easy to align LangFuse spans with explicit counters directly in the graph state for debugging or evaluation. Keep it off in production if you want a clean state.

 ## Evaluation via EvalProtocol

 - `mahsm.testing.PytestHarness` wraps a compiled graph and bridges to EvalProtocol.
 - Use a LangFuse adapter as a data source, or provide your own generators.

 ```python
 import mahsm as ma

 graph = workflow.compile()
 harness = ma.testing.PytestHarness(graph)
 harness.from_langfuse(project="my-project", tag="my-tag")

 # Processor will replay prompts through your graph and collect outcomes
 processor = harness.rollout_processor
 # See eval-protocol docs for configuring metrics/judges
 ```

 With spans in place, you can align evaluation rows from LangFuse with decisions made by edges (fan-out sizes, reducer picks, routing choices) to compute rewards and perform learning in future iterations.
