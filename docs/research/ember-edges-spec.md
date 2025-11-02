 # mahsm Special Edges: Design Spec (inspired by Ember/XCS & JAX transforms)

 Status: Draft (scaffolding)

 ## Goals

 - Introduce graph-level optimization primitives (“special edges”) orthogonal to node logic
 - Enable vectorized fan-out (map), parallel branch execution (wave), and fan-in reducers (reduce)
 - Remain framework-agnostic: no hard dependency on Ember; optional JAX for deterministic numeric transforms
 - Preserve mahsm ergonomics: DSPy nodes + LangGraph state model; full observability via LangFuse

 ## Non-Goals

 - No autodiff across LLM calls (only deterministic numeric functions may use JAX jit)
 - No provider lock-in; edges should be provider-aware but decoupled

 ## Core Concepts

 - Edge = higher-order wiring that controls execution of downstream node(s) and how outputs merge into shared state.
 - Edges expose concurrency, ordering, rate limits, and aggregation policies.

 ## API Surface (proposed)

 ```python
 # mahsm.graph.edges
 vmap(
     target, *,
     batch_key: str,
     item_to_state: callable,   # (item) -> partial_state
     out_key: str,
     max_concurrency: int = 8,
     preserve_order: bool = True,
     rate_limit: dict | None = None,  # {provider, rps, burst, retry}
 ) -> Edge

 parallel(
     branches: list, *,
     merge: callable | None = None,   # (state, branch_updates) -> state_update
     scheduler: str = "parallel",     # sequential|parallel|wave
 ) -> Edge

 reduce_edge(
     *, input_key: str, output_key: str,
     reducer: callable | ReducerSpec,  # majority_vote | score_and_select | llm_judge_select
 ) -> Edge

 # Optional (extras: jax)
 jit_edge(
     fn: callable, *,
     in_keys: list[str], out_key: str,
     device: str | None = None,        # cpu|gpu|tpu
 ) -> Edge
 ```

 ## Behavioral Notes

 - vmap: Launches N calls to `target` from sequence at `state[batch_key]` with bounded concurrency; collects to `state[out_key]`. Ordering controlled by `preserve_order`.
 - parallel: Schedules branches concurrently when possible; `wave` mode groups by readiness level (topological layer). Custom `merge` resolves state updates.
 - reduce_edge: Applies aggregation over `state[input_key]` and writes a single artifact to `state[output_key]`.
 - jit_edge: For pure numeric transforms only; uses JAX jit if extras installed, otherwise falls back to pure Python.

 ## Observability

 - Emit spans per edge invocation with timing, cost (if provider-aware), and fan-out/fan-in sizes.
 - Annotate LangFuse traces with edge metadata (concurrency, scheduler, reducer).

 ## Rate Limiting

 - Basic token-bucket per provider; configurable `rps`, `burst`, `retry`.
 - Backoff and jitter defaults sensible; overridable via `rate_limit`.

 ## Failure Semantics

 - vmap: per-item retries; collect errors with metadata; option to continue-on-error or fail-fast.
 - parallel: fail-fast by default; configurable to isolate failures per branch.
 - reduce_edge: reducer receives successes + error info; implement policies like drop-errors or penalize.

 ## Roadmap

 - 0.3: vmap, parallel, reduce; provider-aware rate limiting; built-in reducers; tracing
 - 0.4: jit_edge (extras), wave scheduler, adaptive concurrency, batch-first adapters
 - Later: device-aware sharding for numeric edges; dynamic budgeting

 ## References

 - Ember v1: XCS transforms, schedulers (sequential/parallel/wave) and operator composition
 - Ember v2: Operators API, XCS @xcs.jit and xcs.vmap
 - JAX: jit/vmap concepts for reference semantics
