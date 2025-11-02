 # PR: Special Edges (Ember/JAX-inspired) — vmap, reduce, parallel, policy router

 Summary
 - Introduces mahsm.edges with vectorized map (vmap), reducers, and parallel scheduler
 - Adds provider-aware pacing via token-bucket
 - Exposes built-in reducers (majority_vote, score_and_select, judge_select)
 - Adds policy_router_select helper for graph.add_conditional_edges
 - LangFuse spans emitted from edges; optional telemetry into state for eval/debug
 - Examples + tests: vmap ordering, best-of-N reduce, policy routing, parallel schedulers

 Why
 - Provide first-class “special edges” to orchestrate fan-out/fan-in, branching, and reductions across DSPy nodes, enabling scalable multi-LLM patterns and learnable routing hooks.

 Highlights
 - vmap: concurrency cap, preserve order, token-bucket rps/burst
 - reduce_edge: pluggable reducers; built-ins shipped under ma.reducers
 - parallel: sequential/parallel schedulers, optional merge, wave placeholder
 - policy_router_select: clean surface for later JAX/RL router
 - tracing: @observe spans for vmap/reduce/parallel
 - telemetry: optional per-edge metadata under `_edges_meta` (fanout, scheduler, reducer)

 Dev Notes
 - No hard dependency on JAX; jit_edge is a stub with feature flagging
 - Telemetry is opt-in to keep state clean in prod
 - Docs: tutorials + research notes (learnable edges, eval loop)

 Tests
 - tests/test_edges_vmap.py — ordering and reduce wiring
 - tests/test_edges_best_of_n.py — fan-out candidates + score reduce
 - tests/test_edges_policy_router.py — conditional routing
 - tests/test_edges_parallel.py — sequential vs parallel semantics
 - tests/test_reducers.py — built-in reducers

 Docs
 - docs/research/ember-integration.md
 - docs/research/ember-edges-spec.md
 - docs/research/learnable-edges.md
 - docs/research/eval-loop.md
 - docs/tutorials/special-edges-quickstart.md
 - docs/tutorials/policy-router-quickstart.md
 - docs/tutorials/best-of-n-judge.md

 Follow-ups
 - Provider-aware token-bucket by provider/model
 - Wave scheduler (batching) and budget-aware best-of-N with judge
 - LangFuse span attributes alignment with telemetry fields
 - JAX extras: simple policy class + REINFORCE loop; checkpointing
 - PR: final polish and examples with DSPy-backed judge
