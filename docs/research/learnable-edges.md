 # Learnable Edges for mahsm (JAX/RL-inspired)

 Status: Draft (concept to guide implementation)

 This document captures ideas from earlier Ember/JAX discussions and maps them to mahsm. These are directional and subject to change; we intentionally avoid hard dependencies on Ember while embracing JAX-style learning where appropriate.

 ## Concept

 - Nodes remain DSPy programs (intelligent, composable units)
 - Edges become decision/control primitives; some can be “learnable,” parameterized by small JAX models or arrays
 - Learning adjusts routing, data selection, and loop halting policies using reward signals from EvalProtocol

 ## Learnable Primitives

 1) Policy Router (Control Flow)
 - Purpose: Choose next node among candidates
 - Params: logits over candidate labels (jnp array) and/or small MLP on state features
 - Inference: softmax -> label (greedy/sample)
 - Training: policy gradient (REINFORCE-style) using episode reward from EvalProtocol

 2) Attention over Retrieved Context (Information Flow)
 - Purpose: Weight/filter retrieved documents before a DSPy node
 - Params: per-doc scorers (linear or small MLP) over features
 - Inference: top-k/threshold via softmax weights
 - Training: reward from final quality; optionally auxiliary scoring signals

 3) Halting Network (Temporal Dynamics)
 - Purpose: Decide whether to continue refinement loops
 - Params: scalar logit -> continuation probability
 - Inference: stop when p(halt) exceeds threshold
 - Training: reward balances quality vs cost (LangFuse metrics)

 ## Learning Loop (Sketch)

 1. Run graph with current params; routers/attention/halting make choices
 2. EvalProtocol computes reward (or loss)
 3. Accumulate (state, action, reward) traces
 4. Apply policy gradient to update params (via JAX)

 Notes:
 - LLM calls are non-differentiable; gradients flow only through the JAX policy components
 - Data is expensive; prefer simple/interpretable parameterizations for sample efficiency

 ## API Direction

 - Short-term helper (shipped): `ma.edges.policy_router_select(prob_fn, labels)` to plug into `add_conditional_edges`
 - Next steps:
   - `mahsm.learn` module: lightweight training loops (trajectory buffer, returns/baselines)
   - `policy_router(params, featurizer)` that returns `prob_fn`; params as JAX arrays when extras installed
   - Checkpointing and load/save of learned params

 ## Examples to Target

 - Greedy vs. sampled routing among two branches (basic viability)
 - Best-of-N with judge where router allocates budget (N) adaptively
 - RAG: attention-based document selection before a DSPy node
 - Halting: learned number of refinement steps in a code critique loop

 ## Observability and Cost

 - Emit spans for each learnable decision with features, logits/probs, chosen action
 - Track tokens, latency, provider costs in LangFuse; include in reward shaping

 ## Risks

 - Overfitting with small datasets; prefer simple models and strong regularization
 - Exploration vs. cost: budget-aware exploration schedules needed
 - Reproducibility: seed management for sampled policies
