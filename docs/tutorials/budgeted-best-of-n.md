# Budgeted Best-of-N via Policy Router

Route to different N based on state (e.g., prompt length or difficulty), then fan out and pick the best.

```python
import mahsm as ma

def gen_candidate(st):
    i = st.get("i", 0)
    return {"text": f"cand {i}", "score": float(i)}

def prob_fn(st):
    return [0.8, 0.2] if len(st.get("prompt", "")) <= 10 else [0.2, 0.8]

route = ma.edges.policy_router_select(prob_fn, labels=["n2", "n5"])

edge_map = ma.edges.vmap(
    target=gen_candidate,
    batch_key="items",
    item_to_state=lambda i: {"i": i},
    out_key="candidates",
)
edge_reduce = ma.edges.reduce_edge(
    input_key="candidates",
    output_key="best",
    reducer=ma.reducers.score_and_select(score_key="score", reverse=True),
)

g = ma.graph.StateGraph(dict)
g.add_node("router", lambda st: {})
g.add_node("seed2", lambda st: {"items": list(range(2))})
g.add_node("seed5", lambda st: {"items": list(range(5))})
g.add_node("gen", ma.edges.make_node(edge_map))
g.add_node("pick", ma.edges.make_node(edge_reduce))

g.add_edge(ma.START, "router")
g.add_conditional_edges("router", route, {"n2": "seed2", "n5": "seed5"})
g.add_edge("seed2", "gen")
g.add_edge("seed5", "gen")
g.add_edge("gen", "pick")
g.add_edge("pick", ma.END)

graph = g.compile()
short = graph.invoke({"prompt": "short"})
long = graph.invoke({"prompt": "this is a very long prompt"})
```

Tips
- Use telemetry on `vmap` and `reduce_edge` to log fanout and reducer choice.
- Swap in `reducers.judge_select` to use an LLM/DSPy judge.
