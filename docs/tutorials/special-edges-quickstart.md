 # Special Edges Quickstart (Experimental)

 This quickstart shows how to fan-out a node over a list (vmap) and then reduce results.

 ```python
 import mahsm as ma
 from typing import TypedDict, List

 class State(TypedDict):
     xs: List[int]
     ys: List[dict]
     total: int

 def target_node(state: dict) -> dict:
     x = state["x"]
     return {"y": x * 2}

 vmap_node = ma.edges.make_node(
     ma.edges.vmap(
         target=target_node,
         batch_key="xs",
         item_to_state=lambda x: {"x": x},
         out_key="ys",
         max_concurrency=4,
         preserve_order=True,
     )
 )

 def sum_reducer(items: List[dict]) -> int:
     return sum(it["y"] for it in items)

 reduce_node = ma.edges.make_node(
     ma.edges.reduce_edge(input_key="ys", output_key="total", reducer=sum_reducer)
 )

 g = ma.graph.StateGraph(State)
 g.add_node("vmap_double", vmap_node)
 g.add_node("sum", reduce_node)
 g.add_edge(ma.START, "vmap_double")
 g.add_edge("vmap_double", "sum")
 g.add_edge("sum", ma.END)

 graph = g.compile()
 out = graph.invoke({"xs": [1, 2, 3]})
 assert out["total"] == 12
 ```

 Notes:
 - `edges.vmap` runs the target concurrently (bounded by `max_concurrency`).
 - `preserve_order=True` ensures outputs align with the input order.
 - `edges.reduce_edge` aggregates a list of per-item updates.
 - Rate limit (RPS) support is naive pacing for now (subject to change).

## See also

- [Best-of-N Judge](best-of-n-judge.md)
- [Policy Router Quickstart](policy-router-quickstart.md)
