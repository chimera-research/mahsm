 # Policy Router Quickstart (Experimental)

 This shows how to route between branches using a probability function.

 ```python
 import mahsm as ma
 from typing import TypedDict

 class State(TypedDict):
     value: int

 def inc(state: dict) -> dict:
     return {"value": state["value"] + 1}

 def dec(state: dict) -> dict:
     return {"value": state["value"] - 1}

 def prob_fn(state: dict):
     v = state["value"]
     # prefer inc if negative, else dec
     return [0.9, 0.1] if v < 0 else [0.1, 0.9]

 route = ma.edges.policy_router_select(prob_fn, labels=["inc", "dec"]) 

 g = ma.graph.StateGraph(State)
 g.add_node("router", lambda st: {})
 g.add_node("inc", inc)
 g.add_node("dec", dec)

 g.add_edge(ma.START, "router")
 g.add_conditional_edges("router", route, {"inc": "inc", "dec": "dec"})
 g.add_edge("inc", ma.END)
 g.add_edge("dec", ma.END)

 graph = g.compile()
 assert graph.invoke({"value": -1})["value"] == 0
 assert graph.invoke({"value": 2})["value"] == 1
 ```

 Notes:
 - `policy_router_select` is a placeholder: greedy selection from probabilities.
 - Future work: JAX-backed parameters, sampling strategies, and policy gradient updates.

## See also

- [Special Edges Quickstart](special-edges-quickstart.md)
- [Cost & Latency Budgeting](cost-latency-budgeting.md)
