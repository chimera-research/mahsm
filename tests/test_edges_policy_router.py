 import mahsm as ma
 from typing import TypedDict


class RState(TypedDict):
    value: int


def inc(state: dict) -> dict:
    return {"value": state["value"] + 1}


def dec(state: dict) -> dict:
    return {"value": state["value"] - 1}


def test_policy_router_select_routes_correctly():
    # Greedy router: if value < 0 prefer inc else dec
    def prob_fn(state: dict):
        v = state["value"]
        if v < 0:
            return [0.9, 0.1]  # inc, dec
        else:
            return [0.1, 0.9]

    route = ma.edges.policy_router_select(prob_fn, labels=["inc", "dec"])

    g = ma.graph.StateGraph(RState)
    # Router node is a pass-through; conditional edges use `route`
    def router_node(state: dict) -> dict:
        return {}

    g.add_node("router", router_node)
    g.add_node("inc", inc)
    g.add_node("dec", dec)

    g.add_edge(ma.START, "router")
    g.add_conditional_edges("router", route, {
        "inc": "inc",
        "dec": "dec",
    })
    g.add_edge("inc", ma.END)
    g.add_edge("dec", ma.END)

    graph = g.compile()

    out1 = graph.invoke({"value": -1})  # should route to inc
    assert out1["value"] == 0

    out2 = graph.invoke({"value": 2})  # should route to dec
    assert out2["value"] == 1
