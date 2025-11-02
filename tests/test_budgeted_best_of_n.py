import mahsm as ma
from typing import TypedDict, List


class State(TypedDict):
    prompt: str
    items: List[int]
    candidates: List[dict]
    best: dict


def gen_candidate(state: dict) -> dict:
    i = state.get("i", 0)
    return {"text": f"cand {i}", "score": float(i)}


def test_budgeted_best_of_n_via_policy_router():
    # Route to N based on prompt length (toy policy)
    def prob_fn(state: dict):
        n_short = 2
        n_long = 5
        is_long = len(state.get("prompt", "")) > 10
        return [0.2, 0.8] if is_long else [0.8, 0.2]  # labels: n2, n5

    route = ma.edges.policy_router_select(prob_fn, labels=["n2", "n5"])

    # fanout and reduce
    edge_map = ma.edges.vmap(
        target=gen_candidate,
        batch_key="items",
        item_to_state=lambda i: {"i": i},
        out_key="candidates",
        max_concurrency=4,
        preserve_order=True,
    )
    reduce = ma.edges.reduce_edge(
        input_key="candidates",
        output_key="best",
        reducer=ma.reducers.score_and_select(score_key="score", reverse=True),
    )

    g = ma.graph.StateGraph(State)

    g.add_node("router", lambda st: {})
    g.add_node("seed2", lambda st: {"items": list(range(2))})
    g.add_node("seed5", lambda st: {"items": list(range(5))})
    g.add_node("gen", ma.edges.make_node(edge_map))
    g.add_node("pick", ma.edges.make_node(reduce))

    g.add_edge(ma.START, "router")
    g.add_conditional_edges("router", route, {"n2": "seed2", "n5": "seed5"})
    g.add_edge("seed2", "gen")
    g.add_edge("seed5", "gen")
    g.add_edge("gen", "pick")
    g.add_edge("pick", ma.END)

    graph = g.compile()

    out_short = graph.invoke({"prompt": "short"})
    assert out_short["best"]["score"] == 1.0

    out_long = graph.invoke({"prompt": "this is a very long prompt"})
    assert out_long["best"]["score"] == 4.0
