 import mahsm as ma
 from typing import TypedDict, List


class State(TypedDict):
    prompt: str
    candidates: List[dict]
    best: dict


def gen_candidate(state: dict) -> dict:
    # toy generator: uses id field to make a score
    i = state.get("i", 0)
    return {
        "text": f"candidate {i} for {state['prompt']}",
        "score": float(i),
    }


def test_best_of_n_with_score_reducer():
    N = 5

    # vmap over items 0..N-1
    edge_map = ma.edges.vmap(
        target=gen_candidate,
        batch_key="items",
        item_to_state=lambda i: {"i": i},
        out_key="candidates",
        max_concurrency=4,
        preserve_order=True,
    )

    # reduce to best by score
    reducer = ma.reducers.score_and_select(score_key="score", reverse=True)
    edge_reduce = ma.edges.reduce_edge(
        input_key="candidates",
        output_key="best",
        reducer=reducer,
    )

    # Build graph
    g = ma.graph.StateGraph(State)

    def seed_items(state: dict) -> dict:
        return {"items": list(range(N))}

    g.add_node("seed", seed_items)
    g.add_node("gen", ma.edges.make_node(edge_map))
    g.add_node("pick", ma.edges.make_node(edge_reduce))

    g.add_edge(ma.START, "seed")
    g.add_edge("seed", "gen")
    g.add_edge("gen", "pick")
    g.add_edge("pick", ma.END)

    graph = g.compile()

    out = graph.invoke({"prompt": "test"})
    assert out["best"]["score"] == float(N - 1)
