 import pytest
 import mahsm as ma
 from typing import TypedDict, List


class S(TypedDict):
    xs: List[int]
    ys: List[dict]


def target_node(state: dict) -> dict:
    x = state["x"]
    return {"y": x * 2}


def test_vmap_edge_preserves_order_and_concurrency():
    workflow = ma.graph.StateGraph(S)

    vmap_edge = ma.edges.vmap(
        target=target_node,
        batch_key="xs",
        item_to_state=lambda x: {"x": x},
        out_key="ys",
        max_concurrency=3,
        preserve_order=True,
    )

    vmap_node = ma.edges.make_node(vmap_edge)

    workflow.add_node("vmap_double", vmap_node)
    workflow.add_edge(ma.START, "vmap_double")
    workflow.add_edge("vmap_double", ma.END)

    graph = workflow.compile()

    data = {"xs": list(range(10))}
    out = graph.invoke(data)

    assert "ys" in out
    ys = out["ys"]
    # Should be list of dict updates from target_node
    assert len(ys) == 10
    # Preserve order: y == 2*i in order
    for i, upd in enumerate(ys):
        assert upd["y"] == 2 * i


def test_reduce_edge_basic():
    # Compose vmap + reduce: extract y values and sum
    class RState(TypedDict):
        xs: List[int]
        ys: List[dict]
        total: int

    workflow = ma.graph.StateGraph(RState)

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

    workflow.add_node("vmap_double", vmap_node)
    workflow.add_node("sum", reduce_node)
    workflow.add_edge(ma.START, "vmap_double")
    workflow.add_edge("vmap_double", "sum")
    workflow.add_edge("sum", ma.END)

    graph = workflow.compile()
    out = graph.invoke({"xs": [1, 2, 3]})
    assert out["total"] == (2 + 4 + 6)
