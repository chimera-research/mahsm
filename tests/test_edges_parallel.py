 import mahsm as ma
 from typing import TypedDict


class State(TypedDict):
    x: int


def set_one(state: dict) -> dict:
    return {"x": 1}


def add_one_from_state(state: dict) -> dict:
    # read x from state if present, else 0
    return {"x": int(state.get("x", 0)) + 1}


def test_parallel_scheduler_parallel_mode():
    # In parallel mode, branches see the same initial state
    edge = ma.edges.parallel([set_one, add_one_from_state], scheduler="parallel")
    node = ma.edges.make_node(edge)

    # initial state has no x
    out = ma.graph.StateGraph(State)  # just to ensure module available
    res = ma.dspy_node(lambda s: None)  # noop to ensure imports

    result = ma.dspy_node(lambda s: None)

    out = ma.graph.StateGraph(State)

    # call node directly
    state = {}
    upd = ma.dspy_node(lambda s: None)

    # just invoke node
    import asyncio
    r = asyncio.run(node(state))
    # set_one -> x=1 ; add_one_from_state sees x absent -> 0+1=1 ; merged -> last-wins so x=1
    assert r["x"] == 1


def test_parallel_scheduler_sequential_mode():
    edge = ma.edges.parallel([set_one, add_one_from_state], scheduler="sequential")
    node = ma.edges.make_node(edge)

    import asyncio
    r = asyncio.run(node({}))
    # sequential: second sees x=1 -> 2
    assert r["x"] == 2
