import mahsm as ma


def set_a(st):
    return {"a": 1}


def set_b_from_a(st):
    return {"b": int(st.get("a", 0)) + 1}


def set_c_from_b(st):
    return {"c": int(st.get("b", 0)) + 1}


def test_parallel_wave_size_affects_state_staging():
    # Wave size 1 behaves like sequential: staged updates between branches
    edge_seq = ma.edges.parallel([set_a, set_b_from_a, set_c_from_b], scheduler="wave", wave_size=1)
    node_seq = ma.edges.make_node(edge_seq)

    import asyncio
    out_seq = asyncio.run(node_seq({}))
    assert out_seq["c"] == 3  # a=1 -> b=2 -> c=3

    # Wave size 2: first two run together on initial state, then third sees merged result
    edge_w2 = ma.edges.parallel([set_a, set_b_from_a, set_c_from_b], scheduler="wave", wave_size=2)
    node_w2 = ma.edges.make_node(edge_w2)
    out_w2 = asyncio.run(node_w2({}))
    assert out_w2["c"] == 2  # wave1: a=1,b=1 -> wave2: c=b+1=2

    # Wave size >= len(branches) equals parallel
    edge_p = ma.edges.parallel([set_a, set_b_from_a, set_c_from_b], scheduler="wave", wave_size=10)
    node_p = ma.edges.make_node(edge_p)
    out_p = asyncio.run(node_p({}))
    assert out_p["c"] == 1  # no staged updates between branches
