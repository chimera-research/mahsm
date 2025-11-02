import mahsm as ma


def test_vmap_rate_limit_key_fn_telemetry_keys():
    # target just echoes item with group to prove batching works
    def target(st: dict):
        return {"y": st["x"], "g": st["group"]}

    edge = ma.edges.vmap(
        target=target,
        batch_key="items",
        item_to_state=lambda it: {"x": it["x"], "group": it["group"]},
        out_key="outs",
        max_concurrency=8,
        preserve_order=True,
        rate_limit={"rps": 1000, "burst": 10, "key_fn": lambda st: st.get("group")},
        telemetry=True,
    )
    node = ma.edges.make_node(edge)

    items = [
        {"x": 1, "group": "a"},
        {"x": 2, "group": "a"},
        {"x": 3, "group": "b"},
        {"x": 4, "group": "b"},
    ]

    import asyncio
    res = asyncio.run(node({"items": items}))

    assert "outs" in res and len(res["outs"]) == 4
    meta = res.get("_edges_meta")
    assert meta and meta.get("edge") == "vmap"
    # two groups -> two buckets
    assert meta.get("distinct_keys") == 2
