 # Provider-Aware Rate Limiting (per-key token buckets)

 The `vmap` edge supports a per-key token bucket via `rate_limit.key_fn`.

 - `rps`: tokens per second per bucket
 - `burst`: initial capacity
 - `key_fn(state) -> key`: determines which bucket to use for each call

 Example:

 ```python
 edge = ma.edges.vmap(
     target=call_provider,
     batch_key="items",
     item_to_state=lambda it: it,
     out_key="outs",
     rate_limit={
         "rps": 10,
         "burst": 20,
         "key_fn": lambda st: (st.get("provider"), st.get("model")),
     },
 )
 ```

 Telemetry
 - Enable `telemetry=True` to record fanout and `distinct_keys` in `_edges_meta`.

 Notes
 - This is a cooperative limiter inside the node; it complements external/global limiters.
 - If `key_fn` is not provided, a single shared bucket is used.