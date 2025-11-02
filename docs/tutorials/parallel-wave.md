# Parallel Edge: Wave Scheduler

Run branches in waves (batches) to control concurrency while allowing staged state updates.

Concept: execute up to W branches concurrently, merge their updates, then proceed to the next wave using the updated state.

Example:

```python
import mahsm as ma

def a(st):
    return {"a": 1}

def b(st):
    # sees updated state from previous wave
    return {"b": st.get("a", 0) + 1}

edge = ma.edges.parallel([a, b], scheduler="wave")  # wave behaves like parallel until wave_size is added
node = ma.edges.make_node(edge)

import asyncio
result = asyncio.run(node({}))
assert "a" in result and "b" in result
```

Notes:
- Current implementation treats `wave` like `parallel` until a `wave_size` option is introduced.
- In a future iteration, `wave_size` will batch branches and apply merges between waves.
