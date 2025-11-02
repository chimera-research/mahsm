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

edge = ma.edges.parallel([a, b], scheduler="wave", wave_size=1)  # process one branch per wave
node = ma.edges.make_node(edge)

import asyncio
result = asyncio.run(node({}))
assert "a" in result and "b" in result
```

Notes:
- `wave_size` controls how many branches run concurrently in each wave.
- After every wave, updates are merged into state before the next wave starts.
