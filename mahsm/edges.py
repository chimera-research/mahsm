 """
 mahsm.edges — scaffolding for Special Edges (vectorized map, parallel branches, reduce, optional jit)

 NOTE: This is scaffolding for experimentation and review. Implementations are placeholders.
 """

 from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence
import asyncio
import inspect
import time

try:
    from .tracing import observe as _observe
except Exception:
    def _observe(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


 class Edge:  # minimal placeholder type for integration discussion
     def __init__(self, kind: str, config: Dict[str, Any]):
         self.kind = kind
         self.config = config

    
 def vmap(
     target: Any,
     *,
     batch_key: str,
     item_to_state: Callable[[Any], Dict[str, Any]],
     out_key: str,
     max_concurrency: int = 8,
     preserve_order: bool = True,
     rate_limit: Optional[Dict[str, Any]] = None,
 ) -> Edge:
     """Vectorized fan-out over a sequence in state; collects results into out_key.
     Placeholder: returns Edge descriptor only.
     """
     return Edge(
         "vmap",
         {
             "target": target,
             "batch_key": batch_key,
             "item_to_state": item_to_state,
             "out_key": out_key,
             "max_concurrency": max_concurrency,
             "preserve_order": preserve_order,
             "rate_limit": rate_limit or {},
         },
     )


 def parallel(
     branches: List[Any],
     *,
     merge: Optional[Callable[..., Dict[str, Any]]] = None,
     scheduler: str = "parallel",  # sequential|parallel|wave
 ) -> Edge:
     """Parallel branch execution with optional custom merge function.
     Placeholder: returns Edge descriptor only.
     """
     return Edge(
         "parallel",
         {
             "branches": branches,
             "merge": merge,
             "scheduler": scheduler,
         },
     )


 def reduce_edge(
     *,
     input_key: str,
     output_key: str,
     reducer: Callable[[List[Any]], Any],
 ) -> Edge:
     """Aggregate a collection in state into a single artifact.
     Placeholder: returns Edge descriptor only.
     """
     return Edge(
         "reduce",
         {
             "input_key": input_key,
             "output_key": output_key,
             "reducer": reducer,
         },
     )


# ---- Conversion helpers: Edge -> LangGraph-compatible node ----

async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def make_node(edge: Edge) -> Callable[[dict], Any]:
    """Convert an Edge descriptor into a LangGraph-compatible node function.

    Supported kinds:
      - vmap: concurrent fan-out over a batch_key
      - reduce: aggregate a list into a single value
    """

    if edge.kind == "vmap":
        cfg = edge.config
        target = cfg["target"]
        batch_key: str = cfg["batch_key"]
        item_to_state: Callable[[Any], Dict[str, Any]] = cfg["item_to_state"]
        out_key: str = cfg["out_key"]
        max_concurrency: int = int(cfg.get("max_concurrency", 8))
        preserve_order: bool = bool(cfg.get("preserve_order", True))
        rate_limit = cfg.get("rate_limit", {}) or {}
        rps = rate_limit.get("rps", None)
        burst = rate_limit.get("burst", 1)

        class _TokenBucket:
            def __init__(self, rate: float, burst: int):
                self.rate = float(rate)
                self.capacity = max(1.0, float(burst))
                self.tokens = self.capacity
                self._last = time.perf_counter()
                self._lock = asyncio.Lock()

            async def acquire(self):
                async with self._lock:
                    while True:
                        now = time.perf_counter()
                        elapsed = now - self._last
                        self._last = now
                        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                        if self.tokens >= 1.0:
                            self.tokens -= 1.0
                            return
                        needed = (1.0 - self.tokens) / self.rate
                        await asyncio.sleep(max(0.0, needed))

        @_observe(name="edges.vmap")
        async def vmap_node(state: dict) -> dict:
            items = list(state.get(batch_key, []))
            if not items:
                return {out_key: []}

            sem = asyncio.Semaphore(max(1, max_concurrency))
            bucket = _TokenBucket(float(rps), int(burst)) if rps else None

            async def run_one(idx_item):
                idx, item = idx_item
                async with sem:
                    if bucket is not None:
                        await bucket.acquire()
                    partial = item_to_state(item)
                    call_state = {**state, **partial}
                    result = await _maybe_await(target(call_state))
                    return idx, result

            coros = [run_one(x) for x in enumerate(items)]
            results = await asyncio.gather(*coros)

            # Extract updates; keep ordering if requested
            if preserve_order:
                results.sort(key=lambda p: p[0])
            outputs = [upd for _, upd in results]
            return {out_key: outputs}

        return vmap_node

    if edge.kind == "reduce":
        cfg = edge.config
        input_key: str = cfg["input_key"]
        output_key: str = cfg["output_key"]
        reducer: Callable[[List[Any]], Any] = cfg["reducer"]

        @_observe(name="edges.reduce")
        def reduce_node(state: dict) -> dict:
            values = state.get(input_key, [])
            result = reducer(values)
            return {output_key: result}

        return reduce_node

    if edge.kind == "parallel":
        cfg = edge.config
        branches: List[Any] = list(cfg.get("branches", []))
        merge: Optional[Callable[..., Dict[str, Any]]] = cfg.get("merge")
        scheduler: str = str(cfg.get("scheduler", "parallel"))

        norm_branches: List[Callable[[dict], Any]] = []
        for br in branches:
            if isinstance(br, Edge):
                norm_branches.append(make_node(br))
            else:
                norm_branches.append(br)

        async def _merge_updates(updates: List[Dict[str, Any]]) -> Dict[str, Any]:
            if merge is not None:
                res = merge(*updates)
                return await _maybe_await(res)
            out: Dict[str, Any] = {}
            for upd in updates:
                out.update(upd or {})
            return out

        @_observe(name="edges.parallel")
        async def parallel_node(state: dict) -> dict:
            if scheduler == "sequential":
                updates: List[Dict[str, Any]] = []
                cur_state = dict(state)
                for br in norm_branches:
                    upd = await _maybe_await(br(cur_state))
                    upd = upd or {}
                    updates.append(upd)
                    cur_state = {**cur_state, **upd}
                return await _merge_updates(updates)
            else:
                coros = [_maybe_await(br(state)) for br in norm_branches]
                results = await asyncio.gather(*coros)
                updates = [r or {} for r in results]
                return await _merge_updates(updates)

        return parallel_node

    # parallel and jit are placeholders for now
    def unsupported_node(state: dict) -> dict:
        raise NotImplementedError(f"Edge kind not yet supported as node: {edge.kind}")

    return unsupported_node


 def _try_import_jax():
     try:
         import jax  # type: ignore
         import jax.numpy as jnp  # noqa: F401
         return jax
     except Exception:
         return None


 def jit_edge(
     fn: Callable[..., Any],
     *,
     in_keys: List[str],
     out_key: str,
     device: Optional[str] = None,
 ) -> Edge:
     """Optional JAX-jitted deterministic transform over parts of state.
     Placeholder: returns Edge descriptor only; JAX is optional.
     """
     jax = _try_import_jax()
     config: Dict[str, Any] = {"in_keys": in_keys, "out_key": out_key, "device": device}
     if jax is not None:
         config["jitted"] = True
     else:
         config["jitted"] = False
     return Edge("jit", config)


# ---- Policy router helper (placeholder for learnable edges) ----

def policy_router_select(
    prob_fn: Callable[[dict], Sequence[float]],
    labels: Sequence[str],
    *,
    strategy: str = "greedy",  # greedy | sample (future)
) -> Callable[[dict], str]:
    """Create a route selector function for add_conditional_edges.

    - prob_fn(state) -> probs over labels (same order)
    - returns a function(state) -> selected label

    Note: Placeholder implementation (no RNG/temperature). Intended as the
    surface for future JAX/RL-backed learnable routers.
    """
    label_list = list(labels)

    def route_fn(state: dict) -> str:
        probs = list(prob_fn(state))
        if not probs or len(probs) != len(label_list):
            raise ValueError("prob_fn must return probs for each label")
        # greedy selection for now
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return label_list[idx]

    return route_fn
