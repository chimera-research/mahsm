 """
 mahsm.edges — scaffolding for Special Edges (vectorized map, parallel branches, reduce, optional jit)

 NOTE: This is scaffolding for experimentation and review. Implementations are placeholders.
 """

 from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import asyncio
import inspect


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

        async def vmap_node(state: dict) -> dict:
            items = list(state.get(batch_key, []))
            if not items:
                return {out_key: []}

            sem = asyncio.Semaphore(max(1, max_concurrency))

            async def run_one(idx_item):
                idx, item = idx_item
                async with sem:
                    if rps:
                        # naive pacing to avoid burst; coarse but predictable
                        await asyncio.sleep(1.0 / float(rps))
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

        def reduce_node(state: dict) -> dict:
            values = state.get(input_key, [])
            result = reducer(values)
            return {output_key: result}

        return reduce_node

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
