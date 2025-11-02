 """
 Built-in reducers for mahsm special edges.

 All reducers here return callables that accept a List[Any] and produce a single value.
 """

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from collections import Counter


def _access(item: Any, key: Optional[str], accessor: Optional[Callable[[Any], Any]]):
    if accessor is not None:
        return accessor(item)
    if key is not None:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key)
    return item


def majority_vote(*, key: Optional[str] = None, accessor: Optional[Callable[[Any], Any]] = None) -> Callable[[List[Any]], Any]:
    """Return the most common label/value in a list.

    - If key/accessor provided, vote over that projection, return the winning original item.
    - Otherwise, vote directly over items and return the winning value.
    """

    def reducer(items: List[Any]) -> Any:
        if not items:
            return None
        if key is None and accessor is None:
            return Counter(items).most_common(1)[0][0]
        counts: Dict[Any, int] = {}
        for it in items:
            val = _access(it, key, accessor)
            counts[val] = counts.get(val, 0) + 1
        # winning value
        win_val = max(counts, key=lambda v: counts[v])
        # return first item matching winning value
        for it in items:
            if _access(it, key, accessor) == win_val:
                return it
        return None

    return reducer


def score_and_select(
    *,
    score_key: Optional[str] = None,
    score_fn: Optional[Callable[[Any], float]] = None,
    reverse: bool = True,
) -> Callable[[List[Any]], Any]:
    """Select item with best score.

    - score_key: pick from dict/attr
    - score_fn: compute score from item
    - reverse: True for max, False for min
    """

    def score(item: Any) -> float:
        if score_fn is not None:
            return float(score_fn(item))
        val = _access(item, score_key, None)
        return float(val) if val is not None else float("-inf")

    def reducer(items: List[Any]) -> Any:
        if not items:
            return None
        return sorted(items, key=score, reverse=reverse)[0]

    return reducer


def judge_select(judge_fn: Callable[[List[Any]], int]) -> Callable[[List[Any]], Any]:
    """Select item by index returned from judge_fn(items) -> index."""

    def reducer(items: List[Any]) -> Any:
        if not items:
            return None
        idx = int(judge_fn(items))
        if 0 <= idx < len(items):
            return items[idx]
        return None

    return reducer
