 # Best-of-N with a Judge Reducer

 This example fans out N candidates and picks the best using a reducer.

 ```python
 import mahsm as ma
 from typing import TypedDict, List

 class State(TypedDict):
     prompt: str
     candidates: List[dict]
     best: dict

 def gen_candidate(state: dict) -> dict:
     i = state.get("i", 0)
     return {"text": f"candidate {i}", "score": float(i)}

 N = 5
 edge_map = ma.edges.vmap(
     target=gen_candidate,
     batch_key="items",
     item_to_state=lambda i: {"i": i},
     out_key="candidates",
 )

 reducer = ma.reducers.score_and_select(score_key="score", reverse=True)
 edge_reduce = ma.edges.reduce_edge(
     input_key="candidates", output_key="best", reducer=reducer
 )

 g = ma.graph.StateGraph(State)
 g.add_node("seed", lambda st: {"items": list(range(N))})
 g.add_node("gen", ma.edges.make_node(edge_map))
 g.add_node("pick", ma.edges.make_node(edge_reduce))
 g.add_edge(ma.START, "seed")
 g.add_edge("seed", "gen")
 g.add_edge("gen", "pick")
 g.add_edge("pick", ma.END)

 graph = g.compile()
 out = graph.invoke({"prompt": "test"})
 assert out["best"]["score"] == float(N - 1)
 ```

 To use an LLM judge, write a `judge_fn(items) -> index` and wrap with `ma.reducers.judge_select(judge_fn)`.

## See also

- [Parameter Sweeps & Comparisons](parameter-sweeps.md)
- [Testing & Evaluation](evaluation-testing.md)
