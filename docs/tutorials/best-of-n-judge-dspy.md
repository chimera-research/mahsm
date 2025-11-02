# Best-of-N with a DSPy-backed Judge (concept)

You can use a small DSPy module as a judge to pick the best candidate by returning an index.

```python
import mahsm as ma

class Judge(ma.Module):
    def forward(self, choices):
        # Replace with a real DSPy comparator; here we pick longest text
        best_idx = max(range(len(choices)), key=lambda i: len(choices[i]["text"]))
        class R:
            def __init__(self):
                self.index = best_idx
        return R()

judge_mod = Judge()

def judge_fn(items):
    r = judge_mod(choices=items)
    return int(r.index)

reducer = ma.reducers.judge_select(judge_fn)
```

Then plug `reducer` into `edges.reduce_edge`. With DSPy LMs configured, the judge can be a proper LLM evaluator.
