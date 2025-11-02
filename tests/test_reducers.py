 import mahsm as ma


def test_majority_vote_raw_values():
    reducer = ma.reducers.majority_vote()
    assert reducer(["a", "b", "a"]) == "a"


def test_majority_vote_by_key_returns_item():
    reducer = ma.reducers.majority_vote(key="label")
    items = [
        {"label": "x", "v": 1},
        {"label": "y", "v": 2},
        {"label": "x", "v": 3},
    ]
    picked = reducer(items)
    assert picked in items and picked["label"] == "x"


def test_score_and_select_max_key():
    reducer = ma.reducers.score_and_select(score_key="score", reverse=True)
    items = [
        {"txt": "a", "score": 0.1},
        {"txt": "b", "score": 0.9},
        {"txt": "c", "score": 0.2},
    ]
    picked = reducer(items)
    assert picked["txt"] == "b"


def test_judge_select_index():
    reducer = ma.reducers.judge_select(lambda items: 2)
    items = [1, 2, 3, 4]
    assert reducer(items) == 3
