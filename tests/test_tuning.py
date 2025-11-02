"""
Unit tests for mahsm.tuning - dataset transforms, apply helpers, and plan runner.
"""
import mahsm as ma
import types
import pytest


def test_tuning_namespace_exists():
    assert hasattr(ma, "tuning")
    assert isinstance(ma.tuning, types.ModuleType)


def test_to_sft_basic_and_filters():
    src = [
        {"prompt": "p1", "output": "o1", "reward": {"value": 0.1}},
        {"prompt": "p2", "output": "o2", "reward": {"value": 0.9}},
        {"prompt": "p2", "output": "o2_dup", "reward": {"value": 0.5}},  # duplicate prompt
        {"step": {"input": {"prompt": "p3"}, "output": {"text": "o3"}}, "reward": {"value": 0.3}},
    ]

    # Basic conversion
    ex = ma.tuning.to_sft(src)
    assert isinstance(ex, list)
    # dedupe drops duplicate p2
    prompts_join = "\n".join([e["text"] for e in ex])
    assert prompts_join.count("p2") == 1

    # Min reward filters out p1 and p3
    ex2 = ma.tuning.to_sft(src, min_reward=0.5)
    assert all("p1" not in e["text"] and "p3" not in e["text"] for e in ex2)

    # top_k keeps only highest reward (p2)
    ex3 = ma.tuning.to_sft(src, top_k=1)
    assert len(ex3) == 1
    assert "p2" in ex3[0]["text"]


def test_to_preferences_basic_and_filters():
    src = [
        {"prompt": "p1", "chosen": "a", "rejected": "b", "reward": {"value": 0.1}},
        {"prompt": "p2", "chosen": "a2", "rejected": "b2", "reward": {"value": 0.9}},
        {"prompt": "p2", "chosen": "a2x", "rejected": "b2x", "reward": {"value": 0.2}},
    ]

    out = ma.tuning.to_preferences(src)
    # dedupe drops duplicate p2
    prompts = [o["prompt"] for o in out]
    assert prompts.count("p2") == 1

    out2 = ma.tuning.to_preferences(src, min_reward=0.5)
    assert all(o["prompt"] != "p1" for o in out2)

    out3 = ma.tuning.to_preferences(src, top_k=1)
    assert len(out3) == 1 and out3[0]["prompt"] == "p2"


def test_to_dspy_examples_from_step_items():
    # Build minimal items resembling LangFuse normalization
    items = [
        {"step": {"input": {"prompt": "Q?"}, "output": {"text": "A!"}}},
        {"step": {"input": {"prompt": "Q2?"}, "output": {"text": "A2!"}}},
    ]

    class Sig(ma.Signature):
        question: str
        answer: str

    exs = ma.tuning.to_dspy_examples(items, signature=Sig, input_fields=["question"], output_field="answer")
    # Should produce dspy.Example instances
    assert len(exs) == 2
    # Examples carry the declared input fields
    assert hasattr(exs[0], "question") and hasattr(exs[0], "answer")


def test_apply_model_uri_parsing(monkeypatch):
    called = {}

    class DummyLM:
        def __init__(self, name, api_base=None, **kwargs):
            called["lm_name"] = name
            called["api_base"] = api_base

    class DummySettings:
        def configure(self, lm):
            called["configured"] = True

    import mahsm.dspy as dsd
    monkeypatch.setattr(dsd, "LM", DummyLM, raising=True)
    monkeypatch.setattr(dsd, "settings", DummySettings(), raising=True)

    art = ma.tuning.Artefact(kind="policy", uri="openai://gpt-5-mini@https://api.example.com/v1")
    ma.tuning.apply_model(art)

    assert called.get("configured") is True
    assert called.get("lm_name").endswith("gpt-5-mini")
    assert called.get("api_base") == "https://api.example.com/v1"


def test_plan_run_with_dummy_adapter(monkeypatch):
    # Build a fake adapter and check that apply_lora records the uri
    class FakeAdapter:
        def fit(self, dataset, config):
            return ma.tuning.Artefact(kind="lora", uri="/tmp/adapter")

    src = ma.tuning.Source("dataset://jsonl/does-not-exist.jsonl")
    # Monkeypatch resolve to return a simple in-memory dataset
    monkeypatch.setattr(ma.tuning, "resolve", lambda s: [{"prompt": "p", "output": "o"}], raising=True)

    phase = ma.tuning.Phase(
        name="sft",
        source=src,
        curate=ma.tuning.to_sft,
        optimize=FakeAdapter(),
        apply=ma.tuning.apply_lora,
    )

    # Reset active lora
    monkeypatch.setattr(ma.tuning, "_ACTIVE_LORA_URI", None, raising=False)
    ma.tuning.run(ma.tuning.Plan(phases=[phase]))
    assert ma.tuning._ACTIVE_LORA_URI == "/tmp/adapter"
