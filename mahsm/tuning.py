from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Callable, Literal, Optional, TypedDict, Any
from urllib.parse import urlparse, parse_qs


class Episode(TypedDict, total=False):
    id: str
    task: Optional[str]
    metadata: dict


class Step(TypedDict, total=False):
    idx: int
    input: dict
    output: dict
    tool_calls: list[dict]
    lat_ms: Optional[int]


class Reward(TypedDict, total=False):
    value: float
    source: str


class Event(TypedDict, total=False):
    episode_id: str
    step: Optional[Step]
    reward: Optional[Reward]
    tags: list[str]
    ts: float

def emit(event: Event) -> None:
    """Best-effort emission of an Event to LangFuse as an observation.

    - If LangFuse SDK or credentials are unavailable, this is a no-op.
    - Uses observation type 'SPAN' and stores event payload in input/output fields.
    """
    try:
        try:
            from langfuse import get_client  # type: ignore
            client = get_client()
        except Exception:
            from langfuse import Langfuse  # type: ignore
            client = Langfuse()
        # observation.create signature may vary across SDK versions; use kwargs defensively
        payload = {k: v for k, v in event.items() if k not in ("ts",)}
        ts = event.get("ts")
        kwargs = {
            "trace_id": event.get("episode_id"),
            "type": "SPAN",
            "name": "mahsm.event",
            "input": payload,
            "output": (event.get("reward") or {}),
        }
        if ts:
            kwargs["start_time"] = ts
        try:
            getattr(client.api.observation, "create")(**kwargs)  # type: ignore[attr-defined]
        except Exception:
            # Fallback: try trace.log if available
            try:
                getattr(client.api.trace, "log")(trace_id=event.get("episode_id"), data=payload)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # Swallow all errors to avoid breaking runtime
        return None


def to_sft(
    source: Any,
    *,
    template: str = "{prompt}\n{output}",
    dedupe: bool = True,
    top_k: Optional[int] = None,
    min_reward: Optional[float] = None,
) -> Iterable[dict]:
    """Convert items/traces into TRL SFT examples.

    Supports basic filters:
    - dedupe: drop duplicate prompts
    - top_k: keep top-k by reward (when available)
    - min_reward: drop items with reward below threshold (when available)
    """
    if not isinstance(source, list):
        return []
    items = list(source)

    def reward_of(it: dict) -> Optional[float]:
        r = it.get("reward") if isinstance(it, dict) else None
        if isinstance(r, dict):
            try:
                return float(r.get("value"))
            except Exception:
                return None
        return None

    # Filter by min_reward if provided
    if min_reward is not None:
        items = [it for it in items if (reward_of(it) or 0.0) >= float(min_reward)]

    # Prepare examples
    prepared: list[tuple[str, dict]] = []  # (prompt, example)
    for item in items:
        if isinstance(item, dict) and "text" in item:
            txt = str(item["text"])
            prepared.append((txt, {"text": txt}))
        elif isinstance(item, dict) and "prompt" in item and "output" in item:
            prompt = str(item["prompt"]) if item.get("prompt") is not None else ""
            output = str(item["output"]) if item.get("output") is not None else ""
            if prompt and output:
                text = template.format(prompt=prompt, output=output)
                prepared.append((prompt, {"text": text}))
        elif isinstance(item, dict) and "step" in item:
            step = item.get("step") or {}
            prompt = (step.get("input") or {}).get("prompt")
            output = (step.get("output") or {}).get("text")
            if prompt and output:
                text = template.format(prompt=prompt, output=output)
                prepared.append((str(prompt), {"text": text}))

    # Dedupe by prompt
    if dedupe:
        seen: set[str] = set()
        deduped: list[tuple[str, dict]] = []
        for k, v in prepared:
            if k in seen:
                continue
            seen.add(k)
            deduped.append((k, v))
        prepared = deduped

    # Rank by reward desc if available
    if top_k is not None and top_k > 0:
        # Attach rewards where possible; none -> -inf
        def key_fn(pair: tuple[str, dict]) -> float:
            # Find original item by prompt; approximate mapping
            prompt = pair[0]
            # Scan items to find first with this prompt and reward
            for it in items:
                if isinstance(it, dict):
                    p = None
                    if "prompt" in it:
                        p = it["prompt"]
                    elif "step" in it:
                        p = ((it.get("step") or {}).get("input") or {}).get("prompt")
                    if p == prompt:
                        val = reward_of(it)
                        return float(val) if val is not None else float("-inf")
            return float("-inf")

        prepared = sorted(prepared, key=key_fn, reverse=True)[: int(top_k)]

    return [ex for _, ex in prepared]


def to_preferences(
    source: Any,
    *,
    dedupe: bool = True,
    top_k: Optional[int] = None,
    min_reward: Optional[float] = None,
) -> Iterable[dict]:
    """Convert items into DPO/ORPO preference pairs.

    Items must contain keys: prompt, chosen, rejected.
    """
    if not isinstance(source, list):
        return []
    items = list(source)

    def reward_of(it: dict) -> Optional[float]:
        r = it.get("reward") if isinstance(it, dict) else None
        if isinstance(r, dict):
            try:
                return float(r.get("value"))
            except Exception:
                return None
        return None

    if min_reward is not None:
        items = [it for it in items if (reward_of(it) or 0.0) >= float(min_reward)]

    out: list[dict] = []
    for item in items:
        if all(k in item for k in ("prompt", "chosen", "rejected")):
            out.append({"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]})

    # Dedupe by prompt
    if dedupe:
        seen: set[str] = set()
        deduped: list[dict] = []
        for it in out:
            p = str(it.get("prompt"))
            if p in seen:
                continue
            seen.add(p)
            deduped.append(it)
        out = deduped

    # Top-k by reward if available
    if top_k is not None and top_k > 0:
        def key_fn(it: dict) -> float:
            # Find matching item to read reward
            p = it.get("prompt")
            for src in items:
                if isinstance(src, dict) and src.get("prompt") == p:
                    val = reward_of(src)
                    return float(val) if val is not None else float("-inf")
            return float("-inf")

        out = sorted(out, key=key_fn, reverse=True)[: int(top_k)]

    return out


def to_trajectories(source: Any, **kwargs) -> Iterable[dict]:
    if isinstance(source, list):
        return source
    return []


def to_dspy_examples(
    source: Any,
    signature: Any,
    input_fields: Optional[list[str]] = None,
    input_path: str = "step.input.prompt",
    output_field: str = "answer",
    output_path: str = "step.output.text",
) -> Iterable[Any]:
    """Convert resolved items (e.g., LangFuse observations) to Iterable[dspy.Example].

    - input_fields: DSPy input field names (default: ["question"]).
    - input_path/output_path: dot-paths into each item (defaults assume our normalized step shape).
    - output_field: DSPy output field name (default: "answer").
    """
    try:
        import dspy
    except ImportError as e:
        raise RuntimeError(f"to_dspy_examples requires DSPy installed: {e}")

    def get_path(obj: Any, path: str):
        cur = obj
        for part in path.split('.'):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    fields = input_fields or ["question"]
    out: list[Any] = []
    if isinstance(source, list):
        for item in source:
            inp_val = get_path(item, input_path)
            out_val = get_path(item, output_path)
            if inp_val is None or out_val is None:
                continue
            payload = {fields[0]: inp_val, output_field: out_val}
            ex = dspy.Example(**payload).with_inputs(*fields)
            out.append(ex)
    return out


def _ensure_sft_examples(dataset: Iterable) -> list[dict]:
    out = []
    for x in dataset:
        if isinstance(x, dict) and "text" in x:
            out.append({"text": x["text"]})
        elif isinstance(x, dict) and "prompt" in x and "output" in x:
            out.append({"text": f"{x['prompt']}\n{x['output']}"})
    return out


def _ensure_dpo_examples(dataset: Iterable) -> list[dict]:
    out = []
    for x in dataset:
        if isinstance(x, dict) and all(k in x for k in ("prompt", "chosen", "rejected")):
            out.append({"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]})
    return out


def pytest_guard(project: str, task: str, metric: str, threshold: float) -> Callable[[Any], bool]:
    def _guard(graph: Any) -> bool:
        from .testing import PytestHarness
        h = PytestHarness(graph)
        h.from_langfuse(project=project, task=task)
        rp = h.rollout_processor
        dl = h.data_loaders
        if hasattr(rp, "run"):
            res = rp.run(dl)
            val = getattr(res, "metrics", {}).get(metric)
            return bool(val is not None and val >= threshold)
        return True
    return _guard


def metric_guard(fn: Callable[[Any], float], threshold: float) -> Callable[[Any], bool]:
    def _guard(graph: Any) -> bool:
        return fn(graph) >= threshold
    return _guard


@dataclass
class Artefact:
    kind: Literal["lora", "full_weights", "prompt", "policy"]
    uri: str


class TrainerAdapter(Protocol):
    def fit(self, dataset: Iterable, config: dict) -> Artefact: ...


class TRL_SFT:
    def __init__(self, model: str, lora: bool = True, **kwargs):
        self.model = model
        self.lora = lora
        self.kwargs = kwargs

    def fit(self, dataset: Iterable, config: dict) -> Artefact:
        try:
            from datasets import Dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
            from trl import SFTTrainer
            try:
                from trl import SFTConfig
                sft_args = SFTConfig(
                    output_dir=self._out_dir("sft"),
                    num_train_epochs=int(config.get("epochs", 1)),
                    per_device_train_batch_size=int(config.get("batch_size", 1)),
                    logging_steps=int(config.get("logging_steps", 10)),
                    save_steps=int(config.get("save_steps", 0)),
                )
                use_sftconfig = True
            except Exception:
                from transformers import TrainingArguments as SFTConfig  # type: ignore
                sft_args = SFTConfig(
                    output_dir=self._out_dir("sft"),
                    num_train_epochs=int(config.get("epochs", 1)),
                    per_device_train_batch_size=int(config.get("batch_size", 1)),
                    logging_steps=int(config.get("logging_steps", 10)),
                    save_steps=int(config.get("save_steps", 0)),
                )
                use_sftconfig = False

            examples = _ensure_sft_examples(dataset)
            ds = Dataset.from_list(examples)
            tok = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(self.model)
            peft_cfg = None
            model_for_train = base
            if self.lora:
                peft_cfg = LoraConfig(r=int(config.get("lora_r", 8)), lora_alpha=int(config.get("lora_alpha", 16)), lora_dropout=float(config.get("lora_dropout", 0.05)))
                model_for_train = get_peft_model(base, peft_cfg)

            if use_sftconfig:
                trainer = SFTTrainer(model=model_for_train, tokenizer=tok, peft_config=peft_cfg, args=sft_args, train_dataset=ds, dataset_text_field="text")
            else:
                trainer = SFTTrainer(model=model_for_train, tokenizer=tok, args=sft_args, train_dataset=ds, dataset_text_field="text")

            trainer.train()
            out = Path(sft_args.output_dir)
            if self.lora:
                adapter_dir = out / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model_for_train.save_pretrained(str(adapter_dir))
                return Artefact(kind="lora", uri=str(adapter_dir))
            else:
                base.save_pretrained(str(out))
                tok.save_pretrained(str(out))
                return Artefact(kind="full_weights", uri=str(out))
        except ImportError as e:
            raise RuntimeError(f"Missing dependency for TRL SFT: {e}")


class TRL_DPO:
    def __init__(self, model: str, lora: bool = True, **kwargs):
        self.model = model
        self.lora = lora
        self.kwargs = kwargs

    def fit(self, dataset: Iterable, config: dict) -> Artefact:
        try:
            from datasets import Dataset
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model
            from trl import DPOTrainer
            examples = _ensure_dpo_examples(dataset)
            ds = Dataset.from_list(examples)
            tok = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(self.model)
            model_for_train = base
            peft_cfg = None
            if self.lora:
                peft_cfg = LoraConfig(r=int(config.get("lora_r", 8)), lora_alpha=int(config.get("lora_alpha", 16)), lora_dropout=float(config.get("lora_dropout", 0.05)))
                model_for_train = get_peft_model(base, peft_cfg)
            trainer = DPOTrainer(model=model_for_train, tokenizer=tok, train_dataset=ds, peft_config=peft_cfg)
            trainer.train()
            out = Path(self._out_dir("dpo"))
            if self.lora:
                adapter_dir = out / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model_for_train.save_pretrained(str(adapter_dir))
                return Artefact(kind="lora", uri=str(adapter_dir))
            else:
                base.save_pretrained(str(out))
                tok.save_pretrained(str(out))
                return Artefact(kind="full_weights", uri=str(out))
        except ImportError as e:
            raise RuntimeError(f"Missing dependency for TRL DPO: {e}")

    def _out_dir(self, tag: str) -> str:
        root = Path(os.environ.get("MAHSM_ARTIFACTS_DIR", ".mahsm/artefacts"))
        ts = int(time.time())
        d = root / f"trl_{tag}_{ts}"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)


class DSPY_Teleprompt:
    """Adapter for DSPy teleprompt/optimizer compilers (e.g., GEPA, BootstrapFewShot).

    Usage:
        DSPY_Teleprompt(student=my_dspy_module, teleprompter="GEPA", metric=my_metric_fn)
        .fit(trainset_of_dspy_examples, config={...})
    """

    def __init__(self, student: Any, teleprompter: str = "GEPA", **kwargs):
        self.student = student
        self.teleprompter = teleprompter
        self.kwargs = kwargs

    def fit(self, dataset: Iterable, config: dict) -> Artefact:
        try:
            import dspy
            # Expect dataset is Iterable[dspy.Example]
            trainset = list(dataset)
            tp_mod = None
            try:
                import dspy.teleprompt as tp
                tp_mod = getattr(tp, self.teleprompter)
            except Exception as e:
                raise RuntimeError(f"DSPy teleprompter '{self.teleprompter}' not found: {e}")
            compiler = tp_mod(**self.kwargs)
            optimized = compiler.compile(student=self.student, trainset=trainset, **config)
            # Persist lightweight artefact
            out_root = Path(os.environ.get("MAHSM_ARTIFACTS_DIR", ".mahsm/artefacts"))
            ts = int(time.time())
            out_dir = out_root / f"dspy_tp_{self.teleprompter.lower()}_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                import pickle
                with (out_dir / "program.pkl").open("wb") as f:
                    pickle.dump(optimized, f)
                return Artefact(kind="prompt", uri=str(out_dir / "program.pkl"))
            except Exception:
                # Fallback to no-op artefact directory
                return Artefact(kind="prompt", uri=str(out_dir))
        except ImportError as e:
            raise RuntimeError(f"DSPy is required for teleprompting: {e}")


_ACTIVE_LORA_URI: Optional[str] = None


def apply_lora(artefact: Artefact) -> None:
    global _ACTIVE_LORA_URI
    if artefact.kind != "lora":
        raise ValueError("apply_lora requires a 'lora' artefact")
    _ACTIVE_LORA_URI = artefact.uri


def apply_model(artefact: Artefact) -> None:
    uri = artefact.uri
    model: Optional[str] = None
    base: Optional[str] = None
    if uri.startswith("openai://"):
        rest = uri[len("openai://"):]
        if "@" in rest:
            model, base = rest.split("@", 1)
        else:
            model = rest
            base = os.environ.get("OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")
    elif uri.startswith("endpoint://"):
        # endpoint://<base>?model=<model>
        parsed = urlparse(uri.replace("endpoint://", "http://", 1))
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".replace("http://", "", 1)
        qs = parse_qs(parsed.query)
        model = (qs.get("model") or [None])[0]
    else:
        # raw model name; rely on OPENAI_BASE env
        model = uri
        base = os.environ.get("OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")

    if not model:
        raise ValueError("apply_model requires a model; use uri like openai://<model>@<base> or set OPENAI_BASE")
    try:
        import dspy
        lm_name = f"openai/{model}" if not model.startswith("openai/") else model
        if base:
            dspy.settings.configure(lm=dspy.LM(lm_name, api_base=base))
        else:
            dspy.settings.configure(lm=dspy.LM(lm_name))
    except ImportError as e:
        raise RuntimeError(f"apply_model requires dspy installed: {e}")


@dataclass
class Source:
    uri: str


@dataclass
class Phase:
    name: str
    source: Source
    curate: Callable[..., Iterable]
    optimize: TrainerAdapter
    apply: Callable[[Artefact], None]


@dataclass
class Plan:
    phases: list[Phase]
    guard: Optional[dict] = None


def resolve(source: Source) -> Any:
    uri = source.uri
    if uri.startswith("trace://langfuse/"):
        try:
            _, _, rest = uri.partition("trace://langfuse/")
            parts = [p for p in rest.split("/") if p]
            project = parts[0] if parts else None
            task = parts[1] if len(parts) > 1 else None
            return _langfuse_fetch(project=project, task=task, limit=int(os.environ.get("MAHSM_LANGFUSE_LIMIT", "500")))
        except Exception as e:
            raise RuntimeError(f"LangFuse fetch failed for {uri}: {e}")
    if uri.startswith("dataset://jsonl/"):
        path = uri.replace("dataset://jsonl/", "", 1)
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {p}")
        items: list[dict] = []
        import json
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        return items
    # Fallback: return as-is
    return source


def run(plan: Plan) -> None:
    for p in plan.phases:
        ds = p.curate(resolve(p.source))
        artefact = p.optimize.fit(ds, config={})
        p.apply(artefact)


class Collector:
    def __init__(self, graph: Any, **kwargs):
        self.graph = graph
        self.kwargs = kwargs

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def collect_from(graph: Any, **kwargs) -> Collector:
    return Collector(graph, **kwargs)


def _langfuse_fetch(project: Optional[str], task: Optional[str], limit: int = 500) -> list[dict]:
    """Fetch traces/observations from LangFuse via SDK; best-effort normalization.

    Returns a list of dicts; downstream to_sft/to_preferences try to extract fields.
    """
    client = None
    try:
        from langfuse import get_client  # type: ignore
        client = get_client()
    except Exception:
        try:
            from langfuse import Langfuse  # type: ignore
            client = Langfuse()
        except Exception as e:
            raise RuntimeError(
                "LangFuse SDK not available. Install 'langfuse' and set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST"
            ) from e

    items: list[dict] = []
    try:
        traces = getattr(client.api.trace, "list")(limit=limit)  # type: ignore[attr-defined]
        trace_list = traces.get("data", traces) if hasattr(traces, "get") else traces
        for tr in trace_list:
            if task:
                tags = tr.get("tags") or []
                name = tr.get("name") or ""
                if task not in tags and task not in name:
                    continue
            trace_id = tr.get("id") or tr.get("traceId")
            if not trace_id:
                continue
            try:
                obs = getattr(client.api.observation, "list")(trace_id=trace_id, limit=1000)  # type: ignore[attr-defined]
                obs_list = obs.get("data", obs) if hasattr(obs, "get") else obs
            except Exception:
                obs_list = []
            for ob in obs_list:
                inp = ob.get("input") or ob.get("inputText") or ob.get("prompt")
                out = ob.get("output") or ob.get("outputText") or ob.get("completion")
                step = {"input": {"prompt": inp}, "output": {"text": out}}
                items.append({"episode_id": trace_id, "step": step, "ts": ob.get("startTime") or ob.get("timestamp")})
    except Exception as e:
        raise RuntimeError(f"LangFuse SDK query failed: {e}")

    return items
