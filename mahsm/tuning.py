from __future__ import annotations
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
