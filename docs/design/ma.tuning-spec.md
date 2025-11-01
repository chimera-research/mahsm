 # mahsm.tuning — A single-module abstraction for SFT/DPO/RL/LoRA

 Status: Draft (for initial PR)

 Owners: @SohumKothavade

 Last updated: 2025-11-01

 ---

 ## Executive summary

 Goal: Add one concise, universal module — `mahsm.tuning` — that lets any mahsm agent participate in post‑training (SFT, DPO, RL) with adapter publication (LoRA or full weights), while keeping the runtime fast and simple. We adopt training–agent disaggregation: the agent runtime emits learning events; a remote trainer consumes events and returns artefacts via an OpenAI‑compatible endpoint. This follows the pattern demonstrated by Microsoft Agent Lightning (trainer/agent disaggregation with OpenAI‑style serving) and Arbor (DSPy‑centric RL server) [1][2][3].

 Design anchors:
 - Stable abstraction is the event/trajectory schema, not any single algorithm [1][3][4].
 - One “stage” contract for all methods: collect → curate → optimize → apply.
 - Trainer adapters (TRL, Agent Lightning/veRL, Arbor) implement a tiny `fit(dataset, config) -> artefact` API.
 - Publishing is pluggable (LoRA adapter, new model version, policy swap).

 Non‑goals (initial PR): implement full PPO/GRPO in‑process, build a bespoke trace store, or replace LangFuse.

 ---

 ## Reusing existing mahsm integrations (zero extra plumbing)

 - Traces: we keep using LangFuse; `tuning.emit()` serializes to LangFuse so downstream is unchanged [7][8].
 - Evals: reuse `mahsm.testing.PytestHarness` (EvalProtocol) to pull traces/datasets and run graph rollouts as guards.

 Example guard flow tying evals into a stage:

 ```python
 from mahsm import testing
 from mahsm import tuning as mt

 plan = mt.Plan(...)

 def guard_ok(graph, metrics_req: dict) -> bool:
     harness = testing.PytestHarness(graph)
     harness.from_langfuse(project="proj1", task="codegen")
     # Run rollouts via EvalProtocol; compute metrics of interest (pseudo)
     results = harness.rollout_processor.run(harness.data_loaders)
     return results.metrics >= metrics_req

 # Inside mt.run(): after s.apply(artefact), call guard_ok(graph, plan.guard)
 ```

 This keeps LangGraph/DSPy/ LangFuse/EvalProtocol exactly as‑is; `mahsm.tuning` just coordinates stages.

 ---

 ## Why this now

 - State of the art converges on: (a) decoupled runtime vs trainer, (b) unified MDP‑like trace schema, (c) OpenAI‑style serving of the optimized policy [1][3][4].
 - Existing trainers (TRL for SFT/DPO; veRL/LightningRL for RL; Arbor for DSPy programs) can be consumed through thin adapters rather than re‑implemented [2][4][5][6].

 ---

 ## Architecture (three planes)

 ```mermaid
 graph LR
   subgraph Runtime plane (mahsm graphs)
     A[Agent graph (LangGraph + DSPy)] -- emits --> E((Learning events))
     A -- queries --> Svc(OpenAI‑ish Inference Endpoint)
   end

   subgraph Data/trace plane
     E
     Store[(Trace store e.g., LangFuse)]
     E -- normalized schema --> Store
   end

   subgraph Learning plane (remote trainer)
     T[Trainer adapters: TRL / Agent Lightning(veRL) / Arbor]
     T -- read --> Store
     T -- artefacts --> Pub[(Artefacts: LoRA, weights)]
     Pub -- served as --> Svc
   end
 ```

 Consequence: the agent stays responsive; training scales independently; algorithms are hot‑swappable.

 ---

 ## Disaggregated trainer–agent: what actually happens

 Steps (typical online RL loop):
 1) Runtime executes graph; `tuning.emit()` logs step/episode events to LangFuse.
 2) Trainer service (e.g., VERL/Lightning) reads trajectories from LangFuse (or a mirrored store) [1][4].
 3) Trainer optimizes (GRPO/PPO/DAPO/…); outputs an artefact (LoRA or weights) [4].
 4) Publisher exposes the artefact behind an OpenAI‑compatible endpoint (could be vLLM/SGLang) [1][4].
 5) Runtime continues calling the same endpoint; it now serves updated weights. No runtime code changes.

 Sequence diagram:

 ```mermaid
 sequenceDiagram
   participant User
   participant Runtime as mahsm runtime (LangGraph+DSPy)
   participant LangFuse as Trace Store
   participant Trainer as Trainer (veRL/Lightning/TRL+server)
   participant Inference as OpenAI‑style Inference

   User->>Runtime: invoke graph(input)
   Runtime->>LangFuse: emit(Event: step/start/tool/reward)
   Runtime->>Inference: POST /chat/completions
   Inference-->>Runtime: completion(tokens)
   Note over Runtime,LangFuse: episode completes
   Trainer->>LangFuse: query trajectories
   Trainer->>Trainer: optimize (GRPO/PPO/DPO/SFT)
   Trainer-->>Inference: publish artefact (LoRA/weights)
   User->>Runtime: next request (same code)
   Runtime->>Inference: POST /chat/completions
   Inference-->>Runtime: improved policy
 ```

 Offline SFT/DPO is identical except source is historic traces/datasets and trainer runs batch jobs (TRL) [5][6][7].

 ---

 ## Minimal surface area (single module)

 Add `mahsm/tuning.py` with four concepts and a small public API. No new top‑level packages.

 ### 1) Event schema (stable core)

 A minimal MDP-ish schema sufficient for SFT/DPO/RL collection.

 ```python
 # sketch only — implemented as dataclasses or TypedDicts
 class Episode: id: str; task: str|None; metadata: dict
 class Step: idx: int; input: dict; output: dict; tool_calls: list[dict]; lat_ms: int|None
 class Reward: value: float; source: str  # e.g., "user", "auto-metric", "rm"
 class Event: episode_id: str; step: Step|None; reward: Reward|None; tags: list[str]; ts: float
 ```

 Emission policy:
 - Node start, node end, tool call(s), final answer, score/reward events.
 - API: `ma.tuning.emit(event)` and convenience wrappers for common node hooks.

 Mapping to LangFuse:
 - Provide a serializer: `Event -> LangFuse trace/observation` and back, keeping all downstream datasets uniform [7][8].

 ### 2) Dataset transforms

 Three canonical transforms from events or traces:
 - `to_sft(events|trace_query, *, template=...) -> Iterable[SFTExample]`
 - `to_preferences(events|trace_query, *, pair_by=..., filters=...) -> Iterable[DPOExample]`
 - `to_trajectories(events|stream, *, window=episode|n_steps, filters=...) -> Iterable[Trajectory]`

 Each supports simple, composable filters: top‑k by reward, dedupe identical prompts, drop unsafe, tool‑success‑only, etc.

 ### 3) Trainer adapters

 Unify all algorithms behind a single protocol.

 ```python
 class Artefact:
     kind: Literal["lora", "full_weights", "prompt", "policy"]
     uri: str  # where to fetch it

 class TrainerAdapter(Protocol):
     def fit(self, dataset: Iterable, config: dict) -> Artefact: ...

 # Built‑ins (thin wrappers)
 TRL_SFT, TRL_DPO, VERL_RL, Arbor_RL
 ```

 - TRL: calls HF TRL’s SFT/DPO trainers [5][6].
 - VERL/Lightning: consumes trajectories, supports disaggregated rollout; expects OpenAI‑ish serving in front [4].
 - Arbor: remote DSPy optimizer server; feeds DSPy programs/episodes [2].

 ### 4) Publishing (apply)

 Minimal application targets:
 - `apply_lora(artefact)` — attach LoRA to the active LM in DSPy (`dspy.settings.configure(lm=...)`).
 - `apply_model(artefact)` — switch to a new base model/endpoint.
 - `apply_policy(artefact)` — hot‑swap a graph policy (advanced/future).

 The apply functions use Mahsm’s existing DSPy/graph wrappers; no trainer coupling.

 ---

## One universal “phase” abstraction

 A stage composes: source → curate → optimize → apply.

 ```python
 @dataclass
 class Source:
     uri: str  # e.g., live://graph/<id>, trace://langfuse/<project>/<task>, dataset://hf/<name>, replay://buffer

 @dataclass
class Phase:
     name: str
     source: Source                      # where experience comes from
     curate: Callable[..., Iterable]     # to_sft / to_preferences / to_trajectories
     optimize: TrainerAdapter            # SFT / DPO / RL (via adapter)
     apply: Callable[[Artefact], None]   # adapter://lora, model://version, policy://swap

 @dataclass
class Plan:
     stages: list[Stage]
     guard: dict | None = None  # simple acceptance criteria, e.g., min metrics

def run(plan: Plan):
    for p in plan.phases:
        ds = p.curate(resolve(p.source))
        artefact = p.optimize.fit(ds, config={})
        p.apply(artefact)
 ```

This keeps all methods uniform. SFT is just `trace -> to_sft -> TRL_SFT -> apply_lora`. Online RL is `live -> to_trajectories -> VERL_RL -> apply_lora`.

 ---

 ## Algorithm catalog (what each needs and how it plugs in)

 | Algorithm | Family | Needs | Online/Offline | Typical adapter | Publish | Notes |
 |---|---|---|---|---|---|---|
 | SFT | Supervised | (prompt, output) pairs | Offline | TRL SFT [6] | LoRA/full | Easiest bootstrap from traces.
 | DPO | Preferences | (prompt, chosen, rejected) | Offline | TRL DPO [5][6] | LoRA/full | Direct preference optimization; no RM.
 | ORPO | Preferences | (prompt, chosen, rejected) | Offline | TRL ORPO [9] | LoRA/full | Odds‑ratio variant; stable.
 | IPO | Preferences | (prompt, chosen, rejected) | Offline | TRL IPO [9] | LoRA/full | Inverse preference opt.
 | KTO | Preferences | (prompt, output, utility) | Offline | TRL KTO [10] | LoRA/full | Prospect‑theory‑inspired.
 | SimPO | Preferences | (prompt, chosen, rejected) | Offline | TRL SimPO [9] | LoRA/full | Simple preference opt.
 | PPO | RL | trajectories + reward fn/RM | Online/Offline | TRL PPO [11], VERL PPO [4] | LoRA/full | Classic on‑policy RL.
 | GRPO | RL | trajectories + relative rewards | Online | VERL/Lightning [4], ART [12] | LoRA/full | Efficient group‑relative PPO.
 | RLOO | RL | trajectories + rewards | Online | TRL RLOO [9] | LoRA/full | Leave‑one‑out RL.
 | XPO | RL | trajectories + rewards | Online | TRL XPO [9] | LoRA/full | Cross‑policy opt.
 | Online DPO | Preferences | streaming prefs | Online | TRL Online DPO [9] | LoRA/full | Preference online.
 | NashMD | RL | multi‑agent trajectories | Online | TRL NashMD [9] | LoRA/full | Nash mean‑field dynamics.
 | DAPO | RL | trajectories + action advantage | Online | VERL DAPO [4] | LoRA/full | Advantage‑based.
 | PRM / RewardModel | RM | (prompt, chosen, rejected) | Offline | TRL Reward/PRM [9] | n/a | Trains RM for RLHF.
 | BCO | Offline RL | logged behavior | Offline | TRL BCO [9] | LoRA/full | Behavioral cloning variants.
 | CPO | Constrained RL | trajectories + constraints | Offline | TRL CPO [9] | LoRA/full | Constraint satisfaction.

 All of these reduce to choosing: source (traces/live) → curate (to_sft/to_preferences/to_trajectories) → optimize (adapter) → apply (LoRA/model).

 ---

 ## Public API sketch (what users write)

 ```python
 import mahsm as ma
 from mahsm import tuning as mt

 # 1) Wrap existing graph (no changes to nodes)
 graph = build_graph_somewhere()

 # 2) Start event collection (LangFuse is already set up in ma.tracing)
 collector = mt.collect_from(graph, project="proj1", task="codegen")

# 3) Define a phased plan
 plan = mt.Plan(
    phases=[
        mt.Phase(
             name="bootstrap-sft",
             source=mt.Source("trace://langfuse/proj1/codegen"),
             curate=mt.to_sft,
             optimize=mt.TRL_SFT(model="Qwen/Qwen2.5-7B", lora=True),
             apply=mt.apply_lora,
         ),
        mt.Phase(
             name="align-dpo",
             source=mt.Source("trace://langfuse/proj1/codegen"),
             curate=mt.to_preferences,
             optimize=mt.TRL_DPO(model="Qwen/Qwen2.5-7B", lora=True),
             apply=mt.apply_lora,
         ),
        mt.Phase(
             name="online-rl",
             source=mt.Source("live://graph/codegen"),
             curate=mt.to_trajectories,
             optimize=mt.VERL_RL(endpoint="http://trainer:8080"),
             apply=mt.apply_lora,
         ),
     ],
     guard={"qa.accuracy": 0.9},
 )

 mt.run(plan)
 ```

 ---

 ## Integration points in mahsm today

 - Graph + nodes: `mahsm.core.dspy_node` already unifies node IO; tuning can wrap these with event emit hooks.
 - Tracing: `mahsm.tracing` (LangFuse) provides spans; tuning serializes its `Event` schema to the same backend [7].
 - Testing/evals: `mahsm.testing.PytestHarness` can be the guard runner, logging back via LangFuse.

 Minimal code changes to existing modules: none. `mahsm.tuning` imports `mahsm.graph`, `mahsm.dspy`, and optional `mahsm.tracing`.

 ---

 ## MVP scope (first PR)

 1) Ship `mahsm/tuning.py` with:
    - Event datatypes + `emit()` + LangFuse serializers.
    - `to_sft`, `to_preferences`, `to_trajectories` with basic filters.
    - Adapters: `TRL_SFT`, `TRL_DPO` (local, single‑GPU happy path; LoRA via PEFT). Config dict passthrough.
    - `apply_lora` (DSPy LM adaptor swap) and `apply_model` (endpoint swap).
    - `Stage`, `Plan`, and `run()`.

 2) Example in `docs/getting-started/`: “Tune your existing mahsm agent in 20 lines”.

 3) Optional: a tiny `LangFuse -> SFT examples` cookbook.

 Out of scope for MVP: online RL training loop, remote rollout orchestration. Those come via adapters next.

 ---

 ## Adapters vs. in‑house implementations

 Positioning: start with adapters (fast, low maintenance, inherit upstream advances). As we standardize our event/dataset contracts and find gaps, we can internalize stable algorithms (copy/port) behind the same `TrainerAdapter` interface without breaking users.

 ---

 ## Phase 2 (follow‑up PRs)

 - Adapter: `VERL_RL` (consumes trajectories, talks to a remote trainer; OpenAI‑compatible serving) [4].
 - Adapter: `Arbor_RL` (connect to Arbor server; DSPy program optimization) [2].
 - Reward model and auto‑metrics wiring (generative RM prototypes; or use existing evaluator outputs) [4].
 - Policy hot‑swap for LangGraph (advanced publisher).

 ---

 ## Design decisions and rationale

 - Event schema first: Algorithms churn; trajectories (episode → steps → rewards → metadata) are the durable contract [1][3][4].
 - Single “stage” abstraction: reduces all tuning to four verbs; easiest conceptual on‑ramp.
 - Adapters over implementations: leverage TRL/veRL/Arbor and inherit their improvements [2][4][5].
 - Disaggregation by default: keeps mahsm runtime responsive; trainers scale independently [1][3][4].

 ---

 ## Open questions for confirmation

 1) Default trace store: standardize on LangFuse for now (yes/no)? If no, ship a neutral JSONL store as fallback.
 2) First adapters to land: TRL SFT+DPO (MVP), then VERL RL, then Arbor — agree?
 3) Publish targets: prioritize LoRA for size/mobility; defer full‑weights? Any preferred PEFT backend/models?
 4) Guard API: reuse `mahsm.testing` evaluators, or keep a lightweight metric callback in `tuning.run()`?

 ---

 ## References

 [1] Agent Lightning: Train ANY AI Agents with Reinforcement Learning (arXiv, 2025) — https://arxiv.org/abs/2508.03680

 [2] Ziems/arbor: A framework for optimizing DSPy programs with RL — https://github.com/ziems/arbor

 [3] Agent Lightning project & docs — https://github.com/microsoft/agent-lightning and https://microsoft.github.io/agent-lightning/latest/

 [4] veRL/VERL (Volcano Engine RL for LLMs) — https://github.com/volcengine/verl and release notes — https://github.com/volcengine/verl/releases

 [5] Hugging Face TRL (SFT/DPO/RL) — https://github.com/huggingface/trl

 [6] TRL documentation: SFT Trainer and DPO Trainer — https://huggingface.co/docs/trl/en/sft_trainer and https://huggingface.co/docs/trl/en/dpo_trainer

 [7] LangFuse data/API: query traces via SDKs — https://langfuse.com/docs/api-and-data-platform/features/query-via-sdk

 [8] LangFuse export options — https://langfuse.com/docs/api-and-data-platform/features/export-from-ui

 [9] TRL index (algorithms overview) — https://huggingface.co/docs/trl/en/index

 [10] TRL KTO Trainer — https://huggingface.co/docs/trl/main/en/kto_trainer

 [11] TRL PPO Trainer — https://huggingface.co/docs/trl/main/en/ppo_trainer

 [12] OpenPipe ART (Agent Reinforcement Trainer, GRPO for agents) — https://github.com/OpenPipe/ART
