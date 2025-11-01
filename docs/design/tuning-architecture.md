# mahsm v0.3.0 — Tuning Architecture and API

Status: Draft for PR v0.3.0

## Executive Summary

mahsm v0.3.0 introduces a unified tuning pipeline that harmonizes prompt optimization and weight tuning under a single Phase/Plan abstraction:

- Data sources: LangFuse traces and JSONL datasets
- Dataset transforms: to_sft, to_preferences, to_trajectories
- Optimizers (adapters): TRL_SFT, TRL_DPO, DSPY_Teleprompt (e.g., GEPA)
- Apply: apply_lora (attach adapters), apply_model (switch LM endpoint in DSPy)

Design principles:
- Training-agent disaggregation: collect → curate → optimize → apply
- Keep existing integrations first-class: LangFuse (tracing), EvalProtocol (guards), DSPy (programs)
- Minimal vendor lock-in via adapter pattern and OpenAI-compatible endpoints

## Unified API Namespace (import mahsm as ma; use ma.tuning)

All tuning entities live under the `ma.tuning` namespace:

- Types: ma.tuning.Artefact, ma.tuning.Source, ma.tuning.Phase, ma.tuning.Plan
- Transforms: ma.tuning.to_sft, ma.tuning.to_preferences, ma.tuning.to_trajectories, ma.tuning.to_dspy_examples
- Optimizers: ma.tuning.TRL_SFT, ma.tuning.TRL_DPO, ma.tuning.DSPY_Teleprompt
- Runners: ma.tuning.run, ma.tuning.collect_from
- Apply: ma.tuning.apply_lora, ma.tuning.apply_model
- Guards: ma.tuning.pytest_guard, ma.tuning.metric_guard

Example:
```python
import mahsm as ma

phase = ma.tuning.Phase(
    name="gepa",
    source=ma.tuning.Source("trace://langfuse/my-project/my-task"),
    curate=lambda s: my_dspy_trainset,  # Iterable[dspy.Example]
    optimize=ma.tuning.DSPY_Teleprompt(student=my_program, teleprompter="GEPA"),
    apply=lambda artefact: None,
)
ma.tuning.run(ma.tuning.Plan(phases=[phase]))
```

## Architecture Overview

```mermaid
flowchart LR
  subgraph Sources
    LF[LangFuse traces\ntrace://langfuse/<project>/<task>]
    J[JSONL dataset\ndataset://jsonl/<path>]
  end

  subgraph Transforms
    SFT[to_sft]
    PREF[to_preferences]
    TRAJ[to_trajectories]
    DEX[to_dspy_examples]
  end

  subgraph Optimizers
    TRLSFT[TRL_SFT\nLoRA-first]
    TRLDPO[TRL_DPO\nLoRA-first]
    TP[DSPY_Teleprompt\n(GEPA, etc.)]
  end

  subgraph Apply
    AL[apply_lora]
    AM[apply_model\n(OpenAI-compatible)]
  end

  LF -->|resolve| SFT
  LF -->|resolve| PREF
  J  -->|resolve| SFT
  J  -->|resolve| PREF
  SFT --> TRLSFT --> AL
  PREF --> TRLDPO --> AL
  %% Teleprompt expects DSPy Examples
  LF -->|resolve| DEX --> TP --> AM
  J  -->|resolve| DEX --> TP --> AM

  classDef src fill:#eef,stroke:#55f;
  classDef tx fill:#efe,stroke:#5a5;
  classDef opt fill:#fee,stroke:#a55;
  classDef apply fill:#ffd,stroke:#aa5;
  class LF,J src;
  class SFT,PREF,TRAJ tx;
  class TRLSFT,TRLDPO,TP opt;
  class AL,AM apply;
```

### Phase/Plan

- Phase: name, source (URI), curate (transform), optimize (adapter), apply (callable)
- Plan: list[Phase], optional guard
- run(plan): executes phases in order: curate → optimize → apply

### Sources and Resolve

- LangFuse via SDK: `trace://langfuse/<project>/<task>` → normalized list of observation dicts
- JSONL datasets: `dataset://jsonl/<path>` → list[dict]

### Transforms

- to_sft: yields `{ "text": ... }` for TRL SFT
- to_preferences: yields `{ "prompt", "chosen", "rejected" }` for TRL DPO
- to_trajectories: pass-through (future RL use)

### Optimizers (Adapters)

- TRL_SFT: Hugging Face TRL SFTTrainer; LoRA-first, full weights optional
- TRL_DPO: Hugging Face TRL DPOTrainer; LoRA-first
- DSPY_Teleprompt: wraps dspy.teleprompt optimizers (e.g., GEPA). Expects `Iterable[dspy.Example]`.

### Apply

- apply_lora: records active LoRA adapter path for serving/attachment downstream
- apply_model: configures DSPy’s LM to an OpenAI-compatible endpoint
  - `openai://<model>@<base>`
  - `endpoint://<base>?model=<model>`
  - `<model>` with `OPENAI_BASE`/`OPENAI_API_BASE`

### Guards

- pytest_guard: invokes EvalProtocol via PytestHarness, asserts metric ≥ threshold
- metric_guard: simple callable threshold check

## Recommended Pipelines

1) Prompt-first, then weights
- GEPA (DSPY_Teleprompt) → apply_model
- TRL SFT (to_sft → TRL_SFT) → apply_lora
- TRL DPO (to_preferences → TRL_DPO) → apply_lora

2) SFT-only quickstart
- LangFuse traces → to_sft → TRL_SFT (LoRA) → apply_lora

## Example Snippets

### GEPA then SFT
```python
import mahsm as ma

# 1) Teleprompt (GEPA)
gepa_phase = ma.tuning.Phase(
    name="gepa",
    source=ma.tuning.Source("trace://langfuse/support/qa"),
    curate=lambda s: ma.tuning.to_dspy_examples(s, signature=MySignature, input_fields=["question"], output_field="answer"),
    optimize=ma.tuning.DSPY_Teleprompt(student=program, teleprompter="GEPA"),
    apply=lambda art: ma.tuning.apply_model(ma.tuning.Artefact(kind="policy", uri="openai://gpt-4o-mini@https://proxy/v1")),
)

# 2) SFT (LoRA)
sft_phase = ma.tuning.Phase(
    name="sft",
    source=ma.tuning.Source("trace://langfuse/support/qa"),
    curate=ma.tuning.to_sft,
    optimize=ma.tuning.TRL_SFT(model="mistralai/Mistral-7B-Instruct-v0.3", lora=True),
    apply=ma.tuning.apply_lora,
)

ma.tuning.run(ma.tuning.Plan(phases=[gepa_phase, sft_phase]))
```

## Testing and Quality

- Unit: dataset transforms (to_sft/to_preferences), adapter argument parsing
- Integration: LangFuse resolve with SDK (env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST)
- E2E (optional): minimal SFT loop on a toy dataset with tiny model

## Future Work

- Add LangFuse→dspy.Example helper (given a target Signature and label mapping)
- RL adapters (veRL/Agent Lightning) and apply_policy hot-swap
- Model/adapter serving integration (runtime attach of LoRA)

## Sources

1. Repository: `mahsm/tuning.py` (Phase/Plan, adapters, apply_*), `mahsm/__init__.py` (re-exports)
2. DSPy Teleprompt: https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt
3. TRL Library: https://github.com/huggingface/trl
4. LangFuse Python SDK: https://github.com/langfuse/langfuse-python
