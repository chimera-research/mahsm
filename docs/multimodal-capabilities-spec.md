# mahsm Multimodal Capabilities Specification
## Vision-First Expansion for Agentic Research

**Version:** 1.0  
**Date:** November 2, 2025  
**Focus:** Vision/Image capabilities as the foundation for multimodal agentic research

---

## Executive Summary

This specification outlines the comprehensive strategy for expanding **mahsm** into a first-class library for multimodal agentic research, with an initial focus on vision/image capabilities. The research reveals that all four core pillars of mahsm (DSPy, LangGraph, LangFuse, EvalProtocol) have substantial multimodal primitives in place, positioning mahsm to integrate these capabilities into a unified, declarative framework.

**Key Finding:** The underlying libraries provide robust multimodal support, but they currently operate in silos. mahsm's value proposition is to unify these capabilities under its declarative, orchestration-first architecture—making vision-language model (VLM) development as seamless as current text-based LLM workflows.

---

## 1. DSPy Multimodal Capabilities

### 1.1 Current State: DSPy.Image

**Status:** ✅ Beta support available (added Q4 2024)

DSPy now includes native multimodal support through the `dspy.Image` type, enabling VLM integration with minimal code changes.

#### Core Primitives

**`dspy.Image` Class:**
```python
import dspy

# Three instantiation methods
image = dspy.Image.from_file("path/to/image.png")
image = dspy.Image.from_url("https://example.com/image.jpg")
image = dspy.Image.from_PIL(pil_image_object)
```

**Key Features:**
- Automatic base64 encoding for API compatibility
- Supports PNG, JPG, JPEG, WEBP formats
- Works with local files, URLs, and PIL Image objects
- Compatible with OpenAI, Anthropic, and other VLM providers

#### Signature Integration

DSPy signatures seamlessly support image inputs:

```python
class ImageAnalysis(dspy.Signature):
    """Analyze image content and extract relevant information."""
    image: dspy.Image = dspy.InputField(desc="Image to analyze")
    query: str = dspy.InputField(desc="Question about the image")
    answer: str = dspy.OutputField(desc="Answer based on visual content")

# Usage
analyzer = dspy.Predict(ImageAnalysis)
result = analyzer(image=dspy.Image.from_file("photo.jpg"), 
                  query="What objects are in this image?")
```

#### VLM Module Support

**Compatible Modules:**
- `dspy.Predict` - Direct VLM inference
- `dspy.ChainOfThought` - Multi-step visual reasoning
- `dspy.ReAct` - Vision-enabled agentic reasoning
- `dspy.ProgramOfThought` - Visual + computational reasoning

**Example: Visual Question Answering:**
```python
class VQA(dspy.Signature):
    """Answer questions about image content."""
    image: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step-by-step analysis")
    answer: str = dspy.OutputField()

vqa_module = dspy.ChainOfThought(VQA)
```

### 1.2 DSPy Optimization for VLMs

**Critical Gap Identified:** Current DSPy optimizers (Teleprompters) have **limited multimodal optimization support**. Most optimizers are designed for text-only workflows.

#### Current Optimizer Compatibility

| Optimizer | Text Support | Multimodal Support | Notes |
|-----------|-------------|-------------------|-------|
| `BootstrapFewShot` | ✅ Full | ⚠️ Limited | Can bootstrap with image examples, but no vision-specific metrics |
| `MIPRO` | ✅ Full | ❌ Not yet | Instruction optimization text-only |
| `SignatureOptimizer` | ✅ Full | ⚠️ Experimental | Can optimize signatures with image fields |
| `BootstrapFinetune` | ✅ Full | ❌ Not yet | Fine-tuning not yet VLM-compatible |

#### Recommended mahsm Integration Strategy

**Phase 1: Signature-Level Support**
- Extend mahsm's `@ma.dspy_node` decorator to handle `dspy.Image` fields automatically
- Create mahsm-specific wrappers for image preprocessing in graph state

**Phase 2: VLM-Specific Optimizers**
- Develop `ma.optimizers.VLMBootstrap` for few-shot learning with vision examples
- Create visual metric functions for VLM evaluation
- Support multi-modal in-context learning

### 1.3 Integration Recommendations for mahsm

```python
# Proposed mahsm API
import mahsm as ma

class ImageAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("image: dspy.Image, query -> analysis, confidence")
    
    def forward(self, image, query):
        return self.analyze(image=image, query=query)

# mahsm decorator with automatic image handling
@ma.dspy_node(image_fields=["screenshot"])
class VisualAgent(dspy.Module):
    def forward(self, screenshot: dspy.Image, task: str):
        # mahsm handles image serialization in graph state
        pass
```

---

## 2. LangGraph Multimodal Workflows

### 2.1 Current State

**Status:** ✅ Full native support for multimodal state

LangGraph's `StateGraph` architecture is **inherently multimodal-ready**—it can store and pass any Python object, including images, through the graph's shared state.

#### Native Multimodal State Support

```python
from typing import TypedDict
from langgraph.graph import StateGraph
import dspy

class MultimodalState(TypedDict):
    # Text data
    query: str
    response: str
    
    # Image data
    screenshot: dspy.Image
    processed_image: dspy.Image
    
    # Mixed data
    visual_annotations: list[dict]
    reasoning_trace: str
```

### 2.2 Multimodal Agent Patterns

LangGraph excels at orchestrating complex multimodal workflows:

**Pattern 1: Vision → Language → Action**
```python
def capture_image(state):
    # Capture screenshot or load image
    state["screenshot"] = dspy.Image.from_file("screen.png")
    return state

def analyze_visual(state):
    # VLM processes image
    vlm = dspy.Predict("image, query -> description")
    result = vlm(image=state["screenshot"], query=state["query"])
    state["description"] = result.description
    return state

def generate_action(state):
    # LLM decides action based on visual description
    state["action"] = decide_action(state["description"])
    return state

# Build graph
graph = StateGraph(MultimodalState)
graph.add_node("capture", capture_image)
graph.add_node("analyze", analyze_visual)
graph.add_node("act", generate_action)
graph.add_edge("capture", "analyze")
graph.add_edge("analyze", "act")
```

**Pattern 2: Iterative Vision Refinement**
```python
def should_refine_visual(state) -> str:
    if state["confidence"] < 0.8:
        return "refine"
    return "complete"

graph.add_conditional_edges("analyze", should_refine_visual, {
    "refine": "capture",  # Re-capture with different parameters
    "complete": END
})
```

### 2.3 mahsm-Specific Enhancements

**Proposed Features:**

1. **Visual State Tracing**
   - Automatic image storage in LangFuse-compatible format
   - Inline rendering of images in mahsm trace viewer

2. **Vision Node Decorators**
   ```python
   @ma.vision_node(input_image="screenshot", output_image="annotated")
   def annotate_image(state):
       # mahsm handles image serialization
       pass
   ```

3. **Multimodal Graph Compilation**
   ```python
   # Enhanced graph.compile() with vision support
   app = graph.compile(
       checkpointer=ma.checkpointer,
       vision_backend="langfuse",  # Auto-upload images
       image_compression="auto"     # Optimize storage
   )
   ```

### 2.4 Example Use Cases

**Use Case 1: Multi-Step Visual Reasoning**
- Screenshot analysis → Object detection → Text extraction → Decision making

**Use Case 2: Human-in-the-Loop Vision Annotation**
- VLM generates initial annotations → Human review → Refinement loop

**Use Case 3: Multi-Agent Vision Collaboration**
- Agent 1: Object detection
- Agent 2: Scene understanding
- Agent 3: Action planning
- Orchestrator: Combines insights

---

## 3. LangFuse Tracing for Multimodal Models

### 3.1 Current State

**Status:** ✅ Full production support (launched November 2024)

LangFuse provides **comprehensive multimodal tracing** including automatic handling of images, audio, and attachments.

#### Supported Media Types

| Type | Formats | Max Size | Notes |
|------|---------|----------|-------|
| Images | PNG, JPG, WEBP | 10MB | Inline rendering in UI |
| Audio | MP3, WAV, MPEG | 10MB | Inline playback |
| Attachments | PDF, TXT | 25MB | Download links |

### 3.2 Automatic Base64 Handling

LangFuse SDKs **automatically detect and extract** base64-encoded images from VLM payloads:

```python
from langfuse import Langfuse
from openai import OpenAI

langfuse = Langfuse()
client = OpenAI()

# Automatic image extraction
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    }]
)
# LangFuse automatically uploads image to object storage
# and displays inline in trace viewer
```

### 3.3 Custom Media Handling

For non-standard scenarios, use `LangfuseMedia`:

```python
from langfuse import Langfuse
from langfuse.media import LangfuseMedia

langfuse = Langfuse()

# Custom upload
media = LangfuseMedia(
    obj=image_bytes,
    content_type="image/png"
)

trace = langfuse.trace(
    name="visual_analysis",
    input={"image": media, "query": "Analyze this"},
    output={"result": "..."}
)
```

### 3.4 mahsm Integration Strategy

**Proposed `ma.init()` Enhancement:**

```python
# Current mahsm initialization
ma.init(langfuse_public_key="...", langfuse_secret_key="...")

# Enhanced for multimodal
ma.init(
    langfuse_public_key="...",
    langfuse_secret_key="...",
    vision_tracking=True,           # Enable image uploads
    image_compression="balanced",    # Storage optimization
    media_retention_days=30         # Auto-cleanup policy
)
```

**Automatic Image Tracing in Nodes:**

```python
@ma.dspy_node
class VisionNode(dspy.Module):
    def forward(self, image: dspy.Image, query: str):
        # mahsm automatically:
        # 1. Extracts image from dspy.Image
        # 2. Uploads to LangFuse
        # 3. Attaches to trace
        result = self.vlm(image=image, query=query)
        return result
```

### 3.5 Trace Viewer Enhancements

**Proposed Features:**

1. **Inline Image Rendering**
   - Display images directly in trace timeline
   - Thumbnail → full-resolution on click

2. **Visual Diff Tools**
   - Compare input vs. output images
   - Highlight regions of interest from VLM attention

3. **Cost Tracking for Vision Tokens**
   - Separate pricing for image vs. text tokens
   - Token count estimation for images

---

## 4. EvalProtocol for Multimodal Evaluation

### 4.1 Current Landscape

**Status:** ⚠️ **No direct EvalProtocol VLM support** (as of Nov 2024)

EvalProtocol primarily focuses on text-based LLM evaluation. However, extensive research exists on VLM benchmarking that mahsm can integrate.

### 4.2 Existing VLM Benchmarks

| Benchmark | Focus | Tasks | Link |
|-----------|-------|-------|------|
| **MM-Vet v2** | Integrated capabilities | Multi-discipline reasoning | [arXiv:2408.00765](https://arxiv.org/abs/2408.00765) |
| **VHELM** | Holistic evaluation | 9 task categories | [arXiv:2410.07112](https://arxiv.org/abs/2410.07112) |
| **EXAMS-V** | Multilingual exams | 20 disciplines, 11 languages | [ACL 2024](https://aclanthology.org/2024.acl-long.420.pdf) |
| **LVLM-EHub** | Large VLM evaluation | Comprehensive metrics | [IEEE](https://ieeexplore.ieee.org/document/10769058/) |
| **HarmonicEval** | Multi-modal, multi-task | Automatic evaluation | [arXiv:2412.14613](https://arxiv.org/abs/2412.14613) |

### 4.3 Recommended Evaluation Framework for mahsm

**Phase 1: Adapter Pattern**

Integrate existing benchmarks via adapters:

```python
# mahsm evaluation adapter
import mahsm as ma
from mahsm.eval import VLMBenchmark

# Use existing benchmark dataset
benchmark = VLMBenchmark.from_huggingface("MMMU/MMMU")

# Evaluate mahsm agent
@ma.dspy_node
class MyVisionAgent(dspy.Module):
    pass

results = ma.testing.evaluate(
    agent=MyVisionAgent,
    benchmark=benchmark,
    metrics=["accuracy", "vision_reasoning", "hallucination"]
)
```

**Phase 2: Custom mahsm VLM Metrics**

```python
# Define mahsm-native VLM metrics
@ma.testing.metric
def vision_grounding_accuracy(prediction, ground_truth, image):
    """Evaluate if prediction is visually grounded."""
    # Use VLM-as-judge pattern
    judge = dspy.Predict("image, prediction, ground_truth -> score, explanation")
    result = judge(
        image=image,
        prediction=prediction,
        ground_truth=ground_truth
    )
    return float(result.score)

@ma.testing.metric
def visual_hallucination_rate(prediction, image):
    """Detect unsupported visual claims."""
    # Check if prediction references objects not in image
    pass
```

### 4.4 VLM-as-a-Judge Pattern

Leverage VLMs for automated evaluation:

```python
class VisionJudge(dspy.Signature):
    """Evaluate VLM output quality using another VLM."""
    image: dspy.Image = dspy.InputField()
    prediction: str = dspy.InputField()
    reference: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Score from 0-1")
    reasoning: str = dspy.OutputField(desc="Justification")

judge = dspy.ChainOfThought(VisionJudge)
```

### 4.5 Integration with EvalProtocol

**Proposed `ma.testing` Enhancements:**

```python
# Visual test case
@ma.testing.test_case
def test_object_detection():
    """Test agent can detect objects in images."""
    image = dspy.Image.from_file("tests/fixtures/kitchen.jpg")
    agent = MyVisionAgent()
    result = agent(image=image, query="List all objects")
    
    # mahsm automatically logs image to trace
    assert "refrigerator" in result.lower()
    assert "stove" in result.lower()

# Visual benchmark suite
class VisionBenchmarkSuite(ma.testing.Suite):
    benchmark = "MMMU/MMMU"
    subset = "validation"
    
    metrics = [
        ma.testing.VisionAccuracy(),
        ma.testing.HallucinationRate(),
        ma.testing.GroundingQuality()
    ]
```

---

## 5. VLM Fine-Tuning: SFT, Preference Optimization, RL

### 5.1 Current Ecosystem

**Key Insight:** VLM fine-tuning is now accessible via standard tools (TRL, Hugging Face), but **mahsm can provide a unified fine-tuning workflow** that integrates with its orchestration layer.

### 5.2 Supervised Fine-Tuning (SFT)

#### Tools and Libraries

**Primary Framework:** [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)

**Supported VLMs:**
- Qwen2-VL (3B, 7B, 72B)
- LLaVA (1.5, 1.6)
- PaliGemma
- Idefics2

#### SFT Example with TRL

```python
from trl import SFTTrainer
from transformers import AutoModelForVision2Seq, AutoProcessor

# Load VLM
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Prepare dataset
def format_dataset(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}]
            }
        ]
    }

# Fine-tune
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset.map(format_dataset),
    args=training_args
)
trainer.train()
```

### 5.3 Preference Optimization (DPO/PPO)

#### Direct Preference Optimization (DPO)

**Status:** ✅ Production-ready for VLMs (TRL support added July 2024)

**Key Papers:**
- "Preference Optimization for Vision Language Models" (Hugging Face, 2024)
- "Re-Align: Retrieval-Augmented DPO" (2025)

**DPO Training Flow:**
1. Collect preference pairs: (image, prompt, chosen_response, rejected_response)
2. Train VLM to prefer human-aligned outputs
3. No explicit reward model needed (unlike RLHF)

**Example DPO Dataset Format:**
```json
{
  "image": "<base64_or_url>",
  "prompt": "Describe this image",
  "chosen": "A serene landscape with mountains and a lake",
  "rejected": "There are purple elephants flying" # hallucination
}
```

**TRL Implementation:**
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=vlm_model,
    ref_model=ref_model,  # Reference model for KL penalty
    train_dataset=preference_dataset,
    tokenizer=processor.tokenizer,
    args=training_args
)
trainer.train()
```

### 5.4 Reinforcement Learning for VLMs

**Status:** 🚀 **Rapidly emerging field** (2024-2025)

#### Key Approaches

**1. Vision-Language Decoupled Actor-Critic (VL-DAC)**
- Paper: "Enhancing VLM Training with RL in Synthetic Worlds" (2025)
- Trains VLMs in simulated environments (MiniWorld, ALFWorld)
- 50% improvement on control tasks

**2. RL4VLM Framework**
- Paper: "Fine-Tuning VLMs as Decision-Making Agents via RL" (Berkeley, 2024)
- Uses Chain-of-Thought (CoT) for exploration
- Outperforms GPT-4V on decision-making benchmarks

**3. RL-VLM-F: Reward from VLM Feedback**
- Uses VLMs to generate preference labels
- Trains reward model from VLM judgments
- No need for human labels or environment code

#### VLM-RL Training Loop

```python
# Simplified RL training loop for VLMs
for episode in range(num_episodes):
    # 1. VLM observes environment
    image = env.get_observation()
    
    # 2. VLM generates action with reasoning
    prompt = f"You are an agent. Observation: [image]. What action should you take?"
    response = vlm(image=image, prompt=prompt)
    action = parse_action(response)
    
    # 3. Execute action in environment
    reward = env.step(action)
    
    # 4. Update VLM policy with RL (PPO)
    update_policy(vlm, reward, trajectory)
```

### 5.5 mahsm Fine-Tuning Integration

**Proposed Unified API:**

```python
import mahsm as ma

# 1. Define mahsm agent
@ma.dspy_node
class VisionAgent(dspy.Module):
    def forward(self, image, task):
        return self.policy(image=image, task=task)

# 2. Supervised fine-tuning
ma.finetune.sft(
    agent=VisionAgent,
    dataset="HuggingFaceH4/rlaif-v_formatted",
    base_model="Qwen/Qwen2-VL-7B-Instruct",
    output_dir="./checkpoints"
)

# 3. Preference optimization
ma.finetune.dpo(
    agent=VisionAgent,
    preference_dataset="openbmb/RLAIF-V-Dataset",
    reference_model="path/to/sft/model"
)

# 4. RL fine-tuning (future)
ma.finetune.rl(
    agent=VisionAgent,
    environment="ALFWorld",
    reward_fn=task_success_rate
)
```

**Key Value Proposition:**
- **Unified API** across SFT, DPO, and RL
- **mahsm-native tracing** during fine-tuning (LangFuse integration)
- **Automatic dataset preparation** from mahsm agent traces
- **Graph-based curriculum learning** (progressive task difficulty)

---

## 6. Gap Analysis and Recommendations

### 6.1 Capability Matrix

| Component | Current State | mahsm Integration | Priority |
|-----------|--------------|-------------------|----------|
| **DSPy Image Support** | ✅ Beta | ✅ Ready | **HIGH** |
| **DSPy VLM Optimizers** | ⚠️ Limited | 🔨 Build custom | **HIGH** |
| **LangGraph Multimodal State** | ✅ Native | ✅ Ready | **MEDIUM** |
| **LangGraph Vision Patterns** | ⚠️ Manual | 🔨 Create templates | **MEDIUM** |
| **LangFuse Image Tracing** | ✅ Full | ✅ Ready | **HIGH** |
| **LangFuse Media UI** | ✅ Full | ✅ Ready | **LOW** |
| **EvalProtocol VLM Tests** | ❌ None | 🔨 Build from scratch | **HIGH** |
| **VLM Benchmarks** | ✅ External | 🔨 Integrate adapters | **MEDIUM** |
| **SFT Support** | ✅ TRL | 🔨 Wrap with mahsm API | **MEDIUM** |
| **DPO Support** | ✅ TRL | 🔨 Wrap with mahsm API | **MEDIUM** |
| **VLM-RL Support** | ⚠️ Research | 🔨 Experimental feature | **LOW** |

### 6.2 Critical Gaps

**Gap 1: DSPy Multimodal Optimizers**
- **Problem:** Bootstrap and MIPRO don't handle vision effectively
- **Solution:** Create `ma.optimizers.VLMBootstrap` with vision-specific metrics

**Gap 2: EvalProtocol for VLMs**
- **Problem:** No pytest harness for visual evaluation
- **Solution:** Extend `ma.testing.PytestHarness` with `@ma.testing.visual_test`

**Gap 3: Graph-Based Vision Workflows**
- **Problem:** No templates for common vision patterns (OCR → Summarize, etc.)
- **Solution:** Create `ma.templates.vision` with pre-built graphs

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Months 1-2)

**Goal:** Seamless image support in mahsm agents

**Deliverables:**
1. **Enhanced `@ma.dspy_node`**
   - Auto-handle `dspy.Image` fields
   - Serialize images in graph state
   - Integration test suite

2. **LangFuse Vision Integration**
   - Automatic image upload from nodes
   - Trace viewer enhancements
   - Cost tracking for vision tokens

3. **Documentation & Examples**
   - "Quick Start: VLM Agent" tutorial
   - Image-based RAG example
   - Multi-step visual reasoning example

### Phase 2: Evaluation (Months 3-4)

**Goal:** Comprehensive VLM testing and benchmarking

**Deliverables:**
1. **`ma.testing` Vision Extensions**
   ```python
   @ma.testing.visual_test
   def test_visual_grounding(image, expected_objects):
       pass
   ```

2. **Benchmark Adapters**
   - MMMU integration
   - MM-Vet integration
   - Custom benchmark loader

3. **VLM-as-Judge Metrics**
   - Hallucination detection
   - Visual grounding quality
   - Reasoning coherence

### Phase 3: Optimization (Months 5-6)

**Goal:** Automated VLM prompt and parameter optimization

**Deliverables:**
1. **VLM-Specific Optimizers**
   ```python
   ma.optimizers.VLMBootstrap(
       metric=ma.metrics.vision_accuracy,
       max_bootstrapped_demos=10,
       vision_augmentation=True
   )
   ```

2. **Few-Shot Vision Learning**
   - Visual example selection
   - Cross-modal retrieval for demos
   - Automatic prompt tuning with images

### Phase 4: Fine-Tuning (Months 7-9)

**Goal:** Unified fine-tuning API for VLMs

**Deliverables:**
1. **SFT Integration**
   ```python
   ma.finetune.sft(agent, dataset, model)
   ```

2. **DPO Integration**
   ```python
   ma.finetune.dpo(agent, preference_dataset)
   ```

3. **Automatic Dataset Generation**
   - Export mahsm traces as training data
   - Human-in-the-loop annotation workflow

### Phase 5: Advanced RL (Months 10-12)

**Goal:** Experimental VLM-RL capabilities

**Deliverables:**
1. **Environment Integration**
   - Simulated vision environments (MiniWorld, ALFWorld)
   - Real-world API interfaces (web browsing, GUIs)

2. **RL Training Loop**
   - PPO for VLM policies
   - Reward shaping from VLM feedback

---

## 8. Conclusion and Next Steps

### 8.1 Summary

mahsm is **exceptionally well-positioned** to become the premier library for multimodal agentic research:

**Strengths:**
1. All four pillars (DSPy, LangGraph, LangFuse, EvalProtocol) have multimodal primitives
2. Graph-based orchestration is ideal for complex vision workflows
3. Built-in observability provides immediate value for VLM debugging
4. Declarative API philosophy extends naturally to vision tasks

**Key Gaps:**
1. DSPy optimizers need VLM-specific enhancements
2. EvalProtocol lacks visual test framework
3. No unified fine-tuning API

### 8.2 Strategic Recommendations

**Priority 1: Quick Wins (Next 30 Days)**
- Enable `dspy.Image` in `@ma.dspy_node` decorator
- Integrate LangFuse image tracing in `ma.init()`
- Create 1-2 vision agent tutorials

**Priority 2: Core Infrastructure (60-90 Days)**
- Build `ma.testing` visual test harness
- Create VLM benchmark adapters
- Develop vision-specific optimizers

**Priority 3: Advanced Features (6-12 Months)**
- Unified fine-tuning API
- Experimental VLM-RL support
- Production-ready vision agent templates

---

## Appendix: Key Resources

### Papers
- "Fine-Tuning Large VLMs as Decision-Making Agents via RL" (Berkeley, 2024)
- "Preference Optimization for Vision Language Models" (Hugging Face, 2024)
- "MM-Vet v2: A Challenging Benchmark for VLMs" (2024)

### Documentation
- [DSPy Multimodal Docs](https://dspy.ai)
- [LangFuse Multi-Modality](https://langfuse.com/docs/observability/features/multi-modality)
- [TRL VLM Fine-Tuning](https://huggingface.co/docs/trl)

---

**Document Version:** 1.0  
**Last Updated:** November 2, 2025  
**Research by:** Droid AI (Factory)
