# mahsm Quickstart Guide 🚀

**Complete End-to-End Guide: Build → Trace → Evaluate**

This guide walks you through building a complete multi-agent LLM application with mahsm, including full observability (Langfuse) and evaluation (EvalProtocol).

---

## Table of Contents

1. [Installation](#1-installation)
2. [Langfuse Setup (Observability)](#2-langfuse-setup-observability)
3. [Build Your First Agent](#3-build-your-first-agent)
4. [Run and Trace](#4-run-and-trace)
5. [View Traces in Langfuse UI](#5-view-traces-in-langfuse-ui)
6. [Run Evaluations](#6-run-evaluations)
7. [View Evaluation Results](#7-view-evaluation-results)
8. [Advanced: Optimize with DSPy](#8-advanced-optimize-with-dspy)

---

## 1. Installation

### From PyPI (Recommended)

```bash
pip install mahsm
```

### From Source

```bash
git clone https://github.com/chimera-research/mahsm.git
cd mahsm
pip install -e .
```

### Verify Installation

```python
import mahsm as ma
print(ma.__version__)  # Should print 0.1.0
```

---

## 2. Langfuse Setup (Observability)

### 2.1 Create Langfuse Account

1. Go to **[Langfuse Cloud](https://cloud.langfuse.com/)** (or [self-host](https://langfuse.com/self-hosting))
2. Sign up for a free account
3. Create a new project (e.g., "mahsm-demo")

### 2.2 Get API Keys

1. In your Langfuse project, go to **Settings** → **API Keys**
2. Click **"Create new API keys"**
3. Copy the keys:
   - **Public Key**: `pk-lf-...`
   - **Secret Key**: `sk-lf-...`
   - **Host**: `https://cloud.langfuse.com` (EU) or `https://us.cloud.langfuse.com` (US)

### 2.3 Set Environment Variables

Create a `.env` file in your project root:

```bash
# Langfuse API Keys
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # Or https://us.cloud.langfuse.com

# OpenAI API Key
OPENAI_API_KEY=sk-proj-your-key-here
```

**Load environment variables:**

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 3. Build Your First Agent

Let's build a **research agent** that searches for information and synthesizes answers.

### 3.1 Create `research_agent.py`

```python
import mahsm as ma
from typing import TypedDict, Optional
import dspy

# Configure DSPy with your preferred model
lm = dspy.OpenAI(model='gpt-4o-mini', max_tokens=2000)
dspy.settings.configure(lm=lm)


# 1. DEFINE STATE
class ResearchState(TypedDict):
    question: str
    search_query: str
    findings: Optional[str]
    answer: Optional[str]
    quality_score: Optional[str]


# 2. DEFINE AGENT NODES
@ma.dspy_node
class QueryGenerator(ma.Module):
    """Generates an optimal search query from a question."""
    def __init__(self):
        super().__init__()
        self.generator = ma.dspy.ChainOfThought("question -> search_query")
    
    def forward(self, question):
        return self.generator(question=question)


@ma.dspy_node
class Researcher(ma.Module):
    """Simulates research by generating findings (replace with real search API)."""
    def __init__(self):
        super().__init__()
        self.researcher = ma.dspy.ChainOfThought("question, search_query -> findings")
    
    def forward(self, question, search_query):
        return self.researcher(question=question, search_query=search_query)


@ma.dspy_node
class Synthesizer(ma.Module):
    """Synthesizes findings into a final answer."""
    def __init__(self):
        super().__init__()
        self.synthesizer = ma.dspy.ChainOfThought("question, findings -> answer")
    
    def forward(self, question, findings):
        return self.synthesizer(question=question, findings=findings)


@ma.dspy_node
class QualityChecker(ma.Module):
    """Evaluates answer quality."""
    def __init__(self):
        super().__init__()
        self.checker = ma.dspy.Predict("question, answer -> quality_score")
    
    def forward(self, question, answer):
        return self.checker(question=question, answer=answer)


# 3. DEFINE ROUTING
def should_continue(state: ResearchState):
    """Route based on quality score."""
    score = state.get("quality_score", "")
    if "EXCELLENT" in score.upper() or "GOOD" in score.upper():
        return ma.END
    return "researcher"  # Loop back for more research


# 4. BUILD GRAPH
workflow = ma.graph.StateGraph(ResearchState)

# Add nodes
workflow.add_node("query_gen", QueryGenerator())
workflow.add_node("researcher", Researcher())
workflow.add_node("synthesizer", Synthesizer())
workflow.add_node("quality_check", QualityChecker())

# Add edges
workflow.add_edge(ma.START, "query_gen")
workflow.add_edge("query_gen", "researcher")
workflow.add_edge("researcher", "synthesizer")
workflow.add_edge("synthesizer", "quality_check")
workflow.add_conditional_edges("quality_check", should_continue)

# Compile
graph = workflow.compile()


# 5. RUN FUNCTION
def run_research(question: str):
    """Run the research agent."""
    result = graph.invoke({"question": question})
    return result


if __name__ == "__main__":
    # Test the agent
    question = "What are the key innovations in DSPy?"
    result = run_research(question)
    print(f"\n📝 Question: {question}")
    print(f"✨ Answer: {result.get('answer', 'N/A')}")
```

---

## 4. Run and Trace

### 4.1 Initialize Tracing

Update your `research_agent.py` to enable automatic tracing:

```python
# Add at the top of your script, after imports
handler = ma.init()  # Initializes Langfuse + DSPy instrumentation
print("✅ Tracing enabled! All LLM calls will be logged to Langfuse.")
```

### 4.2 Run Your Agent

```bash
python research_agent.py
```

You should see:
```
mahsm: Langfuse client initialized.
mahsm: DSPy instrumented for automatic tracing.
✅ Tracing enabled! All LLM calls will be logged to Langfuse.

📝 Question: What are the key innovations in DSPy?
✨ Answer: [Generated answer here]
```

---

## 5. View Traces in Langfuse UI

### 5.1 Open Langfuse Dashboard

1. Go to **[https://cloud.langfuse.com](https://cloud.langfuse.com)**
2. Navigate to your project
3. Click on **"Traces"** in the left sidebar

### 5.2 Explore Your Trace

You'll see a new trace with:
- **Trace ID**: Unique identifier
- **Timestamp**: When it was executed
- **Duration**: Total execution time
- **Cost**: Estimated API cost

**Click on the trace** to see:
- 📊 **Execution tree**: Visualize the agent flow (query_gen → researcher → synthesizer → quality_check)
- 💬 **LLM calls**: All prompts and completions
- 📈 **Token usage**: Input/output tokens for each call
- 💰 **Cost breakdown**: Per-node costs
- 🔗 **Metadata**: Model, temperature, etc.

### 5.3 Key Features to Explore

| Feature | What You Can Do |
|---------|----------------|
| **Sessions** | Group related traces (e.g., user conversation) |
| **Users** | Track per-user usage and behavior |
| **Tags** | Categorize traces (e.g., `production`, `experiment`) |
| **Scores** | Add quality/evaluation scores to traces |
| **Prompts** | Manage and version prompts centrally |
| **Datasets** | Create test datasets from production traces |

---

## 6. Run Evaluations

Now let's evaluate our agent systematically using **EvalProtocol**.

### 6.1 Create `test_research_agent.py`

```python
from research_agent import graph
import mahsm as ma
from dotenv import load_dotenv

load_dotenv()

# 1. Create testing harness
harness = ma.testing.PytestHarness(graph=graph)


# 2. Define test dataset
def research_questions():
    """Test dataset: research questions with expected qualities."""
    return [
        ma.testing.EvaluationRow(
            messages=[
                ma.HumanMessage(
                    content="What are the key innovations in DSPy compared to traditional prompting?"
                )
            ],
            expected_output=None,  # No ground truth needed for LLM-as-judge
        ),
        ma.testing.EvaluationRow(
            messages=[
                ma.HumanMessage(
                    content="Explain how LangGraph enables cyclical agent workflows."
                )
            ],
            expected_output=None,
        ),
        ma.testing.EvaluationRow(
            messages=[
                ma.HumanMessage(
                    content="Compare the tracing capabilities of Langfuse vs LangSmith."
                )
            ],
            expected_output=None,
        ),
    ]


harness._data_loaders = ma.testing.DynamicDataLoader(generators=[research_questions])


# 3. Define evaluation test
@ma.testing.evaluation_test(
    data_loaders=harness.data_loaders,
    rollout_processor=harness.rollout_processor,
    completion_params=[
        {"model": "openai/gpt-4o-mini"},
        {"model": "openai/gpt-4o"},  # Compare models
    ],
)
async def test_research_quality(row: ma.testing.EvaluationRow) -> ma.testing.EvaluationRow:
    """
    Evaluates research answer quality using LLM-as-a-judge.
    """
    return await ma.testing.aha_judge(
        row,
        judge_model="openai/gpt-4o",
        rubric="""
        Evaluate the research answer on:
        1. Accuracy: Is the information correct?
        2. Completeness: Does it address all aspects of the question?
        3. Clarity: Is it well-structured and easy to understand?
        4. Citations: Does it reference relevant sources?
        
        Score from 1-5 (5 = excellent).
        """
    )
```

### 6.2 Run Evaluations

```bash
pytest test_research_agent.py -v
```

You'll see:
```
============================= test session starts ==============================
test_research_agent.py::test_research_quality[openai/gpt-4o-mini] RUNNING
test_research_agent.py::test_research_quality[openai/gpt-4o] RUNNING
...
============================= 6 passed in 45.23s ===============================
```

---

## 7. View Evaluation Results

### 7.1 EvalProtocol Local UI

**EvalProtocol automatically launches a local UI** during test runs.

1. During `pytest` execution, look for:
   ```
   📊 EvalProtocol UI running at: http://localhost:8000
   ```

2. Open **http://localhost:8000** in your browser

### 7.2 Explore Results

The UI shows:

#### **Leaderboard View**
| Model | Avg Score | Pass Rate | Latency | Cost |
|-------|-----------|-----------|---------|------|
| gpt-4o | 4.7/5.0 | 100% | 2.3s | $0.012 |
| gpt-4o-mini | 4.2/5.0 | 100% | 1.8s | $0.003 |

#### **Trace View**
- Click on any test case to see:
  - Full execution trace
  - LLM calls with prompts/completions
  - Scores and feedback
  - Performance metrics

#### **Pivot Table**
- Group by: Model, Dataset, Score Range
- Aggregate: Average score, Total cost, Latency p95

### 7.3 View in Langfuse

Evaluations are also synced to Langfuse:

1. Go to **Langfuse** → **Datasets**
2. You'll see your test dataset with:
   - All test cases
   - Linked traces
   - Evaluation scores

3. Go to **Scores** to see:
   - Score distribution
   - Per-model comparison
   - Score trends over time

---

## 8. Advanced: Optimize with DSPy

### 8.1 Use DSPy Optimizers

DSPy can automatically optimize your prompts:

```python
from dspy.teleprompt import BootstrapFewShot

# Define metric
def accuracy_metric(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

# Optimize
optimizer = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=4)
optimized_researcher = optimizer.compile(Researcher(), trainset=train_examples)

# Replace in graph
workflow.add_node("researcher", ma.dspy_node(optimized_researcher)())
```

### 8.2 Track Optimization in Langfuse

Add tags to track experiments:

```python
from langfuse import observe

@observe(tags=["experiment", "bootstrap-v1"])
def run_optimized():
    return graph.invoke({"question": "..."})
```

---

## 📚 Next Steps

### Learn More
- 📖 **[mahsm Documentation](https://github.com/chimera-research/mahsm)**: Full API reference
- 🎓 **[DSPy Docs](https://dspy.ai/)**: Learn about optimization and advanced modules
- 📊 **[Langfuse Guide](https://langfuse.com/docs)**: Deep dive into observability
- 🧪 **[EvalProtocol](https://pypi.org/project/eval-protocol/)**: Advanced evaluation techniques

### Community
- 💬 **[GitHub Discussions](https://github.com/chimera-research/mahsm/discussions)**: Ask questions
- 🐛 **[Issues](https://github.com/chimera-research/mahsm/issues)**: Report bugs
- ⭐ **[Star the repo](https://github.com/chimera-research/mahsm)**: Support the project

---

## 🎉 You're All Set!

You now have a complete LLM application with:
- ✅ **Production-ready agents** with LangGraph
- ✅ **Deep observability** with Langfuse
- ✅ **Systematic evaluation** with EvalProtocol
- ✅ **Automatic optimization** with DSPy

**Build something amazing!** 🚀
