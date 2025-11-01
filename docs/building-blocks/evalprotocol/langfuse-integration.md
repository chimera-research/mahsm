# Langfuse Integration

> **TL;DR**: When you initialize Langfuse tracing with `ma.tracing.init()`, all evaluation runs are automatically logged to Langfuse, connecting test results with execution traces for complete observability.

## Quick Start

### Automatic Integration

```python
import mahsm as ma
from eval_protocol import EvalTest, EvalSuite, Grader

# 1. Initialize tracing
ma.tracing.init()

# 2. Create tests
suite = EvalSuite(
    name="agent_quality",
    tests=[...]
)

# 3. Run evaluation
results = suite.run(agent)

# ✅ Automatically logged to Langfuse!
```

That's it! Your evaluation results are now in Langfuse.

---

## What Gets Logged

When tracing is enabled, Langfuse captures:

### 1. **Test Suite Execution**

Top-level trace with summary metrics:

```
Trace: "agent_quality_evaluation"
├─ pass_rate: 85%
├─ average_score: 7.8/10
├─ duration: 12.3s
└─ timestamp: 2024-01-15 10:30:00
```

### 2. **Individual Test Results**

Each test creates a span:

```
└─ Span: "test_password_reset"
   ├─ input: "How do I reset my password?"
   ├─ output: "Click 'Forgot Password' and we'll email you a reset link"
   ├─ score: 9.5/10
   ├─ passed: true
   └─ grader: "llm_as_judge"
```

### 3. **Agent Execution Trace**

Full trace of the agent's execution during the test:

```
└─ Span: "agent_execution"
   ├─ Span: "dspy_module_call"
   │  ├─ input: {...}
   │  ├─ output: {...}
   │  └─ duration: 1.2s
   └─ Generation: "llm_call"
      ├─ model: "gpt-4o-mini"
      ├─ tokens: 150
      ├─ cost: $0.002
      └─ latency: 800ms
```

### 4. **Grader Execution** (for LLM-as-Judge)

When using LLM-as-judge, the grading LLM call is also traced:

```
└─ Span: "grader_llm_call"
   ├─ input: "Grade this output: {...}"
   ├─ output: "Score: 8/10. Rationale: ..."
   ├─ model: "gpt-4o"
   └─ duration: 600ms
```

---

## Viewing in Langfuse UI

### 1. Navigate to Traces

Open your Langfuse dashboard → **Traces** tab.

### 2. Filter by Evaluation Runs

```
Tags: ["evaluation", "agent_quality"]
Session ID: "eval-2024-01-15"
```

### 3. Drill Down into Test

Click on a trace to see:

- **Summary**: Pass/fail, score, duration
- **Input/Output**: What was tested and the result
- **Full Trace**: Complete execution path through your agent
- **Costs**: Token usage and API costs for this test

### 4. Compare Over Time

Use Langfuse's dashboard to track:
- Pass rate trends
- Average score over time
- Cost per evaluation
- Latency trends

---

## Complete Example

### Setup

```python
import dspy
import mahsm as ma
from eval_protocol import EvalTest, EvalSuite, Grader

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4o-mini", api_key="...")
dspy.configure(lm=lm)

# Initialize Langfuse tracing
ma.tracing.init(
    tags=["evaluation", "customer_support"],
    release="v1.2.0"
)
```

### Define Agent

```python
@ma.dspy_node
class CustomerSupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> str:
        return self.cot(question=question).answer
```

### Create Test Suite

```python
suite = EvalSuite(
    name="support_quality",
    tests=[
        EvalTest(
            name="password_reset",
            input="How do I reset my password?",
            grader=Grader.llm_as_judge(
                criteria="Response is helpful and actionable"
            )
        ),
        EvalTest(
            name="refund_request",
            input="I want a refund for my order",
            grader=Grader.llm_as_judge(
                criteria="Response shows empathy and provides clear next steps"
            )
        ),
        EvalTest(
            name="account_deletion",
            input="How do I delete my account?",
            grader=Grader.llm_as_judge(
                criteria="Response is clear and includes confirmation steps"
            )
        )
    ]
)
```

### Run Evaluation

```python
# Create agent
agent = CustomerSupportAgent()

# Run evaluation (automatically traced)
results = suite.run(agent)

# Print summary
print(f"Pass rate: {results.pass_rate:.1%}")
print(f"Average score: {results.average_score}/10")

# View in Langfuse:
# → Open dashboard
# → Filter by tags: ["evaluation", "customer_support"]
# → See all 3 test executions with full traces
```

---

## Organizing Evaluations

### Use Session IDs

Group related evaluation runs:

```python
# Daily evaluation run
ma.tracing.init(
    session_id=f"daily-eval-{datetime.now().date()}",
    tags=["scheduled", "daily"]
)

results = suite.run(agent)
```

All tests from this run appear under the same session in Langfuse.

### Use Tags

Tag evaluations by purpose:

```python
# Pre-deployment check
ma.tracing.init(
    tags=["pre-deploy", "quality-gate", "production"]
)

# A/B testing
ma.tracing.init(
    tags=["ab-test", "variant-a"]
)

# Regression testing
ma.tracing.init(
    tags=["regression", "ci-cd"]
)
```

Filter in Langfuse UI by these tags.

### Use Release Versions

Track evaluations per release:

```python
ma.tracing.init(
    release="v2.0.0-beta",
    tags=["evaluation"]
)

results = suite.run(agent)
```

Compare performance across releases in Langfuse.

---

## Advanced Integration

### Custom Metadata

Add custom metadata to evaluation traces:

```python
from langfuse.decorators import observe

@observe(
    name="custom_evaluation",
    metadata={
        "environment": "staging",
        "test_suite": "smoke_tests",
        "engineer": "alice"
    }
)
def run_evaluation():
    results = suite.run(agent)
    return results

results = run_evaluation()
```

### Link to Specific Features

Tag tests by feature area:

```python
suite = EvalSuite(
    name="authentication_tests",
    tests=[
        EvalTest(
            name="login",
            input="...",
            grader=...,
            metadata={"feature": "auth", "priority": "critical"}
        ),
        EvalTest(
            name="password_reset",
            input="...",
            grader=...,
            metadata={"feature": "auth", "priority": "high"}
        )
    ]
)
```

### Track Costs

Calculate evaluation costs:

```python
results = suite.run(agent)

# Langfuse automatically tracks token usage and costs
# View in UI:
# - Total tokens used
# - Cost per test
# - Cost per evaluation run
```

---

## Comparing Evaluation Runs

### Before/After Code Changes

```python
# Before changes
ma.tracing.init(
    session_id="baseline",
    tags=["before-refactor"]
)
baseline_results = suite.run(agent_v1)

# After changes
ma.tracing.init(
    session_id="after-refactor",
    tags=["after-refactor"]
)
new_results = suite.run(agent_v2)

# Compare in Langfuse:
# - Filter by tags
# - View side-by-side metrics
# - Analyze quality improvement
```

### A/B Testing

```python
# Variant A
ma.tracing.init(
    session_id="ab-test-2024-01-15",
    tags=["ab-test", "variant-a"]
)
results_a = suite.run(agent_a)

# Variant B
ma.tracing.init(
    session_id="ab-test-2024-01-15",
    tags=["ab-test", "variant-b"]
)
results_b = suite.run(agent_b)

# Compare in Langfuse dashboard
```

### Regression Detection

```python
import json

# Run evaluation
ma.tracing.init(
    session_id=f"regression-{datetime.now().isoformat()}",
    tags=["regression", "automated"]
)
results = suite.run(agent)

# Save baseline
baseline = {
    "pass_rate": 0.95,
    "average_score": 8.5
}

# Check for regression
if results.pass_rate < baseline["pass_rate"] - 0.05:
    print("⚠️  Regression detected: pass rate dropped!")
    print(f"Baseline: {baseline['pass_rate']:.1%}")
    print(f"Current: {results.pass_rate:.1%}")
    print("View details in Langfuse")
```

---

## Debugging with Langfuse

### Failed Test Investigation

When a test fails:

1. **Open Langfuse** → Find the failed test trace
2. **View full execution** → See exactly what the agent did
3. **Check LLM calls** → Review prompts and responses
4. **Analyze grading** → Understand why it failed

### Example Debugging Flow

```python
# Test fails
results = suite.run(agent)
failed = [t for t in results.test_results if not t.passed]

for test in failed:
    print(f"Failed test: {test.test_name}")
    print(f"Score: {test.score}/10")
    print(f"View trace in Langfuse:")
    print(f"  Session ID: {test.session_id}")
    print(f"  Trace ID: {test.trace_id}")
```

Open Langfuse → Search by trace_id → Inspect execution.

### Common Issues

**Issue 1: LLM call failed**

Langfuse shows:
```
└─ Generation: "llm_call"
   └─ Error: "Rate limit exceeded"
```

**Issue 2: Unexpected output**

Langfuse shows:
```
└─ Span: "dspy_module"
   ├─ input: "What is 2+2?"
   └─ output: "The weather is sunny"  # ← Clearly wrong
```

**Issue 3: Slow evaluation**

Langfuse shows:
```
└─ Span: "test_execution"
   └─ duration: 45s  # ← Much slower than expected
      └─ Span: "llm_call"
         └─ duration: 43s  # ← Bottleneck identified
```

---

## Best Practices

### ✅ Do:

1. **Always initialize tracing before evaluation**
   ```python
   # ✅ Good
   ma.tracing.init()
   results = suite.run(agent)
   ```

2. **Use descriptive session IDs**
   ```python
   # ✅ Good
   ma.tracing.init(
       session_id="weekly-eval-2024-01-15",
       tags=["weekly", "production"]
   )
   ```

3. **Tag by purpose and environment**
   ```python
   # ✅ Good
   ma.tracing.init(
       tags=["ci-cd", "staging", "regression"]
   )
   ```

4. **Include release version**
   ```python
   # ✅ Good
   ma.tracing.init(
       release=os.getenv("APP_VERSION", "dev")
   )
   ```

### ❌ Don't:

1. **Don't forget to initialize tracing**
   ```python
   # ❌ Bad - no tracing
   results = suite.run(agent)
   # Evaluations won't be logged to Langfuse
   ```

2. **Don't use generic session IDs**
   ```python
   # ❌ Bad
   ma.tracing.init(session_id="test")
   
   # ✅ Good
   ma.tracing.init(session_id=f"eval-{datetime.now().isoformat()}")
   ```

3. **Don't mix unrelated evaluations in one session**
   ```python
   # ❌ Bad - confusing
   ma.tracing.init(session_id="everything")
   suite1.run(agent)
   suite2.run(agent)
   suite3.run(agent)
   
   # ✅ Good - separate sessions
   for suite in [suite1, suite2, suite3]:
       ma.tracing.init(session_id=f"eval-{suite.name}")
       suite.run(agent)
   ```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Evaluate and Log

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install eval-protocol
      
      - name: Run evaluation with Langfuse logging
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
        run: python scripts/run_eval.py
```

### Script: `scripts/run_eval.py`

```python
import os
import sys
import mahsm as ma
from eval_protocol import EvalSuite
from tests.suites import agent_quality_suite

# Initialize tracing
ma.tracing.init(
    session_id=f"ci-{os.getenv('GITHUB_SHA', 'local')[:8]}",
    tags=[
        "ci-cd",
        os.getenv("GITHUB_REF_NAME", "main"),
        os.getenv("GITHUB_ACTOR", "unknown")
    ],
    release=os.getenv("GITHUB_SHA", "dev")[:8]
)

# Run evaluation
from my_agent import Agent
agent = Agent()
results = agent_quality_suite.run(agent)

# Print summary
print(f"\\nEvaluation Results:")
print(f"Pass rate: {results.pass_rate:.1%}")
print(f"Average score: {results.average_score}/10")
print(f"\\nView full results in Langfuse")

# Quality gate
if results.pass_rate < 0.9:
    print("❌ Quality gate failed: pass rate too low")
    sys.exit(1)

if results.average_score < 7.0:
    print("❌ Quality gate failed: average score too low")
    sys.exit(1)

print("✅ Quality gate passed")
sys.exit(0)
```

---

## Viewing Trends Over Time

### Langfuse Dashboard

Use Langfuse's dashboard features to:

1. **Track pass rate over time**
   - Graph showing daily/weekly pass rates
   - Identify when quality dropped
   - Correlate with code changes

2. **Monitor costs**
   - Total eval costs per day/week
   - Cost per test
   - Identify expensive tests

3. **Analyze latency**
   - Average evaluation duration
   - Identify slow tests
   - Track performance improvements

4. **Compare releases**
   - Filter by release version
   - Side-by-side comparison
   - Track quality improvements

---

## Custom Analysis

### Export Data from Langfuse

```python
from langfuse import Langfuse

client = Langfuse()

# Fetch evaluation traces
traces = client.fetch_traces(
    tags=["evaluation"],
    from_timestamp="2024-01-01",
    to_timestamp="2024-01-31"
)

# Analyze
pass_rates = []
for trace in traces:
    metadata = trace.metadata
    pass_rates.append(metadata.get("pass_rate", 0))

avg_pass_rate = sum(pass_rates) / len(pass_rates)
print(f"Average pass rate in January: {avg_pass_rate:.1%}")
```

### Custom Dashboards

Create custom visualizations using Langfuse API:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Fetch data
traces = client.fetch_traces(tags=["evaluation"])

# Build dataframe
data = []
for trace in traces:
    data.append({
        "date": trace.timestamp,
        "pass_rate": trace.metadata.get("pass_rate"),
        "score": trace.metadata.get("average_score"),
        "release": trace.release
    })

df = pd.DataFrame(data)

# Plot trends
df.plot(x="date", y=["pass_rate", "score"])
plt.title("Evaluation Trends")
plt.show()
```

---

## Next Steps

1. **[LangGraph Integration →](langgraph-integration.md)** - Evaluate multi-step workflows
2. **[Langfuse Dashboard](https://cloud.langfuse.com)** - View your evaluation traces

---

## External Resources

- **[Langfuse Tracing](https://langfuse.com/docs/tracing)** - Official tracing documentation
- **[Langfuse API](https://langfuse.com/docs/api)** - Programmatic access to traces
- **[EvalProtocol + Langfuse](https://pypi.org/project/eval-protocol/)** - Integration details

---

**Next: Evaluate LangGraph workflows with [LangGraph Integration →](langgraph-integration.md)**
