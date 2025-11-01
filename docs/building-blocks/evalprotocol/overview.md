# EvalProtocol Overview

> **TL;DR**: EvalProtocol provides automated testing and evaluation for LLM agents. Use `ma.testing` to define test suites, run evaluations, and track metrics over time.

## What is EvalProtocol?

**EvalProtocol** is a framework for systematically evaluating LLM-powered applications. It enables you to:

- ✅ **Define test suites** with assertions and grading criteria
- ✅ **Run evaluations** automatically across agent iterations
- ✅ **Track metrics** over time to measure improvements
- ✅ **Integrate with Langfuse** to connect eval results with traces
- ✅ **Grade outputs** using LLM-as-judge or deterministic functions

Think of it as **pytest for LLM agents** - write tests once, run them continuously.

---

## Quick Start

### Installation

```bash
pip install eval-protocol
```

### Basic Example

```python
import mahsm as ma
from eval_protocol import EvalTest, Grader

# Define a test
test = EvalTest(
    name="customer_support_quality",
    input="How do I reset my password?",
    expected_contains=["email", "reset link"],
    grader=Grader.llm_as_judge(
        criteria="Response is helpful and accurate"
    )
)

# Run your agent
@ma.dspy_node
class SupportAgent(dspy.Module):
    def forward(self, question: str) -> str:
        # Your agent logic
        return answer

# Evaluate
result = test.run(SupportAgent())

print(f"Score: {result.score}/10")
print(f"Passed: {result.passed}")
```

---

## Key Concepts

### 1. **EvalTest**

A test case with input, expected output, and grading logic.

```python
from eval_protocol import EvalTest

test = EvalTest(
    name="factual_accuracy",
    input="What is the capital of France?",
    expected_output="Paris",
    grader=Grader.exact_match()
)
```

### 2. **Test Suites**

Collections of related tests.

```python
from eval_protocol import EvalSuite

suite = EvalSuite(
    name="customer_support",
    tests=[test1, test2, test3]
)

# Run entire suite
results = suite.run(agent)
```

### 3. **Graders**

Functions that score agent outputs.

```python
# Built-in graders
Grader.exact_match()           # Exact string match
Grader.contains(text)          # Check substring
Grader.llm_as_judge(criteria)  # Use LLM to grade
Grader.custom(fn)              # Your own function
```

### 4. **Metrics**

Track performance over time.

```python
# Automatically tracked
- Pass rate
- Average score
- Latency
- Token usage
```

---

## mahsm Integration

mahsm provides a unified interface for EvalProtocol:

### Import Structure

```python
import mahsm as ma

# Access EvalProtocol through mahsm
test = ma.testing.EvalTest(...)
suite = ma.testing.EvalSuite(...)
grader = ma.testing.Grader.llm_as_judge(...)
```

### Automatic Tracing

Tests are automatically traced in Langfuse when `ma.tracing.init()` is called:

```python
import mahsm as ma

# Initialize tracing
ma.tracing.init()

# Tests are automatically traced
test = ma.testing.EvalTest(...)
result = test.run(agent)  # ← Logged to Langfuse
```

### Integration with DSPy & LangGraph

mahsm makes it seamless to evaluate both DSPy modules and LangGraph workflows:

```python
# Evaluate DSPy module
@ma.dspy_node
class QAAgent(dspy.Module):
    def forward(self, question: str) -> str:
        return answer

test.run(QAAgent())

# Evaluate LangGraph workflow
app = ma.create_graph(...)
test.run_workflow(app)
```

---

## Complete Example

### Define Agent

```python
import dspy
import mahsm as ma

# Configure DSPy
lm = dspy.LM(model="openai/gpt-4o-mini", api_key="...")
dspy.configure(lm=lm)

# Create agent
@ma.dspy_node
class MathTutor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> str:
        return self.cot(question=question).answer
```

### Create Test Suite

```python
from eval_protocol import EvalTest, EvalSuite, Grader

suite = EvalSuite(
    name="math_tutor_evaluation",
    tests=[
        EvalTest(
            name="basic_addition",
            input="What is 2 + 2?",
            expected_contains=["4"],
            grader=Grader.contains("4")
        ),
        EvalTest(
            name="word_problem",
            input="If I have 5 apples and give away 2, how many do I have?",
            expected_output="3",
            grader=Grader.llm_as_judge(
                criteria="Answer is mathematically correct and well-explained"
            )
        ),
        EvalTest(
            name="multi_step",
            input="Calculate (10 + 5) * 2 - 8",
            expected_output="22",
            grader=Grader.exact_match()
        )
    ]
)
```

### Run Evaluation

```python
# Initialize tracing
ma.tracing.init()

# Create agent
tutor = MathTutor()

# Run evaluation
results = suite.run(tutor)

# View results
print(f"Passed: {results.passed_count}/{results.total_count}")
print(f"Pass rate: {results.pass_rate:.1%}")
print(f"Average score: {results.average_score}/10")

# Individual test results
for test_result in results.test_results:
    print(f"\n{test_result.test_name}:")
    print(f"  Score: {test_result.score}/10")
    print(f"  Passed: {test_result.passed}")
    print(f"  Output: {test_result.output}")
```

---

## Typical Workflow

### 1. **Define Tests** (One-Time)

```python
# tests/eval_suite.py
from eval_protocol import EvalTest, EvalSuite, Grader

suite = EvalSuite(
    name="agent_quality",
    tests=[
        # Add your test cases
    ]
)
```

### 2. **Run During Development**

```python
# Iterate on your agent
agent = MyAgent()

# Run tests
results = suite.run(agent)

# Fix issues, re-run
# ... repeat ...
```

### 3. **Continuous Evaluation**

```python
# In CI/CD
def test_agent_quality():
    agent = MyAgent()
    results = suite.run(agent)
    
    assert results.pass_rate >= 0.9  # 90% pass rate required
    assert results.average_score >= 7.0  # Average score >= 7/10
```

### 4. **Track Over Time**

```python
# Compare versions
v1_results = suite.run(AgentV1())
v2_results = suite.run(AgentV2())

if v2_results.average_score > v1_results.average_score:
    print("✅ V2 is better!")
```

---

## Use Cases

### 1. **Regression Testing**

Ensure new changes don't break existing functionality:

```python
suite = EvalSuite(
    name="regression_tests",
    tests=[
        # Core functionality that must always work
    ]
)

# Run before/after code changes
results = suite.run(agent)
assert results.pass_rate == 1.0  # 100% must pass
```

### 2. **A/B Testing**

Compare different agent configurations:

```python
# Test different prompts
agent_a = Agent(prompt="Be concise")
agent_b = Agent(prompt="Be detailed")

results_a = suite.run(agent_a)
results_b = suite.run(agent_b)

# Choose the better one
best_agent = agent_a if results_a.score > results_b.score else agent_b
```

### 3. **Quality Gates**

Block deployments if quality drops:

```python
def quality_gate():
    results = suite.run(agent)
    
    if results.pass_rate < 0.95:
        raise Exception("Quality gate failed: pass rate too low")
    
    if results.average_score < 8.0:
        raise Exception("Quality gate failed: average score too low")
    
    print("✅ Quality gate passed")
```

### 4. **Prompt Engineering**

Systematically improve prompts:

```python
prompts = [
    "Answer the question directly",
    "Think step by step, then answer",
    "Analyze the question carefully before answering"
]

for prompt in prompts:
    agent = Agent(prompt=prompt)
    results = suite.run(agent)
    print(f"{prompt}: {results.average_score}/10")

# Pick the best prompt
```

---

## Grading Strategies

### Deterministic Grading

Fast, consistent, no LLM calls needed:

```python
# Exact match
Grader.exact_match()

# Contains substring
Grader.contains("expected text")

# Regex match
Grader.regex(r"\d{3}-\d{4}")  # Phone number format

# Custom function
def custom_grader(output: str, expected: str) -> float:
    # Your logic
    return score  # 0-10

Grader.custom(custom_grader)
```

### LLM-as-Judge

Use LLM to evaluate quality:

```python
Grader.llm_as_judge(
    criteria="Response is accurate, helpful, and polite",
    model="openai/gpt-4o",  # Optional, defaults to GPT-4
    scale=10  # Score out of 10
)
```

**Pros:**
- Evaluates semantic correctness
- Handles varied outputs
- Considers nuance

**Cons:**
- Costs money (LLM API calls)
- Slower than deterministic
- Non-deterministic (slight variance)

---

## Best Practices

### ✅ Do:

1. **Start with small test suites**
   ```python
   # Start with 5-10 core tests
   # Expand as you learn what matters
   ```

2. **Mix grading strategies**
   ```python
   suite = EvalSuite(tests=[
       EvalTest(..., grader=Grader.contains(...)),      # Fast check
       EvalTest(..., grader=Grader.llm_as_judge(...)),  # Quality check
   ])
   ```

3. **Test edge cases**
   ```python
   tests = [
       EvalTest(input="normal case", ...),
       EvalTest(input="edge case", ...),
       EvalTest(input="", ...),  # Empty input
       EvalTest(input="very long " * 1000, ...),  # Long input
   ]
   ```

4. **Version your test suites**
   ```python
   # tests/v1/suite.py
   # tests/v2/suite.py
   # Track how tests evolve
   ```

### ❌ Don't:

1. **Don't test implementation details**
   ```python
   # ❌ Bad - tests internal behavior
   EvalTest(input="...", expected_contains=["Chain-of-Thought"])
   
   # ✅ Good - tests output quality
   EvalTest(input="...", grader=Grader.llm_as_judge("Answer is correct"))
   ```

2. **Don't over-rely on exact matches**
   ```python
   # ❌ Bad - brittle
   Grader.exact_match()  # Breaks if wording changes slightly
   
   # ✅ Good - flexible
   Grader.llm_as_judge(criteria="Semantically equivalent")
   ```

3. **Don't skip failing tests**
   ```python
   # ❌ Bad
   # test.skip = True  # "Will fix later"
   
   # ✅ Good - fix or remove the test
   ```

---

## Viewing Results

### Console Output

```python
results = suite.run(agent)

# Summary
print(results.summary())

# Detailed breakdown
print(results.detailed_report())
```

### Langfuse Integration

When tracing is enabled, eval results appear in Langfuse:

```python
ma.tracing.init()

results = suite.run(agent)
# ✅ Results logged to Langfuse automatically
```

View in Langfuse UI:
- Test pass/fail status
- Scores per test
- Full trace of agent execution
- Grading rationale (for LLM-as-judge)

### Export to JSON

```python
import json

results = suite.run(agent)

# Export results
with open("eval_results.json", "w") as f:
    json.dump(results.to_dict(), f, indent=2)
```

---

## Next Steps

1. **[Eval Tests →](eval-tests.md)** - Deep dive into creating effective test cases
2. **[Langfuse Integration →](langfuse-integration.md)** - Connect evals to traces
3. **[LangGraph Integration →](langgraph-integration.md)** - Evaluate multi-step workflows

---

## External Resources

- **[EvalProtocol Documentation](https://pypi.org/project/eval-protocol/)** - Official docs
- **[LLM Evaluation Best Practices](https://eugeneyan.com/writing/llm-evaluation/)** - Eugene Yan's guide
- **[Testing LLM Applications](https://www.confident-ai.com/blog/testing-llm-applications)** - Testing strategies

---

**Next: Create robust test suites with [Eval Tests →](eval-tests.md)**
