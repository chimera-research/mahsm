# DSPy Optimizers

> **TL;DR**: Optimizers (teleprompts) automatically improve your DSPy modules by learning from examples—no manual prompt engineering needed.

## What are DSPy Optimizers?

**DSPy Optimizers** (also called **teleprompts**) are algorithms that automatically improve your modules by:
- Learning from training examples
- Generating better prompts
- Adding few-shot demonstrations
- Tuning instructions

Instead of manually tweaking prompts, you define success criteria and let the optimizer find the best approach.

---

## Why Use Optimizers?

### Manual Prompting (Traditional)

```python
# ❌ Manual iteration
prompt = "Answer the question: {question}"
# Try it... doesn't work well
prompt = "Think step by step and answer: {question}"
# Try again... better but not perfect
prompt = "You are an expert. Think carefully and answer: {question}"
# Keep iterating...
```

### Automatic Optimization (DSPy)

```python
# ✅ Define success metric
def accuracy(example, prediction):
    return example.answer.lower() in prediction.answer.lower()

# ✅ Let optimizer find the best approach
optimizer = BootstrapFewShot(metric=accuracy)
optimized_module = optimizer.compile(my_module, trainset=examples)
# Done! Module is automatically improved
```

---

## Core Concepts

### 1. **Metrics**

A metric function measures success:

```python
def my_metric(example, prediction, trace=None):
    """
    Args:
        example: Ground truth from trainset
        prediction: Module's output
        trace: Optional execution trace
    
    Returns:
        float or bool: Score (higher is better)
    """
    # Simple exact match
    return example.answer == prediction.answer

# Or more nuanced
def f1_metric(example, prediction):
    # Calculate F1 score
    precision = calculate_precision(example, prediction)
    recall = calculate_recall(example, prediction)
    return 2 * (precision * recall) / (precision + recall)
```

### 2. **Training Set**

Examples with inputs and expected outputs:

```python
import dspy

trainset = [
    dspy.Example(
        question="What is 2+2?",
        answer="4"
    ).with_inputs("question"),  # Mark what's an input
    
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("question"),
]
```

### 3. **Compilation**

The optimization process:

```python
optimizer = BootstrapFewShot(metric=accuracy)
optimized = optimizer.compile(
    student=my_module,      # Module to optimize
    trainset=trainset,      # Training examples
    teacher=None            # Optional better model
)
```

---

## Built-in Optimizers

### 1. **BootstrapFewShot**

Learns few-shot examples from your training data.

```python
from dspy.teleprompt import BootstrapFewShot

# Create optimizer
optimizer = BootstrapFewShot(
    metric=accuracy,
    max_bootstrapped_demos=4,  # Number of examples to add
    max_labeled_demos=4         # Max examples per prompt
)

# Optimize module
optimized_qa = optimizer.compile(
    student=qa_module,
    trainset=train_examples
)

# optimized_qa now includes learned few-shot examples!
```

**How it works:**
1. Runs your module on training examples
2. Keeps successful predictions as demonstrations
3. Adds them to future prompts automatically

**When to use:**
- You have labeled training data
- Few-shot learning helps your task
- You want quick improvements

**Example:**

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Define module
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Define metric
def exact_match(example, prediction):
    return example.answer.lower() == prediction.answer.lower()

# Create training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3*3?", answer="9").with_inputs("question"),
    # ... more examples
]

# Optimize
qa = QA()
optimizer = BootstrapFewShot(metric=exact_match)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# Use optimized version
result = optimized_qa(question="What is 5+5?")
print(result.answer)  # More likely to be correct!
```

---

### 2. **MIPRO** (Multi-prompt Instruction Proposal Optimizer)

Advanced optimizer that tunes instructions and demonstrations.

```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=accuracy,
    num_candidates=10,  # Number of prompt variations to try
    init_temperature=1.0
)

optimized = optimizer.compile(
    student=my_module,
    trainset=train_examples,
    valset=val_examples,  # Validation set for selection
    num_trials=20
)
```

**How it works:**
1. Generates multiple prompt instruction variations
2. Tests each on training data
3. Selects best based on validation performance

**When to use:**
- You have both training and validation sets
- You want the best possible performance
- You can afford longer optimization time

---

### 3. **BootstrapFewShotWithRandomSearch**

Combines few-shot learning with random search over hyperparameters.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=accuracy,
    max_bootstrapped_demos=4,
    num_candidate_programs=10,  # Number of variations to try
    num_threads=4               # Parallel evaluation
)

optimized = optimizer.compile(
    student=my_module,
    trainset=train_examples,
    valset=val_examples
)
```

---

### 4. **BayesianSignatureOptimizer**

Uses Bayesian optimization to find best prompts.

```python
from dspy.teleprompt import BayesianSignatureOptimizer

optimizer = BayesianSignatureOptimizer(
    metric=accuracy,
    n=20  # Number of optimization steps
)

optimized = optimizer.compile(
    student=my_module,
    trainset=train_examples
)
```

---

## Advanced: Teacher-Student Optimization

Use a stronger model (teacher) to generate labels for a weaker model (student):

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Configure teacher (expensive, high-quality model)
teacher_lm = dspy.LM('openai/gpt-4o', api_key=os.getenv("OPENAI_API_KEY"))

# Configure student (cheap, fast model)
student_lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))

# Create modules
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Teacher uses GPT-4o
with dspy.settings.context(lm=teacher_lm):
    teacher = QA()

# Student uses GPT-4o-mini
with dspy.settings.context(lm=student_lm):
    student = QA()

# Bootstrap student from teacher
optimizer = BootstrapFewShot(metric=accuracy)
optimized_student = optimizer.compile(
    student=student,
    teacher=teacher,  # Use teacher to generate examples
    trainset=trainset
)

# optimized_student achieves near-teacher quality at student cost!
```

---

## Optimization in mahsm

Optimize mahsm modules just like regular DSPy modules:

```python
import mahsm as ma
from typing import TypedDict
from dspy.teleprompt import BootstrapFewShot

# Define state
class QAState(TypedDict):
    question: str
    answer: str

# Define module
@ma.dspy_node
class QA(ma.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# Create training data
trainset = [
    dspy.Example(question="What is DSPy?", answer="A framework...").with_inputs("question"),
    # ... more examples
]

# Optimize
qa = QA()
optimizer = BootstrapFewShot(metric=lambda e, p: e.answer in p.answer)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# Use in workflow
workflow = ma.graph.StateGraph(QAState)
workflow.add_node("qa", optimized_qa)  # Use optimized version!
```

---

## Best Practices

### ✅ Do:

1. **Start with BootstrapFewShot**
   ```python
   # ✅ Simple and effective
   optimizer = BootstrapFewShot(metric=accuracy)
   ```

2. **Use meaningful metrics**
   ```python
   # ✅ Task-specific
   def f1_score(example, prediction):
       return calculate_f1(example, prediction)
   ```

3. **Use validation sets for selection**
   ```python
   # ✅ Prevents overfitting
   optimizer.compile(student, trainset=train, valset=val)
   ```

4. **Start small, then scale**
   ```python
   # ✅ Iterate on 10 examples first
   trainset_small = trainset[:10]
   optimized = optimizer.compile(module, trainset=trainset_small)
   # Then use full dataset
   ```

5. **Save optimized modules**
   ```python
   # ✅ Don't re-optimize every time
   optimized.save("optimized_qa.json")
   loaded = QA()
   loaded.load("optimized_qa.json")
   ```

### ❌ Don't:

1. **Optimize without evaluation**
   ```python
   # ❌ How do you know it's better?
   optimized = optimizer.compile(module, trainset=data)
   # ✅ Always evaluate
   score = evaluate(optimized, testset)
   ```

2. **Use tiny training sets**
   ```python
   # ❌ 2 examples won't help
   trainset = [example1, example2]
   # ✅ Use at least 20-50 examples
   ```

3. **Over-optimize on training data**
   ```python
   # ❌ Overfitting risk
   max_bootstrapped_demos=100  # Too many!
   # ✅ Use 3-5 demonstrations
   max_bootstrapped_demos=4
   ```

---

## Evaluation After Optimization

Always evaluate on a held-out test set:

```python
from dspy.evaluate import Evaluate

# Create evaluator
evaluator = Evaluate(
    devset=testset,
    metric=accuracy,
    num_threads=4,
    display_progress=True
)

# Compare before and after
baseline_score = evaluator(original_module)
optimized_score = evaluator(optimized_module)

print(f"Baseline: {baseline_score}%")
print(f"Optimized: {optimized_score}%")
print(f"Improvement: {optimized_score - baseline_score}%")
```

---

## Complete Optimization Pipeline

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# 1. Define module
class QA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.qa(question=question)

# 2. Create datasets
full_data = load_data()
train, val, test = split_data(full_data, [0.7, 0.15, 0.15])

# 3. Define metric
def exact_match(example, prediction):
    return example.answer.lower() == prediction.answer.lower()

# 4. Optimize
qa = QA()
optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4)
optimized_qa = optimizer.compile(student=qa, trainset=train, valset=val)

# 5. Evaluate
evaluator = Evaluate(devset=test, metric=exact_match)
baseline_score = evaluator(qa)
optimized_score = evaluator(optimized_qa)

print(f"Baseline: {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")

# 6. Save
if optimized_score > baseline_score:
    optimized_qa.save("best_qa_model.json")
```

---

## Comparison Table

| Optimizer | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| **BootstrapFewShot** | ⚡⚡⚡ Fast | ⭐⭐ Good | Quick improvements |
| **MIPRO** | ⚡ Slow | ⭐⭐⭐ Best | Maximum quality |
| **BootstrapFewShotWithRandomSearch** | ⚡⚡ Medium | ⭐⭐⭐ Better | Balanced |
| **BayesianSignatureOptimizer** | ⚡ Slow | ⭐⭐⭐ Better | Complex tasks |

---

## Next Steps

- **[Best Practices](best-practices.md)** → Production DSPy tips
- **[Optimization Workflow Guide](../../guides/optimization-workflow.md)** → Complete tutorial
- **[LangGraph Overview](../langgraph/overview.md)** → Learn about workflows

---

## External Resources

- **[DSPy Optimizers Docs](https://dspy.ai/docs/building-blocks/optimizers)** - Official guide
- **[DSPy Optimization Paper](https://arxiv.org/abs/2310.03714)** - Research paper

---

**Next: Explore [Best Practices →](best-practices.md)**
