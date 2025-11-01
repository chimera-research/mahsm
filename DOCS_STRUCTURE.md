# mahsm Documentation Structure - Phase 2

## Goal
Create world-class, comprehensive documentation covering all building blocks (DSPy, LangGraph, Langfuse, EvalProtocol) with no prior framework experience needed.

## Documentation Architecture

```
docs/
├── index.md                          # Landing page with overview
├── getting-started/
│   ├── installation.md              # ✅ Exists
│   ├── quickstart.md                # ✅ Exists (needs DSPy fix)
│   └── core-concepts.md             # NEW - mahsm philosophy & workflow
├── building-blocks/
│   ├── dspy/
│   │   ├── overview.md              # What is DSPy, why use it
│   │   ├── signatures.md            # Input/output specifications
│   │   ├── modules.md               # Predict, ChainOfThought, ReAct, etc.
│   │   ├── optimizers.md            # BootstrapFewShot, MIPROv2, etc.
│   │   └── best-practices.md        # Common patterns & tips
│   ├── langgraph/
│   │   ├── overview.md              # What is LangGraph, why use it
│   │   ├── state.md                 # TypedDict, state management
│   │   ├── nodes-edges.md           # Building blocks of graphs
│   │   ├── conditional-routing.md   # Branching logic
│   │   ├── compilation.md           # compile(), invoke(), stream()
│   │   └── visualization.md         # Mermaid diagrams
│   ├── langfuse/
│   │   ├── overview.md              # What is Langfuse, why use it
│   │   ├── initialization.md        # ma.tracing.init()
│   │   ├── dspy-tracing.md          # Automatic DSPy instrumentation
│   │   ├── langgraph-tracing.md     # Callback handlers
│   │   └── manual-tracing.md        # @observe decorator
│   └── evalprotocol/
│       ├── overview.md              # What is EvalProtocol, why use it
│       ├── evaluation-tests.md      # @evaluation_test decorator
│       ├── langfuse-integration.md  # Syncing scores to Langfuse
│       └── langgraph-integration.md # PytestHarness
├── guides/
│   ├── first-agent.md              # Step-by-step: Build your first agent
│   ├── multi-agent-systems.md     # Multi-agent coordination
│   ├── adding-tools.md             # Function calling & tools
│   ├── optimization-workflow.md    # DSPy optimization pipeline
│   └── production-deployment.md    # Best practices for prod
├── examples/
│   ├── basic-qa.md                 # Simple Q&A bot
│   ├── research-agent.md           # Full research pipeline
│   ├── multi-hop-reasoning.md      # Complex reasoning chains
│   └── evaluation-pipeline.md      # Complete eval setup
└── api/
    ├── core.md                     # @dspy_node decorator
    ├── tracing.md                  # ma.tracing module
    └── testing.md                  # ma.testing module
```

## Content Requirements

### For Each Building Block Section:

1. **Overview Page**
   - What is it? (2-3 sentences)
   - Why does mahsm use it?
   - Key concepts (3-5 bullet points)
   - Visual diagram if applicable

2. **Detailed Topic Pages**
   - Clear explanation (no jargon)
   - Code examples (real, working code)
   - Common patterns
   - Gotchas & troubleshooting
   - Links to official docs

3. **Integration with mahsm**
   - How mahsm simplifies it
   - mahsm-specific patterns
   - Migration from vanilla usage

## Priority Order for Creation

### Phase 2A (Critical - Start Here):
1. ✅ mkdocs.yml navigation structure
2. Landing page (index.md) overhaul
3. DSPy overview + signatures + modules
4. LangGraph overview + state + nodes-edges
5. Your First Agent guide

### Phase 2B (High Priority):
6. Langfuse overview + initialization + dspy-tracing
7. EvalProtocol overview + evaluation-tests
8. Research Agent example (complete)
9. API reference for core

### Phase 2C (Complete Coverage):
10. All remaining building block pages
11. All remaining guides
12. All remaining examples
13. API reference for tracing & testing

## Style Guide

### Writing Principles:
- **Clear**: Assume no prior framework knowledge
- **Practical**: Every concept has a code example
- **Progressive**: Start simple, build complexity
- **Integrated**: Show how mahsm ties everything together

### Code Examples:
```python
# ✅ Good: Real, runnable code
import mahsm as ma
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# ❌ Bad: Placeholder/pseudo-code
model = SomeModel(...)  # Configure your model
```

### Structure:
```markdown
# Page Title

> **TL;DR**: One-sentence summary of what reader will learn.

## What is [Topic]?

Brief explanation (2-3 paragraphs).

## Key Concepts

- **Concept 1**: Explanation
- **Concept 2**: Explanation

## Basic Example

\`\`\`python
# Simple, working code
\`\`\`

## Advanced Usage

\`\`\`python
# More complex patterns
\`\`\`

## mahsm Integration

How mahsm makes this easier.

## Next Steps

- Link to related pages
- Link to examples
```

## Success Metrics

- ✅ User can learn all 4 frameworks from our docs alone
- ✅ Every page has working code examples
- ✅ Clear progression from beginner to advanced
- ✅ mahsm integration is always clear
- ✅ Navigation is intuitive

---

**Status**: Structure created, ready for content development
**Next**: Start with Phase 2A critical pages
