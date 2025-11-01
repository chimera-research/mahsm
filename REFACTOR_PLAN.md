# mahsm v0.2.0 Refactoring Plan

**Goal**: Clean up the codebase, fix core DSPy-LangGraph integration issues, improve documentation, and establish a sustainable architecture for future development.

---

## 🗑️ Phase 0: Cleanup & Removal

**Goal**: Remove unnecessary files and get to a clean baseline.

### Files to Delete:
- [x] `PUBLISHING.md` - Not needed in repo (PyPI publishing is automated)
- [x] `SETUP_SUMMARY.md` - Internal-only documentation
- [x] `doc_agent.py` - Test/prototype file
- [x] `test_doc_agent.py` - Test/prototype file
- [x] `tests/proto/` - Entire folder with prototype code

### Files to Keep:
- ✅ `README.md` - Main landing page
- ✅ `QUICKSTART.md` - Will be overhauled but framework is good
- ✅ `.github/workflows/` - CI/CD infrastructure
- ✅ Core library files in `mahsm/`
- ✅ Core tests in `tests/`

**Deliverable**: Clean repository with only essential files.

---

## 🔧 Phase 1: Core Library Fixes

**Goal**: Fix the fundamental DSPy-LangGraph integration and address warnings.

### 1.1 Fix `@dspy_node` Implementation
**Current Issues**:
- Calling `forward()` directly is discouraged by DSPy
- Should use `module(**inputs)` which internally calls forward
- Implementation is verbose with too much introspection
- Warning: "Unable to find output template/schema" from DSPy

**Tasks**:
- [ ] Simplify `@dspy_node` decorator to use `module(**inputs)` instead of `forward()`
- [ ] Reduce complexity - make it a simpler wrapper
- [ ] Fix output template/schema warning
- [ ] Test with built-in DSPy modules (ChainOfThought, Predict, etc.)
- [ ] Document the proper usage pattern

**Success Criteria**:
- No DSPy warnings when using decorated modules
- Simpler, more maintainable code
- All existing tests still pass

### 1.2 Module Architecture Refactoring
**Current State**: Everything in `mahsm/core.py` and `mahsm/testing.py`

**New Architecture**:
```
mahsm/
├── __init__.py           # Main exports
├── core.py              # Core @dspy_node decorator only
├── tracing.py           # NEW: Langfuse integration
│   ├── init()
│   ├── @observe decorator
│   ├── DSPy instrumentation
│   └── LangGraph callbacks
└── testing.py           # EvalProtocol integration (existing)
    ├── PytestHarness
    └── evaluation helpers
```

**Tasks**:
- [ ] Create `mahsm/tracing.py` with Langfuse integration
- [ ] Move `init()` from `core.py` to `tracing.py`
- [ ] Add `@observe` decorator for manual trace decoration
- [ ] Update `mahsm/__init__.py` to expose new structure:
  ```python
  # Core
  from .core import dspy_node
  
  # Tracing
  from .tracing import init, observe
  
  # Testing
  try:
      from .testing import PytestHarness, ...
  except ImportError:
      pass  # Graceful degradation on Windows
  ```
- [ ] Update all imports across codebase

**Deliverable**: Clean, separated concerns with focused modules.

---

## 📚 Phase 2: Documentation Overhaul

**Goal**: Create real, working examples using actual DSPy/LangGraph patterns from their documentation.

### 2.1 MkDocs Setup
**Current State**: `mkdocs.yml` exists but not fully configured

**Tasks**:
- [ ] Set up MkDocs with Material theme
- [ ] Configure local preview: `mkdocs serve`
- [ ] Configure GitHub Pages deployment: `mkdocs gh-deploy`
- [ ] Add to CI: Auto-deploy docs on push to main
- [ ] Add navigation structure:
  ```yaml
  nav:
    - Home: index.md
    - Getting Started: getting-started.md
    - Core Concepts: concepts.md
    - Examples:
      - Basic Agent: examples/basic.md
      - Multi-Agent System: examples/multi-agent.md
      - With Evaluation: examples/evaluation.md
    - API Reference: api.md
    - Tracing Guide: tracing.md
    - Testing Guide: testing.md
  ```

**Deliverable**: Working MkDocs site with live preview capability.

### 2.2 Create Real, Working Example
**Goal**: Replace placeholder code with actual working example based on DSPy/LangGraph tutorials.

**Inspiration Sources**:
- DSPy tutorials: ChainOfThought, ReAct, Multi-hop QA
- LangGraph examples: Agent with tools, multi-agent collaboration
- User's Jupyter notebook concept

**Example Structure**:
```python
# Real DSPy signature
class ResearchSignature(dspy.Signature):
    """Research a topic and provide detailed answer with sources."""
    query: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved context")
    answer: str = dspy.OutputField(desc="Detailed answer")
    sources: list[str] = dspy.OutputField(desc="Source citations")

# Real DSPy module
@ma.dspy_node
class Researcher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research = dspy.ChainOfThought(ResearchSignature)
    
    def forward(self, query, context):
        return self.research(query=query, context=context)

# Real LangGraph state
class AgentState(TypedDict):
    query: str
    context: str
    answer: str
    sources: list[str]
    iterations: int

# Real retrieval step (not placeholder)
def retrieve(state: AgentState) -> AgentState:
    # Actual ColBERT/vector search
    results = dspy.ColBERTv2()(query=state["query"])
    return {"context": "\n".join(results.passages)}

# Real conditional logic
def should_continue(state: AgentState) -> str:
    if state["iterations"] >= 3:
        return "end"
    if "uncertain" in state["answer"].lower():
        return "refine"
    return "end"
```

**Tasks**:
- [ ] Create `examples/research_agent.py` with complete working example
- [ ] Use real DSPy retrieval (ColBERT or similar)
- [ ] Use real DSPy signatures and modules
- [ ] Show proper state updates
- [ ] Include conditional edges and routing
- [ ] Add proper type hints throughout
- [ ] Include visualization: `graph.get_graph().draw_mermaid()`

**Deliverable**: Fully functional example that demonstrates all key features.

### 2.3 Rewrite QUICKSTART.md
**Tasks**:
- [ ] Base on the new working example
- [ ] Use actual DSPy/LangGraph patterns
- [ ] Show tracing setup: `ma.init()`
- [ ] Show evaluation setup with real test
- [ ] Remove placeholder code
- [ ] Add troubleshooting section

### 2.4 Update README.md
**Tasks**:
- [ ] Update quick start with new example
- [ ] Update architecture diagram to show new module structure
- [ ] Add badges for docs (links to GitHub Pages)
- [ ] Update feature comparison if needed

---

## 🧪 Phase 3: Testing & Validation

**Goal**: Ensure everything works end-to-end.

### 3.1 Update Tests
**Tasks**:
- [ ] Update `tests/test_core.py` for new `@dspy_node` implementation
- [ ] Update imports for new module structure (`ma.tracing.init()`)
- [ ] Add test for `@observe` decorator
- [ ] Ensure all 11 tests still pass

### 3.2 Create Integration Example Test
**Tasks**:
- [ ] Create `tests/test_research_agent.py` based on the new example
- [ ] Test with mock LLM responses (no API calls in CI)
- [ ] Verify graph compilation
- [ ] Verify state transitions

### 3.3 Manual Verification
**Tasks**:
- [ ] Run the research agent example manually with real API
- [ ] Verify traces appear in Langfuse
- [ ] Verify no warnings from DSPy
- [ ] Test on Linux, macOS, Windows (CI will help)

**Deliverable**: All tests pass, no warnings, example runs cleanly.

---

## 📦 Phase 4: Version Bump & Release

**Goal**: Package everything up for v0.2.0 release.

### 4.1 Update Package Metadata
**Tasks**:
- [ ] Bump version to `0.2.0` in `pyproject.toml`
- [ ] Update package description if needed
- [ ] Ensure all dependencies are correct

### 4.2 Create CHANGELOG.md
**Tasks**:
- [ ] Document all changes from v0.1.0 to v0.2.0:
  - Fixed DSPy warnings
  - Refactored module architecture
  - Improved documentation
  - Real working examples
  - MkDocs setup

### 4.3 Release
**Tasks**:
- [ ] Create PR for review
- [ ] Merge to main
- [ ] Create GitHub Release: `v0.2.0`
- [ ] Verify PyPI auto-publish
- [ ] Deploy docs to GitHub Pages

---

## 🔮 Future (v0.3.0+)

These are identified for later:
- Tuning/optimization integrations (separate PR in progress)
- Application-specific features
- Additional LangGraph features (checkpointing, etc.)
- More automation helpers
- Performance optimizations

---

## 📋 Summary Checklist

### Phase 0: Cleanup ✅
- [ ] Remove unnecessary MD files
- [ ] Remove doc_agent.py and test_doc_agent.py
- [ ] Remove tests/proto/
- [ ] Commit cleanup

### Phase 1: Core Fixes 🔧
- [ ] Simplify @dspy_node
- [ ] Fix DSPy warnings
- [ ] Create mahsm/tracing.py
- [ ] Refactor module structure
- [ ] Update tests

### Phase 2: Documentation 📚
- [ ] Set up MkDocs
- [ ] Create real working example
- [ ] Rewrite QUICKSTART.md
- [ ] Update README.md

### Phase 3: Validation 🧪
- [ ] All tests pass
- [ ] No warnings
- [ ] Example runs end-to-end

### Phase 4: Release 📦
- [ ] Bump to v0.2.0
- [ ] Create CHANGELOG
- [ ] Release on GitHub/PyPI
- [ ] Deploy docs

---

**Let's tackle these phases one at a time, starting with Phase 0!**
