# mahsm v0.2.0 Testing Plan

## Overview

Comprehensive testing strategy for mahsm covering all modules except `tuning.py` (under active development).

---

## Test Categories

### 1. **Smoke Tests** (Basic Sanity)
Quick tests to verify the system isn't fundamentally broken.

- ✅ Package imports successfully
- ✅ Core modules are accessible
- ✅ Basic decorator works without LLM calls
- ✅ Graph construction doesn't crash

**Location:** `tests/test_smoke.py`

---

### 2. **Unit Tests** (Module Isolation)
Test individual functions and classes in isolation.

#### `mahsm/core.py` (dspy_node decorator)
- ✅ Decorator on class
- ✅ Wrapper on instance  
- ✅ Built-in module wrapping
- ✅ Invalid input handling
- ✅ State field mapping
- ✅ 'self' parameter exclusion
- ✅ Output extraction
- ⬜ Multiple forward parameters
- ⬜ Optional parameters handling
- ⬜ Missing state fields handling
- ⬜ Private field filtering

**Location:** `tests/test_core.py` (expand existing)

#### `mahsm/tracing.py` (Langfuse integration)
- ⬜ Init with valid credentials
- ⬜ Init with missing credentials (warning behavior)
- ⬜ Init with custom host
- ⬜ Environment variable reading
- ⬜ CallbackHandler creation
- ⬜ DSPy instrumentation
- ⬜ observe decorator availability
- ⬜ Graceful degradation when Langfuse unavailable

**Location:** `tests/test_tracing.py` (new)

#### `mahsm/testing.py` (EvalProtocol integration)
- ⬜ PytestHarness initialization
- ⬜ Langfuse data loader configuration
- ⬜ Rollout processor creation
- ⬜ Graph factory behavior
- ⬜ Model override during tests

**Location:** `tests/test_testing_module.py` (new)

#### `mahsm/__init__.py` (Package exports)
- ⬜ All exported symbols are accessible
- ⬜ Version string is present
- ⬜ Namespaces are properly structured
- ⬜ Convenience exports work correctly
- ⬜ Graceful testing module import failure on Windows

**Location:** `tests/test_init.py` (new)

#### `mahsm/dspy.py` & `mahsm/graph.py` (Re-exports)
- ⬜ All DSPy exports are accessible
- ⬜ All LangGraph exports are accessible
- ⬜ No import errors

**Location:** `tests/test_exports.py` (new)

---

### 3. **Integration Tests** (Module Interactions)
Test how modules work together.

#### LangGraph Integration
- ✅ Single node graph execution
- ✅ Multi-node sequential graph
- ✅ Conditional routing
- ✅ Instance-wrapped nodes
- ✅ State preservation
- ⬜ Complex multi-path workflows
- ⬜ Loops and cycles
- ⬜ Error propagation through graph
- ⬜ Graph with mixed node types (dspy + regular functions)

**Location:** `tests/test_graph_integration.py` (expand existing)

#### Tracing Integration
- ⬜ DSPy module execution with tracing enabled
- ⬜ LangGraph workflow with callback handler
- ⬜ Manual @observe decorator usage
- ⬜ Trace visibility in Langfuse (mock verification)
- ⬜ Tracing with missing credentials (graceful failure)

**Location:** `tests/test_tracing_integration.py` (new)

#### Testing Module Integration  
- ⬜ PytestHarness with compiled graph
- ⬜ Evaluation data loading from Langfuse
- ⬜ Rollout processor execution
- ⬜ Model swapping during tests

**Location:** `tests/test_testing_integration.py` (new)

---

### 4. **End-to-End Tests** (Full Workflows)
Test complete user workflows from start to finish.

#### Scenario 1: Simple Agent
- ⬜ Configure DSPy LM
- ⬜ Create DSPy module with @dspy_node
- ⬜ Build LangGraph workflow
- ⬜ Execute workflow
- ⬜ Verify outputs

**Location:** `tests/e2e/test_simple_agent.py` (new)

#### Scenario 2: Multi-Agent System
- ⬜ Multiple DSPy agents
- ⬜ Complex routing logic
- ⬜ State management across agents
- ⬜ End-to-end execution

**Location:** `tests/e2e/test_multi_agent.py` (new)

#### Scenario 3: Traced Workflow
- ⬜ Initialize tracing
- ⬜ Build and execute workflow
- ⬜ Verify traces are created
- ⬜ Manual spans with @observe

**Location:** `tests/e2e/test_traced_workflow.py` (new)

#### Scenario 4: Evaluated Agent
- ⬜ Create agent
- ⬜ Set up test harness
- ⬜ Run evaluation suite
- ⬜ Assert quality metrics

**Location:** `tests/e2e/test_evaluated_agent.py` (new)

---

## Implementation Phases

### **Phase 1: Foundation** (Smoke + Core Unit Tests)
Complete basic sanity checks and core decorator testing.

**Files:**
- `tests/test_smoke.py` (new)
- `tests/test_core.py` (expand)
- `tests/test_init.py` (new)
- `tests/test_exports.py` (new)

**Goal:** Ensure package fundamentals work correctly.

---

### **Phase 2: Module Units** (Individual Module Testing)
Test each module's functionality in isolation.

**Files:**
- `tests/test_tracing.py` (new)
- `tests/test_testing_module.py` (new)

**Goal:** 100% coverage of individual module functionality.

---

### **Phase 3: Integration** (Module Interactions)
Test how modules work together.

**Files:**
- `tests/test_graph_integration.py` (expand)
- `tests/test_tracing_integration.py` (new)
- `tests/test_testing_integration.py` (new)

**Goal:** Verify seamless integration between components.

---

### **Phase 4: End-to-End** (Full Workflows)
Test complete user scenarios.

**Files:**
- `tests/e2e/test_simple_agent.py` (new)
- `tests/e2e/test_multi_agent.py` (new)
- `tests/e2e/test_traced_workflow.py` (new)
- `tests/e2e/test_evaluated_agent.py` (new)

**Goal:** Validate real-world usage patterns.

---

## Test Infrastructure

### Fixtures & Utilities

Create shared test utilities:

**`tests/conftest.py`:**
- Mock DSPy LM configuration
- Mock Langfuse credentials
- Common state type fixtures
- Sample module definitions

**`tests/utils.py`:**
- Mock result generators
- State builders
- Assertion helpers

---

## Coverage Targets

**Minimum Coverage:** 80% overall
- `mahsm/core.py`: 95%
- `mahsm/tracing.py`: 85%
- `mahsm/testing.py`: 85%
- `mahsm/__init__.py`: 90%
- `mahsm/dspy.py`: 100%
- `mahsm/graph.py`: 100%

**Excluded:**
- `mahsm/tuning.py` (under development)

---

## Test Execution

### Local Development
```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/test_smoke.py -v
pytest tests/test_core.py -v
pytest tests/test_graph_integration.py -v

# Run with coverage
pytest tests/ --cov=mahsm --cov-report=html

# Run fast tests only (skip e2e)
pytest tests/ -m "not e2e"
```

### CI/CD
- Run on every PR
- Test on Python 3.10, 3.11, 3.12
- Test on Ubuntu, macOS, Windows
- Generate coverage report
- Fail if coverage < 80%

---

## Success Criteria

For v0.2.0 release, we must have:

✅ All Phase 1 tests passing (smoke + core)
✅ All Phase 2 tests passing (module units)
✅ All Phase 3 tests passing (integration)
✅ At least 50% of Phase 4 tests passing (e2e)
✅ Overall coverage >= 80%
✅ CI/CD pipeline green on all platforms
✅ No critical bugs or regressions

---

## Notes

- **Mock LLM calls:** Use `unittest.mock` to avoid actual API calls in unit tests
- **Integration tests:** May use mock LLMs or skip if credentials unavailable
- **E2E tests:** Mark as `@pytest.mark.e2e` and make optional (require API keys)
- **Windows compatibility:** Handle eval-protocol import failures gracefully
- **Parallel execution:** Tests should be independent and parallelizable

---

**Status:** Planning Complete
**Next:** Begin Phase 1 Implementation
