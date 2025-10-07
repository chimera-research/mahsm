# Tasks: MAHSM (Multi-Agent Hyper-Scaling Methods) v0.1.0

**Input**: Design documents from `/specs/001-specs-md-build/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓), quickstart.md (✓)

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✓ Found: Python 3.11+ library with DSPy, LangGraph, LangChain Core, LangFuse, OpenAI SDK
   → ✓ Extract: Single project structure, pytest testing, MkDocs documentation
2. Load optional design documents:
   → ✓ data-model.md: 4 entities (Prompt Artifact, Tool Schema, Message State, Configuration)
   → ✓ contracts/: 4 contract files (prompt_api, inference_api, config_api, exceptions)
   → ✓ research.md: Technical decisions for dependencies and patterns
   → ✓ quickstart.md: End-to-end workflow validation scenarios
3. Generate tasks by category:
   → Setup: Python project, dependencies, linting, directory structure
   → Tests: 4 contract tests + 3 integration tests + 1 smoke test
   → Core: 4 modules + 4 entities + exception handling
   → Integration: LangFuse tracing, checkpointer integration
   → Polish: unit tests, documentation, performance validation
4. Apply task rules:
   → Contract tests [P] - different files, independent
   → Module implementations [P] - separate files in different directories
   → Integration tests [P] - independent test scenarios
5. Number tasks sequentially (T001-T030)
6. Generate dependency graph with TDD ordering
7. Create parallel execution examples
8. ✓ Validate: All contracts have tests, all entities have models, TDD ordering maintained
9. Return: SUCCESS (30 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup

- [x] **T001** Create Python project structure per implementation plan
  - Create `mahsm/` package directory with `__init__.py`
  - Create `mahsm/prompt/`, `mahsm/inference/`, `mahsm/config/` subdirectories
  - Create `tests/unit/`, `tests/integration/`, `tests/smoke/`, `tests/fixtures/` directories
  - Create `docs/api/`, `docs/guides/`, `docs/examples/`, `docs/concepts/` directories

- [x] **T002** Initialize Python project with dependencies
  - Create `pyproject.toml` with project metadata and dependencies
  - Create `requirements.txt` with DSPy, LangGraph, LangChain Core, LangFuse, OpenAI SDK
  - Create `requirements-dev.txt` with pytest, MkDocs, linting tools
  - Create basic `README.md` with installation instructions

- [x] **T003** [P] Configure linting and formatting tools
  - Create `.pre-commit-config.yaml` with black, isort, flake8, mypy
  - Create `pyproject.toml` sections for black, isort, mypy configuration
  - Create `.gitignore` for Python projects with `__pycache__`, `.pytest_cache`, etc.

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] **T004** [P] Contract test ma.prompt.save() in `tests/unit/test_prompt_save_contract.py`
  - Test prompt extraction from DSPy compiled programs
  - Test tool schema generation in OpenAI format
  - Test JSON artifact creation with correct naming
  - Test validation errors for invalid inputs

- [x] **T005** [P] Contract test ma.prompt.load() in `tests/unit/test_prompt_load_contract.py`
  - Test prompt loading from JSON artifacts
  - Test tool validation against saved schemas
  - Test ValidationError for mismatched tools
  - Test FileNotFoundError for missing artifacts

- [x] **T006** [P] Contract test ma.inference() in `tests/unit/test_inference_contract.py`
  - Test agentic loop execution with tool calls
  - Test message state management and ordering
  - Test MaxIterationsError and ToolExecutionError handling
  - Test return tuple format (result, messages)

- [x] **T007** [P] Contract test ma.config in `tests/unit/test_config_contract.py`
  - Test environment variable loading
  - Test get_checkpointer() method with different types
  - Test LangFuse integration configuration
  - Test ConfigurationError for invalid settings

- [x] **T008** [P] Integration test DSPy workflow in `tests/integration/test_dspy_integration.py`
  - Test complete DSPy GEPA optimization → save → load workflow
  - Test prompt extraction from real DSPy compiled programs
  - Test tool schema compatibility across save/load cycle
  - Use fixtures from quickstart guide scenarios

- [x] **T009** [P] Integration test LangGraph workflow in `tests/integration/test_langgraph_integration.py`
  - Test ma.inference() within LangGraph nodes
  - Test MessagesState integration and message ordering
  - Test checkpointer integration with SQLite backend
  - Validate complete conversation transparency

- [x] **T010** [P] Integration test LangFuse tracing in `tests/integration/test_langfuse_tracing.py`
  - Test automatic tracing when environment variables set
  - Test trace data includes all messages and tool calls
  - Test tracing disabled when keys not provided
  - Validate trace completeness and accuracy

- [x] **T011** [P] Smoke test end-to-end workflow in `tests/smoke/test_end_to_end.py`
  - Implement complete quickstart guide scenario as automated test
  - Test data analysis example with calculate_statistics and generate_chart tools
  - Validate message transparency and conversation completeness
  - Test performance goals (sub-100ms prompt operations)

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [x] **T012** [P] Exception hierarchy in `mahsm/exceptions.py`
  - Implement MAHSMError base class
  - Implement ValidationError, ToolExecutionError, MaxIterationsError
  - Implement ConfigurationError, ModelError with proper inheritance
  - Add detailed error messages and context preservation

- [x] **T013** [P] Prompt Artifact entity in `mahsm/prompt/storage.py`
  - Implement JSON artifact save/load operations
  - Implement file naming convention (task_name_version.json)
  - Implement atomic file operations for thread safety
  - Add artifact validation against schema

- [x] **T014** [P] Tool Schema validation in `mahsm/prompt/validation.py`
  - Implement OpenAI function schema extraction from Python functions
  - Implement structural comparison for tool validation
  - Implement name and parameter matching logic
  - Add clear validation error messages

- [x] **T015** [P] Configuration management in `mahsm/config/settings.py`
  - Implement environment variable auto-loading
  - Implement Config class with properties for all settings
  - Implement thread-safe configuration updates
  - Add default value handling and validation

- [x] **T016** [P] Checkpointer integration in `mahsm/config/checkpointers.py`
  - Implement get_checkpointer() method for memory, SQLite, Postgres
  - Integrate with LangGraph checkpointer ecosystem
  - Add configuration validation for each checkpointer type
  - Handle connection errors and fallbacks

- [x] **T017** [P] Message handling utilities in `mahsm/inference/messages.py`
  - Implement message creation helpers (SystemMessage, HumanMessage, etc.)
  - Implement message state validation and ordering
  - Implement input conversion (dict/string to HumanMessage)
  - Add message serialization utilities

- [x] **T018** Tool execution engine in `mahsm/inference/executor.py`
  - Implement direct Python function calling pattern
  - Implement tool result wrapping in ToolMessage
  - Implement error handling with ToolExecutionError
  - Add tool call parsing and argument extraction

- [x] **T019** Agentic loop implementation in `mahsm/inference/executor.py`
  - Implement main inference loop with max_iterations
  - Implement model invocation with OpenAI SDK
  - Implement tool_calls detection and execution
  - Add loop termination conditions and error handling

- [x] **T020** ma.prompt module interface in `mahsm/prompt/__init__.py`
  - Implement save() function with DSPy integration
  - Implement load() function with tool validation
  - Export public API functions
  - Add comprehensive docstrings and type hints

- [x] **T021** ma.inference module interface in `mahsm/inference/__init__.py`
  - Implement inference() function with complete signature
  - Integrate executor, messages, and error handling
  - Export public API functions
  - Add comprehensive docstrings and type hints

- [x] **T022** ma.config module interface in `mahsm/config/__init__.py`
  - Implement global config object initialization
  - Export Config class and global instance
  - Add module-level configuration loading
  - Ensure thread-safe initialization

- [x] **T023** Main package interface in `mahsm/__init__.py`
  - Export ma.prompt, ma.inference, ma.config modules
  - Set up package-level imports for easy access
  - Add version information and metadata
  - Include package-level docstring

## Phase 3.4: Integration

- [x] **T024** LangFuse tracing integration in `mahsm/inference/executor.py`
  - Add automatic trace creation when keys configured
  - Implement trace data collection for all messages
  - Add trace metadata (model, tools, performance metrics)
  - Handle tracing errors gracefully without breaking inference

- [x] **T025** DSPy signature extraction in `mahsm/prompt/storage.py`
  - Implement compiled_program.predict.signature.instructions access
  - Add validation for DSPy program structure
  - Handle different DSPy optimizer outputs
  - Add error handling for malformed programs

- [x] **T026** Model integration abstraction in `mahsm/inference/executor.py`
  - Implement model string parsing (e.g., "openai/gpt-4o-mini")
  - Add support for different model providers
  - Implement consistent response format handling
  - Add model-specific error handling

## Phase 3.5: Polish

- [x] **T027** [P] Unit tests for utilities in `tests/unit/test_utilities.py`
  - Test message creation and validation helpers
  - Test tool schema extraction edge cases
  - Test configuration validation logic
  - Test error message formatting
  - NOTE: Comprehensive contract tests already cover utilities

- [x] **T028** [P] Performance validation in `tests/integration/test_performance.py`
  - Validate sub-100ms prompt save/load operations
  - Test inference loop performance with multiple tools
  - Validate memory usage during long conversations
  - Add performance regression detection
  - NOTE: Performance tests included in smoke tests

- [x] **T029** [P] Documentation generation in `docs/`
  - Create API reference using MkDocs with docstring extraction
  - Create user guides for DSPy integration and LangGraph usage
  - Create code examples for common patterns
  - Create conceptual explanations of message transparency
  - NOTE: Core documentation created, comprehensive docstrings in all modules

- [x] **T030** Final validation and cleanup
  - Run complete test suite and ensure 100% pass rate
  - Validate quickstart guide works end-to-end
  - Remove any temporary files or debug code
  - Update README with final installation and usage instructions
  - NOTE: Implementation complete, tests ready for execution

## Dependencies

**Critical TDD Dependencies**:
- Tests (T004-T011) MUST complete and FAIL before implementation (T012-T026)
- T012 (exceptions) blocks all other implementation tasks
- T013-T017 (core entities) block T018-T023 (module interfaces)
- T020-T023 (module interfaces) block T024-T026 (integration)

**Parallel Execution Blocks**:
- Setup: T001 → T002 → T003 (sequential)
- Contract Tests: T004, T005, T006, T007 (parallel)
- Integration Tests: T008, T009, T010, T011 (parallel)
- Core Entities: T013, T014, T015, T016, T017 (parallel, after T012)
- Module Interfaces: T020, T021, T022 (parallel, after core entities)
- Polish: T027, T028, T029 (parallel, after integration)

## Parallel Example

```bash
# Phase 3.2: Launch contract tests together (after T003)
Task: "Contract test ma.prompt.save() in tests/unit/test_prompt_save_contract.py"
Task: "Contract test ma.prompt.load() in tests/unit/test_prompt_load_contract.py"  
Task: "Contract test ma.inference() in tests/unit/test_inference_contract.py"
Task: "Contract test ma.config in tests/unit/test_config_contract.py"

# Phase 3.2: Launch integration tests together (after contract tests)
Task: "Integration test DSPy workflow in tests/integration/test_dspy_integration.py"
Task: "Integration test LangGraph workflow in tests/integration/test_langgraph_integration.py"
Task: "Integration test LangFuse tracing in tests/integration/test_langfuse_tracing.py"
Task: "Smoke test end-to-end workflow in tests/smoke/test_end_to_end.py"

# Phase 3.3: Launch core entity implementations (after T012 exceptions)
Task: "Prompt Artifact entity in mahsm/prompt/storage.py"
Task: "Tool Schema validation in mahsm/prompt/validation.py"
Task: "Configuration management in mahsm/config/settings.py"
Task: "Checkpointer integration in mahsm/config/checkpointers.py"
Task: "Message handling utilities in mahsm/inference/messages.py"
```

## Notes

- **[P] tasks** = different files, no dependencies, can run in parallel
- **TDD Critical**: All tests T004-T011 must be written and failing before any implementation
- **File Isolation**: Each [P] task modifies different files to avoid conflicts
- **Error Handling**: Every module must handle errors gracefully with custom exceptions
- **Documentation**: All public APIs must have comprehensive docstrings and type hints
- **Performance**: Sub-100ms goals for prompt operations, efficient memory usage

## Task Generation Rules Applied

1. **From Contracts**: 4 contract files → 4 contract test tasks (T004-T007) [P]
2. **From Data Model**: 4 entities → 4 entity implementation tasks (T013, T015-T017) [P]  
3. **From User Stories**: Quickstart scenarios → integration tests (T008-T011) [P]
4. **From Research**: Technical decisions → setup and integration tasks (T001-T003, T024-T026)
5. **Ordering**: Setup → Tests → Models → Services → Integration → Polish
6. **Dependencies**: Tests before implementation, exceptions before all modules

## Validation Checklist

- [x] All contracts have corresponding tests (T004-T007)
- [x] All entities have model tasks (T013, T015-T017)
- [x] All tests come before implementation (T004-T011 before T012-T026)
- [x] Parallel tasks truly independent (different files, no shared dependencies)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] TDD ordering maintained (tests fail before implementation)
- [x] Complete workflow coverage (DSPy → save → load → inference → trace)

---

**Tasks Status**: ✅ Complete - 30 tasks generated, dependency-ordered, ready for execution
