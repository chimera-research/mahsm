# Data Model: MAHSM v0.1.0

**Date**: October 7, 2025  
**Feature**: MAHSM (Multi-Agent Hyper-Scaling Methods) v0.1.0  
**Status**: Complete

## Core Entities

### Prompt Artifact
**Purpose**: Persistent storage of optimized prompts with validation metadata  
**Storage**: JSON files in ~/.mahsm/prompts/

**Fields**:
- `prompt` (string): The optimized prompt text extracted from DSPy
- `tools` (array): Tool schemas in OpenAI function calling format
- `metadata` (object): Creation and optimization metadata
  - `created_at` (ISO timestamp): When artifact was saved
  - `optimizer` (string): DSPy optimizer used (e.g., "GEPA")
  - `version` (string): Artifact version for compatibility
- `task_name` (string): Identifier for the prompt's purpose
- `task_version` (string): Version of this specific prompt

**Validation Rules**:
- `prompt` must be non-empty string
- `tools` array must contain valid OpenAI function schemas
- `task_name` must be valid filename (no special chars)
- `task_version` must follow semantic versioning pattern

**File Naming**: `{task_name}_{task_version}.json`

**Example**:
```json
{
  "prompt": "You are an expert data analyst. Analyze the provided data and generate insights...",
  "tools": [
    {
      "name": "calculate_statistics",
      "description": "Calculate statistical measures",
      "parameters": {
        "type": "object",
        "properties": {
          "data": {"type": "array", "items": {"type": "number"}},
          "measures": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["data", "measures"]
      }
    }
  ],
  "metadata": {
    "created_at": "2025-10-07T10:30:00Z",
    "optimizer": "GEPA",
    "version": "1.0.0"
  },
  "task_name": "data_analysis",
  "task_version": "v1"
}
```

### Tool Schema
**Purpose**: OpenAI function calling format specification for validation  
**Storage**: Embedded within Prompt Artifacts

**Fields**:
- `name` (string): Function name identifier
- `description` (string): Human-readable function description
- `parameters` (object): JSON Schema for function parameters
  - `type`: Always "object"
  - `properties` (object): Parameter definitions
  - `required` (array): Required parameter names

**Validation Rules**:
- `name` must be valid Python function name
- `parameters` must be valid JSON Schema
- Required parameters must exist in properties

### Message State
**Purpose**: LangGraph-compatible conversation history  
**Storage**: In-memory during inference, optionally persisted via checkpointers

**Structure**: List of message objects in chronological order
- `SystemMessage`: Contains the loaded prompt
- `HumanMessage`: Contains user input
- `AIMessage`: Contains model responses, may include tool_calls
- `ToolMessage`: Contains tool execution results

**State Transitions**:
1. Initialize with SystemMessage (prompt)
2. Add HumanMessage (user input)
3. Add AIMessage (model response)
4. If tool_calls present: Add ToolMessage for each execution
5. Repeat steps 3-4 until no tool_calls or max_iterations

**Example Flow**:
```python
[
  SystemMessage(content="You are an expert..."),
  HumanMessage(content="Analyze this data: [1,2,3,4,5]"),
  AIMessage(content="I'll analyze the data", tool_calls=[...]),
  ToolMessage(content="Statistics: mean=3, std=1.58", tool_call_id="call_123"),
  AIMessage(content="Based on the analysis, the data shows...")
]
```

### Configuration
**Purpose**: Global settings and environment integration  
**Storage**: Environment variables and runtime configuration

**Fields**:
- `langfuse_public_key` (string, optional): LangFuse tracing public key
- `langfuse_secret_key` (string, optional): LangFuse tracing secret key
- `mahsm_home` (string): Base directory for artifacts (default: ~/.mahsm)
- `default_max_iterations` (int): Default inference loop limit (default: 10)
- `checkpointer_type` (string): Storage backend type (sqlite/postgres/memory)

**Environment Variable Mapping**:
- `LANGFUSE_PUBLIC_KEY` → `langfuse_public_key`
- `LANGFUSE_SECRET_KEY` → `langfuse_secret_key`
- `MAHSM_HOME` → `mahsm_home`
- `MAHSM_MAX_ITERATIONS` → `default_max_iterations`

## Entity Relationships

```
DSPy Program → [extract] → Prompt Artifact → [load] → Runtime Prompt
                                ↓
Tool Functions → [schema] → Tool Schemas → [validate] → Runtime Tools
                                ↓
User Input → [inference] → Message State → [trace] → LangFuse
```

## Validation Matrix

| Entity | Validation Point | Rule | Error Type |
|--------|------------------|------|------------|
| Prompt Artifact | Save | Non-empty prompt | ValueError |
| Prompt Artifact | Save | Valid tool schemas | ValidationError |
| Tool Schema | Load | Name/parameter match | ValidationError |
| Message State | Inference | Valid message types | TypeError |
| Configuration | Initialize | Valid environment vars | ConfigurationError |

## State Management

### Persistence Strategy
- **Prompt Artifacts**: Always persisted to filesystem
- **Message State**: In-memory by default, optionally persisted via LangGraph checkpointers
- **Configuration**: Loaded once at import, cached in global object

### Concurrency Considerations
- **File Operations**: Thread-safe via atomic writes
- **Message State**: Single-threaded per inference session
- **Configuration**: Read-only after initialization

---

**Data Model Status**: ✅ Complete - All entities defined with validation rules and relationships
