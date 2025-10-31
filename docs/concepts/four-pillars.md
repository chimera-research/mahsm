# The Four Pillars of `mahsm`

`mahsm` achieves its power by deeply integrating four essential, best-in-class libraries into a single, seamless experience. All functionality is exposed through the unified `import mahsm as ma` API, giving you a consistent and clean developer experience.

---

### 1. DSPy: The Reasoning Engine (`ma.dspy`)

*   **What it is:** A framework from Stanford NLP for programming—not just prompting—language models. It separates program flow from parameters (prompts and model weights) and uses optimizers to tune them for maximum performance.
*   **How `mahsm` Fuses it:** `mahsm` treats DSPy modules as the fundamental building blocks of agent intelligence. The core innovation is the **`@ma.dspy_node`** decorator. This tool instantly transforms any `dspy.Module` into a fully compliant LangGraph node, automatically handling the complex mapping of data from the shared graph `State` to the module's inputs and back.

---

### 2. LangGraph: The Orchestration Scaffolding (`ma.graph`)

*   **What it is:** A library for building stateful, multi-agent applications by representing them as cyclical graphs. It provides the primitives of `State`, `Nodes`, and `Edges` to create complex, long-running agentic workflows.
*   **How `mahsm` Fuses it:** LangGraph provides the skeleton, and `mahsm` provides the intelligent organs. By making DSPy modules the primary type of "thinking" node, `mahsm` supercharges LangGraph development. You define your application's `State` and use `ma.graph.StateGraph` to wire together your `@ma.dspy_node` agents.

---

### 3. LangFuse: The Unified Observability Layer

*   **What it is:** A comprehensive open-source platform for LLM observability, providing detailed tracing, debugging, and analytics for AI applications.
*   **How `mahsm` Fuses it:** `mahsm` makes deep, hierarchical tracing an automatic, zero-effort feature. The single **`ma.init()`** function simultaneously instruments both LangGraph and DSPy. When you run your graph, `mahsm` creates a single, unified trace in LangFuse that captures both the high-level graph flow and the low-level DSPy execution details (prompts, tool calls, etc.), solving the massive pain point of achieving end-to-end observability.

---

### 4. EvalProtocol: The Quality Control & Testing Framework (`ma.testing`)

*   **What it is:** A standardized, `pytest`-based framework for evaluating the performance of AI systems using LLM-as-a-judge and other metrics.
*   **How `mahsm` Fuses it:** `mahsm` bridges the gap between your built application and your test suite. The **`ma.testing.PytestHarness`** class radically simplifies setup by automatically generating the boilerplate processors required by `eval-protocol`. The harness can even pull evaluation datasets directly from your production LangFuse traces, enabling a tight, continuous loop of deploying, observing, and evaluating your system's real-world performance.