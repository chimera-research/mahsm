# Declarative by Design

The central philosophy of `mahsm` is its declarative approach. Instead of manually writing imperative "glue code" to connect different libraries, you **declare** the components of your system, and `mahsm` handles the integration and boilerplate.

This "convention over configuration" approach is designed to let you focus entirely on your agent's business logic, not the plumbing.

### The `mahsm` Approach

*   **You declare your agent's reasoning** by writing standard `dspy.Module` classes. The powerful `@ma.dspy_node` decorator instantly makes them compatible with the orchestration layer.
*   **You declare your workflow's structure** by adding your nodes to a `ma.graph.StateGraph` and defining the edges between them.
*   **You declare your evaluation criteria** by configuring the `ma.testing.PytestHarness` to run your graph against a dataset.

### The Benefits

This philosophy drastically reduces boilerplate, improves code readability, and embeds best practices for observability and testing directly into the development process. The result is a workflow that is faster, more robust, and produces systems that are understandable by default.