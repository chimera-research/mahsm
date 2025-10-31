# The `mahsm` Development Workflow

Developing with `mahsm` follows a simple, iterative, and powerful three-step loop: **Build, Trace, and Evaluate**. This cycle is designed to be fast and data-driven, ensuring you are creating high-quality, robust systems.

![mahsm Workflow](https://i.imgur.com/your-diagram-url.png) <!-- It's highly recommended to create a simple diagram for this! -->

### 1. Build

This is the core development step where you write idiomatic `mahsm` code.
- **Define State:** Create a `TypedDict` that represents the shared state of your application.
- **Create Nodes:** Write `dspy.Module` classes to encapsulate the reasoning logic for your agents. Decorate them with `@ma.dspy_node`.
- **Wire Graph:** Add your nodes to a `ma.graph.StateGraph` and define the edges to control the flow of execution.
- **Compile:** Call `.compile()` on your graph to create the runnable application.

### 2. Trace

Once built, you run your application.
    *   With `ma.init()` called at the start of your script, every execution is automatically and deeply traced in LangFuse.
    *   You use the LangFuse UI to inspect the full decision-making process of your agent, debug issues, understand latency, and analyze token usage.
    *   You can tag interesting traces to save them as examples for regression testing or for creating evaluation datasets.

### 3. Evaluate

Finally, you verify the quality of your agent's output.
    *   **Write a Test File:** Create a standard `pytest` file.
    *   **Configure the Harness:** Use the `ma.testing.PytestHarness` to connect your compiled `mahsm` graph to the evaluation protocol.
    *   **Run Eval:** Use datasets (potentially generated from your production traces in LangFuse) to run an evaluation.
    *   **Analyze & Iterate:** Use the evaluation leaderboards and results to identify weaknesses in your agent's logic, then go back to the **BUILD** step to improve it.