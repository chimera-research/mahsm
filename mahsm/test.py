from eval_protocol import (
    DynamicDataLoader,
    LangGraphRolloutProcessor,
    create_langfuse_adapter,
    RolloutProcessorConfig,
    evaluation_test,
    aha_judge,
    EvaluationRow,
)
import dspy

class PytestHarness:
    """A testing harness to bridge a compiled mahsm graph with EvalProtocol."""
    def __init__(self, graph):
        self.graph = graph
        self._data_loaders = None

    def from_langfuse(self, **kwargs):
        """Configures the data source to be a Langfuse deployment."""
        adapter = create_langfuse_adapter()
        self._data_loaders = DynamicDataLoader(
            generators=[lambda: adapter.get_evaluation_rows(**kwargs)]
        )

    @property
    def data_loaders(self):
        if self._data_loaders is None:
            raise RuntimeError("Data source not configured. Call .from_langfuse() first.")
        return self._data_loaders

    @property
    def rollout_processor(self):
        """Returns a pre-configured LangGraphRolloutProcessor."""
        return LangGraphRolloutProcessor(graph_factory=self._graph_factory)

    def _graph_factory(self, config: RolloutProcessorConfig):
        """
        Internal factory to re-configure the graph's LM for a test run.
        """
        new_model = config.completion_params.get("model")
        if new_model:
            # This is a temporary override for the duration of the test run.
            # In a real implementation, you might use contextvars for safety.
            dspy.settings.configure(lm=dspy.LM(new_model))
        
        return self.graph