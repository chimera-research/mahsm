from doc_agent import graph
import mahsm as ma

# 1. Create a testing harness from our compiled application graph.
harness = ma.testing.PytestHarness(graph=graph)

# 2. Configure the data source. For this alpha, we will create a dummy
#    dataset in code instead of pulling from Langfuse to ensure it runs
#    out of the box. Replace with harness.from_langfuse() for real evals.

def dummy_data_generator():
    # In production, this data comes from LangFuse traces.
    return [
        ma.testing.EvaluationRow(
            messages=[
                ma.HumanMessage(
                    content="Use FastAPI to create a GET endpoint at '/items/{item_id}' that returns the item_id."
                )
            ],
            # This is where ground truth would go if we had it.
            expected_output=None,
        )
    ]

harness._data_loaders = ma.testing.DynamicDataLoader(generators=[dummy_data_generator])

# 3. Write the evaluation test.
@ma.testing.evaluation_test(
    data_loaders=harness.data_loaders,
    rollout_processor=harness.rollout_processor,
    completion_params=[
        {"model": "openai/gpt-5-mini"},
        {"model": "openai/gpt-5"}, # Comparing two models
    ],
)
async def test_code_generation_quality(row: ma.testing.EvaluationRow) -> ma.testing.EvaluationRow:
    """
    Evaluates the quality of the generated code using an LLM-as-a-judge.
    """
    return await ma.testing.aha_judge(
        row,
        judge_model="openai/gpt-4o-mini",
        rubric="Does the final 'generated_code' in the assistant's message correctly implement the use case from the user's message? The code must be complete and correct."
    ))