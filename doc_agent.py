import mahsm as ma
from typing import TypedDict, List, Optional
import dspy # Still needed for settings

# --- Configure DSPy for direct execution ---
# This will be the default model unless overridden by the testing harness.
turbo = dspy.OpenAI(model='gpt-5-mini', max_tokens=4000)
dspy.settings.configure(lm=turbo)


# 1. DEFINE THE SHARED STATE
class CodeGenerationState(TypedDict):
    doc_content: str
    use_case: str
    generated_code: Optional[str]
    critique: Optional[str]
    refinement_attempts: int

# 2. DEFINE THE AGENT NODES
@ma.dspy_node
class CodeGenerator(ma.Module):
    def __init__(self):
        super().__init__()
        self.generator = ma.dspy.ChainOfThought("doc_content, use_case, critique -> generated_code")
    def forward(self, doc_content, use_case, critique=None):
        return self.generator(doc_content=doc_content, use_case=use_case, critique=critique)

@ma.dspy_node
class CodeCritic(ma.Module):
    def __init__(self):
        super().__init__()
        self.critic = ma.dspy.Predict("generated_code, use_case -> critique", n=1)
    def forward(self, generated_code, use_case):
        return self.critic(generated_code=generated_code, use_case=use_case)

# 3. DEFINE THE EDGES
def should_refine_code(state: CodeGenerationState):
    # Increment refinement attempts
    state['refinement_attempts'] = state.get('refinement_attempts', 0) + 1
    
    if "PERFECT" in state["critique"].upper() or state["refinement_attempts"] >= 2:
        return ma.END
    return "generator"

# 4. CONSTRUCT THE GRAPH
workflow = ma.graph.StateGraph(CodeGenerationState)
workflow.add_node("generator", CodeGenerator())
workflow.add_node("critic", CodeCritic())
workflow.add_edge(ma.START, "generator")
workflow.add_conditional_edges("generator", "critic", { "critic": "critic" }) # Always go to critic after generating
workflow.add_conditional_edges("critic", should_refine_code)

# 5. COMPILE THE GRAPH
graph = workflow.compile()

# --- Example of running the application directly ---
if __name__ == "__main__":
    # handler = ma.init()
    
    inputs = {
        "doc_content": "In FastAPI, you use the @app.get('/') decorator to define a GET endpoint.",
        "use_case": "Create a simple hello world endpoint in FastAPI.",
        "refinement_attempts": 0,
    }

    # config = {"callbacks": [handler]} if handler else {}
    for event in graph.stream(inputs):
        print(f"--- Event: {list(event.keys())[0]} ---")
        print(event[list(event.keys())[0]])