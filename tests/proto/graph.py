from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, AnyMessage

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model = "gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature = 0.2)

# The chatbot node function takes the current State as input and returns an updated messages list. This is the basic pattern for all LangGraph node functions.
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
 
# Add a "chatbot" node. Nodes represent units of work. They are typically regular python functions.
graph_builder.add_node("chatbot", chatbot)
 
# Add an entry point. This tells our graph where to start its work each time we run it.
graph_builder.set_entry_point("chatbot")
 
# Set a finish point. This instructs the graph "any time this node is run, you can exit."
graph_builder.set_finish_point("chatbot")
 
# To be able to run our graph, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

for s in graph.stream(
    {"messages": [HumanMessage(content = "What is Langfuse?")]}):
    print(s)