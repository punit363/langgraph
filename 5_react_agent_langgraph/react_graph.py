from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph, END
from nodes import act_node, reason_node
from react_state import AgentState

load_dotenv()

REASON_NODE = "reason_node"
ACT_NODE = "act_node"

def should_continue(state:AgentState)->str:
    if isinstance(state['agent_outcome'], AgentFinish):
        return END
    else:
        return ACT_NODE
    
graph = StateGraph(AgentState)

graph.add_node(REASON_NODE,reason_node)
graph.set_entry_point(REASON_NODE)

graph.add_node(ACT_NODE,act_node)

graph.add_conditional_edges(REASON_NODE,should_continue)
graph.add_edge(ACT_NODE, REASON_NODE)

app = graph.compile()
app.get_graph().draw_ascii()

response = app.invoke({
    "input":"How many day's ago was skyroot's launch?",
    "agent_outcome":None,
    "intermediate_steps":[]
})

print(f"reponse: {response}")
print(f"\n\n\n")
print(response["agent_outcome"].return_values["output"], "final result")