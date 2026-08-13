from typing import List

from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

MAX_ITERATIONS=2

graph = MessageGraph()

RESPONDER="RESPONDER"
SEARCH_TOOL="SEARCH_TOOL"
REVISER="REVISER"


graph.add_node(RESPONDER,first_responder_chain)
graph.add_node(SEARCH_TOOL,execute_tools)
graph.add_node(REVISER,revisor_chain)

graph.set_entry_point(RESPONDER)

graph.add_edge(RESPONDER,SEARCH_TOOL)
graph.add_edge(SEARCH_TOOL,REVISER)


def event_loop(state: List[BaseMessage])->str:
    count_tool_visits = sum(isinstance(item,ToolMessage) for item in state)

    if(count_tool_visits>MAX_ITERATIONS):
        return END
    return SEARCH_TOOL

graph.add_conditional_edges(REVISER,event_loop,{
        SEARCH_TOOL: SEARCH_TOOL,
        END: END
    }) #right after generator node it branches off to two seperate nodes as defined in the function

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().draw_ascii()

response = app.invoke([HumanMessage(content="Write me Blog on gRPC protocol with key findings like history, problem why it was created, how it solves that, how it works, where it is used etc")])
print(response[-1].tool_calls[0]["args"]["answer"])
print(f"\n\n\n")
print(f"reponse: {response}")