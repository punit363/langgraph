from typing import TypedDict
from langgraph.graph import StateGraph, END

class SimpleState(TypedDict):
    count : int

def increment(state:SimpleState)->SimpleState:
    return {
        "count":state["count"]+1
    }
# the state from increment does not update the previous state rather replaces the entire state

def should_continue(state):
    if(state["count"]<5):
        return "continue"
    else:
        return "stop"
    

graph = StateGraph(SimpleState)

INCREMENT = "INCREMENT"

graph.add_node(INCREMENT,increment)

graph.set_entry_point(INCREMENT)

graph.add_conditional_edges(INCREMENT,should_continue,{
    "continue":INCREMENT,
    "stop":END
})

app = graph.compile()

state = {
    "count":0
}

result = app.invoke(state)

print(result)