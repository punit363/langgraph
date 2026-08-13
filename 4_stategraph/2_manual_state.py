from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class SimpleState(TypedDict):
    count : int
    sum : int
    history: List[int]

def increment(state:SimpleState)->SimpleState:
    new_count = state["count"]+1
    return {
        "count": new_count,
        "sum": state["sum"] + new_count,
        "history":state["history"] + [new_count]
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
    "count":0,
    "sum":0,
    "history":[0]
}

result = app.invoke(state)

print(result)