from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generator_node(state):
    return generation_chain.invoke({
        "messages":state
    })

def reflector_node(state):
    response = reflection_chain.invoke({
        "messages":state
    })
    return [HumanMessage(content=response.content )] # just to let llm assume that this critque was from a human

graph.add_node(GENERATE,generator_node)
graph.add_node(REFLECT,reflector_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if(len(state)>4):
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE,should_continue,{
        REFLECT: REFLECT,
        END: END
    }) #right after generator node it branches off to two seperate nodes as defined in the function

graph.add_edge(REFLECT,GENERATE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().draw_ascii()

response = app.invoke([HumanMessage(content="AI taking over human jobs")])
print(f"agentic reponse: {response}")