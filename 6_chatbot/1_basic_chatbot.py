from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

class BasicChatState(TypedDict):
    messages:Annotated[list, add_messages]

def chat_bot(state:BasicChatState):
    return{
        "messages":[llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

CHAT_BOT = "chat_bot"

graph.add_node(CHAT_BOT,chat_bot)
graph.set_entry_point(CHAT_BOT)
graph.add_edge(CHAT_BOT,END)


app = graph.compile()

while True :
    user_input = input("User: ")
    if user_input in ["EXIT","END","end","exit"]:
        break
    else:
        result = app.invoke({
            "messages":[HumanMessage(content=user_input)]
        })

        print(result)