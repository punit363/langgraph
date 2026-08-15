from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import TavilySearchResults
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
# memory = MemorySaver()
sqlite_connect = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_connect)

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools=tools)

class BasicChatState(TypedDict):
    messages:Annotated[list, add_messages]

def chat_bot(state:BasicChatState):
    return{
        "messages":[llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state:BasicChatState):
    last_message = state["messages"][-1]

    if(hasattr(last_message,"tool_calls") & len(last_message.tool_calls)>0):
        return TOOL_CALL
    else:
        return END
    

graph = StateGraph(BasicChatState)

CHAT_BOT = "chat_bot"
TOOL_CALL = "tool_node"

tool_node = ToolNode(tools=tools)

graph.add_node(CHAT_BOT,chat_bot)
graph.add_node(TOOL_CALL,tool_node)

graph.set_entry_point(CHAT_BOT)

graph.add_edge(TOOL_CALL,CHAT_BOT)

graph.add_conditional_edges(CHAT_BOT,tools_router,{
    TOOL_CALL:TOOL_CALL,
    END:END
})


app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id":1
}}

while True :
    user_input = input("User: ")
    if user_input in ["EXIT","END","end","exit"]:
        break
    else:
        result = app.invoke({
            "messages":[HumanMessage(content=user_input)]
        },config=config)

    print(f"RESPONSE-----> {result} \n==========================")
