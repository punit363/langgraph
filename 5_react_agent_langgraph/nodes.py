from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()

def reason_node(state:AgentState):
    # Format intermediate steps for standard ReAct parser
    agent_outcome = react_agent_runnable.invoke({
        "input": state["input"],
        "intermediate_steps": state["intermediate_steps"]
    })
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    agent_action = state["agent_outcome"]

    # If already finished, return no-op
    if isinstance(agent_action, AgentFinish):
        return {}
    #extract tool name and input from AgentAction
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    #find the matching tool in avaiblable tool
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break

    #exectue the tool with the function
    if tool_function:
        if isinstance(tool_input,dict):
            output = tool_function.invoke(**tool_input) #spreadout if multiple inputs
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool '{tool_name}' not found"

    return {"intermediate_steps":[(agent_action,str(output))]}