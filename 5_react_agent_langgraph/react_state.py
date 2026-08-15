import operator
from typing import Annotated,TypedDict,Union

from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction,str]], operator.add]

#agentstate defines the output stucture from every node 
#input: what prompt was fed to the agent node
#agent_outcome: what was the outcome from each node, initially it is None
#intermediate_steps: all outputs are saved here after each tool call
