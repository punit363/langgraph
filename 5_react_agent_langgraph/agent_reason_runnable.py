from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub

llm = ChatOpenAI(model="gpt-4o")


search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_current_time(format:str = "%Y-%m-%d %H:%M:%S"):
    """Return the current data and time in specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools=[search_tool,get_current_time]

react_prompt_template =hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(llm,tools,prompt=react_prompt_template)