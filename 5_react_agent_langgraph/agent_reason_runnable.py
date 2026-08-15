import datetime
from dotenv import load_dotenv
from langchain.agents import create_react_agent, tool
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current date and time in specified format"""
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

tools = [search_tool, get_system_time]

# Standard ReAct Prompt Template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (or cannot find more details)
Final Answer: the final answer to the original input question (always prefix your final response with 'Final Answer:')

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt_template = PromptTemplate.from_template(template)

# Create runnable and executor
react_agent_runnable = create_react_agent(llm, tools, prompt=react_prompt_template)
