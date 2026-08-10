from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_current_time(format:str = "%Y-%m-%d %H:%M:%S"):
    """Return the current data and time in specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

agent = initialize_agent(tools=[search_tool,get_current_time],llm=llm, agent="zero-shot-react-description",verbose=True) #Zero shot --> it is doing things without us providing any prior knowledge

agent.invoke("When was the first private orbital space launch in India? How much time has passed since then?")

