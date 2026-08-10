from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

search_tool = TavilySearchResults(search_depth="basic")

agent = initialize_agent(tools=[search_tool],llm=llm, agent="zero-shot-react-description",verbose=True) #Zero shot --> it is doing things without us providing any prior knowledge

agent.invoke("Bring me the last tweet by Nimsdai before his demise")

