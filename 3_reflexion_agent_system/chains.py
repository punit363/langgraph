from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

#Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [(
        "system",
        """You are an expert AI researcher
        current time: {time}
        1. {first_instruction}
        2. reflect and critique your answer. Be severe to maximize your improvement.
        3. After the reflection, **list 1-3 search queries seperately** for 
        researching improvements. Do not include them inside the reflection.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("system","Answer the user's question using the required format.")]
).partial(#prepopulate data before invoking the prompt
    time= lambda:datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 words answer"
)

llm = ChatOpenAI(model="gpt-4o")

first_responder_chain= first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice='AnswerQuestion'#mandating llm to use only this tool
) | pydantic_parser

response = first_responder_chain.invoke({
    "messages":[HumanMessage(content="Write me Blog on gRPC protocol with key findings like history, problem why it was created, how it solves that, how it works, where it is used etc")]
})

print(f"RESPONSE: {response}")