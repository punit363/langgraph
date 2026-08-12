from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assitant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request"
            " If the user provides a crtique, respond with a revised version of your previous attempt."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate a critique and recommendation for user's tweet."
            " Always provide detailed recommendations, including request for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm =ChatGoogleGenerativeAI(model="gemini-2.5-flash")

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm