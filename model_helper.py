from langchain.chat_models import init_chat_model
import os 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

async def call_open_ai_model(question: str) -> str:
  model = init_chat_model("gpt-4o-mini", model_provider="openai",temperature=0.2)
  
  system_template = """You are a knowledgeable, concise, and helpful assistant. Answer questions clearly, accurately friendly. If you don’t know the answer, 
                    say so honestly. Do not make up information.
                   """
  
  prompt = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{user_input}")]
  )
  messages = prompt.format_messages(user_input=question)

  response = model.invoke(messages)
  return response.content

async def call_gemini_model(question: str) -> str:
  model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
  response = await model.invoke(question)
  return response.content