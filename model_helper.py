from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

async def call_open_ai_model(question: str) -> str:

  model = init_chat_model("gpt-4o-mini", model_provider="openai",temperature=0.2)
  response = await model.invoke(question)
  return response.content

async def call_gemini_model(question: str) -> str:
  model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
  response = await model.invoke(question)
  return response.content