from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
import os 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

async def call_open_ai_model(question: str,llm_model: str, style: str,domain:str, lan:str) -> str:
  model = ChatOpenAI(
               model_name=llm_model,  # preferred and correct
               temperature=0.1
             )
  
  style="UK english in a polite way"
  
  system_template = f"""
                      You are a knowledgeable, concise, and helpful assistant working in the domain of {domain}.
                      You respond in {lan}, following this answering style: {style}.
                      Answer questions clearly, accurately, and in a friendly tone. 
                      If you don’t know the answer, say so honestly. Do not make up information.
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