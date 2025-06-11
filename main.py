from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from model_helper import call_open_ai_model,call_gemini_model
from pydantic import BaseModel

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    model: str
    llm: str
    domain:str
    style: str
    lan:str

@app.post("/question")
async def question(request: QuestionRequest):
    if request.model == "openai":
        print("openai")
        answer = await call_open_ai_model(request.question,request.llm,request.style,request.domain,request.lan)
    else:
        print("gemini")
        answer = await call_gemini_model(request.question)
        
    return {"answer": answer}
