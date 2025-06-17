from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
import os 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import Groq
load_dotenv()


#OPENAI MODEL
async def call_open_ai_model(question: str,llm_model: str, style: str,domain:str, language:str,detail_level:str) -> str:
  model = ChatOpenAI(
               model_name=llm_model,  # preferred and correct
               temperature=0.1
             )
  
  style="UK english in a polite way"
  
  system_template = f"""You are a highly intelligent and helpful AI assistant with expertise in the field of {domain}.
           Your goal is to deliver accurate, complete, and user-friendly answers to questions, tailored to the user’s needs.

           **Communication Instructions**
           - **Language:** Respond strictly in {language}.
           - **Tone & Style:** Use a {style} tone. Be friendly, respectful, and professional at all times.
           - **Depth:** Provide responses with a {detail_level} level of detail appropriate to the question.

           **Content Guidelines**
          - **Accuracy:** Only provide factually correct and verifiable information.
          - **Completeness:** Address all relevant parts of the question. Go beyond the surface if needed.
          - **Clarity:** Be concise and avoid unnecessary jargon. Structure responses logically.
          - **Ethics & Safety:** Do not generate harmful, offensive, or biased content.
          - **Transparency:** If a question is unclear or information is unavailable, ask for clarification or state that you don’t know. Never fabricate facts.
          - **Neutrality:** Stay impartial and objective unless explicitly asked for an opinion.

      Format your response clearly using markdown if necessary (e.g., for lists or code)."""

  
  prompt = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{user_input}")]
  )
  llm_chain = LLMChain(
        llm=model,
        prompt=prompt
    )

  response = await llm_chain.ainvoke({"user_input": question})

  return response.get("text", str(response))


# GOOGLE GEMINI MODEL
async def call_gemini_model(question: str,llm_model: str, style: str,domain:str, language:str,detail_level:str) -> str:
  try:
    model = ChatGoogleGenerativeAI(model=llm_model)

    system_template = f"""You are an expert AI Assistant specializing in the domain of {domain}.
                     Your primary goal is to provide accurate, comprehensive, and helpful answers to user questions.
                     
                     **Communication Guidelines:**
                     - **Language:** Respond strictly in {language}.
                     - **Answering Style:** Adopt a {style} style. Be friendly, approachable, and professional.
                     - **Detail Level:** Provide answers with a {detail_level} level of detail.
                     
                     **Content Guidelines:**
                     - **Accuracy:** Ensure all information is factually correct.
                     - **Completeness:** Strive to cover all aspects of the user's question.
                     - **Conciseness:** Be to the point without omitting important details.
                     - **Safety & Ethics:** Avoid generating any harmful, biased, or unethical content.
                     - **Handling Unknowns:** If you do not have sufficient information to answer a question, or if the question is ambiguous, clearly state that you don't know the answer or ask for clarification. **Do not fabricate information.**
                     - **Neutrality:** Maintain a neutral and objective tone.
                     
                     Please provide your response in a clear and well-structured format."""
  
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{user_input}")
    ])

    # Bind model and prompt using LLMChain
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Invoke the chain with all variables
    response = await llm_chain.ainvoke({"user_input": question})

    return response.get("text", str(response))
  except Exception as e:
    print(e)






