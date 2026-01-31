from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGEngine
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
rag = RAGEngine()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(q: Query):
    try:
        user_q = q.question
        print("USER QUESTION:", user_q)

        context = rag.search(user_q)
        print("CONTEXT RETURNED:", context[:500])

        prompt = f"""
You are Devraj Singh Chouhan's personal AI assistant.

Your job: 
- Answer the user's question in a clean, natural and friendly way.
- Do NOT list too many points unless necessary.
- Summarize and speak like a human, not like a PDF.
- Keep answers short unless user asks for detailed explanation.
- Do NOT mention “context”, “chunks”, “RAG”, or anything technical.

Use ONLY the information from the context below.

### CONTEXT:
{context}

### QUESTION:
{user_q}
###FINAL ANSWER(clean and natural):
"""

        print("SENDING TO GROQ...")
        response = groq_llm.invoke(prompt)
        print("GROQ RESPONSE:", response)

        return {"answer": response.content}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
