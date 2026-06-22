import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from src.core.agent import Agent
from src.core.retrieval import add_document, list_documents

load_dotenv()

app = FastAPI(
    title="Personal Knowledge Assistant API",
    description="A RAG-based agentic assistant with memory, tool use, and document search",
    version="1.0.0"
)

# One agent instance per server session — holds memory across requests
agent = Agent()


# --- Pydantic models (request/response schemas) ---

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    question: str

class StatusResponse(BaseModel):
    status: str
    message: str


# --- Endpoints ---

@app.get("/", response_model=StatusResponse)
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Personal Knowledge Assistant API is running"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Send a question to the agent. Maintains conversation memory across calls.
    The agent will automatically decide whether to search documents,
    use web search, calculate, or answer directly.
    """

    try:
        answer = agent.ask(request.question)
        return {"answer": answer, "question": request.question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=StatusResponse)
def reset_memory():
    """Clears the agent's conversation memory, starting fresh."""
    
    agent.conversation_history = []
    agent.conversation_summary = ""

    return {"status": "ok", "message": "Memory reset successfully"}


@app.get("/history", response_model=list)
def get_history():
    """Returns the current conversation history."""

    return agent.conversation_history

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and add it to the knowledge base."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        file_bytes = await file.read()
        chunks_added = add_document(file_bytes, file.filename)
        return {"status": "ok", "filename": file.filename, "chunks_added": chunks_added}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/documents")
def get_documents():
    """Returns list of documents currently in the knowledge base."""
    try:
        docs = list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))