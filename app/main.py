from fastapi import FastAPI, UploadFile, Form, HTTPException
import shutil, os, uuid
from sentence_transformers import SentenceTransformer
from app.ingest import ingest_pdf
from app.retriever import retrieve
from app.generator import generate_answer  # Updated generator
from app.memory import memory_store
from app.config import EMBEDDING_MODEL

# Ensure necessary folders exist
os.makedirs("sample_pdfs", exist_ok=True)

app = FastAPI(title="RAG Chatbot with Citations")

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# -------------------- Start a new session --------------------
@app.get("/start_session")
def start_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

# -------------------- Ingest a PDF --------------------
@app.post("/ingest")
async def ingest(file: UploadFile):
    # Save file with unique name
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("sample_pdfs", unique_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        result = ingest_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {"message": "PDF ingested successfully", "file_path": file_path, "details": result}

# -------------------- Chat --------------------
@app.post("/chat")
async def chat(session_id: str = Form(...), message: str = Form(...)):
    # Retrieve relevant contexts from vector DB
    try:
        contexts = retrieve(message, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Generate grounded answer using local LLM
    try:
        answer, citations = generate_answer(message, contexts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    # Save chat in memory for multi-turn follow-up
    memory_store.add_turn(session_id, message, answer)

    return {"answer": answer, "citations": citations}

# -------------------- Get chat history --------------------
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    history = memory_store.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": history}