import os
import hashlib
import PyPDF2
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from app.config import VECTOR_DB_PATH, EMBEDDING_MODEL

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def load_pdf(file_path: str):
    text_data = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_data.append({"page": i+1, "text": text})
    return text_data

def chunk_text(text, chunk_size=800, overlap=120):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_pdf(file_path: str, chunk_size=800, overlap=120):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_or_create_collection("documents")

    pdf_text = load_pdf(file_path)
    total_chunks = 0
    for entry in pdf_text:
        chunks = chunk_text(entry["text"], chunk_size, overlap)
        embeddings = model.encode(chunks).tolist()
        total_chunks += len(chunks)
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"file": os.path.basename(file_path), "page": entry["page"]}],
                ids=[f"{os.path.basename(file_path)}_p{entry['page']}_{chunk_id}"]
            )

    return {"status": "success", "file": file_path, "chunks": total_chunks}
