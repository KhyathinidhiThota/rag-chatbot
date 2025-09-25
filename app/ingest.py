import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# 🔹 Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 FAISS index for cosine similarity
dimension = embedder.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)

# 🔹 Store documents as dicts {text, file, page}
documents = []

def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def chunk_text(text, file, page, chunk_size=150, overlap=30):
    """Split text into overlapping chunks for better retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append({"text": chunk, "file": file, "page": page})
    return chunks

def ingest_pdf(file_path: str):
    """Read, chunk, embed, and store a PDF in FAISS index."""
    global documents, index
    text = load_pdf(file_path)
    if not text:
        return "No text found in PDF."

    # Reset FAISS index & documents when re-ingesting
    index.reset()
    documents.clear()

    # Extract all pages
    all_chunks = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks = chunk_text(page_text, file=file_path, page=page_num)
        all_chunks.extend(chunks)

    if not all_chunks:
        return "No chunks could be created from PDF."

    # Encode & add to FAISS
    embeddings = embedder.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
    index.add(np.array(embeddings, dtype=np.float32))
    documents.extend(all_chunks)

    return f"Ingested {len(all_chunks)} chunks from {file_path}"

def retrieve(query: str, top_k: int = 3):
    """Retrieve top-k most relevant chunks for a query."""
    if not documents:
        return []

    query_vec = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(np.array(query_vec, dtype=np.float32), top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(documents):
            results.append(documents[idx])
    return results
