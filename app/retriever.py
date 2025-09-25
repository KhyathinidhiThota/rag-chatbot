from chromadb import PersistentClient
from app.config import VECTOR_DB_PATH

client = PersistentClient(path=VECTOR_DB_PATH)

def retrieve(query, model, top_k=5):
    # encode query
    query_emb = model.encode(query).tolist()
    
    # fetch all embeddings from your Chroma DB collection
    collection = client.get_collection("documents")
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    contexts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append({
            "text": doc,
            "file": meta["file"],
            "page": meta["page"]
        })
    return contexts