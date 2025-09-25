# app/chat_handler.py
from app.generator import generate_answer
from app.retriever import retrieve_top_k

REFUSAL_TEMPLATE = "I can’t find this in the uploaded documents. Please rephrase or upload a PDF that contains it."

def handle_chat(question: str, session_id: str):
    """
    Orchestrates retrieval + generation.
    Returns dict with answer, citations, and trace info.
    """
    # Step 1: retrieve top chunks
    retrieved_chunks = retrieve_top_k(question, k=5)

    if not retrieved_chunks:
        return {
            "answer": REFUSAL_TEMPLATE,
            "citations": [],
            "trace": {"retrieved": []}
        }

    # Step 2: call generator
    answer, citations = generate_answer(question, retrieved_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "trace": {"retrieved": retrieved_chunks}
    }
