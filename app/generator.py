from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"  # better than t5-small
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(question, contexts, max_input_tokens=768, max_output_tokens=200):
    """
    Generate concise, factual answers with citations from retrieved contexts.
    Returns: (answer, citations)
    """
    try:
        if not contexts:
            return "No relevant information found in PDFs.", []

        # 🔹 Only keep top 3–5 most relevant chunks (avoid dumping full PDF!)
        top_contexts = contexts[:5]

        combined_text = " ".join(
            f"{c['text']} (Source: {c['file']}, Page {c['page']})"
            for c in top_contexts
        )

        # 🔹 Strong instruction prompt
        prompt = (
            "You are a helpful assistant. Answer the question ONLY using the context. "
            "If the answer is not in the context, say 'No relevant information found'.\n\n"
            f"Context:\n{combined_text}\n\nQuestion: {question}\nAnswer concisely:"
        )

        # 🔹 Tokenize with truncation
        inputs = tokenizer(
            prompt,
            max_length=max_input_tokens,
            truncation=True,
            return_tensors="pt"
        )

        # 🔹 Generate with beam search + control
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_tokens,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,  # 🔑 Avoid copying context
                early_stopping=True
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        citations = [{"file": c["file"], "page": c["page"]} for c in top_contexts]

        return answer.strip(), citations

    except Exception as e:
        return f"Error generating answer: {str(e)}", []
