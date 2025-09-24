from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(question, contexts, max_input_tokens=512, max_output_tokens=150):
    """
    Generate concise answer from retrieved contexts with citations.
    Always returns (answer, citations).
    """
    try:
        if not contexts:
            return "No relevant information found in PDFs.", []

        # Prepare context
        combined_text = ""
        for c in contexts:
            chunk_text = f"{c['text']} (Source: {c['file']}, Page: {c['page']})"
            combined_text += chunk_text + " "

        # Truncate input for model
        inputs = tokenizer(
            f"answer the question based on the context:\n{combined_text}\nQuestion: {question}",
            max_length=max_input_tokens,
            truncation=True,
            return_tensors="pt"
        )

        # Generate output
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_tokens,
            num_beams=4,
            early_stopping=True
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Prepare citations
        citations = [{"file": c["file"], "page": c["page"]} for c in contexts]

        return answer, citations

    except Exception as e:
        # Always return a safe fallback
        return f"Error generating answer: {str(e)}", []
