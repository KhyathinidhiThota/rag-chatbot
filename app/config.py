import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"
TOP_K = 5
SCORE_THRESHOLD = 0.4
