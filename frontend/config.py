
"""
Configuration file for Type 2 Diabetes RAG Frontend
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============= PINECONE CONFIG =============
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "type2diabetes")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west4-gcp")

# ============= HUGGING FACE CONFIG =============
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ============= BACKEND CONFIG =============
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ============= RETRIEVAL CONFIG =============
TOP_K_CHUNKS = 5  # Number of chunks to retrieve
RERANK_THRESHOLD = 0.5  # Re-ranking threshold
USE_HYBRID_SEARCH = True  # Use hybrid search (BM25 + semantic)
USE_RERANKING = True  # Enable re-ranking

# ============= LLM CONFIG =============
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95

# ============= EVALUATION CONFIG =============
EVALUATE_FAITHFULNESS = True
EVALUATE_RELEVANCY = True

# ============= CHAINLIT CONFIG =============
APP_TITLE = "Type 2 Diabetes Management RAG System"
APP_DESCRIPTION = """
Ask questions about Type 2 Diabetes management, treatment options, and prevention strategies.

This system uses:
- **Hybrid Search**: BM25 + Semantic search with Re-ranking
- **LLM Generation**: Advanced language model for accurate answers
- **Quality Evaluation**: Automatic faithfulness and relevancy scoring
"""

# Example queries for welcome screen
EXAMPLE_QUERIES = [
    "What are the latest medications for Type 2 Diabetes?",
    "How does insulin resistance relate to Type 2 Diabetes?",
    "What lifestyle changes help manage Type 2 Diabetes?",
    "What are the complications of untreated Type 2 Diabetes?",
    "How is Type 2 Diabetes diagnosed?",
]

# Validation
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in .env file")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set in .env file")
