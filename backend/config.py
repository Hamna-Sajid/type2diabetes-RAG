"""
config.py — single source of truth for all backend settings.
All values are read from environment variables (set in HF Spaces secrets).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Pinecone ─────────────────────────────────────────────────────────────
    PINECONE_API_KEY:    str = ""
    PINECONE_INDEX_NAME: str = "diabetes-rag"
    PINECONE_REGION:     str = "us-east-1"
    PINECONE_CLOUD:      str = "aws"

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL:   str = "llama-3.3-70b-versatile"
    GROQ_URL:     str = "https://api.groq.com/openai/v1/chat/completions"

    # ── Models ───────────────────────────────────────────────────────────────
    EMBED_MODEL_NAME:    str = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Retrieval knobs ───────────────────────────────────────────────────────
    BM25_TOP_K:   int   = 20
    SEM_TOP_K:    int   = 20
    RRF_K:        int   = 60
    RERANK_TOP_K: int   = 5

    # ── LLM generation ────────────────────────────────────────────────────────
    LLM_MAX_TOKENS:  int   = 400
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_RETRIES: int   = 4

    # ── Rate-limit guard (seconds to sleep after each Groq call) ─────────────
    # Set to 0 in prod if you have a paid Groq plan
    GROQ_INTER_CALL_DELAY: float = 2.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_ALT_QUESTIONS: int = 3   # questions generated for relevancy


settings = Settings()
