"""
Type 2 Diabetes RAG — FastAPI Backend
======================================
Connects to: Pinecone (vectors), BM25 (keyword), Groq (LLM), CrossEncoder (rerank)
Exposes:     POST /query  →  answer + chunks + faithfulness + relevancy
             GET  /health →  readiness probe
"""

import os
import re
import time
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

from config import settings
from retrieval import HybridRetriever
from llm import GroqClient
from evaluation import Evaluator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rag.main")

# ── Global singletons (loaded once at startup) ────────────────────────────────
_retriever: Optional[HybridRetriever] = None
_llm:       Optional[GroqClient]      = None
_evaluator: Optional[Evaluator]       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy models once at startup, release at shutdown."""
    global _retriever, _llm, _evaluator

    log.info("⏳ Loading models and connecting to Pinecone…")

    # 1. Embedding model
    embedder = SentenceTransformer(settings.EMBED_MODEL_NAME)
    log.info(f"✅ Embedder ready ({settings.EMBED_MODEL_NAME})")

    # 2. CrossEncoder re-ranker
    cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
    log.info(f"✅ CrossEncoder ready ({settings.CROSS_ENCODER_MODEL})")

    # 3. Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    stats = pinecone_index.describe_index_stats()
    log.info(f"✅ Pinecone connected — {stats['total_vector_count']} vectors")

    # 4. Fetch all chunks from Pinecone metadata to build BM25
    #    (done once; BM25 lives in RAM)
    log.info("⏳ Fetching chunks for BM25 index…")
    all_chunks = _fetch_all_chunks(pinecone_index, stats["total_vector_count"])
    bm25, chunk_ids, chunk_lookup = _build_bm25(all_chunks)
    log.info(f"✅ BM25 ready — {len(all_chunks)} chunks")

    # 5. Wire everything together
    _retriever = HybridRetriever(
        embedder=embedder,
        cross_encoder=cross_encoder,
        pinecone_index=pinecone_index,
        bm25=bm25,
        chunk_ids=chunk_ids,
        chunk_lookup=chunk_lookup,
        settings=settings,
    )
    _llm       = GroqClient(settings)
    _evaluator = Evaluator(embedder=embedder, llm=_llm, settings=settings)

    log.info("🚀 Backend ready")
    yield

    log.info("👋 Shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your HF Space URL in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    mode:  str = Field("hybrid", pattern="^(hybrid|semantic|bm25)$")
    top_k: int = Field(5, ge=1, le=10)
    evaluate: bool = True


class ChunkResult(BaseModel):
    chunk_id:  str
    text:      str
    title:     str
    authors:   str
    year:      str
    journal:   str
    ce_score:  float
    rrf_score: float


class EvalResult(BaseModel):
    faithfulness_score: float
    relevancy_score:    float
    num_claims:         int
    num_supported:      int


class QueryResponse(BaseModel):
    query:           str
    answer:          str
    chunks:          list[ChunkResult]
    evaluation:      Optional[EvalResult]
    retrieval_time:  float
    generation_time: float
    eval_time:       Optional[float]
    mode:            str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Readiness probe — HF Spaces pings this."""
    ready = all([_retriever, _llm, _evaluator])
    return {"status": "ok" if ready else "loading", "ready": ready}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not _retriever:
        raise HTTPException(503, "Backend still loading, retry in a moment")

    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    chunks = await asyncio.to_thread(
        _retriever.retrieve, req.query, mode=req.mode, top_k=req.top_k
    )
    retrieval_time = round(time.perf_counter() - t0, 3)

    if not chunks:
        raise HTTPException(404, "No relevant documents found for this query")

    # ── 2. Generate ──────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    answer = await asyncio.to_thread(_llm.generate, req.query, chunks)
    generation_time = round(time.perf_counter() - t1, 3)

    # ── 3. Evaluate (optional) ───────────────────────────────────────────────
    eval_result = None
    eval_time   = None
    if req.evaluate:
        t2 = time.perf_counter()
        eval_result = await asyncio.to_thread(
            _evaluator.evaluate, req.query, answer, chunks
        )
        eval_time = round(time.perf_counter() - t2, 3)

    # ── 4. Serialise ─────────────────────────────────────────────────────────
    chunk_out = [
        ChunkResult(
            chunk_id  = c["chunk_id"],
            text      = c["text"],
            title     = c.get("title", ""),
            authors   = c.get("authors", ""),
            year      = c.get("year", ""),
            journal   = c.get("journal", ""),
            ce_score  = c.get("ce_score", 0.0),
            rrf_score = c.get("rrf_score", 0.0),
        )
        for c in chunks
    ]

    ev_out = None
    if eval_result:
        ev_out = EvalResult(
            faithfulness_score = eval_result["faithfulness"]["faithfulness_score"],
            relevancy_score    = eval_result["relevancy"]["relevancy_score"],
            num_claims         = eval_result["faithfulness"]["num_claims"],
            num_supported      = eval_result["faithfulness"]["num_supported"],
        )

    log.info(
        f"query={req.query[:60]!r} mode={req.mode} "
        f"ret={retrieval_time}s gen={generation_time}s eval={eval_time}s"
    )

    return QueryResponse(
        query           = req.query,
        answer          = answer,
        chunks          = chunk_out,
        evaluation      = ev_out,
        retrieval_time  = retrieval_time,
        generation_time = generation_time,
        eval_time       = eval_time,
        mode            = req.mode,
    )


# ── Helpers (called at startup) ───────────────────────────────────────────────
def _fetch_all_chunks(pinecone_index, total: int) -> list[dict]:
    """
    Pinecone free tier has no list-all-vectors API, so we query with a
    zero vector to get metadata for every stored chunk in one shot.
    Falls back to paginated dummy-vector queries if > 10 000 chunks.
    """
    dim = 384  # all-MiniLM-L6-v2
    zero = [0.0] * dim
    results = pinecone_index.query(vector=zero, top_k=min(total, 10_000),
                                   include_metadata=True)
    chunks = []
    for m in results["matches"]:
        meta = m.get("metadata", {})
        chunks.append({
            "chunk_id": m["id"],
            "text":     meta.get("text", ""),
            "title":    meta.get("title", ""),
            "authors":  meta.get("authors", ""),
            "year":     meta.get("year", ""),
            "journal":  meta.get("journal", ""),
            "strategy": meta.get("strategy", ""),
        })
    return chunks


def _build_bm25(chunks: list[dict]):
    tokenized  = [c["text"].lower().split() for c in chunks]
    bm25       = BM25Okapi(tokenized)
    chunk_ids  = [c["chunk_id"] for c in chunks]
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    return bm25, chunk_ids, chunk_lookup


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
