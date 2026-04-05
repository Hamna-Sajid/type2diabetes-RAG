"""
retrieval.py — Hybrid retrieval pipeline
=========================================
  1. BM25 keyword search (in-RAM)
  2. Semantic search (Pinecone ANN)
  3. RRF fusion
  4. CrossEncoder re-ranking
"""

import time
import logging
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

log = logging.getLogger("rag.retrieval")


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder:       SentenceTransformer,
        cross_encoder:  CrossEncoder,
        pinecone_index,
        bm25:           BM25Okapi,
        chunk_ids:      list[str],
        chunk_lookup:   dict[str, dict],
        settings,
    ):
        self.embedder      = embedder
        self.ce            = cross_encoder
        self.index         = pinecone_index
        self.bm25          = bm25
        self.chunk_ids     = chunk_ids
        self.chunk_lookup  = chunk_lookup
        self.s             = settings

    # ── Public entry point ────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        mode:  Literal["hybrid", "semantic", "bm25"] = "hybrid",
        top_k: int = 5,
    ) -> list[dict]:
        """
        Returns up to `top_k` re-ranked chunks as dicts with keys:
        chunk_id, text, title, authors, year, journal, ce_score, rrf_score
        """
        rerank_k = max(top_k, self.s.RERANK_TOP_K)

        if mode == "bm25":
            candidates = self._bm25_search(query, top_k=self.s.BM25_TOP_K + self.s.SEM_TOP_K)
        elif mode == "semantic":
            candidates = self._semantic_search(query, top_k=self.s.BM25_TOP_K + self.s.SEM_TOP_K)
        else:  # hybrid
            bm25_res  = self._bm25_search(query, top_k=self.s.BM25_TOP_K)
            sem_res   = self._semantic_search(query, top_k=self.s.SEM_TOP_K)
            candidates = self._rrf_fusion(bm25_res, sem_res)

        return self._rerank(query, candidates, top_k=top_k)

    # ── BM25 ─────────────────────────────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens  = query.lower().split()
        scores  = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in indices]

    # ── Semantic (Pinecone) ───────────────────────────────────────────────────
    def _semantic_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_emb   = self.embedder.encode(query, normalize_embeddings=True).tolist()
        result  = self.index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        return [(m["id"], float(m["score"])) for m in result["matches"]]

    # ── RRF fusion ───────────────────────────────────────────────────────────
    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        sem_results:  list[tuple[str, float]],
        k: int = None,
    ) -> list[tuple[str, float]]:
        k = k or self.s.RRF_K
        rrf: dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_results):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(sem_results):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    # ── CrossEncoder re-ranking ───────────────────────────────────────────────
    def _rerank(
        self,
        query:      str,
        candidates: list[tuple[str, float]],
        top_k:      int,
    ) -> list[dict]:
        pairs, valid = [], []
        for cid, rrf_score in candidates:
            chunk = self.chunk_lookup.get(cid)
            if chunk:
                pairs.append([query, chunk["text"]])
                valid.append((cid, rrf_score))

        if not pairs:
            return []

        ce_scores = self.ce.predict(pairs)

        reranked = []
        for i, (cid, rrf_score) in enumerate(valid):
            chunk = self.chunk_lookup[cid]
            reranked.append({
                **chunk,
                "rrf_score": round(float(rrf_score), 5),
                "ce_score":  round(float(ce_scores[i]), 4),
            })

        reranked.sort(key=lambda x: x["ce_score"], reverse=True)
        return reranked[:top_k]
