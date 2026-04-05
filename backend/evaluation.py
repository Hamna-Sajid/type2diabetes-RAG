"""
evaluation.py — LLM-as-a-Judge
================================
Faithfulness
  1. Extract claims from the generated answer (LLM)
  2. Verify each claim against the retrieved context (LLM)
  3. Score = supported_claims / total_claims

Relevancy
  1. Generate N alternative questions from the answer (LLM)
  2. Cosine similarity between each alt-question and the original query
  3. Score = mean similarity
"""

import re
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("rag.evaluation")


class Evaluator:
    def __init__(self, *, embedder: SentenceTransformer, llm, settings):
        self.embedder = embedder
        self.llm      = llm
        self.s        = settings

    # ── Public entry point ────────────────────────────────────────────────────
    def evaluate(self, query: str, answer: str, chunks: list[dict]) -> dict:
        faith = self._faithfulness(answer, chunks)
        relev = self._relevancy(query, answer)
        return {
            "faithfulness": faith,
            "relevancy":    relev,
            "overall_score": round(
                (faith["faithfulness_score"] + relev["relevancy_score"]) / 2, 3
            ),
        }

    # ── Faithfulness ──────────────────────────────────────────────────────────
    def _faithfulness(self, answer: str, chunks: list[dict]) -> dict:
        context = " ".join(c["text"] for c in chunks)[:3000]  # cap context length
        claims  = self._extract_claims(answer)
        log.info(f"Evaluating faithfulness for {len(claims)} claims")

        verifications = []
        for claim in claims:
            supported = self._verify_claim(claim, context)
            verifications.append({"claim": claim, "supported": supported})

        n_supported = sum(1 for v in verifications if v["supported"])
        score = n_supported / len(verifications) if verifications else 0.0

        return {
            "faithfulness_score": round(score, 3),
            "num_claims":         len(claims),
            "num_supported":      n_supported,
            "verifications":      verifications,
        }

    def _extract_claims(self, answer: str) -> list[str]:
        prompt = (
            "Extract all factual claims from the answer below. "
            "Return ONLY a numbered list, one claim per line. "
            "Each claim must be a single verifiable statement.\n\n"
            f"ANSWER:\n{answer[:1500]}\n\nExtract claims:"
        )
        raw    = self.llm.complete(prompt, max_tokens=400)
        claims = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                claim = re.sub(r"^[\d\-\.\)\s]+", "", line).strip()
                if len(claim) > 15:
                    claims.append(claim)

        # Fallback: sentence-split the answer
        if not claims:
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            claims = [s.strip() for s in sentences if len(s.strip()) > 20][:6]

        return claims or [answer[:300]]

    def _verify_claim(self, claim: str, context: str) -> bool:
        prompt = (
            "You are evaluating whether a research claim is supported by the context.\n"
            "Answer YES if the claim is directly or indirectly supported, "
            "or NO if the context clearly contradicts or does not mention it.\n\n"
            f"CONTEXT: {context[:1200]}\n\n"
            f"CLAIM: {claim}\n\n"
            "Answer YES or NO only:"
        )
        result = self.llm.complete(prompt, max_tokens=5)
        return result.strip().upper().startswith("YES")

    # ── Relevancy ─────────────────────────────────────────────────────────────
    def _relevancy(self, query: str, answer: str) -> dict:
        n = self.s.EVAL_ALT_QUESTIONS
        prompt = (
            f"Given this answer about Type 2 Diabetes, generate exactly {n} "
            "different questions that this answer addresses. "
            f"Return ONLY the questions numbered 1 to {n}, one per line.\n\n"
            f"ANSWER: {answer[:600]}\n\nGenerate {n} questions:"
        )
        raw = self.llm.complete(prompt, max_tokens=200)

        alt_questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                q = re.sub(r"^[\d\-\.\)\s]+", "", line).strip()
                if len(q) > 10:
                    alt_questions.append(q)
        alt_questions = alt_questions[:n]

        # Fallback: use the answer itself
        method = "llm_alt_questions"
        if not alt_questions:
            alt_questions = [answer[:300]]
            method        = "embedding_fallback"

        orig_emb = self.embedder.encode(query,         normalize_embeddings=True)
        alt_embs = self.embedder.encode(alt_questions, normalize_embeddings=True)
        sims     = [float(np.dot(orig_emb, ae)) for ae in alt_embs]

        return {
            "relevancy_score": round(float(np.mean(sims)), 3),
            "alt_questions":   alt_questions,
            "similarities":    [round(s, 3) for s in sims],
            "method":          method,
        }
