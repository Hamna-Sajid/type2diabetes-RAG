"""
llm.py — Groq LLM client
==========================
- Exponential backoff on 429s (reads Retry-After header when available)
- Configurable inter-call sleep to stay inside free-tier token budget
- Single `generate()` for RAG answers, single `complete()` for eval prompts
"""

import re
import time
import logging

import requests

log = logging.getLogger("rag.llm")

_SYSTEM_PROMPT = (
    "You are a medical research assistant specialising in Type 2 Diabetes. "
    "Answer the question using ONLY the provided context. "
    "Cite sources as [Source N] where relevant. "
    "If the context is insufficient, say so clearly and do not invent facts."
)


class GroqClient:
    def __init__(self, settings):
        self.s       = settings
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(self, query: str, chunks: list[dict]) -> str:
        """Build context from chunks, call Groq, return answer string."""
        prompt = self._build_rag_prompt(query, chunks)
        result = self._call(prompt, max_tokens=self.s.LLM_MAX_TOKENS,
                            temperature=self.s.LLM_TEMPERATURE)
        return result or "Error: could not generate answer."

    def complete(self, prompt: str, max_tokens: int = 300,
                 temperature: float = 0.05) -> str:
        """Low-temp completion used by the evaluator."""
        result = self._call(prompt, max_tokens=max_tokens,
                            temperature=temperature)
        return result or ""

    # ── Internal ──────────────────────────────────────────────────────────────
    def _build_rag_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            header = f"[Source {i}] {c.get('title', 'Unknown')} ({c.get('year', '')})"
            context_parts.append(f"{header}\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}"
        )

    def _call(self, prompt: str, *, max_tokens: int,
              temperature: float) -> str | None:
        payload = {
            "model":       self.s.GROQ_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        for attempt in range(self.s.LLM_MAX_RETRIES):
            try:
                resp = requests.post(
                    self.s.GROQ_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip()
                    # Polite sleep so we don't burn the free-tier rate limit
                    time.sleep(self.s.GROQ_INTER_CALL_DELAY)
                    return text

                elif resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else min(2 ** (attempt + 2), 60)
                    log.warning(f"Rate limited — waiting {wait:.0f}s (attempt {attempt+1})")
                    time.sleep(wait)

                elif resp.status_code == 401:
                    log.error("Invalid GROQ_API_KEY")
                    return None

                else:
                    log.warning(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(5)

            except requests.exceptions.Timeout:
                log.warning(f"Timeout on attempt {attempt+1}")
                time.sleep(3)
            except Exception as e:
                log.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(3)

        log.error(f"All {self.s.LLM_MAX_RETRIES} attempts failed")
        return None
