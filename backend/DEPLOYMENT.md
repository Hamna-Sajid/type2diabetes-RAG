# Diabetes RAG — Deployment Guide

## File layout

```
backend/          ← FastAPI HF Space
  main.py         ← app entry point + lifespan loader
  config.py       ← all settings via env vars
  retrieval.py    ← BM25 + Semantic + RRF + CrossEncoder
  llm.py          ← Groq client with backoff + throttle
  evaluation.py   ← faithfulness + relevancy (LLM-as-a-Judge)
  requirements_backend.txt

frontend/         ← Chainlit HF Space
  app.py          ← updated — calls real backend
  config.py       ← frontend config (points to BACKEND_URL)
  requirements_frontend.txt
```

---

## Step 1 — Deploy the FastAPI backend on HF Spaces

1. Create a new Space → **Docker** SDK → name it e.g. `diabetes-rag-backend`
2. Upload `main.py`, `config.py`, `retrieval.py`, `llm.py`, `evaluation.py`,
   `requirements_backend.txt`
3. Add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

4. Set **Repository secrets** (Settings → Variables and secrets):
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`   (default: `diabetes-rag`)
   - `GROQ_API_KEY`

5. Wait for the Space to build. Test:
   `https://<your-username>-diabetes-rag-backend.hf.space/health`
   → should return `{"status":"ok","ready":true}`

---

## Step 2 — Deploy the Chainlit frontend on HF Spaces

1. Create a new Space → **Docker** SDK → name it e.g. `diabetes-rag-frontend`
2. Upload `app.py`, `config.py` (frontend version), `requirements_frontend.txt`
3. Add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements_frontend.txt .
RUN pip install --no-cache-dir -r requirements_frontend.txt
COPY . .
EXPOSE 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
```

4. Set **Repository secrets**:
   - `BACKEND_URL` = `https://<your-username>-diabetes-rag-backend.hf.space`

5. Done — open the frontend Space URL.

---

## Rate-limit strategy (Groq free tier)

| Setting | Value | Effect |
|---|---|---|
| `GROQ_INTER_CALL_DELAY` | 2.0 s | Sleep after every successful call |
| `LLM_MAX_RETRIES` | 4 | Retry up to 4× on 429 |
| Backoff | reads `Retry-After` header | Honours Groq's own window |
| Eval claims cap | 6 claims | Limits verify calls per query |
| Context cap | 3 000 chars | Keeps prompt tokens low |

For a paid Groq plan set `GROQ_INTER_CALL_DELAY=0` in secrets.

---

## API reference

### `GET /health`
Returns `{"status":"ok","ready":true}` once models are loaded.

### `POST /query`
```json
{
  "query":    "What is metformin?",
  "mode":     "hybrid",   // "hybrid" | "semantic" | "bm25"
  "top_k":    5,
  "evaluate": true
}
```
Response:
```json
{
  "query":           "...",
  "answer":          "...",
  "chunks":          [{"chunk_id":"...","text":"...","title":"...","ce_score":4.47,...}],
  "evaluation":      {"faithfulness_score":0.8,"relevancy_score":0.77,"num_claims":5,"num_supported":4},
  "retrieval_time":  0.42,
  "generation_time": 0.91,
  "eval_time":       3.2,
  "mode":            "hybrid"
}
```
