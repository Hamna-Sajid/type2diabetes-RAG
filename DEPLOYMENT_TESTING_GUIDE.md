# Type 2 Diabetes RAG - Complete Deployment & Testing Guide

## 📋 Overview

You have two applications that need to be deployed on Hugging Face Spaces:

1. **Backend** (FastAPI) - Does retrieval, generation, and evaluation
2. **Frontend** (Chainlit) - User interface that calls the backend

---

## 🚀 STEP 1: Deploy Backend on Hugging Face Spaces

### 1.1 Create a New HF Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `diabetes-rag-backend` (or your choice)
   - **License**: MIT
   - **SDK**: **Docker** (important!)
   - **Private**: Yes (optional, but recommended)
4. Click **Create Space**

### 1.2 Upload Backend Files

Once the Space is created, upload these files from `backend/` folder:

```
main.py
config.py
retrieval.py
llm.py
evaluation.py
requirements_backend.txt
Dockerfile
```

**How to upload:**
- Go to the Space
- Click **Files** → **Add file** → select files or drag-and-drop
- Commit the files

### 1.3 Set Up Secrets

1. Go to **Settings** → **Repository secrets**
2. Click **"New secret"** and add these:

| Secret Name | Value | Where to Get |
|---|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key | [console.pinecone.io](https://console.pinecone.io) |
| `PINECONE_INDEX_NAME` | `diabetes-rag` | Your Pinecone index name |
| `GROQ_API_KEY` | Your Groq API key | [console.groq.com](https://console.groq.com) |

3. Click **"Save"** for each secret

### 1.4 Wait for Build

1. Go to **Logs** tab
2. Watch for the build to complete
3. Look for this message:
   ```
   INFO | rag.main | 🚀 Backend ready
   ```

---

## ✅ STEP 2: Test Backend Health

Once backend shows "Running", test if it's ready:

### 2.1 Check Health Endpoint

```bash
curl https://<your-username>-diabetes-rag-backend.hf.space/health
```

Expected response (if backend is still loading):
```json
{"status": "loading", "ready": false}
```

**Wait and check again every 30 seconds until you see:**
```json
{"status": "ok", "ready": true}
```

### 2.2 Example Test Query (Optional)

Once `ready: true`, try a simple query:

```bash
curl -X POST https://<your-username>-diabetes-rag-backend.hf.space/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is metformin?",
    "mode": "hybrid",
    "top_k": 5,
    "evaluate": true
  }'
```

You should get a response like:
```json
{
  "query": "What is metformin?",
  "answer": "Metformin is...",
  "chunks": [...],
  "evaluation": {
    "faithfulness_score": 0.95,
    "relevancy_score": 0.87
  },
  "retrieval_time": 0.45,
  "generation_time": 2.15,
  "eval_time": 0.8
}
```

**✓ If you see this, your backend is working!**

---

## 📋 Copy Backend URL

From the browser URL bar, copy:
```
https://<your-username>-diabetes-rag-backend.hf.space
```

You'll need this for the frontend.

---

## 🚀 STEP 3: Deploy Frontend on Hugging Face Spaces

### 3.1 Create a New HF Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `diabetes-rag-frontend`
   - **License**: MIT
   - **SDK**: **Docker** (important!)
4. Click **Create Space**

### 3.2 Upload Frontend Files

Upload these files from `frontend/` folder:

```
app.py
config.py
requirements_frontend.txt
Dockerfile
```

Or upload from root:
- `Dockerfile` (in root, not renamed)

### 3.3 Set Up Secrets

1. Go to **Settings** → **Repository secrets**
2. Add this secret:

| Secret Name | Value |
|---|---|
| `BACKEND_URL` | `https://<your-username>-diabetes-rag-backend.hf.space` |

**Replace** `<your-username>` with your actual HF username!

### 3.4 Wait for Build

Watch the **Logs** tab until you see:
```
✓ Chainlit app configured and ready
```

---

## 🧪 STEP 4: Test Frontend in Browser

Once frontend Space shows "Running":

1. Click on the **Space URL** to open it
2. You should see the Chainlit chat interface
3. Try typing a question, e.g.:
   ```
   What are the medications for type 2 diabetes?
   ```

### Expected Behavior:

1. ✅ Chat interface loads with welcome message
2. ✅ Example questions are displayed
3. ✅ You can type a question
4. ✅ After ~5-10 seconds, you get an answer with:
   - 📖 **Answer** section with generated text
   - 📚 **Retrieved Sources** with titles, authors, years
   - 📊 **Quality Metrics** showing faithfulness & relevancy scores

---

## 🔍 Troubleshooting

### Problem: Frontend shows error "Could not connect to backend"

**Solution:**
1. Check that **backend** Space is running (go to backend Space URL)
2. Check that **backend health** returns `{"ready": true}`
3. Check that **`BACKEND_URL` secret** is set correctly on frontend:
   - Should be: `https://<your-username>-diabetes-rag-backend.hf.space`
   - Check for typos!
4. Try **restarting frontend** Space:
   - Go to **Settings** → **Restart Space**

### Problem: Backend takes long time to load

**Reason:** First-time load requires downloading models (~2-3GB)

**Solution:**
- Wait 5-10 minutes for models to download
- Check logs for progress
- Once you see `🚀 Backend ready`, it will be much faster

### Problem: Response says "timeout" or "no answer"

**Solution:**
1. Check backend logs for errors
2. Make sure Pinecone index has data
3. Try simpler questions first
4. Wait if backend is still loading

---

## ✅ HAcritical Verification Checklist

Use this checklist to verify everything works:

- [ ] **Backend Space created** at `diabetes-rag-backend`
- [ ] **Backend Dockerfile uploaded** and builds successfully
- [ ] **Backend secrets set**: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `GROQ_API_KEY`
- [ ] **Backend /health returns** `{"ready": true}`
- [ ] **Backend /query works** with curl test
- [ ] **Backend URL copied**: `https://<username>-diabetes-rag-backend.hf.space`
- [ ] **Frontend Space created** at `diabetes-rag-frontend`
- [ ] **Frontend Dockerfile uploaded** and builds successfully
- [ ] **Frontend BACKEND_URL secret set** correctly
- [ ] **Frontend loads in browser** without errors
- [ ] **Frontend displays** welcome message and example questions
- [ ] **Frontend accepts** your question
- [ ] **Frontend displays** answer from backend
- [ ] **Frontend shows** retrieved sources
- [ ] **Frontend shows** faithfulness & relevancy scores (if enabled)

---

## 🎯 Testing Workflow

### Test 1: Simple Question
```
Question: "What is diabetes?"
Expected: 2-3 sentence answer with sources
```

### Test 2: Specific Question  
```
Question: "How does metformin work?"
Expected: Detailed answer explaining mechanism, with medical sources
```

### Test 3: Multiple Follow-ups
```
Ask 3-4 related questions in sequence
Expected: Each gets answer + sources, no errors
```

### Test 4: Quality Metrics
```
If evaluation enabled, check scores
Expected: Faithfulness > 0.8, Relevancy > 0.7
```

---

## 📊 Understanding the Response

Each answer includes:

1. **Answer**: AI-generated response based on medical sources
2. **Sources**: Retrieved papers with:
   - Title
   - Authors
   - Year
   - Relevance Score (0-1, higher = more relevant)
3. **Quality Metrics**:
   - **Faithfulness** (0-1): % of claims supported by sources
   - **Relevancy** (0-1): How well answer matches question

---

## 🔗 Useful Links

- Backend Space URL: `https://<username>-diabetes-rag-backend.hf.space`
- Frontend Space URL: `https://<username>-diabetes-rag-frontend.hf.space`
- Backend Health: `https://<username>-diabetes-rag-backend.hf.space/health`
- API Docs: `https://<username>-diabetes-rag-backend.hf.space/docs`

---

## 💡 Tips for Success

1. **Wait for first build**: First-time model loading takes 5-10 minutes
2. **Test backend first**: Always verify backend health before testing frontend
3. **Check logs**: Both Spaces have logs - check them if something breaks
4. **Copy URLs carefully**: Small typos in BACKEND_URL will break frontend
5. **Give it time**: LLM generation can take 5-10 seconds per question
6. **Check secrets**: Many errors are due to missing or wrong secret values

---

## 🎉 Success!

Once everything passes the checklist, your RAG system is **fully deployed and working!**

You can now:
- ✅ Ask questions about Type 2 Diabetes
- ✅ Get answers based on real medical research
- ✅ See retrieved sources and quality metrics
- ✅ Share the link with others to use the system
