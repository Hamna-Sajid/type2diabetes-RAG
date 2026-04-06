---
title: Type 2 Diabetes RAG
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Type 2 Diabetes RAG

> Medical research synthesis via advanced RAG - PubMed abstract retrieval, hybrid BM25+dense search, cross-encoder re-ranking, and academic citation formatting.

**Demo query:**
```
"What are the most effective medications for treating Type 2 Diabetes and how do they work?"
```

## Setup (to run locally)

```bash
# 1. Clone this repo
git clone https://github.com/Hamna-Sajid/type2diabetes-RAG
cd type2diabetes-RAG

# 2. Create virtualenv
python -m venv .venv # python3 for macOS/Linux
.venv/Scripts/Activate.ps1  # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env - add your HuggingFace API token (optional for local LLM)
```

## Local LLM Setup (Required)

Install and start Ollama for local inference:

```bash
# Install Ollama: https://ollama.ai

# Pull required models
ollama pull llama3.1:8b

# Start Ollama server (runs at localhost:11434)
ollama serve
```

## Pipeline (run once, offline)

The pipeline fetches diabetes abstracts from PubMed, chunks them, embeds them, and builds keyword indices.

```bash
# Step 1: Fetch and chunk diabetes abstracts from PubMed
python pipeline/prepare_data.py --strategy fixed
# Repeat for: --strategy recursive

# Step 2: Build BM25 keyword index
python pipeline/build_bm25.py --strategy fixed

# Step 3: Embed chunks and upsert to vector database
python pipeline/embed_and_upsert.py --strategy fixed
```

## Running the App locally

The Chainlit app provides an interactive interface to ask questions about diabetes research.

```bash
# Start the app
chainlit run app/app.py --port 8000
# Open http://localhost:8000 in your browser
```

## Features

- **Hybrid Retrieval**: BM25 keyword search + semantic embeddings + reciprocal rank fusion
- **Cross-encoder Reranking**: Improves relevance of top results
- **Academic Citations**: Inline citation format `Authors (Year)` - clickable and traceable
- **PubMed Integration**: ~600 Type 2 Diabetes abstracts covering pathophysiology, treatment, complications
- **Local LLM Support**: Run with Ollama (llama3.1:8b) or cloud provider (HuggingFace Router)

## Configuration

All settings are in `config.yaml`:
- **Chunking strategy**: fixed (512 tokens) or recursive (sentence-based)
- **Vector store**: ChromaDB (local) or Pinecone (cloud)
- **Embedding model**: all-MiniLM-L6-v2 (384-dim)
- **LLM provider**: local (Ollama) or cloud (HuggingFace Router)
- **System prompt**: Enforces citation format and factual grounding

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Medical Disclaimer

This repository is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.- **Embedding model**: all-MiniLM-L6-v2 (384-dim)
- **LLM provider**: local (Ollama) or cloud (HuggingFace Router)
- **System prompt**: Enforces citation format and factual grounding

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Medical Disclaimer

This repository is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

