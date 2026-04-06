"""
app/app.py - Diabetes RAG
Run: chainlit run app/app.py --port 8000
"""

import sys, logging, json, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
import yaml
from dotenv import load_dotenv
load_dotenv()

from app.retrieval import Retriever
from app.generation import Generator

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Load once at startup 

log.info("Loading retrieval indexes...")
retriever = Retriever()
retriever.load()
generator = Generator()

with open("config.yaml") as f:
    _cfg = yaml.safe_load(f)

SYSTEM_PROMPT = _cfg["generation"]["system_prompt"].strip()
MODEL      = generator.model

log.info("Diabetes RAG ready — model: %s (provider: %s)", MODEL, generator.provider)

# Retrieval routing 
ROUTE_PROMPT = """You are a routing assistant. Decide if the user's message requires
searching a collection of Type 2 Diabetes research abstracts from PubMed.

Answer with exactly one word: YES or NO.

Search is needed for:
- Questions about diabetes pathophysiology or mechanisms
- Questions about diabetes treatment options or medications
- Questions about diabetes complications
- Questions about diabetes diagnosis or monitoring
- Questions about specific diabetes-related topics mentioned in research
- Any question where the user explicitly asks for research evidence, studies, or abstracts

Search is NOT needed for:
- Greetings (hello, hi, how are you)
- Questions about what this tool is or how to use it
- General conversation
- Questions completely unrelated to health/diabetes

User message: {query}
Answer (YES or NO):"""

OBVIOUS_NO = {
    "hello", "hi", "hey", "thanks", "thank you", "bye",
    "good morning", "good afternoon", "good evening",
}

async def needs_retrieval(query: str, use_rag: bool) -> bool:
    if not use_rag:
        return False
    q = query.lower().strip().rstrip("!?.")
    if q in OBVIOUS_NO or len(q.split()) <= 2:
        return False
    # Ask the model for routing decision (don't store in history)
    result = generator.generate(query=ROUTE_PROMPT.format(query=query), chunks=None, use_history=False)
    decision = result.get("answer", "")
    return "YES" in decision.upper()

# Format context for LLM 
def build_context(chunks: list[dict]) -> str:
    """
    Format chunks (abstracts) for the LLM. Each block is labelled with inline citation,
    which is exactly what the LLM should cite in its answer.
    """
    parts = []
    for c in chunks:
        authors = c.get("authors", "?")
        year = c.get("year", "?")
        abstract = c.get("abstract", c.get("_text", ""))[:1000]
        citation = f"{authors} ({year})"
        parts.append(f"**{citation}**\n\n{abstract}")
    return "\n\n".join(parts)

# Settings 
@cl.on_chat_start
async def start():
    generator.clear_history()  # Reset history for new session
    cl.user_session.set("settings", {"use_rag": True, "method": "hybrid", "top_k": 5})
    cl.user_session.set("retrieved_chunks", {})  # Accumulate chunks across conversation
    await cl.ChatSettings([
        cl.input_widget.Switch(
            id="use_rag", label="Enable RAG", initial=True,
            description="Retrieve diabetes research abstracts before answering",
        ),
        cl.input_widget.Select(
            id="method", label="Retrieval method", initial_index=0,
            values=["hybrid", "dense", "bm25"],
            description="hybrid = BM25 + dense + RRF + reranking",
        ),
        cl.input_widget.Slider(
            id="top_k", label="Chunks to retrieve",
            initial=5, min=1, max=10, step=1,
        ),
    ]).send()

@cl.on_settings_update
async def update_settings(s: dict):
    cl.user_session.set("settings", {
        "use_rag": s.get("use_rag", True),
        "method":  s.get("method", "hybrid"),
        "top_k":   int(s.get("top_k", 5)),
    })

# Main handler 
@cl.on_message
async def on_message(msg: cl.Message):
    s       = cl.user_session.get("settings") or {}
    use_rag = s.get("use_rag", True)
    method  = s.get("method", "hybrid")
    top_k   = int(s.get("top_k", 5))
    query   = msg.content.strip()
    if not query:
        return

    chunks = []
    do_retrieve = await needs_retrieval(query, use_rag)

    # Retrieval 
    if do_retrieve:
        async with cl.Step(name="Searching abstracts", type="retrieval") as step:
            chunks = retriever.retrieve(query, method=method, top_k=top_k)
            step.output = ""
            for c in chunks:
                authors = c.get("authors", "?")
                year   = c.get("year", "?")
                step.output += f"{authors} ({year})\n"
            
            # Accumulate chunks in session (use pmid::chunk_idx as key to avoid duplicates)
            accumulated = cl.user_session.get("retrieved_chunks", {})
            for c in chunks:
                key = c.get("chunk_id", "?")
                accumulated[key] = c
            cl.user_session.set("retrieved_chunks", accumulated)

    # Generate answer via LLM (handles RAG formatting internally)
    # Generator maintains internal conversation history
    async with cl.Step(name="Generating", type="llm") as step:
        result = generator.generate(query=query, chunks=chunks if do_retrieve else None, use_history=True)
        answer = result.get("answer", "Error generating response")
        duration = result.get("duration_s", 0)
        step.output = f"`{MODEL.split('/')[-1]}` — {duration}s"

    # Source elements - use ALL accumulated chunks, not just current retrieval
    # This way, if the model cites a chunk from earlier, it's still available
    accumulated = cl.user_session.get("retrieved_chunks", {})
    elements: list[cl.Text] = []
    for key, c in accumulated.items():
        pmid   = c.get("chunk_id", "").split("::")[0]
        title  = c.get("title", "?")
        authors = c.get("authors", "?")
        year   = c.get("year", "?")
        journal = c.get("journal", "?")
        score  = c.get("_rerank_score", c.get("_score", 0))
        abstract = c.get("abstract", c.get("_text", ""))

        citation = f"{authors} ({year})"
        elements.append(cl.Text(
            name=citation,
            content=(
                f"### {title}\n\n"
                f"**Citation:** {citation}  \n"
                f"**Journal:** {journal}  ·  **PMID:** {pmid}  \n"
                f"**Relevance score:** `{score:.3f}`\n\n"
                f"{abstract}"
            ),
            display="side",
        ))

    # Send 
    rag_info = f"{method} · {len(chunks)} chunks" if do_retrieve else "no retrieval"
    meta = f"\n\n---\n*`{MODEL.split('/')[-1]}` · {rag_info} · {duration}s*"

    await cl.Message(
        content=answer + meta,
        elements=elements,
    ).send()