"""
Type 2 Diabetes RAG - Chainlit Frontend

A conversational AI system that answers questions about Type 2 Diabetes
using Retrieval-Augmented Generation (RAG) with advanced retrieval strategies.

This version connects to a FastAPI backend for all operations:
  - Document retrieval
  - Answer generation
  - Quality evaluation (faithfulness, relevancy)
"""

import os
import asyncio
from typing import Optional, List, Dict

# Chainlit imports
import chainlit as cl

# HTTP imports
import httpx

from config import (
    BACKEND_URL,
    APP_TITLE,
    APP_DESCRIPTION,
    EXAMPLE_QUERIES,
    USE_HYBRID_SEARCH,
    EVALUATE_FAITHFULNESS,
    EVALUATE_RELEVANCY,
)

print(f"🚀 Frontend starting")
print(f"   Backend URL: {BACKEND_URL}")
print(f"   Hybrid search: {USE_HYBRID_SEARCH}")
print(f"   Evaluation: {EVALUATE_FAITHFULNESS or EVALUATE_RELEVANCY}")

# ============= CHAINLIT LIFECYCLE HOOKS =============

@cl.on_chat_start
async def start():
    """Called when a new user starts a chat session."""
    
    print("âœ“ Chat started")  # Debug print
    
    # STEP 1: Send title message
    await cl.Message(
        content=f"# {APP_TITLE}\n\n{APP_DESCRIPTION}"
    ).send()
    
    # STEP 2: Send instructions
    await cl.Message(
        content="""
## How to Use:
1. **Ask a question** about Type 2 Diabetes
2. **View the answer** from medical research
3. **Check sources** that were retrieved
4. **See quality scores** (faithfulness & relevancy)

## Example Questions:
"""
    ).send()
    
    # STEP 3: Send example queries
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        await cl.Message(
            content=f"{i}. {query}"
        ).send()
    
    # STEP 4: Send ready message
    await cl.Message(
        content="**Type your question below to get started!**"
    ).send()
    
    # STEP 5: Initialize session
    cl.user_session.set("message_count", 0)
    cl.user_session.set("chat_history", [])
    
    print("Chat initialization complete")


@cl.on_message
async def main(message: cl.Message):
    """Process user query through RAG pipeline."""
    
    print(f"\n>>> User message: {message.content}")
    
    # Get the query
    user_query = message.content
    
    # Update message count
    message_count = cl.user_session.get("message_count", 0)
    cl.user_session.set("message_count", message_count + 1)
    
    # Add to history
    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": user_query})
    
    # Step 1: Show loading message
    thinking_msg = cl.Message(content="Searching medical literature...")
    await thinking_msg.send()
    
    try:
        # Step 2: RETRIEVE documents
        print("  â†’ Retrieving documents...")
        thinking_msg.content = "Retrieving relevant documents..."
        await thinking_msg.update()
        
        retrieved_context, sources = await retrieve_documents(user_query)
        print(f"  Retrieved {len(sources)} sources")
        
        # Step 3: GENERATE answer
        print("  â†’ Generating answer...")
        thinking_msg.content = "Generating answer..."
        await thinking_msg.update()
        
        answer = await generate_answer(user_query, retrieved_context)
        print(f"  Generated answer ({len(answer)} chars)")
        
        # Step 4: EVALUATE quality
        faithfulness_score = None
        relevancy_score = None
        
        if EVALUATE_FAITHFULNESS or EVALUATE_RELEVANCY:
            print("  â†’ Evaluating quality...")
            thinking_msg.content = " Evaluating answer quality..."
            await thinking_msg.update()
            
            if EVALUATE_FAITHFULNESS:
                faithfulness_score = await evaluate_faithfulness(answer, retrieved_context)
            
            if EVALUATE_RELEVANCY:
                relevancy_score = await evaluate_relevancy(answer, user_query)
            
            print(f"  Faithfulness: {faithfulness_score}, Relevancy: {relevancy_score}")
        
        # Remove thinking message
        await thinking_msg.remove()
        
        # Step 5: DISPLAY results
        await display_results(
            answer=answer,
            sources=sources,
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score
        )
        
        # Update chat history
        chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "scores": {
                "faithfulness": faithfulness_score,
                "relevancy": relevancy_score
            }
        })
        cl.user_session.set("chat_history", chat_history)
        
        print("Message processing complete\n")
        
    except Exception as e:
        print(f"  Error: {str(e)}")
        await thinking_msg.remove()
        error_content = f""" **Error Processing Query**
    
An error occurred while processing your question. Please try again later.
Error details: {str(e)}
""" 
        await cl.Message(content=error_content).send()


# ============= BACKEND COMMUNICATION =============

async def call_backend(query: str) -> dict:
    """
    Call the FastAPI backend /query endpoint.
    Returns the complete response with answer, chunks, and scores.
    """
    
    if not BACKEND_URL:
        raise Exception("❌ BACKEND_URL not configured. Set it in your HF Spaces secrets.")
    
    payload = {
        "query": query,
        "mode": "hybrid" if USE_HYBRID_SEARCH else "semantic",
        "top_k": 5,
        "evaluate": EVALUATE_FAITHFULNESS or EVALUATE_RELEVANCY
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{BACKEND_URL}/query"
            print(f"   POST {url}")
            
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                error_text = response.text[:500]
                raise Exception(f"Backend returned {response.status_code}: {error_text}")
            
            data = response.json()
            return data
    
    except httpx.ConnectError as e:
        raise Exception(
            f"❌ Could not connect to backend at {BACKEND_URL}\n"
            f"Make sure the backend is deployed and ready.\n"
            f"Error: {str(e)}"
        )
    except httpx.TimeoutException:
        raise Exception(
            f"❌ Backend request timed out. It may still be loading.\n"
            f"Try again in a moment."
        )
    except Exception as e:
        raise Exception(f"❌ Backend error: {str(e)}")


# ============= RETRIEVAL FUNCTIONS =============

async def retrieve_documents(query: str) -> tuple:
    """Kept for backwards compatibility - now calls backend."""
    backend_response = await call_backend(query)
    return backend_response, backend_response.get('chunks', [])


async def generate_answer(query: str, backend_response: dict) -> str:
    """Extract answer from backend response."""
    answer = backend_response.get('answer', 'No answer generated')
    print(f"  ✓ Got answer from backend ({len(answer)} chars)")
    return answer


async def evaluate_faithfulness(answer: str, backend_response: dict) -> Optional[float]:
    """Extract faithfulness score from backend response."""
    evaluation = backend_response.get('evaluation', {})
    if evaluation:
        score = evaluation.get('faithfulness_score')
        print(f"  ✓ Faithfulness score: {score}")
        return score
    return None


async def evaluate_relevancy(answer: str, backend_response: dict) -> Optional[float]:
    """Extract relevancy score from backend response."""
    evaluation = backend_response.get('evaluation', {})
    if evaluation:
        score = evaluation.get('relevancy_score')
        print(f"  ✓ Relevancy score: {score}")
        return score
    return None


# ============= DISPLAY FUNCTIONS =============

async def display_results(
    answer: str,
    sources: list,
    faithfulness_score: float = None,
    relevancy_score: float = None
):
    """Display answer, sources, and evaluation scores."""
    
    try:
        # Display the answer
        answer_msg = f"""## Answer

{answer}"""
        
        await cl.Message(content=answer_msg).send()
        
        # Display sources
        if sources:
            sources_msg = "## Retrieved Sources\n\n"
            
            for i, source in enumerate(sources, 1):
                sources_msg += f"""### Source {i}: {source.get('title', 'Unknown')}

- **Authors**: {source.get('authors', 'N/A')}
- **Year**: {source.get('year', 'N/A')}
- **DOI**: {source.get('doi', 'N/A')}
- **Relevance Score**: {source.get('relevance_score', 0):.1%}

> {source.get('chunk_preview', 'N/A')}

---
"""
            
            await cl.Message(content=sources_msg).send()
        
        # Display evaluation scores
        if faithfulness_score is not None or relevancy_score is not None:
            scores_msg = "## Answer Quality Metrics\n\n"
            
            if faithfulness_score is not None:
                # Create a visual score bar
                bar_length = int(faithfulness_score * 20)
                bar = "â–ˆ" * bar_length + "â–‘" * (20 - bar_length)
                scores_msg += f"**Faithfulness Score**: {faithfulness_score:.1%}\n"
                scores_msg += f"`{bar}`\n"
                scores_msg += "*This score indicates what percentage of claims in the answer are supported by the source documents.*\n\n"
            
            if relevancy_score is not None:
                bar_length = int(relevancy_score * 20)
                bar = "â–ˆ" * bar_length + "â–‘" * (20 - bar_length)
                scores_msg += f"**Relevancy Score**: {relevancy_score:.2f} / 1.00\n"
                scores_msg += f"`{bar}`\n"
                scores_msg += "*This score indicates how well the answer matches the original question.*\n"
            
            await cl.Message(content=scores_msg).send()
        
    except Exception as e:
        print(f"Error in display_results: {e}")
        await cl.Message(content=f"Error displaying results: {e}").send()


# ============= UTILITY FUNCTIONS =============

def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    """Format sources into readable markdown."""
    if not sources:
        return "No sources found"
    
    formatted = "### Sources\n\n"
    for i, source in enumerate(sources, 1):
        formatted += f"{i}. **{source['title']}** ({source['year']})\n"
        formatted += f"   Authors: {source['authors']}\n"
        formatted += f"   Relevance: {source['relevance_score']:.2%}\n\n"
    
    return formatted


# ============= RUN APP =============

if __name__ == "__main__":
    print("Starting Type 2 Diabetes RAG Frontend with Chainlit...")
    print(f"App Title: {APP_TITLE}")
    print(f"Hybrid Search: {USE_HYBRID_SEARCH}")
    print(f"Re-ranking: {USE_RERANKING}")
    print(f"Evaluation Enabled: {EVALUATE_FAITHFULNESS and EVALUATE_RELEVANCY}")
