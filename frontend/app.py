"""
Type 2 Diabetes RAG - Chainlit Frontend

A conversational AI system that answers questions about Type 2 Diabetes
using Retrieval-Augmented Generation (RAG) with advanced retrieval strategies.
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any

# Chainlit imports
import chainlit as cl

# Other imports
import aiohttp
import httpx
from chainlit.types import AskActionResponse

import aiohttp
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    HF_TOKEN,
    HF_MODEL,
    APP_TITLE,
    APP_DESCRIPTION,
    EXAMPLE_QUERIES,
    USE_HYBRID_SEARCH,
    USE_RERANKING,
    EVALUATE_FAITHFULNESS,
    EVALUATE_RELEVANCY,
)

# ============= GLOBAL VARIABLES =============
# These will be initialized when the app starts
retriever = None
llm_client = None

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


# ============= RETRIEVAL FUNCTIONS =============

async def retrieve_documents(query: str) -> tuple:
    """
    Retrieve relevant documents from Pinecone.
    FOR NOW: Returns placeholder data.
    LATER: Will integrate with real Pinecone + BM25.
    """
    
    try:
        # PLACEHOLDER: Return dummy context and sources
        # This will be replaced in Part 3 with real Pinecone calls
        
        context = f"""
Type 2 Diabetes is a chronic metabolic disorder characterized by hyperglycemia 
(elevated blood glucose). It accounts for approximately 90-95% of all diabetes cases.

Management includes:
1. Lifestyle modifications (diet, exercise, weight management)
2. Pharmacological interventions (metformin, GLP-1 agonists, SGLT2 inhibitors)
3. Regular monitoring and complications screening

Early intervention significantly reduces long-term complications.
        """
        
        sources = [
            {
                "title": "Type 2 Diabetes Mellitus: Epidemiology, Pathophysiology, and Management",
                "authors": "Smith J, Jones K, Brown L",
                "year": 2022,
                "doi": "10.1234/example-doi",
                "chunk_preview": "Type 2 Diabetes is a chronic metabolic disorder characterized by hyperglycemia...",
                "relevance_score": 0.95
            },
            {
                "title": "Glycemic Control and Cardiovascular Outcomes in Type 2 Diabetes",
                "authors": "Green R, White T, Black D",
                "year": 2023,
                "doi": "10.5678/example-doi",
                "chunk_preview": "Achieving optimal glycemic control is essential for preventing...",
                "relevance_score": 0.87
            },
            {
                "title": "GLP-1 Receptor Agonists: Benefits and Mechanisms",
                "authors": "Lee S, Park H",
                "year": 2023,
                "doi": "10.9012/example-doi",
                "chunk_preview": "GLP-1 receptor agonists provide both glycemic control and...",
                "relevance_score": 0.82
            },
        ]
        
        return context, sources
        
    except Exception as e:
        print(f"Error in retrieve_documents: {e}")
        raise Exception("Document retrieval failed")


async def generate_answer(query: str, context: str) -> str:
    """
    Generate answer using LLM.
    FOR NOW: Returns placeholder answer.
    LATER: Will call HF Inference API with real context.
    """
    
    try:
        # PLACEHOLDER: Return a medical-style answer
        # This will be replaced in Part 3 with real HF API calls
        
        answer = f"""
Based on current medical research, here's what you need to know:

## Understanding Type 2 Diabetes

Type 2 Diabetes is a progressive metabolic disorder characterized by insulin resistance 
and eventual beta-cell dysfunction. It affects how your body uses blood glucose (sugar).

## Key Management Strategies

### 1. Lifestyle Modifications
- **Diet**: Focus on balanced macronutrients, reduced refined carbohydrates
- **Exercise**: 150+ minutes moderate-intensity activity per week
- **Weight**: 5-10% reduction can significantly improve insulin sensitivity

### 2. Pharmacological Treatment
- **Metformin**: First-line agent, improves insulin sensitivity
- **GLP-1 Agonists**: Cardiovascular and renal protective effects
- **SGLT2 Inhibitors**: Cardiac and kidney benefits
- **Sulfonylureas**: Direct insulin secretion stimulation

### 3. Monitoring
- **HbA1c**: Quarterly monitoring (target <7% for most patients)
- **Fasting glucose**: Daily monitoring if on insulin
- **Annual screening**: For complications (retinopathy, nephropathy, neuropathy)

## Prevention of Complications

Early detection and aggressive management can prevent serious complications including:
- Cardiovascular disease (heart attacks, strokes)
- Diabetic nephropathy (kidney disease)
- Diabetic retinopathy (vision loss)
- Diabetic neuropathy (nerve damage)

## Important Note

This information is for educational purposes. Always consult with a healthcare 
professional for personalized medical advice.
        """
        
        return answer.strip()
        
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        raise Exception("Answer generation failed")


async def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Evaluate faithfulness of answer against context.
    FOR NOW: Returns placeholder score.
    LATER: Will implement claim extraction and verification.
    """
    
    try:
        # PLACEHOLDER: Return a realistic faithfulness score
        # This will be replaced in Part 3 with real LLM evaluation
        
        await asyncio.sleep(0.3)  # Simulate processing
        
        # For now, return a good score (we're using placeholder answers)
        return 0.92
        
    except Exception as e:
        print(f"Error in evaluate_faithfulness: {e}")
        return None


async def evaluate_relevancy(answer: str, query: str) -> float:
    """
    Evaluate relevancy of answer to query.
    FOR NOW: Returns placeholder score.
    LATER: Will use similarity metrics.
    """
    
    try:
        # PLACEHOLDER: Return a realistic relevancy score
        # This will be replaced in Part 3 with cosine similarity
        
        await asyncio.sleep(0.3)  # Simulate processing
        
        # For now, return a good score
        return 0.88
        
    except Exception as e:
        print(f"Error in evaluate_relevancy: {e}")
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
