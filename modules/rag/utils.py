"""Shared utilities for RAG implementations"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .config import (
    MAX_DOCUMENT_LENGTH,
    MAX_CONTEXT_LENGTH
)

# Configure logging
logger = logging.getLogger(__name__)

def get_content_from_llm_response(response: Any) -> str:
    """Extract string content from various types of LLM responses."""
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "content" in response:
        return response["content"]
    elif isinstance(response, list) and len(response) > 0:
        last_item = response[-1]
        if hasattr(last_item, "content"):
            return last_item.content
        elif isinstance(last_item, dict) and "content" in last_item:
            return last_item["content"]
    return ""

def summarize_document(doc: Document) -> Document:
    """Summarize a document if it exceeds the maximum length."""
    if len(doc.page_content) <= MAX_DOCUMENT_LENGTH:
        return doc
        
    # Create a summarization prompt
    prompt = f"""
    Please summarize the following text while preserving the key information:
    
    {doc.page_content[:MAX_DOCUMENT_LENGTH * 2]}
    
    Provide a concise summary that captures the main points.
    """
    
    try:
        # Use ChatGPT to generate a summary
        llm = ChatOpenAI(temperature=0)
        summary_response = llm.invoke(prompt)
        summary = get_content_from_llm_response(summary_response)
        
        # Create a new document with the summary
        return Document(
            page_content=summary,
            metadata={
                **doc.metadata,
                "summarized": True,
                "original_length": len(doc.page_content)
            }
        )
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        # Return truncated original if summarization fails
        return Document(
            page_content=doc.page_content[:MAX_DOCUMENT_LENGTH],
            metadata={
                **doc.metadata,
                "truncated": True,
                "original_length": len(doc.page_content)
            }
        )

def optimize_context(documents: List[Document], query: str) -> str:
    """Optimize context for a query by scoring and processing documents."""
    if not documents:
        return ""
    
    # Score documents if not already scored
    scored_docs = []
    for doc in documents:
        # Get existing score or calculate simple relevance score
        if "score" in doc.metadata:
            score = float(doc.metadata["score"])  # Convert to float to ensure comparability
        else:
            # Simple relevance scoring based on term overlap
            query_terms = set(query.lower().split())
            content_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms.intersection(content_terms))
            score = overlap / len(query_terms) if query_terms else 0
        
        scored_docs.append((score, doc))
    
    # Sort by score
    scored_docs = sorted(scored_docs, key=lambda x: float(x[0]), reverse=True)
    
    # Build context string, respecting max length
    context = ""
    current_length = 0
    
    for _, doc in scored_docs:
        # Skip empty documents
        if not doc.page_content.strip():
            continue
            
        # Format document content
        doc_content = f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}\n\n"
        
        # Check if adding this document would exceed max length
        if current_length + len(doc_content) > MAX_CONTEXT_LENGTH:
            # If this is the first document, include a truncated version
            if not context:
                context = doc_content[:MAX_CONTEXT_LENGTH]
                current_length = len(context)
            break
        
        # Add document to context
        context += doc_content
        current_length = len(context)
        
        # Break if we've reached max length
        if current_length >= MAX_CONTEXT_LENGTH:
            break
    
    return context.strip() 