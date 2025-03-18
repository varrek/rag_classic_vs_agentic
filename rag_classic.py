import os
from typing import Callable, List, Dict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from knowledge_base import get_document_store

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    def __init__(self, callback_fn: Callable[[str], None]):
        self.callback_fn = callback_fn
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token."""
        self.callback_fn(token)

def retrieve_relevant_context(query: str, top_k: int = 5, fetch_k: int = None, filter_threshold: float = None) -> List[Document]:
    """
    Retrieve the most relevant documents for the query with enhanced capabilities.
    
    Args:
        query: The search query
        top_k: Number of documents to retrieve
        fetch_k: Number of documents to initially fetch before filtering (if None, uses top_k * 2)
        filter_threshold: Optional similarity threshold for filtering results
        
    Returns:
        List of retrieved documents
    """
    try:
        vectorstore = get_document_store()
        
        # Default fetch_k to double top_k for more diverse candidate pool
        if fetch_k is None:
            fetch_k = top_k * 2
        
        # First attempt: standard similarity search
        if filter_threshold is not None:
            # Use score threshold if provided
            documents = vectorstore.similarity_search_with_score(
                query, 
                k=fetch_k
            )
            # Filter by threshold and take top_k
            filtered_docs = [(doc, score) for doc, score in documents if score >= filter_threshold]
            # Sort by score and take top k
            sorted_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)[:top_k]
            documents = [doc for doc, _ in sorted_docs]
        else:
            # Standard similarity search
            documents = vectorstore.similarity_search(query, k=top_k)
        
        # If we didn't get enough docs, try a more lenient search
        if len(documents) < top_k:
            additional_docs = vectorstore.similarity_search(
                query, 
                k=top_k - len(documents)
            )
            # Add only non-duplicate documents
            existing_content = {doc.page_content for doc in documents}
            for doc in additional_docs:
                if doc.page_content not in existing_content:
                    documents.append(doc)
                    existing_content.add(doc.page_content)
        
        return documents
        
    except Exception as e:
        print(f"Error in retrieve_relevant_context: {e}")
        # Return an empty list if retrieval fails
        return []

def format_context(documents: List[Document]) -> str:
    """Format the retrieved documents into a context string."""
    context = ""
    for i, doc in enumerate(documents):
        # Include metadata if available
        metadata_str = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'Unknown source')
            article_title = doc.metadata.get('article_title', 'Unknown article')
            metadata_str = f"Source: {source}, Article: {article_title}\n"
        
        context += f"Document {i+1}:\n{metadata_str}{doc.page_content}\n\n"
    return context

def run_classic_rag(query: str, stream_callback: Callable[[str], None]) -> None:
    """Run the classic RAG pipeline with a single retrieval step."""
    # 1. Retrieve relevant documents
    documents = retrieve_relevant_context(query)
    
    # 2. Format context
    context = format_context(documents)
    
    # 3. Create the prompt
    prompt_template = """You are a helpful AI assistant that answers questions based on the provided context. 
If the information needed to answer the question is not in the context, say "I don't have enough information to answer this question."
Do not make up or guess at information that is not provided in the context.

Context:
{context}

Question: {query}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )
    
    # 4. Get answer from LLM with streaming
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        streaming=True,
        callbacks=[StreamingCallbackHandler(stream_callback)]
    )
    
    # 5. Generate answer
    formatted_prompt = prompt.format(context=context, query=query)
    _ = llm.predict(formatted_prompt)  # The handler will take care of streaming
    
    return 