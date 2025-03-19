"""
Classic RAG Implementation

This module implements a traditional Retrieval-Augmented Generation (RAG) approach, 
where documents are retrieved based on vector similarity and then provided as context 
to the language model for answer generation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports for chains and prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler

# Define the StreamingCallbackHandler class
class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM output to the UI."""
    
    def __init__(self, callback_function: Callable[[str], None]):
        """Initialize the callback handler with a streaming function."""
        self.callback_function = callback_function
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens as they become available."""
        try:
            self.callback_function(token)
        except Exception as e:
            logger.error(f"Error in callback function: {e}")

def get_retriever(vectorstore, k: int = 4):
    """Get the retriever from the document store."""
    try:
        if vectorstore is None:
            logger.error("Document store is None")
            return None
            
        # Set the search parameters
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        return retriever
    except Exception as e:
        logger.error(f"Error getting retriever: {e}")
        return None

def format_docs(docs: List[Document]) -> str:
    """Format the documents into a string for the prompt."""
    if not docs:
        return "No relevant documents found."
        
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content
        formatted_doc = f"Document {i} (Source: {source}):\n{content}"
        formatted_docs.append(formatted_doc)
    
    return "\n\n".join(formatted_docs)

# Alias format_context to format_docs for backward compatibility 
format_context = format_docs

def retrieve_relevant_context(retriever, query: str, top_k: int = 3) -> List[Document]:
    """Retrieve the most relevant documents for a query."""
    try:
        # Log the query
        logger.info(f"Retrieving documents for query: {query}")
        
        if retriever:
            docs = retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        logger.warning("No retriever available")
        return []
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def create_classical_rag_chain(retriever):
    """Create a classic RAG chain."""
    try:
        # Initialize the language model
        llm = ChatOpenAI(temperature=0)
        
        # Define the prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, just say you don't know and keep your answer short.
Do not make up information that is not provided in the context.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        if retriever is None:
            # Create a chain that just uses the LLM without retrieval
            logger.warning("Retriever is not available. Creating a chain without retrieval.")
            chain = (
                {"context": lambda x: "No context available.", "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        else:
            # Create a chain that combines retrieval and the LLM
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        
        return chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        raise

def query_rag(vectorstore, query: str) -> Dict[str, Any]:
    """Query the RAG system with a question."""
    try:
        # Log the start of processing
        logger.info(f"Processing RAG query: {query}")
        
        # Get the retriever
        retriever = get_retriever(vectorstore, k=3)
        
        # Create the chain
        chain = create_classical_rag_chain(retriever)
        
        # Get the documents from the retriever separately to include them in the output
        docs = []
        
        if retriever:
            try:
                docs = retriever.invoke(query)
                logger.info(f"Retrieved {len(docs)} documents for presentation")
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
        else:
            logger.warning("No retriever available for document retrieval")
        
        # Get the answer from the chain
        try:
            logger.info("Generating answer with LLM")
            answer = chain.invoke(query)
            logger.info("Answer generation complete")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"I encountered an error while trying to answer your question: {str(e)}"
        
        # Format and return the result
        result = {
            "query": query,
            "answer": answer,
            "documents": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown source")
                }
                for doc in docs
            ]
        }
        
        logger.info("RAG query processing complete")
        return result
        
    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        # Return a minimal result with the error
        return {
            "query": query,
            "answer": f"An error occurred: {str(e)}",
            "documents": []
        } 