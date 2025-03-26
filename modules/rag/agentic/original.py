"""
Original Agentic RAG Implementation

This module implements the original custom-built agentic RAG approach with iterative 
retrieval, planning, and specialized tools. It uses a procedural, function-based 
approach rather than a graph-based structure.
"""

import logging
import time
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Import shared modules
from ..types import RAGResult
from ..config import (
    MAX_ITERATIONS,
    INITIAL_TOP_K,
    ENABLE_PLANNING,
    ENABLE_SELF_CRITIQUE,
    WEB_SEARCH_ENABLED,
    SYNTHETIC_DATA_ENABLED,
    GOOGLE_CSE_API_KEY,
    GOOGLE_CSE_ENGINE_ID
)
from ..utils import (
    get_content_from_llm_response,
    optimize_context,
    summarize_document
)
from ..prompts import (
    SUFFICIENCY_TEMPLATE,
    ANSWER_TEMPLATE,
    PLANNING_TEMPLATE,
    CRITIQUE_TEMPLATE
)

# Import internal modules
from modules.knowledge_base import get_document_store

def web_search(query: str, num_results: int = 3) -> List[Document]:
    """Perform a web search using Google Custom Search API."""
    # Check if web search is enabled and API keys are available
    if not WEB_SEARCH_ENABLED or not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ENGINE_ID:
        logger.warning("Web search is disabled or API keys are missing")
        return []
    
    try:
        # Import the Google API client library
        from googleapiclient.discovery import build
        
        # Build the service
        service = build("customsearch", "v1", developerKey=GOOGLE_CSE_API_KEY)
        
        # Execute the search
        result = service.cse().list(
            q=query,
            cx=GOOGLE_CSE_ENGINE_ID,
            num=num_results
        ).execute()
        
        # Process the results
        documents = []
        if "items" in result:
            for item in result["items"]:
                # Extract title, snippet, and link
                title = item.get("title", "No title")
                snippet = item.get("snippet", "No description")
                link = item.get("link", "No link")
                
                # Create a document with the search result
                content = f"Title: {title}\nDescription: {snippet}\nURL: {link}"
                
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": link,
                            "title": title,
                            "search_query": query
                        }
                    )
                )
        
        logger.info(f"Web search returned {len(documents)} results")
        return documents
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return []

def generate_synthetic_data(query: str, existing_context: str) -> Optional[Document]:
    """Generate synthetic document to fill knowledge gaps."""
    if not SYNTHETIC_DATA_ENABLED:
        return None
    
    try:
        # Create the prompt with the current context
        prompt = PLANNING_TEMPLATE.format(
            query=query,
            context=existing_context[:5000] if existing_context else "No relevant context found."
        )
        
        # Generate the synthetic content using an LLM
        llm = ChatOpenAI(temperature=0.2)
        synthetic_response = llm.invoke(prompt)
        synthetic_content = get_content_from_llm_response(synthetic_response)
        
        # Create a document with the synthetic content
        return Document(
            page_content=synthetic_content,
            metadata={
                "source": "synthetic_data",
                "query": query,
                "type": "generated_content"
            }
        )
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        return None

def fallback_search(query: str, log_prefix: str, stream_callback: Callable[[str], None], existing_context: str = "") -> List[Document]:
    """Perform fallback searches when initial retrieval fails."""
    fallback_docs = []
    
    # Try web search if enabled
    if WEB_SEARCH_ENABLED and GOOGLE_CSE_API_KEY and GOOGLE_CSE_ENGINE_ID:
        stream_callback(f"{log_prefix}Attempting web search to find more information...\n\n")
        web_docs = web_search(query)
        if web_docs:
            fallback_docs.extend(web_docs)
            stream_callback(f"{log_prefix}Found {len(web_docs)} results from web search.\n\n")
    
    # Try synthetic data generation if enabled
    if SYNTHETIC_DATA_ENABLED and not fallback_docs:
        stream_callback(f"{log_prefix}Generating knowledge to help answer the query...\n\n")
        synthetic_doc = generate_synthetic_data(query, existing_context)
        if synthetic_doc:
            fallback_docs.append(synthetic_doc)
            stream_callback(f"{log_prefix}Generated knowledge document.\n\n")
    
    return fallback_docs

def analyze_context_sufficiency(query: str, context: str) -> Dict[str, Any]:
    """Analyze if the retrieved context is sufficient to answer the query."""
    try:
        # Create the prompt with the current context
        sufficiency_prompt = SUFFICIENCY_TEMPLATE.format(
            query=query, 
            context=context or "No relevant context found."
        )
        
        # Get the evaluation from the LLM
        llm = ChatOpenAI(temperature=0)
        sufficiency_response = llm.invoke(sufficiency_prompt)
        sufficiency_eval = get_content_from_llm_response(sufficiency_response)
        
        # Check if the context is deemed sufficient
        is_sufficient = "SUFFICIENT" in sufficiency_eval.upper()
        
        # Extract refined query if context is insufficient
        refined_query = None
        if not is_sufficient and "INSUFFICIENT" in sufficiency_eval.upper():
            # Try to extract the refined query - look for text after "INSUFFICIENT"
            import re
            insufficient_pattern = r"INSUFFICIENT\s*(.*)"
            match = re.search(insufficient_pattern, sufficiency_eval, re.IGNORECASE | re.DOTALL)
            if match:
                refined_query = match.group(1).strip().split("\n")[0].strip()
                # Remove any quotation marks around the query
                refined_query = refined_query.strip('"\'')
        
        # Return the analysis results
        return {
            "is_sufficient": is_sufficient,
            "refined_query": refined_query,
            "full_analysis": sufficiency_eval
        }
    except Exception as e:
        logger.error(f"Error analyzing context sufficiency: {e}")
        # In case of error, assume context is sufficient to prevent infinite loops
        return {
            "is_sufficient": True,
            "refined_query": None,
            "full_analysis": f"Error during analysis: {str(e)}"
        }

def retrieve_with_multiple_strategies(query: str, iteration: int, previous_docs: List[Document] = None) -> List[Document]:
    """Retrieve documents using multiple strategies."""
    all_documents = []
    
    # Get vectorstore
    try:
        vectorstore = get_document_store()
        
        # Initial retrieval with higher k for first iteration
        k = INITIAL_TOP_K if iteration == 1 else INITIAL_TOP_K
        
        # Use the retriever to get relevant documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Retrieve documents
        initial_docs = retriever.invoke(query)
        
        # Add retrieved documents to our list
        if initial_docs:
            all_documents.extend(initial_docs)
            logger.info(f"Retrieved {len(initial_docs)} documents from vector store")
        
        # If we didn't get any documents and web search is enabled, try it
        if not all_documents and WEB_SEARCH_ENABLED and GOOGLE_CSE_API_KEY:
            web_docs = web_search(query)
            if web_docs:
                all_documents.extend(web_docs)
                logger.info(f"Retrieved {len(web_docs)} documents from web search")
        
        # If we still don't have documents and synthetic data is enabled, generate some
        if not all_documents and SYNTHETIC_DATA_ENABLED:
            synthetic_doc = generate_synthetic_data(query, "")
            if synthetic_doc:
                all_documents.append(synthetic_doc)
                logger.info("Generated synthetic document as fallback")
        
        return all_documents
    except Exception as e:
        logger.error(f"Error in multi-strategy retrieval: {e}")
        return []

def create_query_plan(query: str) -> Dict[str, Any]:
    """Create a plan for complex queries by breaking them down into sub-questions."""
    if not ENABLE_PLANNING:
        return {"is_complex": False, "sub_queries": []}
    
    try:
        # Create the prompt
        prompt = PLANNING_TEMPLATE.format(query=query)
        
        # Get the planning analysis from the LLM
        llm = ChatOpenAI(temperature=0)
        planning_response = llm.invoke(prompt)
        planning_content = get_content_from_llm_response(planning_response)
        
        # Check if the query is considered complex
        is_complex = "COMPLEX" in planning_content.upper()
        
        # Extract sub-questions if complex
        sub_queries = []
        if is_complex:
            # Look for numbered lists of sub-questions
            import re
            lines = planning_content.split('\n')
            for line in lines:
                # Skip lines until we find the COMPLEX marker
                if "COMPLEX" in line.upper():
                    continue
                    
                # Check for numbered list items
                match = re.match(r'^\s*\d+\.\s*(.+)$', line)
                if match:
                    sub_question = match.group(1).strip()
                    if sub_question:
                        sub_queries.append(sub_question)
        
        # Return the planning results
        return {
            "is_complex": is_complex,
            "sub_queries": sub_queries,
            "full_analysis": planning_content
        }
    except Exception as e:
        logger.error(f"Error in query planning: {e}")
        return {"is_complex": False, "sub_queries": [], "full_analysis": f"Error: {str(e)}"}

def perform_self_critique(answer: str, query: str, context: str) -> str:
    """Perform self-critique on the generated answer to improve quality."""
    if not ENABLE_SELF_CRITIQUE:
        return answer
        
    try:
        # Get the critique from the LLM
        critique_model = ChatOpenAI(temperature=0)
        critique_response = critique_model.invoke(CRITIQUE_TEMPLATE.format(
            query=query,
            answer=answer,
            context=context[:6000] if len(context) > 6000 else context
        ))
        critique_content = get_content_from_llm_response(critique_response)
        
        # Extract the improved answer from the critique
        # Look for common patterns in the critique response
        improved_answer = critique_content
        
        # Try to extract just the revised answer part
        patterns = [
            r"Revised Answer:\s*(.*)",
            r"Improved Answer:\s*(.*)",
            r"Rewritten Answer:\s*(.*)",
            r"Here's the revised answer:\s*(.*)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, critique_content, re.DOTALL | re.IGNORECASE)
            if match:
                improved_answer = match.group(1).strip()
                break
        
        return improved_answer if improved_answer else answer
    except Exception as e:
        logger.error(f"Error in self-critique: {e}")
        return answer  # Return original answer if critique fails

def run_agentic_rag(query: str, stream_callback: Callable[[str], None]) -> RAGResult:
    """Run the agentic RAG pipeline with iterative retrieval."""
    logger.info(f"Starting agentic RAG for query: '{query}'")
    
    # Create timestamp for logging prefix
    log_prefix = "[Agent] "
    start_time = time.time()
    
    # Track iteration number
    current_iteration = 1
    
    # Track retrieved documents across iterations
    all_documents = []
    
    # Initialize a flag to track if we should continue
    continue_retrieval = True
    
    # Wrap the callback in a streaming message
    stream_callback(f"{log_prefix}Starting retrieval process for query: '{query}'\n\n")
    
    try:
        # PHASE 1: PLANNING - For complex queries, break down into sub-questions
        if ENABLE_PLANNING:
            stream_callback(f"{log_prefix}Analyzing query complexity and creating retrieval plan...\n\n")
            query_plan = create_query_plan(query)
            
            if query_plan["is_complex"]:
                stream_callback(f"{log_prefix}Query identified as complex. Breaking down into sub-questions:\n\n")
                for i, subq in enumerate(query_plan["sub_queries"]):
                    stream_callback(f"{log_prefix}Sub-question {i+1}: {subq}\n")
                stream_callback("\n")
                
                # For complex queries, we'll process each sub-question separately
                # and combine the results
                for i, subq in enumerate(query_plan["sub_queries"]):
                    stream_callback(f"{log_prefix}Processing sub-question {i+1}: '{subq}'\n\n")
                    
                    # Retrieve documents for the sub-question
                    subq_docs = retrieve_with_multiple_strategies(subq, 1)
                    
                    # Add only non-duplicate documents
                    existing_content = {doc.page_content for doc in all_documents}
                    new_docs = [doc for doc in subq_docs if doc.page_content not in existing_content]
                    
                    if new_docs:
                        all_documents.extend(new_docs)
                        stream_callback(f"{log_prefix}Added {len(new_docs)} documents for sub-question {i+1}.\n\n")
        
        # PHASE 2: INITIAL RETRIEVAL
        if not all_documents:  # Only do initial retrieval if we haven't already from sub-questions
            initial_documents = retrieve_with_multiple_strategies(query, current_iteration)
            all_documents.extend(initial_documents)
        
        # Use optimized context processing
        current_context = optimize_context(all_documents, query)
        
        stream_callback(f"{log_prefix}Initial retrieval complete. Analyzing if context is sufficient...\n\n")
        
        # PHASE 3: ITERATIVE RETRIEVAL
        while continue_retrieval and current_iteration < MAX_ITERATIONS:
            # Analyze if the current context is sufficient
            sufficiency_analysis = analyze_context_sufficiency(query, current_context)
            
            if sufficiency_analysis["is_sufficient"]:
                stream_callback(f"{log_prefix}Context deemed sufficient to answer the query.\n\n")
                continue_retrieval = False
            else:
                # Increment iteration counter
                current_iteration += 1
                
                # Use the refined query if available, otherwise use the original query
                refined_query = sufficiency_analysis["refined_query"]
                if refined_query:
                    stream_callback(f"{log_prefix}Context insufficient. Refined query: '{refined_query}'\n\n")
                    retrieval_query = refined_query
                else:
                    stream_callback(f"{log_prefix}Context insufficient. Continuing with original query.\n\n")
                    retrieval_query = query
                
                # Retrieve additional documents
                additional_docs = retrieve_with_multiple_strategies(retrieval_query, current_iteration, all_documents)
                
                # Process new documents
                if additional_docs:
                    # Add only non-duplicate documents
                    existing_content = {doc.page_content for doc in all_documents}
                    new_docs = [doc for doc in additional_docs if doc.page_content not in existing_content]
                    
                    if new_docs:
                        all_documents.extend(new_docs)
                        stream_callback(f"{log_prefix}Retrieved {len(new_docs)} additional documents.\n\n")
                        current_context = optimize_context(all_documents, query)
                    else:
                        # Try fallback strategies
                        fallback_docs = fallback_search(retrieval_query, log_prefix, stream_callback, current_context)
                        if fallback_docs:
                            # Add only non-duplicate documents
                            new_fallback_docs = [doc for doc in fallback_docs if doc.page_content not in existing_content]
                            if new_fallback_docs:
                                all_documents.extend(new_fallback_docs)
                                stream_callback(f"{log_prefix}Added {len(new_fallback_docs)} documents from fallback methods.\n\n")
                                current_context = optimize_context(all_documents, query)
                            else:
                                continue_retrieval = False
                        else:
                            continue_retrieval = False
                else:
                    # Try fallback strategies
                    fallback_docs = fallback_search(retrieval_query, log_prefix, stream_callback, current_context)
                    if fallback_docs:
                        # Add only non-duplicate documents
                        existing_content = {doc.page_content for doc in all_documents}
                        new_fallback_docs = [doc for doc in fallback_docs if doc.page_content not in existing_content]
                        if new_fallback_docs:
                            all_documents.extend(new_fallback_docs)
                            stream_callback(f"{log_prefix}Added {len(new_fallback_docs)} documents from fallback methods.\n\n")
                            current_context = optimize_context(all_documents, query)
                        else:
                            continue_retrieval = False
                    else:
                        continue_retrieval = False
        
        # PHASE 4: ANSWER GENERATION
        stream_callback(f"{log_prefix}Generating answer from {len(all_documents)} documents...\n\n")
        
        try:
            # Create the prompt with the retrieved context
            answer_prompt = ANSWER_TEMPLATE.format(
                context=current_context or "No relevant context found to answer this query.",
                query=query
            )
            
            # Generate the answer
            llm = ChatOpenAI(temperature=0)
            answer_response = llm.invoke(answer_prompt)
            initial_answer = get_content_from_llm_response(answer_response)
            
            # Apply self-critique if enabled
            if ENABLE_SELF_CRITIQUE:
                logger.info("Applying self-critique to improve answer")
                final_answer = perform_self_critique(initial_answer, query, current_context)
            else:
                final_answer = initial_answer
            
            # Stream the answer
            stream_callback(final_answer)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Return the result
            return {
                "query": query,
                "answer": final_answer,
                "refined_query": sufficiency_analysis.get("refined_query"),
                "num_documents": len(all_documents),
                "iterations": current_iteration,
                "time_taken": total_time,
                "metadata": {
                    "total_time": total_time,
                    "iterations": current_iteration,
                    "documents_retrieved": len(all_documents)
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "query": query,
                "answer": f"Error generating answer: {str(e)}",
                "refined_query": None,
                "num_documents": len(all_documents),
                "iterations": current_iteration,
                "time_taken": time.time() - start_time,
                "metadata": {
                    "error": True,
                    "error_message": str(e)
                },
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error in agentic RAG: {e}")
        return {
            "query": query,
            "answer": f"Error in RAG processing: {str(e)}",
            "refined_query": None,
            "num_documents": 0,
            "iterations": 0,
            "time_taken": time.time() - start_time,
            "metadata": {
                "error": True,
                "error_message": str(e)
            },
            "error": str(e)
        } 