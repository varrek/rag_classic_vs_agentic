"""
Original Agentic RAG Implementation

This module implements the original custom-built agentic RAG approach with iterative 
retrieval, planning, and specialized tools. It uses a procedural, function-based 
approach rather than a graph-based structure.
"""

import os
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import streamlit as st
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# Local imports
from modules.knowledge_base import get_document_store
from modules.rag.classic import StreamingCallbackHandler, retrieve_relevant_context, format_context

# Configuration parameters
MAX_ITERATIONS = 5
INITIAL_TOP_K = 5
ADDITIONAL_TOP_K = 3
WEB_SEARCH_ENABLED = st.secrets.get("google_search", {}).get("WEB_SEARCH_ENABLED", True)
SYNTHETIC_DATA_ENABLED = True
ENABLE_PLANNING = True
ENABLE_SELF_CRITIQUE = True
ENABLE_ANIMAL_DATA_TOOL = True
MAX_CONTEXT_LENGTH = 10000
MAX_DOCUMENT_LENGTH = 1500

# Google Custom Search API Key and Engine ID from Streamlit secrets
GOOGLE_CSE_API_KEY = st.secrets.get("google_search", {}).get("api_key", "")
GOOGLE_CSE_ENGINE_ID = st.secrets.get("google_search", {}).get("search_engine_id", "")

# Print debug information
if GOOGLE_CSE_API_KEY:
    logger.info("Google Custom Search API Key found in secrets")
else:
    logger.warning("Google Custom Search API Key not found in secrets")

if GOOGLE_CSE_ENGINE_ID:
    logger.info("Google Custom Search Engine ID found in secrets")
else:
    logger.warning("Google Custom Search Engine ID not found in secrets")

# Animal data for specialized knowledge tool
ANIMAL_DATA = {
    "otter": {
        "behavior": "Otters are known for their playful behavior. They often float on their backs, using their chests as 'tables' for cracking open shellfish with rocks. They're one of the few animals that use tools. They're very social animals and live in family groups. Baby otters (pups) cannot swim when born and are taught by their mothers.",
        "diet": "Otters primarily eat fish, crustaceans, and mollusks. Sea otters in particular are known for using rocks to crack open shellfish. They have a high metabolism and need to eat approximately 25% of their body weight daily.",
        "habitat": "Different otter species inhabit various aquatic environments. Sea otters live in coastal marine habitats, river otters in freshwater rivers, streams and lakes, while some species adapt to brackish water environments. They typically prefer areas with clean water and abundant prey.",
        "tools": "Sea otters are one of the few non-primate animals known to use tools. They often use rocks to crack open hard-shelled prey like clams, mussels, and crabs. They may store their favorite rocks in the pouches of loose skin under their forelimbs. This tool use is not taught by mothers but appears to be an innate behavior that develops as they grow."
    },
    "dolphin": {
        "behavior": "Dolphins are highly intelligent marine mammals known for their playful behavior and complex social structures. They communicate using clicks, whistles, and body language. They live in groups called pods and are known to help injured members. They sleep with one brain hemisphere at a time, keeping one eye open.",
        "diet": "Dolphins primarily feed on fish and squid. They use echolocation to find prey, sometimes working in groups to herd fish. Some dolphins use a technique called 'fish whacking' where they strike fish with their tails to stun them before eating.",
        "habitat": "Dolphins inhabit oceans worldwide, from shallow coastal waters to deep offshore environments. Different species have adapted to specific habitats, from warm tropical waters to colder regions. Some dolphin species even live in rivers.",
    },
    "elephant": {
        "behavior": "Elephants are highly social animals with complex emotional lives. They live in matriarchal groups led by the oldest female. They display behaviors suggesting grief, joy, and self-awareness. They communicate through rumbles, some too low for humans to hear. Elephants have excellent memories and can recognize hundreds of individuals.",
        "diet": "Elephants are herbivores, consuming up to 300 pounds of plant matter daily. African elephants primarily browse, eating leaves, bark, and branches from trees and shrubs. Asian elephants graze more, eating grasses, as well as browsing. They spend 12-18 hours per day feeding.",
        "habitat": "African elephants inhabit savannas, forests, deserts, and marshes. Asian elephants prefer forested areas and transitional zones between forests and grasslands. Both species need large territories with access to water and abundant vegetation. Human encroachment has significantly reduced their natural habitats.",
    }
}

def get_content_from_llm_response(response):
    """
    Extract string content from various types of LLM responses.
    Handles both string responses and LangChain AIMessage objects.
    """
    if isinstance(response, str):
        return response
    elif hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "content" in response:
        return response["content"]
    elif isinstance(response, list) and len(response) > 0:
        # For list responses, try to get content from the last item
        last_item = response[-1]
        if hasattr(last_item, "content"):
            return last_item.content
        elif isinstance(last_item, dict) and "content" in last_item:
            return last_item["content"]
    # Return empty string as fallback
    return ""

def get_animal_data(query: str) -> Optional[Document]:
    """
    Retrieve specialized animal data for specific queries.
    
    Args:
        query: The user's query
        
    Returns:
        Document with animal information or None if not applicable
    """
    if not ENABLE_ANIMAL_DATA_TOOL:
        return None
        
    # Convert query to lowercase for matching
    query_lower = query.lower()
    
    # Check for animal-related queries
    for animal, data in ANIMAL_DATA.items():
        if animal in query_lower:
            # Determine which aspect of the animal the query is about
            aspect = None
            
            # Special case for otter tool use
            if animal == "otter" and any(word in query_lower for word in ["tool", "rock", "crack", "shellfish"]):
                aspect = "tools"
            elif "behavior" in query_lower or "do" in query_lower or "act" in query_lower or "play" in query_lower:
                aspect = "behavior"
            elif "eat" in query_lower or "food" in query_lower or "diet" in query_lower:
                aspect = "diet"
            elif "live" in query_lower or "habitat" in query_lower or "environment" in query_lower:
                aspect = "habitat"
                
            # Construct response based on query focus
            content = f"Information about {animal.capitalize()}:\n\n"
            
            if aspect and aspect in data:
                # Just include the specific aspect
                content += f"{aspect.capitalize()}: {data[aspect]}\n\n"
            else:
                # Include all information
                for key, value in data.items():
                    content += f"{key.capitalize()}: {value}\n\n"
            
            return Document(
                page_content=content,
                metadata={
                    "source": "animal_data_tool",
                    "animal": animal,
                    "aspect": aspect
                }
            )
    
    return None

def is_animal_query(query: str) -> bool:
    """
    Check if a query is about animals in our database.
    
    Args:
        query: The user's query
        
    Returns:
        True if query is about a known animal, False otherwise
    """
    if not ENABLE_ANIMAL_DATA_TOOL:
        return False
        
    query_lower = query.lower()
    
    # Check if any of our known animals are in the query
    for animal in ANIMAL_DATA.keys():
        if animal in query_lower:
            return True
            
    return False

def summarize_document(document: Document) -> Document:
    """
    Summarize a long document to make it more concise.
    
    Args:
        document: The document to summarize
        
    Returns:
        Summarized document
    """
    # If document is not too long, return as is
    if len(document.page_content) <= MAX_DOCUMENT_LENGTH:
        return document
    
    # Create a truncated version
    truncated_content = document.page_content[:MAX_DOCUMENT_LENGTH] + "... [content truncated for length]"
    return Document(
        page_content=truncated_content,
        metadata=document.metadata
    )

def optimize_context(documents: List[Document], query: str) -> str:
    """
    Optimize context for the query by prioritizing and processing documents.
    
    Args:
        documents: List of retrieved documents
        query: The user's query
        
    Returns:
        Optimized context string
    """
    if not documents:
        return ""
    
    # 1. Re-rank documents to ensure most relevant are included
    # Use a simple scoring function based on keyword matching
    def simple_score(doc: Document) -> float:
        query_terms = set(query.lower().split())
        content_terms = set(doc.page_content.lower().split())
        
        # Count overlapping terms
        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms) if query_terms else 0
    
    # Score and sort documents by relevance
    scored_docs = [(doc, simple_score(doc)) for doc in documents]
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # 2. Process documents, summarizing if needed
    processed_docs = []
    total_length = 0
    
    for doc, score in sorted_docs:
        # Summarize long documents
        processed_doc = summarize_document(doc)
        
        # Check if adding this document would exceed our context limit
        if total_length + len(processed_doc.page_content) > MAX_CONTEXT_LENGTH:
            # If we have at least one document, stop adding more
            if processed_docs:
                break
            # If this is the first document and it's too long, summarize aggressively
            else:
                # Truncate to fit within the max context length
                truncated_content = processed_doc.page_content[:MAX_CONTEXT_LENGTH - 100] + "... [truncated]"
                processed_doc = Document(
                    page_content=truncated_content,
                    metadata=processed_doc.metadata
                )
        
        # Add the document and update total length
        processed_docs.append(processed_doc)
        total_length += len(processed_doc.page_content)
    
    # 3. Format the context string
    # Include only a limited number of documents, prioritize the highest-scored ones
    formatted_docs = []
    for i, doc in enumerate(processed_docs):
        source = doc.metadata.get("source", "Unknown source")
        source_name = Path(source).name if isinstance(source, str) and "/" in source else source
        formatted_doc = f"Document {i+1} (Source: {source_name}, Relevance: {'High' if i < 2 else 'Medium' if i < 4 else 'Low'}):\n{doc.page_content}"
        formatted_docs.append(formatted_doc)
    
    context = "\n\n".join(formatted_docs)
    
    # Add a header with the number of documents
    header = f"Retrieved {len(formatted_docs)} documents for query: '{query}'\n\n"
    
    return header + context

def web_search(query: str, num_results: int = 3) -> List[Document]:
    """
    Perform a web search using Google Custom Search API.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of Documents containing search results
    """
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

def generate_synthetic_data(query: str, existing_context: str) -> Document:
    """
    Generate synthetic document to fill knowledge gaps.
    
    Args:
        query: The user's query
        existing_context: Existing retrieval context
        
    Returns:
        Document containing synthetic knowledge
    """
    if not SYNTHETIC_DATA_ENABLED:
        return None
    
    # Create a prompt for synthetic data generation
    synthetic_template = """You are a knowledgeable assistant that generates helpful content when information is missing.
A user has asked a question, but the retrieved context doesn't fully answer it.

User question: {query}

Retrieved context: {context}

Generate a synthetic document that contains factual, accurate information that would help answer the question.
Focus on generally known facts rather than speculative or opinion-based content.
Structure your response as a cohesive, informative document about the topic.
Do not mention that this is synthetic or generated content; just provide the information directly.
Write in a neutral, informative style similar to an encyclopedia article.

Generated document:"""
    
    try:
        # Create the prompt
        prompt = synthetic_template.format(
            query=query,
            context=existing_context[:5000] if existing_context else "No relevant context found."
        )
        
        # Generate the synthetic content using an LLM
        llm = ChatOpenAI(temperature=0.2, model="gpt-4o")
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
    """
    Perform fallback searches when initial retrieval fails.
    
    Args:
        query: The user's query
        log_prefix: Prefix for log messages
        stream_callback: Callback function for streaming output
        existing_context: Existing retrieval context
        
    Returns:
        List of documents from fallback sources
    """
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
    
    # Check for animal-related data
    if is_animal_query(query):
        animal_doc = get_animal_data(query)
        if animal_doc:
            fallback_docs.append(animal_doc)
            stream_callback(f"{log_prefix}Added specialized animal data.\n\n")
    
    return fallback_docs

def analyze_context_sufficiency(query: str, context: str) -> Dict[str, Any]:
    """
    Analyze if the retrieved context is sufficient to answer the query.
    
    Args:
        query: The user's query
        context: The retrieved context
        
    Returns:
        Dict with analysis results
    """
    # Create a prompt for context analysis
    sufficiency_template = """You are a helpful assistant evaluating if the retrieved context is sufficient to answer a question.

Question: {query}

Retrieved Context:
{context}

First, analyze what information is needed to answer the question completely and accurately.
Then, evaluate if the retrieved context contains the necessary information.

If the context is sufficient to provide a complete and accurate answer, respond with "SUFFICIENT".
If the context is missing important information, respond with "INSUFFICIENT" followed by a specific refined search query 
that would help retrieve the missing information.

Your evaluation:"""
    
    # Invoke the model to evaluate context sufficiency
    try:
        # Create the prompt with the current context
        sufficiency_prompt = sufficiency_template.format(
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
    """
    Retrieve documents using multiple strategies.
    
    Args:
        query: The query to search for
        iteration: Current iteration number
        previous_docs: Previously retrieved documents
        
    Returns:
        List of retrieved documents
    """
    all_documents = []
    
    # Get vectorstore
    try:
        vectorstore = get_document_store()
        
        # Initial retrieval with higher k for first iteration
        k = INITIAL_TOP_K if iteration == 1 else ADDITIONAL_TOP_K
        
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
        
        # Check for animal data if applicable
        if is_animal_query(query):
            animal_doc = get_animal_data(query)
            if animal_doc:
                all_documents.append(animal_doc)
                logger.info("Added specialized animal data document")
        
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
    """
    Create a plan for complex queries by breaking them down into sub-questions.
    
    Args:
        query: The user's query
        
    Returns:
        Dict with query planning information
    """
    if not ENABLE_PLANNING:
        return {"is_complex": False, "sub_queries": []}
    
    # Create a prompt for query planning
    planning_template = """You're an AI assistant that helps break down complex questions into simpler sub-questions.

Question: {query}

First, analyze if this question is complex (requires multiple pieces of information or multiple steps to answer completely).

If the question is SIMPLE (can be answered in one step with a single piece of information), respond with:
"SIMPLE"

If the question is COMPLEX, break it down into 2-4 sub-questions that would help answer the main question.
Format your response as:
"COMPLEX
1. [First sub-question]
2. [Second sub-question]
..."

Your analysis:"""
    
    try:
        # Create the prompt
        prompt = planning_template.format(query=query)
        
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
    """
    Perform self-critique on the generated answer to improve quality.
    
    Args:
        answer: The answer to critique
        query: The original user query
        context: The retrieval context
    
    Returns:
        Improved answer after self-critique
    """
    if not ENABLE_SELF_CRITIQUE:
        return answer
        
    # Create a critique prompt
    critique_template = """You are a critical evaluator reviewing an answer to a user's question.
Question: {query}

Provided Answer: {answer}

Retrieved Context:
{context}

First, analyze if the answer:
1. Directly addresses the user's question
2. Is factually correct according to the context
3. Avoids making up information not in the context
4. Is complete and thorough
5. Is well-organized and clear

Then, identify any issues or ways to improve the answer.
Finally, rewrite the answer to address these issues. The revised answer should be comprehensive, accurate, and well-structured.

Your critique:"""

    try:
        # Truncate context if needed to fit within token limits
        truncated_context = context
        if len(context) > 6000:  # Approximate value to avoid token limits
            truncated_context = context[:6000] + "... [truncated]"
        
        # Format the prompt
        critique_prompt = critique_template.format(
            query=query,
            answer=answer,
            context=truncated_context
        )
        
        # Get the critique from the LLM
        critique_model = ChatOpenAI(temperature=0)
        critique_response = critique_model.invoke(critique_prompt)
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
        
        for pattern in patterns:
            match = re.search(pattern, critique_content, re.DOTALL | re.IGNORECASE)
            if match:
                improved_answer = match.group(1).strip()
                break
        
        return improved_answer if improved_answer else answer
    except Exception as e:
        logger.error(f"Error in self-critique: {e}")
        return answer  # Return original answer if critique fails

def run_agentic_rag(query: str, stream_callback: Callable[[str], None]) -> None:
    """Run the agentic RAG pipeline with iterative retrieval."""
    logger.info(f"Starting agentic RAG for query: '{query}'")
    
    # Create timestamp for logging prefix
    log_prefix = "[Agent] "
    
    # Track iteration number
    current_iteration = 1
    max_iterations = MAX_ITERATIONS
    
    # Track retrieved documents across iterations
    all_documents = []
    
    # Initialize a flag to track if we should continue
    continue_retrieval = True
    
    # Wrap the callback in a streaming message
    stream_callback(f"{log_prefix}Starting retrieval process for query: '{query}'\n\n")
    
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
    
    # Use optimized context processing instead of simple formatting
    current_context = optimize_context(all_documents, query)
    
    stream_callback(f"{log_prefix}Initial retrieval complete. Analyzing if context is sufficient...\n\n")
    
    # PHASE 3: ITERATIVE RETRIEVAL
    # Continue retrieving documents until we have sufficient context or hit max iterations
    while continue_retrieval and current_iteration < max_iterations:
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
            
            # Check if we got new documents
            if additional_docs:
                # Add only non-duplicate documents
                existing_content = {doc.page_content for doc in all_documents}
                new_docs = [doc for doc in additional_docs if doc.page_content not in existing_content]
                
                if new_docs:
                    all_documents.extend(new_docs)
                    stream_callback(f"{log_prefix}Retrieved {len(new_docs)} additional documents.\n\n")
                    # Update the context with all documents
                    current_context = optimize_context(all_documents, query)
                else:
                    stream_callback(f"{log_prefix}No new documents found.\n\n")
                    
                    # Try fallback strategies (web search, synthetic data)
                    fallback_docs = fallback_search(retrieval_query, log_prefix, stream_callback, current_context)
                    
                    if fallback_docs:
                        # Add only non-duplicate documents
                        existing_content = {doc.page_content for doc in all_documents}
                        new_fallback_docs = [doc for doc in fallback_docs if doc.page_content not in existing_content]
                        
                        if new_fallback_docs:
                            all_documents.extend(new_fallback_docs)
                            stream_callback(f"{log_prefix}Added {len(new_fallback_docs)} documents from fallback methods.\n\n")
                            # Update the context with all documents
                            current_context = optimize_context(all_documents, query)
                        else:
                            # No new documents from fallbacks either, stop retrieving
                            stream_callback(f"{log_prefix}No new documents found from fallback methods. Moving to answer generation.\n\n")
                            continue_retrieval = False
                    else:
                        # No documents from fallbacks, stop retrieving
                        stream_callback(f"{log_prefix}No fallback documents found. Moving to answer generation.\n\n")
                        continue_retrieval = False
            else:
                # No additional documents found, try fallbacks
                fallback_docs = fallback_search(retrieval_query, log_prefix, stream_callback, current_context)
                
                if fallback_docs:
                    # Add only non-duplicate documents
                    existing_content = {doc.page_content for doc in all_documents}
                    new_fallback_docs = [doc for doc in fallback_docs if doc.page_content not in existing_content]
                    
                    if new_fallback_docs:
                        all_documents.extend(new_fallback_docs)
                        stream_callback(f"{log_prefix}Added {len(new_fallback_docs)} documents from fallback methods.\n\n")
                        # Update the context with all documents
                        current_context = optimize_context(all_documents, query)
                    else:
                        # No new documents from fallbacks either, stop retrieving
                        stream_callback(f"{log_prefix}No new documents found from fallback methods. Moving to answer generation.\n\n")
                        continue_retrieval = False
                else:
                    # No documents from fallbacks, stop retrieving
                    stream_callback(f"{log_prefix}No fallback documents found. Moving to answer generation.\n\n")
                    continue_retrieval = False
    
    # PHASE 4: ANSWER GENERATION
    stream_callback(f"{log_prefix}Generating answer from {len(all_documents)} documents...\n\n")
    
    # Create the answer generation prompt
    answer_template = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain the answer, just say you don't know and keep your answer short.
Do not make up information that is not provided in the context.

Context:
{context}

Question: {query}

Provide a comprehensive, accurate, and well-structured answer to the question based on the provided context."""
    
    try:
        # Create the prompt with the retrieved context
        answer_prompt = answer_template.format(
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
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        stream_callback(f"\n\nError generating answer: {str(e)}") 