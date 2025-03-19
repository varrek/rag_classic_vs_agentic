import os
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path
import requests
import json
import streamlit as st
import re

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from knowledge_base import get_document_store
from rag_classic import StreamingCallbackHandler, retrieve_relevant_context, format_context

# Maximum number of iterations for the agent loop
MAX_ITERATIONS = 5  # Increased from 3 to 5

# Initial and subsequent retrieval parameters
INITIAL_TOP_K = 5  # Increased from 3 to 5
ADDITIONAL_TOP_K = 3  # Increased from 2 to 3

# Flag to enable web search fallback - can be configured in secrets.toml
WEB_SEARCH_ENABLED = st.secrets.get("google_search", {}).get("WEB_SEARCH_ENABLED", True)

# Flag to enable synthetic data generation
SYNTHETIC_DATA_ENABLED = True

# Flag to enable planning and self-critique steps
ENABLE_PLANNING = True
ENABLE_SELF_CRITIQUE = True

# Flag to enable animal data tool
ENABLE_ANIMAL_DATA_TOOL = True

# Maximum context length (in characters) to manage token limits
MAX_CONTEXT_LENGTH = 10000
MAX_DOCUMENT_LENGTH = 1500

# Google Custom Search API Key and Engine ID from Streamlit secrets
# Update to match the keys in secrets.toml
GOOGLE_CSE_API_KEY = st.secrets.get("google_search", {}).get("api_key", "")
GOOGLE_CSE_ENGINE_ID = st.secrets.get("google_search", {}).get("search_engine_id", "")

# Print statuses for debugging (will show in the Streamlit server logs)
if GOOGLE_CSE_API_KEY:
    print("Google Custom Search API Key found in secrets")
else:
    print("WARNING: Google Custom Search API Key not found in secrets")

if GOOGLE_CSE_ENGINE_ID:
    print("Google Custom Search Engine ID found in secrets")
else:
    print("WARNING: Google Custom Search Engine ID not found in secrets")

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
    
    Args:
        response: LLM response (string or AIMessage)
        
    Returns:
        String content from the response
    """
    # If it's already a string, return it
    if isinstance(response, str):
        return response
    
    # If it's an AIMessage, extract the content
    if hasattr(response, 'content'):
        return response.content
        
    # If it's any other object with string representation, convert to string
    return str(response)

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

def summarize_document(document: Document) -> Document:
    """
    Summarize a document if it's too long to fit in context window.
    
    Args:
        document: The document to potentially summarize
        
    Returns:
        Original document or summarized version
    """
    if len(document.page_content) <= MAX_DOCUMENT_LENGTH:
        return document
    
    # Use GPT to create a concise summary
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    prompt = f"""Please summarize the following text concisely while preserving all key information, facts, numbers and details:
    
{document.page_content}

Summary:"""
    
    try:
        response = llm.invoke(prompt)
        summary = get_content_from_llm_response(response)
        return Document(
            page_content=summary,
            metadata={**document.metadata, "summarized": True}
        )
    except Exception as e:
        print(f"Error summarizing document: {e}")
        # If summarization fails, truncate the document
        return Document(
            page_content=document.page_content[:MAX_DOCUMENT_LENGTH] + "... [truncated]",
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
    
    # Process high-scoring documents first
    for doc, score in sorted_docs:
        # Summarize long documents
        if len(doc.page_content) > MAX_DOCUMENT_LENGTH:
            processed_doc = summarize_document(doc)
        else:
            processed_doc = doc
        
        # Check if adding this document would exceed our context limit
        doc_length = len(processed_doc.page_content)
        if total_length + doc_length <= MAX_CONTEXT_LENGTH:
            processed_docs.append(processed_doc)
            total_length += doc_length
        elif len(processed_docs) < 3:
            # Always include at least 3 docs, even if we have to summarize aggressively
            try:
                # More aggressive summarization for the most important docs
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                response = llm.invoke(f"Summarize this text in 3-4 sentences, focusing on facts relevant to '{query}':\n\n{doc.page_content}")
                concise_summary = get_content_from_llm_response(response)
                
                processed_doc = Document(
                    page_content=concise_summary,
                    metadata={**doc.metadata, "summarized": "aggressive"}
                )
                
                processed_docs.append(processed_doc)
                total_length += len(processed_doc.page_content)
            except Exception as e:
                print(f"Error with aggressive summarization: {e}")
        
        # Stop if we've added enough documents
        if len(processed_docs) >= len(documents) or total_length >= MAX_CONTEXT_LENGTH:
            break
    
    # 3. Format the optimized context
    return format_context(processed_docs)

def web_search(query: str, num_results: int = 3) -> List[Document]:
    """
    Perform a web search using Google Custom Search JSON API as a fallback
    when the knowledge base doesn't have enough information.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of documents from web search
    """
    try:
        # Verify we have required API credentials
        if not GOOGLE_CSE_API_KEY:
            print("Web search not available: Google Custom Search API key not found in secrets.toml (missing 'api_key' in [google_search])")
            return []
            
        if not GOOGLE_CSE_ENGINE_ID:
            print("Web search not available: Google Custom Search Engine ID not found in secrets.toml (missing 'search_engine_id' in [google_search])")
            return []
        
        print(f"Performing web search with API key: {GOOGLE_CSE_API_KEY[:4]}... and engine ID: {GOOGLE_CSE_ENGINE_ID}")
        
        # Clean up and prepare the search query
        # Remove any numbering, phrases like "keywords include", etc.
        clean_query = query
        
        # Remove common prefixes that might be part of query refinements
        prefixes_to_remove = [
            "Search query:", "Refined search query:", 
            "Keywords:", "Specific keywords", 
            "Keywords that might be helpful",
            "include:"
        ]
        
        for prefix in prefixes_to_remove:
            if prefix.lower() in clean_query.lower():
                # Split by the prefix and take what's after it
                parts = clean_query.lower().split(prefix.lower(), 1)
                if len(parts) > 1:
                    clean_query = parts[1].strip()
        
        # Remove numbering (e.g., "1.", "2.", etc.)
        clean_query = re.sub(r'^\d+\.\s*', '', clean_query)
        
        # Remove quotes that might be present
        clean_query = clean_query.strip('"\'')
        
        # Limit query length
        if len(clean_query) > 150:
            clean_query = clean_query[:150]
        
        # Ensure we actually have something to search for
        if not clean_query.strip():
            clean_query = query  # Fallback to original if cleaning removed everything
        
        print(f"Performing web search with clean query: '{clean_query}'")
        
        # Use Google Custom Search JSON API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_CSE_API_KEY,
            "cx": GOOGLE_CSE_ENGINE_ID,
            "q": clean_query,
            "num": min(num_results, 10),  # API allows max 10 results per request
            "safe": "active"
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Web search error: {response.status_code} - {response.text}")
            return []
        
        results = response.json()
        
        # Add debug information to help diagnose API issues
        if "searchInformation" in results:
            print(f"Search information: {results.get('searchInformation')}")
        
        # Extract search results
        documents = []
        if "items" in results:
            for item in results["items"][:num_results]:
                # Extract the snippet, title and link
                content = f"Title: {item.get('title', '')}\n"
                content += f"Snippet: {item.get('snippet', '')}\n"
                content += f"URL: {item.get('link', '')}\n"
                
                # Check if there's a pagemap with more details
                if "pagemap" in item and "metatags" in item["pagemap"]:
                    metatags = item["pagemap"]["metatags"][0]
                    if "og:description" in metatags:
                        content += f"Description: {metatags['og:description']}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "web_search",
                        "url": item.get("link", ""),
                        "title": item.get("title", "")
                    }
                )
                documents.append(doc)
                
            print(f"Web search returned {len(documents)} results")
        else:
            print("Web search returned no items in the response")
            if "error" in results:
                print(f"Error details: {results['error']}")
        
        return documents
        
    except Exception as e:
        print(f"Error performing web search: {e}")
        return []

def generate_synthetic_data(query: str, existing_context: str) -> Document:
    """
    Generate synthetic data when the knowledge base lacks information.
    This simulates having access to a larger knowledge base by generating
    plausible information based on the query and what we already know.
    
    Args:
        query: The user's query
        existing_context: Any context we already have
        
    Returns:
        Document with synthetic information
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)  # Slight temperature for reasonable variation
    
    # First, analyze what information we're missing
    analysis_prompt = f"""Based on the following query and the context we already have, 
determine what specific information is missing to properly answer the query.

Query: {query}

Existing context:
{existing_context}

What specific information is missing that would help answer this query?
Be precise about the facts, data, or explanations needed:"""
    
    try:
        response = llm.invoke(analysis_prompt)
        missing_info_analysis = get_content_from_llm_response(response)
        
        # Now generate plausible information to fill the gaps
        generation_prompt = f"""You are a highly knowledgeable AI that has been asked to provide plausible information 
to supplement an existing knowledge base. Based on your training data (but without making up completely false information),
generate a factual-sounding document that would help answer the query below.

The document should ONLY include information relevant to the query and required missing information. 
Format it like an encyclopedia or textbook entry - factual, clear, and well-structured.

Query: {query}

Missing information needed (based on analysis): 
{missing_info_analysis}

Generate a factual-sounding, helpful document that addresses this information gap:"""
        
        response = llm.invoke(generation_prompt)
        synthetic_content = get_content_from_llm_response(response)
        
        # Add clear labeling that this is synthetic data
        labeled_content = f"""[SYNTHETIC INFORMATION - Generated based on general knowledge, not from specific documents]

{synthetic_content}

[Note: The above information was not found in the knowledge base but was generated to help answer the query.]"""
        
        return Document(
            page_content=labeled_content,
            metadata={
                "source": "synthetic_generation",
                "query": query,
                "synthetic": True
            }
        )
        
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return Document(
            page_content="[Failed to generate synthetic information]",
            metadata={"source": "synthetic_generation", "error": str(e)}
        )

def fallback_search(query: str, log_prefix: str, stream_callback: Callable[[str], None], existing_context: str = "") -> List[Document]:
    """
    Attempt fallback search methods when standard retrieval yields insufficient results.
    
    Args:
        query: The search query
        log_prefix: Prefix for logging messages
        stream_callback: Callback for streaming output
        existing_context: Any context we already have
        
    Returns:
        List of documents from fallback methods
    """
    documents = []
    
    # First check if this is an animal-related query
    if ENABLE_ANIMAL_DATA_TOOL:
        animal_doc = get_animal_data(query)
        if animal_doc:
            stream_callback(f"{log_prefix}Found specialized animal information relevant to your query.\n\n")
            documents.append(animal_doc)
            return documents  # If we have specialized animal data, return it immediately
    
    # Try web search next
    if WEB_SEARCH_ENABLED:
        stream_callback(f"{log_prefix}Knowledge base search insufficient. Attempting web search...\n\n")
        web_docs = web_search(query)
        
        if web_docs:
            stream_callback(f"{log_prefix}Found {len(web_docs)} results from web search.\n\n")
            documents.extend(web_docs)
        else:
            stream_callback(f"{log_prefix}Web search returned no results.\n\n")
    
    # If web search failed or is disabled, try synthetic data generation
    if SYNTHETIC_DATA_ENABLED and (not documents or len(documents) < 2):
        stream_callback(f"{log_prefix}Attempting to generate synthetic information to fill knowledge gaps...\n\n")
        synthetic_doc = generate_synthetic_data(query, existing_context)
        documents.append(synthetic_doc)
        stream_callback(f"{log_prefix}Added synthetic information to supplement available knowledge.\n\n")
    
    return documents

def analyze_context_sufficiency(query: str, context: str) -> Dict[str, Any]:
    """
    Analyze if the current context is sufficient to answer the query.
    Returns a dict with keys:
    - is_sufficient: bool
    - missing_info: str, description of missing information
    - search_query: str, refined search query for next retrieval
    """
    # Create a prompt for the agent to analyze context sufficiency
    prompt_template = """You are an AI agent that determines if the retrieved context is sufficient to answer a user's question.
Analyze the context and determine if it contains enough information to provide a complete and accurate answer.

Question: {query}

Retrieved Context:
{context}

Please evaluate if this context is sufficient to answer the question fully.
If it is sufficient, respond with "YES" followed by a brief explanation.
If it is not sufficient, respond with "NO" followed by:
1. What specific information is missing
2. A refined search query that would help find the missing information (provide ONLY the search query with no prefixes or numbering)
3. 2-3 specific keywords that would be helpful for retrieval (comma separated list, no numbering)

Your response (start with YES or NO):"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"]
    )
    
    # Use the LLM to analyze context sufficiency
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0
    )
    
    # Get agent's analysis
    response = llm.invoke(prompt.format(query=query, context=context))
    response_text = get_content_from_llm_response(response)
    
    # Parse the response
    is_sufficient = response_text.strip().upper().startswith("YES")
    
    result = {
        "is_sufficient": is_sufficient,
        "analysis": response_text
    }
    
    # If insufficient, extract the refined search query
    if not is_sufficient:
        # Try to find a refined search query in the response
        lines = response_text.strip().split("\n")
        search_query = query  # Default to original query
        keywords = []
        
        for i, line in enumerate(lines):
            # Look for lines with search query information
            if any(term in line.lower() for term in ["search query", "refined query", "query:"]):
                if i + 1 < len(lines) and not any(prefix in lines[i+1].lower() for prefix in ["keyword", "specific", "missing"]):
                    search_query = lines[i + 1].strip()
                    # Clean up the search query
                    search_query = re.sub(r'^\d+\.\s*', '', search_query)  # Remove numbering
                    search_query = search_query.strip('"\'')  # Remove quotes
                    break
                    
            # Look for lines with keywords
            if any(term in line.lower() for term in ["keyword", "term", "specific word"]):
                if i + 1 < len(lines):
                    keywords_line = lines[i + 1].strip()
                    # Clean up the keywords
                    keywords_line = re.sub(r'^\d+\.\s*', '', keywords_line)  # Remove numbering
                    keywords = [k.strip() for k in keywords_line.split(',')]
        
        # If we couldn't find a better search query in the response, 
        # combine the original query with some keywords
        if search_query == query and keywords:
            search_query = f"{query} {' '.join(keywords[:2])}"
        
        result["search_query"] = search_query
        result["keywords"] = keywords
    
    return result

def retrieve_with_multiple_strategies(query: str, iteration: int, previous_docs: List[Document] = None) -> List[Document]:
    """
    Use multiple retrieval strategies to get more diverse and comprehensive results.
    
    Args:
        query: The user's query or refined search query
        iteration: Current iteration number
        previous_docs: Documents retrieved in previous iterations
        
    Returns:
        List of retrieved documents
    """
    # Start with standard retrieval
    documents = retrieve_relevant_context(query, top_k=INITIAL_TOP_K if iteration == 1 else ADDITIONAL_TOP_K)
    
    # For later iterations, try alternative strategies
    if iteration > 1 and previous_docs:
        # Strategy 1: Use higher similarity threshold but with more documents
        try:
            vectorstore = get_document_store()
            additional_docs = vectorstore.similarity_search(
                query, 
                k=ADDITIONAL_TOP_K + iteration,  # Gradually increase with iterations
                fetch_k=ADDITIONAL_TOP_K * 3     # Fetch more and filter
            )
            documents.extend(additional_docs)
        except Exception as e:
            print(f"Error in additional similarity search: {e}")
        
        # Strategy 2: Try keyword extraction from previous context
        # Use the existing documents to extract potential keywords
        if len(previous_docs) > 0:
            combined_text = " ".join([doc.page_content for doc in previous_docs])
            # Simple keyword extraction - split by space and take unique words
            keywords = set([word.lower() for word in combined_text.split() if len(word) > 5])
            for keyword in list(keywords)[:5]:  # Use top 5 keywords
                try:
                    keyword_docs = retrieve_relevant_context(f"{query} {keyword}", top_k=2)
                    documents.extend(keyword_docs)
                except Exception as e:
                    print(f"Error in keyword retrieval: {e}")
    
    # Remove duplicates by content
    unique_docs = []
    seen_content = set()
    for doc in documents:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)
    
    return unique_docs

def create_query_plan(query: str) -> Dict[str, Any]:
    """
    Create a structured plan for answering complex queries by breaking them down.
    
    Args:
        query: The user's original query
        
    Returns:
        Dictionary containing query plan information
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    planning_prompt = f"""Analyze the following query and create a structured plan to answer it.
If the query is simple and straightforward, just identify it as such.
If the query is complex or multi-part, break it down into sub-questions or research tasks.

Query: {query}

Your response should follow this format:
1. Query complexity: [Simple/Complex]
2. Main information need: [Core information the user is seeking]
3. Sub-questions (if complex):
   - [Sub-question 1]
   - [Sub-question 2]
   - etc.
4. Knowledge likely needed: [Types of information that would be useful to answer this]
5. Best approach: [Brief description of approach to answering this query]"""

    try:
        response = llm.invoke(planning_prompt)
        planning_result = get_content_from_llm_response(response)
        
        # Parse the planning result to create a structured plan
        lines = planning_result.strip().split('\n')
        plan = {
            "original_query": query,
            "planning_output": planning_result,
            "sub_queries": []
        }
        
        # Extract sub-questions for complex queries
        in_subquestions = False
        for line in lines:
            line = line.strip()
            if "Sub-questions" in line:
                in_subquestions = True
                continue
                
            if in_subquestions and line.startswith("-"):
                subq = line[1:].strip()
                if subq:
                    plan["sub_queries"].append(subq)
            
            elif in_subquestions and line.startswith("4."):
                in_subquestions = False
        
        # If no sub-queries were found but complexity is mentioned as complex,
        # generate some reasonable sub-queries
        if "complexity: Complex" in planning_result.lower() and not plan["sub_queries"]:
            subq_prompt = f"Break down this complex query into 2-3 specific sub-questions: {query}"
            response = llm.invoke(subq_prompt)
            subq_result = get_content_from_llm_response(response)
            for line in subq_result.strip().split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '-')) and len(line) > 5:
                    clean_line = line.lstrip('123.- ')
                    plan["sub_queries"].append(clean_line)
        
        # Determine if the query is complex
        plan["is_complex"] = "complexity: Complex" in planning_result.lower() or len(plan["sub_queries"]) > 0
        
        return plan
    
    except Exception as e:
        print(f"Error creating query plan: {e}")
        # Return a basic plan on failure
        return {
            "original_query": query,
            "is_complex": False,
            "planning_output": "Failed to create plan",
            "sub_queries": []
        }

def perform_self_critique(answer: str, query: str, context: str) -> str:
    """
    Critique and improve an answer by checking for errors or missing information.
    
    Args:
        answer: The generated answer to critique
        query: The original query
        context: The context used to generate the answer
        
    Returns:
        Improved answer after self-critique
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    critique_prompt = f"""Evaluate the following answer to a query. Identify any issues such as:
1. Factual errors or unsupported claims
2. Missing important information from the context
3. Irrelevant information
4. Logical inconsistencies
5. Incomplete response to the query

Query: {query}

Context snippets: 
{context[:2000]}... [context truncated for brevity]

Generated Answer:
{answer}

Your critique:
"""
    
    try:
        response = llm.invoke(critique_prompt)
        critique = get_content_from_llm_response(response)
        
        # Based on the critique, improve the answer
        improvement_prompt = f"""Based on the following critique, improve the answer to the query.
Maintain the same general structure but fix any identified issues.
If the critique suggests the answer is good, you can keep it largely the same.

Query: {query}

Original Answer:
{answer}

Critique:
{critique}

Improved Answer:"""
        
        response = llm.invoke(improvement_prompt)
        improved_answer = get_content_from_llm_response(response)
        return improved_answer
    
    except Exception as e:
        print(f"Error performing self-critique: {e}")
        return answer  # Return original if critique fails

def is_animal_query(query: str) -> bool:
    """
    Check if a query is specifically about animals in our database.
    
    Args:
        query: The user's query
        
    Returns:
        Boolean indicating if this is an animal-related query
    """
    query_lower = query.lower()
    
    # Special case for otter-tool questions
    if "otter" in query_lower and ("tool" in query_lower or "rock" in query_lower or "crack" in query_lower or "shellfish" in query_lower):
        return True
    
    # Look for animal names in our database
    for animal in ANIMAL_DATA.keys():
        if animal in query_lower:
            # Look for specific question patterns
            behavior_patterns = ["do", "behavior", "act", "play", "social", "sleep", "live"]
            diet_patterns = ["eat", "food", "diet", "hunting", "prey", "feed"]
            habitat_patterns = ["habitat", "environment", "live", "home", "found", "water", "land"]
            tool_patterns = ["tool", "use", "rock", "crack", "open"]
            
            for pattern in behavior_patterns + diet_patterns + habitat_patterns + tool_patterns:
                if pattern in query_lower:
                    return True
                    
            # If query contains both animal name and a question, it's likely about the animal
            question_markers = ["?", "what", "how", "why", "where", "when", "is", "are", "can", "do"]
            for marker in question_markers:
                if marker in query_lower:
                    return True
    
    return False

def run_agentic_rag(query: str, stream_callback: Callable[[str], None]) -> None:
    """Run the agentic RAG pipeline with iterative retrieval."""
    # Initial logs
    current_iteration = 1
    all_documents = []
    log_prefix = "[Agent] "
    
    # Stream the agent's thought process to the user
    stream_callback(f"{log_prefix}Starting retrieval process for query: '{query}'\n\n")
    
    # Step 1: Check if this is an animal-related query that can be answered directly
    if ENABLE_ANIMAL_DATA_TOOL and is_animal_query(query):
        stream_callback(f"{log_prefix}Detected animal-related query. Checking specialized animal knowledge...\n\n")
        animal_doc = get_animal_data(query)
        
        if animal_doc:
            stream_callback(f"{log_prefix}Found specialized animal information directly relevant to your query.\n\n")
            all_documents.append(animal_doc)
            # Skip to answer generation with this specialized information
            current_context = animal_doc.page_content
            stream_callback(f"{log_prefix}Using specialized animal information to generate answer...\n\n")
            
            # PHASE 4: ANSWER GENERATION with specialized animal data
            prompt_template = """You are a helpful AI assistant with expertise in animal behavior and biology.
Please answer the question based on the specialized animal information provided.
Use a friendly, informative tone and focus on accuracy.

Specialized Animal Information:
{context}

Question: {query}

Answer:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "query"]
            )
            
            # Generate answer without streaming first (for potential critique)
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0
            )
            
            formatted_prompt = prompt.format(context=current_context, query=query)
            response = llm.invoke(formatted_prompt)
            initial_answer = get_content_from_llm_response(response)
            
            # Self-critique if enabled
            final_answer = initial_answer
            if ENABLE_SELF_CRITIQUE:
                stream_callback(f"{log_prefix}Reviewing answer for accuracy and completeness...\n\n")
                final_answer = perform_self_critique(initial_answer, query, current_context)
            
            # Stream the final answer
            streaming_llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0,
                streaming=True,
                callbacks=[StreamingCallbackHandler(stream_callback)]
            )
            
            repeat_prompt = f"Provide this exact answer to the user's question about animals: {final_answer}"
            _ = streaming_llm.invoke(repeat_prompt)
            
            return
    
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
    # Agent loop - analyze context and potentially retrieve more information
    while current_iteration < MAX_ITERATIONS:
        # Analyze if current context is sufficient
        analysis = analyze_context_sufficiency(query, current_context)
        is_sufficient = analysis["is_sufficient"]
        
        if is_sufficient:
            stream_callback(f"{log_prefix}Context is sufficient. Generating final answer...\n\n")
            break
        
        # Context is insufficient, so do another retrieval
        stream_callback(f"{log_prefix}Context is insufficient. {current_iteration}/{MAX_ITERATIONS} iterations completed.\n\n")
        
        # Use the refined search query for the next retrieval
        refined_query = analysis.get("search_query", query)
        stream_callback(f"{log_prefix}Refined search query: '{refined_query}'\n\n")
        
        # Increment iteration before getting more documents
        current_iteration += 1
        
        # Retrieve more documents with multiple strategies
        additional_documents = retrieve_with_multiple_strategies(
            refined_query, 
            current_iteration,
            all_documents
        )
        
        # Add new documents to the collection, avoiding duplicates by checking content
        existing_content = {doc.page_content for doc in all_documents}
        new_docs = [doc for doc in additional_documents if doc.page_content not in existing_content]
        
        if new_docs:
            all_documents.extend(new_docs)
            # Use optimized context processing
            current_context = optimize_context(all_documents, query)
            stream_callback(f"{log_prefix}Added {len(new_docs)} new documents to context.\n\n")
        else:
            stream_callback(f"{log_prefix}No new information found in knowledge base.\n\n")
            
            # If we're in the last iteration or no new docs, try fallback methods like web search
            if current_iteration >= MAX_ITERATIONS - 1:
                fallback_docs = fallback_search(refined_query, log_prefix, stream_callback, current_context)
                
                # Add fallback docs if found
                new_fallback_docs = [doc for doc in fallback_docs if doc.page_content not in existing_content]
                if new_fallback_docs:
                    all_documents.extend(new_fallback_docs)
                    # Use optimized context processing
                    current_context = optimize_context(all_documents, query)
                    stream_callback(f"{log_prefix}Added {len(new_fallback_docs)} documents from fallback search.\n\n")
                    # Skip to final answer if we found something useful
                    break
                else:
                    stream_callback(f"{log_prefix}No additional information found. Moving to final answer generation.\n\n")
                    break
            
    # If we've reached max iterations but context is still insufficient
    if current_iteration >= MAX_ITERATIONS and not is_sufficient:
        stream_callback(f"{log_prefix}Reached maximum iterations. Using best available context for answer.\n\n")
    
    # PHASE 4: ANSWER GENERATION
    stream_callback(f"{log_prefix}Generating final answer using {len(all_documents)} retrieved documents...\n\n")
    
    # Enhanced answer generation prompt
    prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.
Answer the question based ONLY on the provided context.
If the information needed to answer the question is not in the context, acknowledge what you do know from the context
and then clearly state: "I don't have enough information to fully answer this question."

Context:
{context}

Question: {query}

Answer the question thoroughly and thoughtfully, making sure to address all parts of the question.
If the answer is a simple yes/no, provide that answer first, then explain the reasoning based on the context.

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )
    
    # Get answer from LLM without streaming first (so we can critique it)
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0
    )
    
    # Generate initial answer
    formatted_prompt = prompt.format(context=current_context, query=query)
    response = llm.invoke(formatted_prompt)
    initial_answer = get_content_from_llm_response(response)
    
    # PHASE 5: SELF-CRITIQUE (if enabled)
    final_answer = initial_answer
    if ENABLE_SELF_CRITIQUE:
        stream_callback(f"{log_prefix}Reviewing initial answer for accuracy and completeness...\n\n")
        final_answer = perform_self_critique(initial_answer, query, current_context)
    
    # Stream the final answer
    streaming_llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        streaming=True,
        callbacks=[StreamingCallbackHandler(stream_callback)]
    )
    
    # Now we ask the LLM to just repeat our already-critique answer
    repeat_prompt = f"Provide this exact answer to the user's question: {final_answer}"
    _ = streaming_llm.invoke(repeat_prompt)
    
    return 