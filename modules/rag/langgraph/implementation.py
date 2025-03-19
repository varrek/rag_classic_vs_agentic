"""
LangGraph RAG Implementation

This module implements a RAG system using LangGraph for more structured control flow.
It organizes the retrieval and generation process as a graph of components.
"""

import os
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence, TypedDict, Annotated, Literal
import operator
import traceback
from pathlib import Path
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import Annotated
    # For LangGraph 0.3.x compatibility
except ImportError as e:
    logger.error(f"Error importing LangGraph components: {e}")
    raise ImportError("LangGraph is required for this implementation")

# Import LangChain components
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain.tools.retriever import create_retriever_tool
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain import hub
except ImportError as e:
    logger.error(f"Error importing LangChain components: {e}")
    raise ImportError("LangChain is required for this implementation")

# Try to import RAGAS for evaluation
try:
    # Import various RAGAS metrics - try to import multiple variants to ensure compatibility
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
    from ragas.metrics import AnswerCorrectness, AnswerSimilarity
    # Also try to import non-LLM versions if available
    try:
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference, 
            NonLLMContextPrecisionWithoutReference
        )
        non_llm_metrics_available = True
    except ImportError:
        non_llm_metrics_available = False
    
    # Import SingleTurnSample for the newer RAGAS API
    from ragas import SingleTurnSample
    ragas_available = True
    logger.info("RAGAS evaluation metrics available")
except ImportError as e:
    logger.warning(f"RAGAS not available, falling back to LLM evaluation: {e}")
    ragas_available = False
    non_llm_metrics_available = False

# Import internal modules
from modules.retrieval import retrieve_documents
from modules.knowledge_base import get_document_store
from langchain_core.documents import Document

# Configuration parameters
MAX_ITERATIONS = 5
INITIAL_TOP_K = 5
ADDITIONAL_TOP_K = 3
WEB_SEARCH_ENABLED = False
SYNTHETIC_DATA_ENABLED = True
ENABLE_PLANNING = True
ENABLE_SELF_CRITIQUE = True
ENABLE_ANIMAL_DATA_TOOL = True
MAX_CONTEXT_LENGTH = 10000
MAX_DOCUMENT_LENGTH = 1500

# Model configuration
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
DEFAULT_RETRIEVAL_K = 5

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

# Define our state schema using TypedDict
class AgentState(TypedDict):
    """State for the agentic RAG system"""
    messages: List[BaseMessage]
    documents: List[Document]
    current_iteration: int
    is_sufficient: bool
    refined_query: Optional[str]
    final_answer: Optional[str]

# Utility functions
def string_reducer(a: str, b: str) -> str:
    """Custom reducer for string values that always takes the latest value."""
    return b

def get_content_from_llm_response(response):
    """Extract string content from various types of LLM responses."""
    # If it's already a string, return it
    if isinstance(response, str):
        return response
    
    # If it's an AIMessage, extract the content
    if hasattr(response, 'content'):
        return response.content
        
    # If it's any other object with string representation, convert to string
    return str(response)

def get_animal_data(query: str) -> Optional[Document]:
    """Retrieve specialized animal data for specific queries."""
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
    """Determine if a query is about an animal we have specialized data for."""
    query_lower = query.lower()
    return any(animal in query_lower for animal in ANIMAL_DATA.keys())

def summarize_document(document: Document) -> Document:
    """Summarize a document if it exceeds the maximum length."""
    if len(document.page_content) <= MAX_DOCUMENT_LENGTH:
        return document
    
    # Create a summary using truncation for now
    # A more sophisticated approach would use an LLM to summarize
    truncated_content = document.page_content[:MAX_DOCUMENT_LENGTH] + "... [content truncated]"
    
    # Create a new document with the truncated content
    summarized_doc = Document(
        page_content=truncated_content,
        metadata={
            **document.metadata,
            "summarized": True,
            "original_length": len(document.page_content)
        }
    )
    
    return summarized_doc

def optimize_context(documents: List[Document], query: str) -> str:
    """Optimize the context by selecting and ordering the most relevant documents."""
    if not documents:
        return ""
    
    # Score documents based on relevance to query
    def simple_score(doc: Document) -> float:
        """Simple relevance scoring function."""
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Count query term occurrences (simple TF scoring)
        query_terms = set(query_lower.split())
        content_terms = content_lower.split()
        
        term_count = sum(1 for term in content_terms if term in query_terms)
        
        # Calculate the percentage of query terms that appear in the document
        query_coverage = len([term for term in query_terms if term in content_lower]) / len(query_terms) if query_terms else 0
        
        # Source boosting
        source_boost = 1.0
        if "source" in doc.metadata:
            if "animal_data_tool" in doc.metadata["source"]:
                source_boost = 2.0  # Boost specialized data
                
        # Final score
        return (0.5 * term_count + 0.5 * query_coverage) * source_boost
    
    # Score and sort documents
    scored_docs = [(doc, simple_score(doc)) for doc in documents]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Build context string with document scores
    context_parts = []
    for i, (doc, score) in enumerate(scored_docs):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content
        context_parts.append(f"[Document {i+1}] Relevance: {score:.2f}, Source: {source}\n{content}\n")
    
    return "\n\n".join(context_parts)

def perform_self_critique(answer: str, query: str, context: str) -> str:
    """Have the LLM perform self-critique and improve its answer."""
    if not ENABLE_SELF_CRITIQUE:
        return answer
    
    try:
        # Use the LLM to perform self-critique and refinement
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
        
        system_prompt = """
        You are a critical evaluator of AI-generated answers. Your task is to analyze the 
        provided answer to a user's question, and improve it by:
        
        1. Checking if it directly addresses the question
        2. Verifying if it uses the context information effectively
        3. Identifying any inaccuracies or unsupported statements
        4. Suggesting improvements to make the answer more helpful and accurate
        
        After your analysis, provide an improved version of the answer that:
        - Is more accurate and better supported by the context
        - Is more directly relevant to the question
        - Has better structure and clarity
        - Cites sources appropriately (if applicable)
        
        Focus your critique on substantial improvements only.
        """
        
        # Truncate context if needed
        context_excerpt = context[:5000] + "..." if len(context) > 5000 else context
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Question: {query}
            
            Context Information (excerpt): 
            {context_excerpt}
            
            Original Answer: 
            {answer}
            
            Please critique and improve this answer.
            """)
        ]
        
        improved_response = llm.invoke(messages)
        improved_answer = get_content_from_llm_response(improved_response)
        
        # Check if the improved answer is significantly different
        # For now, a simple comparison
        if len(improved_answer) > len(answer) * 0.8 and improved_answer != answer:
            logger.info("Self-critique produced a significantly improved answer")
            return improved_answer
        else:
            logger.info("Self-critique did not significantly improve the answer")
            return answer
            
    except Exception as e:
        logger.error(f"Error during self-critique: {e}")
        return answer  # Return original answer if self-critique fails

def create_retriever_tools():
    """Create specialized retriever tools for the agent."""
    try:
        # Get vector store
        vectorstore = get_document_store()
        
        # Create the standard retriever tool
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": INITIAL_TOP_K}
        )
        
        retriever_tool = create_retriever_tool(
            retriever,
            name="knowledge_base",
            description="Search the knowledge base for relevant information"
        )
        
        return [retriever_tool]
    except Exception as e:
        logger.error(f"Error creating retriever tools: {e}")
        return []

# Node functions for LangGraph
def agent(state: AgentState):
    """Agent node that plans the retrieval approach."""
    # Extract the query
    current_messages = state["messages"]
    latest_message = None
    for message in reversed(current_messages):
        if isinstance(message, HumanMessage):
            latest_message = message
            break
    
    if not latest_message:
        return {
            **state,
            "refined_query": None,
            "is_sufficient": False
        }
    
    query = latest_message.content
    logger.info(f"Agent planning for query: {query}")
    
    # Check if animal data tool would be useful
    if is_animal_query(query):
        logger.info("Detected animal query, will use specialized tool")
        # Get animal data
        animal_doc = get_animal_data(query)
        if animal_doc:
            state["documents"].append(animal_doc)
    
    # Create an initial plan using an LLM
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    
    # Create a planning prompt
    planning_prompt = f"""
    You are a research and retrieval agent. Analyze this question:
    
    "{query}"
    
    What information should be retrieved to answer this question comprehensively?
    Identify key search terms, entities, and aspects to focus on.
    
    Your plan should be clear and focused on gathering the most relevant information.
    """
    
    # Generate the retrieval plan
    try:
        planning_response = llm.invoke(planning_prompt)
        planning_output = get_content_from_llm_response(planning_response)
        
        # Determine if we should refine the query
        refine_prompt = f"""
        Based on the original query: "{query}"
        
        And your retrieval plan: 
        {planning_output}
        
        Formulate a more effective search query that will yield better search results.
        If the original query is already optimal, return it unchanged.
        
        Return ONLY the refined query text with no explanation or additional text.
        """
        
        refine_response = llm.invoke(refine_prompt)
        refined_query = get_content_from_llm_response(refine_response).strip()
        
        # Only use the refined query if it's meaningfully different
        if refined_query and refined_query != query and len(refined_query) > 5:
            logger.info(f"Refined query: {refined_query}")
            state["refined_query"] = refined_query
        else:
            logger.info("Using original query for retrieval")
            state["refined_query"] = query
            
    except Exception as e:
        logger.error(f"Error in agent planning: {e}")
        state["refined_query"] = query
    
    return state

def retrieve(state: AgentState):
    """Node for document retrieval."""
    # Get appropriate query
    query = state["refined_query"] or state["messages"][-1].content
    current_iteration = state["current_iteration"]
    
    logger.info(f"Retrieval iteration {current_iteration} for query: {query}")
    
    try:
        # Get vector store
        vectorstore = get_document_store()
        
        # Adjust k based on iteration
        k = ADDITIONAL_TOP_K if current_iteration > 1 else INITIAL_TOP_K
        
        # Get retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Retrieve documents
        new_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(new_docs)} documents")
        
        # Process documents (deduplicate, summarize if needed)
        existing_content = {doc.page_content for doc in state["documents"]}
        unique_new_docs = []
        
        for doc in new_docs:
            if doc.page_content not in existing_content:
                # Summarize if needed
                processed_doc = summarize_document(doc)
                unique_new_docs.append(processed_doc)
                existing_content.add(doc.page_content)
        
        logger.info(f"Adding {len(unique_new_docs)} unique new documents")
        
        # Add to state
        updated_docs = state["documents"] + unique_new_docs
        
        # Return updated state
        return {
            **state,
            "documents": updated_docs,
            "current_iteration": current_iteration + 1
        }
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return state

def analyze_sufficiency(state: AgentState):
    """Node to analyze if retrieved context is sufficient."""
    query = state["refined_query"] or state["messages"][-1].content
    documents = state["documents"]
    current_iteration = state["current_iteration"]
    
    logger.info(f"Analyzing context sufficiency (iteration {current_iteration})")
    
    # If we've reached max iterations, consider context sufficient regardless
    if current_iteration > MAX_ITERATIONS:
        logger.info(f"Reached maximum iterations ({MAX_ITERATIONS}), proceeding with generation")
        return {
            **state,
            "is_sufficient": True
        }
    
    if not documents:
        logger.info("No documents retrieved, context is insufficient")
        return {
            **state,
            "is_sufficient": False
        }
    
    # Format context for analysis
    context = optimize_context(documents, query)
    
    # Use LLM to determine if context is sufficient
    try:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        
        sufficiency_prompt = f"""
        You are an information analyst. Determine if the retrieved information is sufficient to answer this question:
        
        Question: "{query}"
        
        Retrieved Information:
        {context[:5000]}  # Truncate if too long
        
        Is this information sufficient to provide a complete, accurate answer to the question?
        Respond with "SUFFICIENT" if the context contains enough information to answer the question.
        Respond with "INSUFFICIENT" if critical information is missing.
        
        If insufficient, briefly explain what specific information is still needed.
        """
        
        sufficiency_response = llm.invoke(sufficiency_prompt)
        sufficiency_analysis = get_content_from_llm_response(sufficiency_response)
        
        is_sufficient = "SUFFICIENT" in sufficiency_analysis.upper()
        logger.info(f"Context sufficiency analysis: {is_sufficient}")
        
        # If insufficient and we have iterations left, continue retrieval
        if not is_sufficient and current_iteration <= MAX_ITERATIONS:
            # Try to extract what information is still needed
            extract_prompt = f"""
            Based on the analysis:
            {sufficiency_analysis}
            
            Formulate a focused search query to find the specific missing information.
            Return ONLY the search query with no explanation or additional text.
            """
            
            extract_response = llm.invoke(extract_prompt)
            new_query = get_content_from_llm_response(extract_response).strip()
            
            if new_query and new_query != query and len(new_query) > 5:
                logger.info(f"Using focused query for next iteration: {new_query}")
                return {
                    **state,
                    "is_sufficient": False,
                    "refined_query": new_query
                }
        
        return {
            **state,
            "is_sufficient": is_sufficient
        }
    except Exception as e:
        logger.error(f"Error analyzing context sufficiency: {e}")
        # Default to proceed with generation if analysis fails
        return {
            **state,
            "is_sufficient": True
        }

def generate_answer(state: AgentState):
    """Node to generate the final answer."""
    query = state["refined_query"] or state["messages"][-1].content
    documents = state["documents"]
    
    logger.info("Generating answer from retrieved context")
    
    if not documents:
        logger.info("No documents available, generating answer without context")
        return {
            **state,
            "final_answer": "I don't have enough information to answer this question accurately."
        }
    
    try:
        # Format context for answer generation
        context = optimize_context(documents, query)
        
        # Use LLM to generate answer
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
        
        generation_prompt = f"""
        You are a helpful AI assistant that answers questions based on retrieved information.
        
        Question: {query}
        
        Retrieved Information:
        {context}
        
        Answer the question based ONLY on the provided information. 
        If the retrieved information isn't sufficient, acknowledge the limitations in your answer.
        Cite your sources using the document numbers [Document X] where appropriate.
        Structure your answer clearly and concisely.
        """
        
        generation_response = llm.invoke(generation_prompt)
        answer = get_content_from_llm_response(generation_response)
        
        # Perform self-critique and improvement if enabled
        if ENABLE_SELF_CRITIQUE:
            improved_answer = perform_self_critique(answer, query, context)
            answer = improved_answer
        
        logger.info("Answer generation complete")
        
        return {
            **state,
            "final_answer": answer
        }
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            **state,
            "final_answer": f"I encountered an error while generating an answer: {str(e)}"
        }

def should_continue_retrieval(state: AgentState) -> Literal["retrieve_more", "generate", "end"]:
    """Decide whether to retrieve more, generate an answer, or end."""
    if state["final_answer"]:
        return "end"
    
    if state["is_sufficient"]:
        return "generate"
        
    if state["current_iteration"] > MAX_ITERATIONS:
        return "generate"
        
    return "retrieve_more"

def check_tools_condition(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to use tools or end the process."""
    animal_query = is_animal_query(state["messages"][-1].content)
    
    if animal_query and ENABLE_ANIMAL_DATA_TOOL:
        return "tools"
    
    return "end"

def build_rag_graph():
    """Build the RAG graph using LangGraph."""
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("analyze", analyze_sufficiency)
    workflow.add_node("generate", generate_answer)
    
    # Define edges
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "retrieve")
    workflow.add_edge("retrieve", "analyze")
    
    # Conditional edges based on analysis
    workflow.add_conditional_edges(
        "analyze",
        should_continue_retrieval,
        {
            "retrieve_more": "retrieve",
            "generate": "generate",
            "end": END
        }
    )
    
    workflow.add_edge("generate", END)
    
    # Compile the graph
    return workflow.compile()

def run_langgraph_rag(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the LangGraph RAG system with the given query.
    
    Args:
        query: The user query
        config: Optional configuration parameters
        
    Returns:
        Dict containing the answer and metadata
    """
    if config is None:
        config = {}
    
    start_time = time.time()
    
    # Extract stream handler if provided
    stream_handler = config.get("stream_handler")
    if stream_handler and callable(stream_handler):
        # Send initial message to stream handler
        stream_handler(f"[LangGraph] Starting retrieval process for query: '{query}'\n\n")
    else:
        stream_handler = None
    
    try:
        # Create the initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "documents": [],
            "current_iteration": 1,
            "is_sufficient": False,
            "refined_query": None,
            "final_answer": None
        }
        
        # Build and run the graph
        graph = build_rag_graph()
        logger.info(f"Running LangGraph RAG with query: {query}")
        
        # Track iterations and final state
        iterations = 0
        final_state = None
        
        # Stream execution if handler provided
        # Also capture the last state to avoid needing to call get_state()
        for output in graph.stream(initial_state):
            iterations += 1
            # Extract and log the current node and state
            for node, state in output.items():
                logger.info(f"Completed node: {node}")
                # Save the latest state - this will become our final state
                final_state = state
                
                # Stream update based on node if handler exists
                if stream_handler:
                    if node == "agent":
                        stream_handler(f"[Agent] Analyzing query and deciding approach...\n\n")
                    elif node == "retrieve":
                        if state["documents"]:
                            stream_handler(f"[Agent] Retrieved {len(state['documents'])} documents.\n\n")
                        else:
                            stream_handler(f"[Agent] No documents found in retrieval.\n\n")
                    elif node == "analyze":
                        if state["is_sufficient"]:
                            stream_handler(f"[Agent] Context is sufficient. Generating final answer...\n\n")
                        else:
                            refined = state["refined_query"]
                            stream_handler(f"[Agent] Context is insufficient. Need more information.\n")
                            if refined and refined != query:
                                stream_handler(f"[Agent] Refining query to: '{refined}'\n\n")
                    elif node == "generate":
                        if state["final_answer"]:
                            stream_handler(f"[Agent] Final answer generated.\n\n")
        
        # Fallback if we couldn't get a final state from the stream
        if final_state is None:
            logger.warning("Could not get final state from stream, using initial state")
            final_state = initial_state
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Send the final answer to the stream handler if available
        if stream_handler and final_state.get("final_answer"):
            stream_handler("\n\n[LangGraph] Final answer:\n")
            stream_handler(final_state["final_answer"])
        
        # Prepare result
        result = {
            "query": query,
            "answer": final_state.get("final_answer") or "No answer was generated.",
            "refined_query": final_state.get("refined_query"),
            "num_documents": len(final_state.get("documents", [])),
            "iterations": iterations,
            "time_taken": total_time,
            "metadata": {
                "total_time": total_time,
                "iterations": iterations,
                "documents_retrieved": len(final_state.get("documents", []))
            }
        }
        
        logger.info(f"Completed LangGraph RAG in {total_time:.2f} seconds with {iterations} iterations")
        return result
    except Exception as e:
        logger.error(f"Error running LangGraph RAG: {e}", exc_info=True)
        traceback.print_exc()
        
        # Return error result
        return {
            "query": query,
            "answer": f"I'm sorry, an error occurred while processing your question: {str(e)}",
            "error": str(e),
            "metadata": {
                "error": True,
                "error_message": str(e),
                "error_traceback": traceback.format_exc()
            }
        } 