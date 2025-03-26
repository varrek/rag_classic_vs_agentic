"""
LangGraph RAG Implementation

This module implements a RAG system using LangGraph for more structured control flow.
It organizes the retrieval and generation process as a graph of components.
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LangGraph components
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Import LangChain components
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

# Import shared modules
from ..types import AgentState, RAGResult
from ..config import (
    MAX_ITERATIONS,
    INITIAL_TOP_K,
    ENABLE_ANIMAL_DATA_TOOL,
    MODEL_NAME
)
from ..utils import (
    get_content_from_llm_response,
    get_animal_data,
    is_animal_query,
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
    llm = ChatOpenAI(temperature=0.2, model=MODEL_NAME)
    
    # Generate the retrieval plan
    try:
        planning_response = llm.invoke(PLANNING_TEMPLATE.format(query=query))
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
        k = INITIAL_TOP_K
        
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
        llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
        
        sufficiency_response = llm.invoke(SUFFICIENCY_TEMPLATE.format(
            query=query,
            context=context[:5000]  # Truncate if too long
        ))
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
        llm = ChatOpenAI(temperature=0.2, model=MODEL_NAME)
        
        generation_response = llm.invoke(ANSWER_TEMPLATE.format(
            query=query,
            context=context
        ))
        answer = get_content_from_llm_response(generation_response)
        
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

def run_langgraph_rag(query: str, config: Optional[Dict[str, Any]] = None) -> RAGResult:
    """
    Run the LangGraph RAG system with the given query.
    
    Args:
        query: The user query
        config: Optional configuration parameters
        
    Returns:
        RAGResult containing the answer and metadata
    """
    if config is None:
        config = {}
    
    start_time = time.time()
    
    # Extract stream handler if provided
    stream_handler = config.get("stream_handler")
    if stream_handler and callable(stream_handler):
        # Send initial message to stream handler
        stream_handler(f"[LangGraph] Starting retrieval process for query: '{query}'\n\n")
    
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
        for output in graph.stream(initial_state):
            iterations += 1
            # Extract and log the current node and state
            for node, state in output.items():
                logger.info(f"Completed node: {node}")
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
        result: RAGResult = {
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
            },
            "error": None
        }
        
        logger.info(f"Completed LangGraph RAG in {total_time:.2f} seconds with {iterations} iterations")
        return result
    except Exception as e:
        logger.error(f"Error running LangGraph RAG: {e}", exc_info=True)
        
        # Return error result
        return {
            "query": query,
            "answer": f"I'm sorry, an error occurred while processing your question: {str(e)}",
            "refined_query": None,
            "num_documents": 0,
            "iterations": 0,
            "time_taken": time.time() - start_time,
            "metadata": {
                "error": True,
                "error_message": str(e),
                "error_traceback": traceback.format_exc()
            },
            "error": str(e)
        } 