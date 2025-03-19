"""
Agentic RAG Implementation using LangGraph

This module implements an agentic RAG approach using LangGraph for structured
graph-based execution flow. It includes specialized agents for retrieval, 
analysis, and answer generation, connected in a directed graph.
"""

import os
from typing import List, Dict, Any, Optional, Callable, Sequence, Tuple, TypedDict, Annotated, Literal, Union
from pathlib import Path
import logging
import streamlit as st
import re
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain import hub

# LangGraph specific imports
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# Local imports - adjusted for new module structure
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

# Google Custom Search API credentials
GOOGLE_CSE_API_KEY = st.secrets.get("google_search", {}).get("api_key", "")
GOOGLE_CSE_ENGINE_ID = st.secrets.get("google_search", {}).get("search_engine_id", "")

# Print debug information
if GOOGLE_CSE_API_KEY:
    logger.info("Google Custom Search API Key found in secrets")
else:
    logger.warning("Google Custom Search API Key not found in secrets")

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

# Define the AgentState TypedDict
class AgentState(TypedDict):
    """State for the agentic RAG system"""
    messages: List[BaseMessage]
    documents: List[Document]
    current_iteration: int
    is_sufficient: bool
    refined_query: Optional[str]
    final_answer: Optional[str]

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
    """Optimize context for the query by prioritizing and processing documents."""
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

# Create the retriever tool for the graph
def create_retriever_tools():
    """Create retriever tools for the agent to use."""
    vectorstore = get_document_store()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": INITIAL_TOP_K}
    )
    
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and retrieve documents that can help answer the user's question. Use this tool when you need specific information from the knowledge base."
    )
    
    return [retriever_tool]

# Define the graph nodes for LangGraph

def agent(state: AgentState):
    """
    Main agent node that decides what to do based on the current state.
    Can choose to use retrieval tools or answer directly.
    """
    logger.info("Agent node: Analyzing query and deciding next steps")
    
    # Extract the relevant information from state
    messages = state["messages"]
    current_iteration = state["current_iteration"]
    
    # Get the original query from the first message
    if isinstance(messages[0], HumanMessage):
        query = messages[0].content
    elif isinstance(messages[0], tuple) and messages[0][0] == "user":
        query = messages[0][1]
    else:
        query = str(messages[0])
    
    # Log the current iteration
    log_message = f"[Agent] Processing iteration {current_iteration}/{MAX_ITERATIONS} for query: '{query}'"
    logger.info(log_message)
    
    # Create the agent with the retrieval tool
    tools = create_retriever_tools()
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    model = model.bind_tools(tools)
    
    # Decide whether to retrieve or answer directly
    if current_iteration == 1:
        # First iteration - introduce thinking step
        thinking_prompt = f"""You're an AI assistant with RAG capabilities. 
You need to answer the user's question: "{query}"

Before answering:
1. Analyze what information you need to answer this question
2. Consider if you need to retrieve documents or if you already know the answer
3. If you need to retrieve documents, use the retrieve_documents tool
4. Be specific in your retrieval query to get the most relevant information

Think carefully about what information would help answer this query."""

        # Create a system message for the first iteration
        if isinstance(messages[0], tuple):
            # Convert tuple format to HumanMessage
            messages = [HumanMessage(content=messages[0][1])]
        
        # Add the thinking prompt as a system message
        messages = [
            HumanMessage(content=thinking_prompt)
        ] + messages
    
    # Invoke the model with the current messages
    response = model.invoke(messages)
    
    # Return updated state with the agent's response
    return {
        "messages": state["messages"] + [response],
        "current_iteration": current_iteration,
        "documents": state["documents"],
        "is_sufficient": state["is_sufficient"],
        "refined_query": state["refined_query"],
        "final_answer": state["final_answer"]
    }

def retrieve(state: AgentState):
    """
    Retrieve documents based on the agent's decision.
    """
    logger.info("Retrieve node: Getting documents from vector store")
    
    # Extract information from the state
    messages = state["messages"]
    documents = state["documents"]
    current_iteration = state["current_iteration"]
    
    # Get the last message (agent's response)
    last_message = messages[-1]
    content = get_content_from_llm_response(last_message)
    
    # Get the original query from the first message
    if isinstance(messages[0], HumanMessage):
        query = messages[0].content
    elif isinstance(messages[0], tuple) and messages[0][0] == "user":
        query = messages[0][1]
    else:
        query = str(messages[0])
    
    # Determine if there are any tool calls in the last message
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("Tool calls found in last message")
        for tool_call in last_message.tool_calls:
            if tool_call.name == "retrieve_documents":
                # Extract the retrieval query from the tool call
                retrieval_query = None
                try:
                    if isinstance(tool_call.args, dict) and "query" in tool_call.args:
                        retrieval_query = tool_call.args["query"]
                    elif isinstance(tool_call.args, str):
                        # Try to parse as JSON
                        args_dict = json.loads(tool_call.args)
                        if "query" in args_dict:
                            retrieval_query = args_dict["query"]
                except Exception as e:
                    logger.error(f"Error parsing tool call args: {e}")
                
                if retrieval_query:
                    logger.info(f"Using retrieval query from tool call: {retrieval_query}")
                    # Use the retrieval query to get documents
                    vectorstore = get_document_store()
                    retriever = vectorstore.as_retriever(search_kwargs={"k": INITIAL_TOP_K})
                    new_docs = retriever.invoke(retrieval_query)
                    documents.extend(new_docs)
                    logger.info(f"Retrieved {len(new_docs)} documents using query: {retrieval_query}")
                    
                    # Check for animal-specific data
                    if is_animal_query(retrieval_query):
                        animal_doc = get_animal_data(retrieval_query)
                        if animal_doc:
                            documents.append(animal_doc)
                            logger.info("Added specialized animal data document")
                
    # If no documents were retrieved through tool calls, use the original query
    if not documents:
        logger.info("No tool calls found or no documents retrieved, using original query")
        vectorstore = get_document_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": INITIAL_TOP_K})
        documents = retriever.invoke(query)
        logger.info(f"Retrieved {len(documents)} documents using original query")
        
        # Check for animal-specific data for the original query
        if is_animal_query(query):
            animal_doc = get_animal_data(query)
            if animal_doc:
                documents.append(animal_doc)
                logger.info("Added specialized animal data document")
    
    # Return updated state
    return {
        "messages": state["messages"],
        "documents": documents,
        "current_iteration": current_iteration,
        "is_sufficient": state["is_sufficient"],
        "refined_query": state["refined_query"],
        "final_answer": state["final_answer"]
    }

def analyze_sufficiency(state: AgentState):
    """
    Analyze if the retrieved documents are sufficient to answer the query.
    """
    logger.info("Analyze node: Checking if context is sufficient")
    
    # Extract information from the state
    messages = state["messages"]
    documents = state["documents"]
    current_iteration = state["current_iteration"]
    
    # Get the original query from the first message
    if isinstance(messages[0], HumanMessage):
        query = messages[0].content
    elif isinstance(messages[0], tuple) and messages[0][0] == "user":
        query = messages[0][1]
    else:
        query = str(messages[0])
    
    # Format the documents into a context string
    context = optimize_context(documents, query)
    
    # Check if the context is sufficient using LLM
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
        sufficiency_prompt = sufficiency_template.format(query=query, context=context or "No relevant context found.")
        
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
        
        # Log the result
        if is_sufficient:
            logger.info("Context deemed sufficient to answer query")
        else:
            logger.info(f"Context deemed insufficient. Refined query: {refined_query}")
        
        # Increment iteration for the next round
        next_iteration = current_iteration + 1
        
        # Return updated state
        return {
            "messages": state["messages"],
            "documents": state["documents"],
            "current_iteration": next_iteration,
            "is_sufficient": is_sufficient,
            "refined_query": refined_query,
            "final_answer": state["final_answer"]
        }
    except Exception as e:
        logger.error(f"Error analyzing context sufficiency: {e}")
        # In case of error, assume context is sufficient to prevent infinite loops
        return {
            "messages": state["messages"],
            "documents": state["documents"],
            "current_iteration": current_iteration + 1,
            "is_sufficient": True,
            "refined_query": None,
            "final_answer": state["final_answer"]
        }

def generate_answer(state: AgentState):
    """
    Generate the final answer based on the retrieved documents.
    """
    logger.info("Generate node: Creating final answer")
    
    # Extract information from the state
    messages = state["messages"]
    documents = state["documents"]
    
    # Get the original query from the first message
    if isinstance(messages[0], HumanMessage):
        query = messages[0].content
    elif isinstance(messages[0], tuple) and messages[0][0] == "user":
        query = messages[0][1]
    else:
        query = str(messages[0])
    
    # Format the documents into a context string
    context = optimize_context(documents, query)
    
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
            context=context or "No relevant context found to answer this query.",
            query=query
        )
        
        # Generate the answer
        llm = ChatOpenAI(temperature=0)
        answer_response = llm.invoke(answer_prompt)
        initial_answer = get_content_from_llm_response(answer_response)
        
        # Apply self-critique if enabled
        if ENABLE_SELF_CRITIQUE:
            logger.info("Applying self-critique to improve answer")
            final_answer = perform_self_critique(initial_answer, query, context)
        else:
            final_answer = initial_answer
        
        # Create an AI message with the final answer
        answer_message = AIMessage(content=final_answer)
        
        # Return updated state with the answer
        return {
            "messages": state["messages"] + [answer_message],
            "documents": state["documents"],
            "current_iteration": state["current_iteration"],
            "is_sufficient": state["is_sufficient"],
            "refined_query": state["refined_query"],
            "final_answer": final_answer
        }
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        error_message = f"I encountered an error while generating an answer: {str(e)}"
        return {
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "documents": state["documents"],
            "current_iteration": state["current_iteration"],
            "is_sufficient": state["is_sufficient"],
            "refined_query": state["refined_query"],
            "final_answer": error_message
        }

# Define edge routing conditions

def should_continue_retrieval(state: AgentState) -> Literal["retrieve_more", "generate", "end"]:
    """
    Decide whether to continue with retrieval or generate an answer.
    """
    # Check if we've reached the maximum number of iterations
    if state["current_iteration"] >= MAX_ITERATIONS:
        logger.info(f"Reached maximum iterations ({MAX_ITERATIONS}), moving to generate answer")
        return "generate"
    
    # Check if the context is sufficient
    if state["is_sufficient"]:
        logger.info("Context is sufficient, moving to generate answer")
        return "generate"
    
    # Continue with retrieval
    logger.info("Context is insufficient, continuing with retrieval")
    return "retrieve_more"

def check_tools_condition(state: AgentState) -> Literal["tools", "end"]:
    """
    Check if the agent used tools in its response.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message contains tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("Agent used tools, proceeding to retrieval")
        return "tools"
    
    # No tools used, proceed to generate
    logger.info("Agent did not use tools, proceeding to generate answer")
    return "end"

# Build the LangGraph
def build_rag_graph():
    """
    Build and return the LangGraph for the agentic RAG system.
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("analyze", analyze_sufficiency)
    workflow.add_node("generate", generate_answer)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Add conditional edges for agent decision
    workflow.add_conditional_edges(
        "agent",
        check_tools_condition,
        {
            "tools": "retrieve",
            "end": "generate"
        }
    )
    
    # Add edge from retrieve to analyze
    workflow.add_edge("retrieve", "analyze")
    
    # Add conditional edges for analysis decision
    workflow.add_conditional_edges(
        "analyze",
        should_continue_retrieval,
        {
            "retrieve_more": "agent",
            "generate": "generate",
            "end": END
        }
    )
    
    # Add edge from generate to end
    workflow.add_edge("generate", END)
    
    # Compile the graph
    return workflow.compile()

# Wrapper function to run agentic RAG
def run_agentic_rag(query: str, stream_callback: Callable[[str], None]) -> None:
    """
    Run the agentic RAG pipeline with the LangGraph implementation.
    
    Args:
        query: The user's query
        stream_callback: Callback function for streaming output
    """
    logger.info(f"Starting agentic RAG for query: {query}")
    
    # Create a callback handler that will call our stream_callback
    callback_handler = StreamingCallbackHandler(stream_callback)
    
    # Wrap the callback in a streaming message
    stream_callback(f"[Agent] Starting retrieval process for query: '{query}'\n\n")
    
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "documents": [],
        "current_iteration": 1,
        "is_sufficient": False,
        "refined_query": None,
        "final_answer": None
    }
    
    # Build the graph
    try:
        graph = build_rag_graph()
        
        # Run the graph
        for output in graph.stream(initial_state):
            # Extract and log the current node and state
            for node, state in output.items():
                logger.info(f"Completed node: {node}")
                
                # Stream update based on node
                if node == "agent":
                    stream_callback(f"[Agent] Analyzing query and deciding approach...\n\n")
                elif node == "retrieve":
                    if state["documents"]:
                        stream_callback(f"[Agent] Retrieved {len(state['documents'])} documents.\n\n")
                    else:
                        stream_callback(f"[Agent] No documents found in retrieval.\n\n")
                elif node == "analyze":
                    if state["is_sufficient"]:
                        stream_callback(f"[Agent] Context is sufficient. Generating final answer...\n\n")
                    else:
                        refined = state["refined_query"]
                        stream_callback(f"[Agent] Context is insufficient. Need more information.\n")
                        if refined and refined != query:
                            stream_callback(f"[Agent] Refined query: '{refined}'\n\n")
                elif node == "generate":
                    # Check if we have a final answer
                    if state["final_answer"]:
                        # Stream the final answer token by token for a better UI experience
                        stream_callback(state["final_answer"])
                    else:
                        # Fallback if final_answer is not available
                        last_message = state["messages"][-1]
                        if hasattr(last_message, "content"):
                            stream_callback(last_message.content)
        
        logger.info("Agentic RAG processing complete")
    except Exception as e:
        logger.error(f"Error in agentic RAG: {e}")
        stream_callback(f"\n\nError: {str(e)}") 