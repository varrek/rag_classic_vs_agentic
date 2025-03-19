import os
from typing import List, Dict, Any, Optional, Callable, Sequence, Tuple, TypedDict, Annotated, Literal, Union
from pathlib import Path
import logging
import streamlit as st
import re
import json
from collections import defaultdict

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

# Local imports
from knowledge_base import get_document_store
from rag_classic import StreamingCallbackHandler, retrieve_relevant_context, format_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration parameters (migrated from rag_agentic.py)
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

# Google Custom Search API credentials (migrated from rag_agentic.py)
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

# Animal data for specialized knowledge tool (migrated from rag_agentic.py)
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

# Utility functions migrated from rag_agentic.py

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
    """Check if a query is specifically about animals in our database."""
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

def summarize_document(document: Document) -> Document:
    """Summarize a document if it's too long to fit in context window."""
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
        logger.error(f"Error summarizing document: {e}")
        # If summarization fails, truncate the document
        return Document(
            page_content=document.page_content[:MAX_DOCUMENT_LENGTH] + "... [truncated]",
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
                logger.error(f"Error with aggressive summarization: {e}")
        
        # Stop if we've added enough documents
        if len(processed_docs) >= len(documents) or total_length >= MAX_CONTEXT_LENGTH:
            break
    
    # 3. Format the optimized context
    return format_context(processed_docs)

def perform_self_critique(answer: str, query: str, context: str) -> str:
    """Critique and improve an answer by checking for errors or missing information."""
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
        logger.error(f"Error performing self-critique: {e}")
        return answer  # Return original if critique fails

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
    Execute retrieval based on tool calls in the agent's response.
    """
    logger.info("Retrieve node: Retrieving documents")
    
    # Get the last message (which should be the agent's response)
    messages = state["messages"]
    last_message = messages[-1]
    
    # Initialize variables
    new_documents = []
    tool_response = None
    
    # Extract query from the tool call
    query = ""
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Extract query from the first tool call
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "retrieve_documents":
                try:
                    args = tool_call["args"]
                    if isinstance(args, str):
                        # Try to parse JSON if it's a string
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    
                    if isinstance(args, dict) and "query" in args:
                        query = args["query"]
                    else:
                        # Fallback to the original query if args parsing fails
                        query = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else str(state["messages"][0])
                except Exception as e:
                    logger.error(f"Error extracting query from tool call: {e}")
                    query = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else str(state["messages"][0])
                break
    
    if not query:
        # Fallback to the original query if extraction fails
        query = state["messages"][0].content if isinstance(state["messages"][0], HumanMessage) else str(state["messages"][0])
    
    # Log the retrieval query
    logger.info(f"Retrieving documents with query: {query}")
    
    # Retrieve documents
    documents = retrieve_relevant_context(query, top_k=INITIAL_TOP_K)
    
    # Check if we have specialized animal data
    if is_animal_query(query):
        animal_doc = get_animal_data(query)
        if animal_doc:
            documents.append(animal_doc)
            logger.info(f"Added specialized animal data: {animal_doc.metadata.get('animal')}")
    
    # Optimize the context
    if documents:
        context = optimize_context(documents, query)
        
        # Create tool response
        tool_response = ToolMessage(
            content=context,
            name="retrieve_documents",
            tool_call_id="retrieval_call"
        )
    else:
        # No documents found
        tool_response = ToolMessage(
            content="No relevant documents found in the knowledge base.",
            name="retrieve_documents",
            tool_call_id="retrieval_call"
        )
    
    # Return updated state with the retrieved documents
    return {
        "messages": state["messages"] + [tool_response] if tool_response else state["messages"],
        "documents": state["documents"] + documents,
        "current_iteration": state["current_iteration"] + 1,
        "is_sufficient": state["is_sufficient"],
        "refined_query": state["refined_query"],
        "final_answer": state["final_answer"]
    }

def analyze_sufficiency(state: AgentState):
    """
    Analyze if the retrieved context is sufficient to answer the query.
    """
    logger.info("Analyze node: Checking if context is sufficient")
    
    # Extract the relevant information
    messages = state["messages"]
    documents = state["documents"]
    
    # Get the original query
    query = messages[0].content if isinstance(messages[0], HumanMessage) else str(messages[0])
    
    # Get the context from documents
    context = optimize_context(documents, query)
    
    # Create a prompt for analyzing context sufficiency
    prompt_template = """You are an AI agent that determines if the retrieved context is sufficient to answer a user's question.
Analyze the context and determine if it contains enough information to provide a complete and accurate answer.

Question: {query}

Retrieved Context:
{context}

Please evaluate if this context is sufficient to answer the question fully.
If it is sufficient, respond with "YES" followed by a brief explanation.
If it is not sufficient, respond with "NO" followed by:
1. What specific information is missing
2. A refined search query that would help find the missing information

Your response (start with YES or NO):"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query", "context"]
    )
    
    # Use LLM to analyze context sufficiency
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Get analysis
    response = llm.invoke(prompt.format(query=query, context=context))
    analysis = get_content_from_llm_response(response)
    
    # Parse the response
    is_sufficient = analysis.strip().upper().startswith("YES")
    
    # Extract refined query if context is insufficient
    refined_query = query  # default to original query
    if not is_sufficient:
        # Try to find a refined search query in the response
        lines = analysis.strip().split("\n")
        for i, line in enumerate(lines):
            # Look for lines with search query information
            if any(term in line.lower() for term in ["search query", "refined query", "query:"]):
                if i + 1 < len(lines):
                    refined_query = lines[i + 1].strip()
                    # Clean up the search query
                    refined_query = re.sub(r'^\d+\.\s*', '', refined_query)  # Remove numbering
                    refined_query = refined_query.strip('"\'')  # Remove quotes
                    break
    
    # Create analysis message
    analysis_message = AIMessage(content=analysis)
    
    # Return updated state with analysis results
    return {
        "messages": state["messages"] + [analysis_message],
        "documents": state["documents"],
        "current_iteration": state["current_iteration"],
        "is_sufficient": is_sufficient,
        "refined_query": refined_query,
        "final_answer": state["final_answer"]
    }

def generate_answer(state: AgentState):
    """
    Generate the final answer based on the retrieved context.
    """
    logger.info("Generate node: Creating final answer")
    
    # Extract relevant information
    messages = state["messages"]
    documents = state["documents"]
    
    # Get the original query
    query = messages[0].content if isinstance(messages[0], HumanMessage) else str(messages[0])
    
    # Get the optimized context
    context = optimize_context(documents, query)
    
    # Create the prompt for answer generation
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
    
    # Generate initial answer
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    formatted_prompt = prompt.format(context=context, query=query)
    response = llm.invoke(formatted_prompt)
    initial_answer = get_content_from_llm_response(response)
    
    # Perform self-critique if enabled
    final_answer = initial_answer
    if ENABLE_SELF_CRITIQUE:
        logger.info("Applying self-critique to improve answer")
        final_answer = perform_self_critique(initial_answer, query, context)
    
    # Create answer message
    answer_message = AIMessage(content=final_answer)
    
    # Return the final state with the answer
    return {
        "messages": state["messages"] + [answer_message],
        "documents": state["documents"],
        "current_iteration": state["current_iteration"],
        "is_sufficient": state["is_sufficient"],
        "refined_query": state["refined_query"],
        "final_answer": final_answer
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
    Check if the agent wants to use tools.
    """
    # Get the last message
    last_message = state["messages"][-1]
    
    # Check if the message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info("Agent decided to use tools")
        return "tools"
    else:
        logger.info("Agent decided not to use tools")
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
        logger.error(f"Error in agentic RAG: {e}", exc_info=True)
        stream_callback(f"\n\n❌ Error in Agentic RAG: {str(e)}")
