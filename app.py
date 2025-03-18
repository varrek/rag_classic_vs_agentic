import streamlit as st
import os
import random
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple

# Set page config
st.set_page_config(
    page_title="RAG Comparison: Classic vs Agentic",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set OpenAI API key from environment variable or Streamlit secrets
import openai
if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ['OPENAI_API_KEY']
elif 'openai' in st.secrets:
    openai.api_key = st.secrets['openai']['api_key']
else:
    st.warning("Please set the OpenAI API key in the environment variable OPENAI_API_KEY or in the Streamlit secrets.")
    st.stop()

# Import our custom modules
from knowledge_base import (
    check_knowledge_base_exists, 
    create_knowledge_base, 
    get_document_store,
    get_random_example_questions
)
from rag_classic import run_classic_rag
from rag_agentic import run_agentic_rag
from evaluation import compare_answers

# Initialize session state
if 'knowledge_base_created' not in st.session_state:
    st.session_state.knowledge_base_created = check_knowledge_base_exists()
if 'example_questions' not in st.session_state:
    st.session_state.example_questions = []
if 'classic_rag_answer' not in st.session_state:
    st.session_state.classic_rag_answer = ""
if 'agentic_rag_answer' not in st.session_state:
    st.session_state.agentic_rag_answer = ""
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = ""
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'classic_score' not in st.session_state:
    st.session_state.classic_score = None
if 'agentic_score' not in st.session_state:
    st.session_state.agentic_score = None
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def process_query(query: str, ground_truth: str = ""):
    """Process a query through both RAG pipelines and update session state."""
    # Store query details
    st.session_state.current_query = query
    st.session_state.ground_truth = ground_truth
    
    # Process Classic RAG
    classic_answer = ""
    def classic_callback(token: str):
        nonlocal classic_answer
        classic_answer += token
    
    try:
        run_classic_rag(query, classic_callback)
        st.session_state.classic_rag_answer = classic_answer
    except Exception as e:
        st.session_state.classic_rag_answer = f"Error: {str(e)}"
    
    # Process Agentic RAG
    agentic_answer = ""
    def agentic_callback(token: str):
        nonlocal agentic_answer
        agentic_answer += token
    
    try:
        run_agentic_rag(query, agentic_callback)
        st.session_state.agentic_rag_answer = agentic_answer
    except Exception as e:
        st.session_state.agentic_rag_answer = f"Error: {str(e)}"
    
    # Compare answers if ground truth is available
    if ground_truth:
        try:
            st.session_state.classic_score = compare_answers(classic_answer, ground_truth)
            st.session_state.agentic_score = compare_answers(agentic_answer, ground_truth)
        except Exception as e:
            st.error(f"Error comparing answers: {str(e)}")
    
    # Show results
    st.session_state.show_results = True

def main():
    """Main function to run the Streamlit app."""
    st.title("📚 RAG Comparison: Classic vs Agentic")
    
    # Knowledge Base Creation
    if not st.session_state.knowledge_base_created:
        st.warning("Knowledge base not found. Please create it first.")
        if st.button("Create Knowledge Base"):
            with st.spinner("Creating knowledge base..."):
                try:
                    create_knowledge_base(lambda p, s: None)
                    st.session_state.knowledge_base_created = True
                    st.session_state.example_questions = get_random_example_questions(10)
                    st.success("Knowledge base created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create knowledge base: {str(e)}")
        return
    
    # Load random example questions if not already loaded
    if not st.session_state.example_questions:
        st.session_state.example_questions = get_random_example_questions(10)
    
    # Input Section
    st.subheader("Enter your query")
    query = st.text_input("Type your question here:", key="query_input")
    
    # Example Questions Section
    st.subheader("Or select an example question")
    
    # Create two columns for questions
    cols = st.columns(2)
    
    # Display example questions
    for i, (question, answer) in enumerate(st.session_state.example_questions):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"Q: {question}", key=f"q_{i}", use_container_width=True):
                # Process the example question directly
                process_query(question, answer)
                st.rerun()
    
    # Custom query submission
    if st.button("Submit Query", disabled=not query, key="submit", use_container_width=True):
        process_query(query)
        st.rerun()
    
    # Results Section
    if st.session_state.show_results:
        st.header("Results")
        st.subheader(f"Query: {st.session_state.current_query}")
        
        # Split screen for results
        col1, col2 = st.columns(2)
        
        # Determine highlight background
        classic_bg_color = "transparent"
        agentic_bg_color = "transparent"
        
        if st.session_state.classic_score is not None and st.session_state.agentic_score is not None:
            if st.session_state.classic_score > st.session_state.agentic_score:
                classic_bg_color = "#C2E7C8"  # Light green
            elif st.session_state.agentic_score > st.session_state.classic_score:
                agentic_bg_color = "#C2E7C8"  # Light green
        
        # Classic RAG results
        with col1:
            st.subheader("Classic RAG")
            st.markdown(
                f"""<div style="background-color: {classic_bg_color}; padding: 10px; border-radius: 5px;">
                {st.session_state.classic_rag_answer}
                </div>""", 
                unsafe_allow_html=True
            )
            if st.session_state.classic_score is not None:
                st.metric("Similarity Score", f"{st.session_state.classic_score:.2f}")
        
        # Agentic RAG results
        with col2:
            st.subheader("Agentic RAG")
            st.markdown(
                f"""<div style="background-color: {agentic_bg_color}; padding: 10px; border-radius: 5px;">
                {st.session_state.agentic_rag_answer}
                </div>""", 
                unsafe_allow_html=True
            )
            if st.session_state.agentic_score is not None:
                st.metric("Similarity Score", f"{st.session_state.agentic_score:.2f}")
        
        # Ground Truth (if available)
        if st.session_state.ground_truth:
            st.subheader("Ground Truth")
            st.info(st.session_state.ground_truth)
        
        # New query button
        if st.button("New Query", key="new_query"):
            st.session_state.show_results = False
            st.session_state.current_query = ""
            st.rerun()

if __name__ == "__main__":
    main() 