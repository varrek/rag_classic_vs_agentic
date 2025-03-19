import os
import streamlit as st
import logging
import time
import random
import hashlib
from typing import Dict, List, Any, Callable, Optional, Tuple

# Local imports
from knowledge_base import (
    check_knowledge_base_exists, 
    create_knowledge_base, 
    get_random_example_questions
)
from rag_classic import query_rag
from rag_agentic import run_agentic_rag
from evaluation import compare_answers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page configuration with a clean slate
st.set_page_config(
    page_title="RAG Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define a clear session state initialization function
def init_session_state():
    """Initialize session state variables with default values."""
    defaults = {
        'user_query': "",
        'ground_truth': "",
        'processing': False,
        'selected_question_index': -1,
        'processed_query': "",
        'classic_answer': "",
        'agentic_answer': "",
        'classic_docs': [],
        'classic_logs': "🔍 Waiting to start Classic RAG processing...\n",
        'agentic_logs': "🔍 Waiting to start Agentic RAG processing...\n",
        'processing_complete': False,
        'app_version': "1.0.2",
    }
    
    # Initialize each variable if not already present
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Initialize session state
init_session_state()

# Check for OpenAI API key
def check_api_key():
    """Check if the OpenAI API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        st.error("OpenAI API key not found. Please enter it below:")
        user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if user_api_key:
            os.environ["OPENAI_API_KEY"] = user_api_key
            st.success("API key set successfully! Reloading app...")
            time.sleep(1)
            st.rerun()
        return False
    return True

# Function to create knowledge base with progress reporting
def initialize_knowledge_base():
    """Create the knowledge base with a nice progress bar."""
    
    # Create a progress placeholder
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    status_text = st.empty()
    
    # Define the callback for progress updates
    def progress_callback(progress: float, status: str):
        # Update progress bar
        progress_bar.progress(progress)
        status_text.text(status)
    
    # Create the knowledge base
    try:
        create_knowledge_base(progress_callback)
        # Complete the progress
        progress_bar.progress(1.0)
        status_text.text("Knowledge base created successfully!")
        time.sleep(1)
        progress_container.empty()
        status_text.empty()
        st.success("Knowledge base created successfully!")
        st.rerun()
    except Exception as e:
        progress_container.empty()
        status_text.empty()
        st.error(f"Error creating knowledge base: {str(e)}")

# Simple log update functions
def update_classic_logs(message: str):
    """Update Classic RAG logs in session state"""
    if 'classic_logs' not in st.session_state:
        st.session_state.classic_logs = ""
    st.session_state.classic_logs += message + "\n"

def update_agentic_logs(token: str):
    """Update Agentic RAG logs and answer in session state"""
    if 'agentic_logs' not in st.session_state:
        st.session_state.agentic_logs = ""
    if 'agentic_answer' not in st.session_state:
        st.session_state.agentic_answer = ""
    st.session_state.agentic_logs += token
    st.session_state.agentic_answer += token

# Process query function - runs the actual RAG processing
def process_query(query: str):
    """Process a query with both RAG methods and store results in session state"""
    if not query:
        return
    
    logger.info(f"Starting to process query: '{query}'")
    
    # Reset answers and status through session state
    st.session_state.classic_answer = ""
    st.session_state.agentic_answer = ""
    st.session_state.classic_docs = []
    st.session_state.classic_logs = "🔍 Starting Classic RAG processing...\n"
    st.session_state.agentic_logs = "🔍 Starting Agentic RAG processing...\n"
    st.session_state.processing_complete = False
    
    logger.info("Reset session state and starting Classic RAG processing")
    
    # We'll use a placeholder to show real-time progress
    status_placeholder = st.empty()
    
    # Process Classic RAG first
    try:
        # Update status
        status_placeholder.info("Processing with Classic RAG...")
        update_classic_logs("Retrieving relevant documents from knowledge base...")
        
        # Process query
        logger.info("Calling query_rag function")
        result = query_rag(query)
        logger.info(f"Classic RAG processing completed with {len(result['documents'])} documents")
        
        # Store results in session state
        st.session_state.classic_answer = result["answer"]
        st.session_state.classic_docs = result["documents"]
        update_classic_logs("✅ Classic RAG processing complete!")
    except Exception as e:
        logger.error(f"Error in Classic RAG: {e}", exc_info=True)
        error_msg = f"Error in Classic RAG: {str(e)}"
        update_classic_logs(f"❌ {error_msg}")
        st.session_state.classic_answer = f"Error occurred: {str(e)}"
    
    logger.info("Starting Agentic RAG processing")
    
    # Update status for Agentic RAG
    status_placeholder.info("Processing with Agentic RAG...")
    
    # Then process Agentic RAG
    try:
        # Process with Agentic RAG - this will stream results via callback
        run_agentic_rag(query, update_agentic_logs)
        logger.info("Agentic RAG processing completed")
    except Exception as e:
        logger.error(f"Error in Agentic RAG: {e}", exc_info=True)
        error_msg = f"Error in Agentic RAG: {str(e)}"
        update_agentic_logs(f"\n\n❌ {error_msg}")
        st.session_state.agentic_answer = f"Error occurred: {str(e)}"
    
    # Clear the status placeholder
    status_placeholder.empty()
    
    # CRITICAL FIX: Mark processing as complete AND disable processing flag
    # This combination enables inputs and shows results
    logger.info("Query processing complete, updating session state")
    st.session_state.processing_complete = True
    st.session_state.processing = False
    
    # Force a rerun to update the UI with the results
    st.rerun()

# Main application
def main():
    """Main app function."""
    
    # Check for API key
    if not check_api_key():
        return
    
    # Check if knowledge base exists
    kb_exists = check_knowledge_base_exists()
    
    if not kb_exists:
        st.warning("Knowledge base not found. Click below to create it.")
        if st.button("Create Knowledge Base"):
            initialize_knowledge_base()
        return
    
    # Title and description
    st.title("🤖 RAG Comparison: Classic vs Agentic")
    
    st.markdown("""
    This application demonstrates the differences between Classic RAG and Agentic RAG approaches.
    
    - **Classic RAG**: Simple retrieval followed by generation using the retrieved context
    - **Agentic RAG**: Iterative retrieval with planning, self-critique, and specialized tools
    
    Try a sample question or enter your own to see how both approaches handle the same query.
    """)
    
    # Create a layout with two columns
    input_col, status_col = st.columns([3, 1])
    
    with input_col:
        # Text input for question
        user_query = st.text_input(
            "Enter your question:", 
            key="input_question",
            disabled=st.session_state.processing
        )
        
        # Submit button
        submit_button = st.button(
            "Submit", 
            type="primary",
            disabled=st.session_state.processing,
            key="submit_button"
        )
        
        # Process query when Submit is clicked
        if submit_button and user_query:
            st.session_state.user_query = user_query
            st.session_state.processed_query = ""
            st.session_state.processing = True
            st.session_state.processing_complete = False
            process_query(user_query)
    
    with status_col:
        # Show processing status
        if st.session_state.processing:
            st.info("⏳ Processing query...")
        elif st.session_state.processing_complete:
            st.success("✅ Processing complete")
        else:
            st.info("Ready to process query")
        
        # Reset button
        if st.button("Reset", disabled=st.session_state.processing):
            # Clear all state and force rerun
            for key in ['processing', 'processing_complete', 'user_query', 
                       'classic_answer', 'agentic_answer', 'classic_logs', 
                       'agentic_logs', 'classic_docs', 'processed_query']:
                if key in st.session_state:
                    st.session_state[key] = "" if isinstance(st.session_state[key], str) else (
                        [] if isinstance(st.session_state[key], list) else False
                    )
            # Rerun to refresh UI
            st.rerun()
    
    # Sample questions section - wrap in expander to save space
    with st.expander("Sample Questions", expanded=True):
        # Create a container for sample questions to prevent UI shifts
        sample_container = st.container()
        
        # Set random seed for reproducibility before getting sample questions
        random.seed(42)
        # Get 10 sample questions without passing the seed parameter
        sample_questions = get_random_example_questions(10)
        
        # Create buttons for sample questions (2 columns of 5 questions)
        with sample_container:
            cols = st.columns(2)
            for i, (question, answer) in enumerate(sample_questions):
                col_idx = i % 2
                with cols[col_idx]:
                    # Generate a hash for this question to use as a stable key
                    question_hash = hashlib.md5(question.encode()).hexdigest()
                    
                    # Set button color and disabled state
                    button_type = "primary" if i == st.session_state.get('selected_question_index', -1) else "secondary"
                    # FIXED: Only disable buttons during active processing, but ENABLE if processing is complete
                    button_disabled = st.session_state.get('processing', False) and not st.session_state.get('processing_complete', False)
                    
                    # Create button with direct callback code instead of using on_click
                    if st.button(
                        f"Q: {question}", 
                        key=f"sample_{question_hash}",
                        disabled=button_disabled,
                        type=button_type,
                        use_container_width=True
                    ):
                        # Only process if not already processing
                        if not st.session_state.processing or st.session_state.processing_complete:
                            # Update session state directly
                            st.session_state.user_query = question
                            st.session_state.ground_truth = answer
                            st.session_state.selected_question_index = i
                            st.session_state.processing = True
                            st.session_state.processed_query = ""  # Reset to force processing
                            st.session_state.processing_complete = False
                            # Process the query
                            process_query(question)
    
    # Results section - only show if processing is complete
    if st.session_state.processing_complete:
        st.header("Results")
        
        # Use tabs for classic and agentic results
        tabs = st.tabs(["Classic RAG", "Agentic RAG", "Comparison"])
        
        with tabs[0]:  # Classic RAG
            st.subheader("Classic RAG Results")
            st.markdown("##### Answer")
            st.markdown(st.session_state.classic_answer or "No answer generated")
            
            st.markdown("##### Log")
            st.text(st.session_state.classic_logs)
            
            if st.session_state.classic_docs:
                st.markdown("##### Retrieved Documents")
                for i, doc in enumerate(st.session_state.classic_docs):
                    with st.expander(f"Document {i+1} - {doc['source']}"):
                        st.markdown(doc["content"])
        
        with tabs[1]:  # Agentic RAG
            st.subheader("Agentic RAG Results")
            st.markdown("##### Answer")
            st.markdown(st.session_state.agentic_answer or "No answer generated")
            
            st.markdown("##### Log")
            st.text(st.session_state.agentic_logs)
        
        with tabs[2]:  # Comparison
            st.subheader("Comparison Results")
            
            # Only compare if both results are available
            classic_answer = st.session_state.classic_answer
            agentic_answer = st.session_state.agentic_answer
            ground_truth = st.session_state.ground_truth
            
            if ground_truth and classic_answer and agentic_answer:
                try:
                    # Calculate scores
                    classic_similarity, classic_explanation = compare_answers(classic_answer, ground_truth)
                    agentic_similarity, agentic_explanation = compare_answers(agentic_answer, ground_truth)
                    
                    # Display scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Classic RAG Score", f"{classic_similarity:.2f}")
                        st.write(f"**Interpretation:** {classic_explanation}")
                    
                    with col2:
                        st.metric("Agentic RAG Score", f"{agentic_similarity:.2f}")
                        st.write(f"**Interpretation:** {agentic_explanation}")
                    
                    # Determine winner
                    if classic_similarity > agentic_similarity:
                        st.success("Classic RAG performed better!")
                    elif agentic_similarity > classic_similarity:
                        st.success("Agentic RAG performed better!")
                    else:
                        st.info("Both approaches performed equally well.")
                except Exception as e:
                    st.error(f"Error comparing answers: {str(e)}")
            else:
                st.info("No ground truth available for comparison, or answers not generated.")
    
    # If we have a query but haven't processed it yet, process it
    if st.session_state.user_query and not st.session_state.processed_query and st.session_state.processing:
        st.session_state.processed_query = st.session_state.user_query
        process_query(st.session_state.user_query)

# Initialize PyTorch settings and run the app
if __name__ == "__main__":
    # Disable file watching to prevent inotify watch limit errors
    os.environ['STREAMLIT_SERVER_WATCH_CHANGES'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Configure PyTorch to reduce memory usage and prevent issues
    try:
        import torch
        torch.set_grad_enabled(False)
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)
    except ImportError:
        pass
    
    # Run the main app
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"Error in application: {str(e)}")
        st.code(f"Exception details: {str(e)}") 