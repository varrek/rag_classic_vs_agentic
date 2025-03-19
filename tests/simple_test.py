import streamlit as st
import time

# Initialize session state values
if "processing" not in st.session_state:
    st.session_state.processing = False
if "result" not in st.session_state:
    st.session_state.result = ""
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# Set page configuration
st.set_page_config(
    page_title="Streamlit Test",
    page_icon="🔍",
    layout="wide"
)

# Simple processing function
def process_query(query):
    """Simulate processing a query."""
    # Display processing message
    status = st.empty()
    status.info(f"Processing query: {query}")
    
    # Simulate processing time
    for i in range(5):
        time.sleep(1)
        status.info(f"Processing step {i+1}/5...")
    
    # Clear status message when done
    status.empty()
    
    # Update session state
    st.session_state.result = f"This is the result for: {query}"
    st.session_state.processing_complete = True
    st.session_state.processing = False
    
    # Force a rerun to update UI with the result
    st.rerun()

# Main app
st.title("Streamlit Test App")
st.write("This is a simple test app to debug Streamlit state issues.")

# User input
user_query = st.text_input(
    "Enter a query:", 
    key="query_input",
    disabled=st.session_state.processing
)

# Process button
if st.button("Process", disabled=st.session_state.processing):
    if user_query:
        st.session_state.processing = True
        st.session_state.processing_complete = False
        st.session_state.result = ""
        process_query(user_query)

# Reset button
if st.button("Reset", disabled=st.session_state.processing):
    st.session_state.processing = False
    st.session_state.processing_complete = False
    st.session_state.result = ""
    st.rerun()

# Show status
if st.session_state.processing:
    st.info("⏳ Processing in progress...")
elif st.session_state.processing_complete:
    st.success("✅ Processing complete")

# Show result
if st.session_state.processing_complete and st.session_state.result:
    st.header("Result")
    st.write(st.session_state.result) 