#!/bin/bash

# Set environment variables to prevent PyTorch module scanning issues
export PYTHONPATH=${PYTHONPATH}:$(pwd)
export STREAMLIT_SERVER_WATCH_CHANGES=false
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set PyTorch configuration to minimize resource usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Run with these settings
echo "Starting Streamlit app with optimized settings..."
streamlit run app.py "$@" 