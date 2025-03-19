#!/bin/bash
# Script to restart Streamlit and run the application

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Restarting Streamlit RAG Comparison Application...${NC}"

# Kill existing Streamlit processes
echo -e "${YELLOW}Stopping existing Streamlit processes...${NC}"
pkill -f streamlit
sleep 2

# Check if venv_latest exists
if [ -d "venv_latest" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv_latest/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Creating it...${NC}"
    python3 -m venv venv_latest
    source venv_latest/bin/activate
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}OPENAI_API_KEY environment variable not set.${NC}"
    echo -e "${YELLOW}You will be prompted to enter it in the application.${NC}"
fi

# Run Streamlit with error handling
echo -e "${GREEN}Starting Streamlit application...${NC}"
echo -e "${YELLOW}The application will be available at http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"

# Run with increased inotify limit and disabled file watching
streamlit run app.py

# Check exit status
if [ $? -ne 0 ]; then
    echo -e "${RED}Error running Streamlit. Check logs for details.${NC}"
    echo -e "${YELLOW}You may need to increase the inotify watch limit:${NC}"
    echo -e "${YELLOW}Run: sudo bash increase_inotify_watches.sh${NC}"
    exit 1
fi 