#!/bin/bash

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up RAG environment...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv_latest" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv_latest
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv_latest/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment!${NC}"
    exit 1
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Failed to upgrade pip.${NC}"
fi

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install requirements!${NC}"
    exit 1
fi
echo -e "${GREEN}Requirements installed successfully.${NC}"

# Run the test script
echo -e "${YELLOW}Running tests...${NC}"
python tests/test_rag.py
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}Tests completed successfully!${NC}"
else
    echo -e "${RED}Tests failed with exit code ${TEST_RESULT}${NC}"
fi

echo -e "${YELLOW}You can now run the Streamlit app with:${NC}"
echo -e "${GREEN}source venv_latest/bin/activate && streamlit run app.py${NC}"

exit $TEST_RESULT 