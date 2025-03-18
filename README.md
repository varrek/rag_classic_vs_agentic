# RAG Comparison: Classic vs Agentic

A Streamlit application for comparing the performance of Classic RAG and Agentic RAG systems on various queries.

## Overview

This application allows users to:
- Run queries through both Classic and Agentic RAG pipelines
- Compare the generated answers side-by-side
- Evaluate responses against ground truth answers (when available)
- Visualize which system performs better for different types of queries

## Features

- **Interactive UI**: Simple, user-friendly interface for entering custom queries or selecting example questions
- **Real-time Results**: See the answers from both systems simultaneously
- **Performance Metrics**: Automatic evaluation of answers when ground truth is available
- **Knowledge Base Integration**: Easy-to-use knowledge base creation and management
- **Example Questions**: Pre-loaded example questions with known answers for testing
- **Google Web Search Integration**: Enhanced information retrieval with Google Custom Search
- **Animal Data Tool**: Specialized capabilities for answering animal-related queries

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git LFS (for handling large data files)
- (Optional) Google Custom Search API key and Search Engine ID

### Data Directory Setup

The application requires a specific data structure:

```
data/
├── *.txt.clean (text files from RAG Mini Wikipedia)
└── qa/
    └── *.txt (question-answer files)
```

To create this structure:

1. Create the necessary directories:
   ```bash
   mkdir -p data/qa
   ```

2. Download text data files from [HuggingFace RAG Mini Wikipedia text data](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/tree/main/raw_data/text_data) and place them in the `data` directory.

3. Download question-answer pairs from [HuggingFace RAG Mini Wikipedia QA data](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia/tree/main/raw_data) (files like `S08_question_answer_pairs.txt`) and place them in the `data/qa` directory.

Quick download using huggingface-cli:
```bash
# Install the huggingface_hub package if you don't have it
pip install huggingface_hub

# Download the text files
python -c "from huggingface_hub import hf_hub_download; import os; [hf_hub_download(repo_id='rag-datasets/rag-mini-wikipedia', filename=f'raw_data/text_data/{file}', repo_type='dataset', local_dir='.', local_dir_use_symlinks=False) for file in ['S08_set1_a1.txt.clean', 'S08_set1_a2.txt.clean', 'S08_set1_a3.txt.clean']]"

# Download the QA files
python -c "from huggingface_hub import hf_hub_download; import os; [hf_hub_download(repo_id='rag-datasets/rag-mini-wikipedia', filename=f'raw_data/{file}', repo_type='dataset', local_dir='.', local_dir_use_symlinks=False) for file in ['S08_question_answer_pairs.txt', 'S09_question_answer_pairs.txt']]"

# Move the files to the correct directories
mkdir -p data/qa
mv raw_data/text_data/*.clean data/
mv raw_data/*.txt data/qa/
```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/varrek/rag_classic_vs_agentic.git
   cd rag_classic_vs_agentic
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file with:
     ```
     [openai]
     api_key = "your-openai-api-key"
     ```

4. (Optional) Set up Google Custom Search:
   - Add to your `.streamlit/secrets.toml`:
     ```
     [google_search]
     api_key = "your-google-api-key"
     search_engine_id = "your-search-engine-id"
     ```

### Creating the Knowledge Base

The knowledge base needs to be created before using the application. This process:
1. Processes all `.clean` files in the `data/` directory
2. Creates embeddings using OpenAI's embedding model
3. Builds a FAISS index for fast retrieval
4. Loads and stores Q&A pairs from the `data/qa/` directory

You can create the knowledge base in two ways:

1. **Using the UI**:
   - Start the application: `streamlit run app.py`
   - Click the "Create Knowledge Base" button when prompted
   - Wait for the process to complete (this may take several minutes)

2. **Using Python**:
   ```bash
   python -c "from knowledge_base import create_knowledge_base; create_knowledge_base()"
   ```

The process will:
- Create a `vectorstore/` directory with the FAISS index
- Process approximately 40-50 documents
- Generate embeddings for each document
- Index all documents for semantic search
- Load example Q&A pairs for testing

### Setting Up Google Custom Search

To enable the web search capability in Agentic RAG:

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable billing for your project

2. **Enable the Custom Search API**:
   - Go to [Google Cloud API Library](https://console.cloud.google.com/apis/library)
   - Search for "Custom Search API" and enable it

3. **Create API Key**:
   - Go to [Credentials](https://console.cloud.google.com/apis/credentials)
   - Click "Create credentials" > "API key"
   - Copy the generated API key

4. **Create a Programmable Search Engine**:
   - Go to [Programmable Search Engine](https://programmablesearchengine.google.com/about/)
   - Click "Create a search engine"
   - Configure your search engine:
     - Select sites to search or search the entire web
     - Enable "Search the entire web" option
   - Get your Search Engine ID from the setup page

5. **Pricing Information** (as of 2023):
   - The Custom Search API has a free tier of 100 search queries per day
   - Beyond the free tier, each 1,000 additional queries costs approximately $5
   - Refer to [Google's pricing page](https://developers.google.com/custom-search/v1/overview#pricing) for current rates
   - Set usage limits in the Google Cloud Console to control costs

### Running the Application

Start the Streamlit server:
```
streamlit run app.py
```

Access the application at http://localhost:8501

## Project Structure

- `app.py`: Main Streamlit application
- `knowledge_base.py`: Knowledge base creation and management
- `rag_classic.py`: Classic RAG implementation
- `rag_agentic.py`: Agentic RAG with advanced features
- `evaluation.py`: Answer evaluation utilities
- `data/`: Source data for the knowledge base
- `vectorstore/`: Storage for vector embeddings (created during setup)

## How It Works

### Classic RAG

The Classic RAG system follows the standard Retrieval-Augmented Generation pattern:
1. User query is processed
2. Relevant documents are retrieved from the knowledge base
3. Retrieved context is sent to the LLM along with the query
4. LLM generates an answer based on the provided context

### Agentic RAG

The Agentic RAG system enhances the classic approach with:
1. **Multi-task Planning**: Breaks complex queries into manageable sub-questions
2. **Web Search Integration**: Falls back to web search when knowledge base is insufficient
3. **Self-critique**: Reviews and refines answers for factual accuracy
4. **Synthetic Data Generation**: Creates plausible information when data is unavailable
5. **Animal Data Tool**: Direct access to structured animal information
6. **Context Sufficiency Analysis**: Determines if available context is adequate for answering

## Usage Examples

1. **Simple queries**: Enter factual questions to test basic retrieval capabilities
2. **Complex queries**: Try multi-part questions to test planning capabilities
3. **Animal queries**: Ask about animal behavior, habitats, or characteristics
4. **Out-of-knowledge-base queries**: Test how systems handle information not in their knowledge base

## Evaluation

Answers are evaluated based on:
- Semantic similarity to ground truth
- Factual accuracy
- Completeness of information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for RAG components
- OpenAI for language models
- Streamlit for the user interface
- HuggingFace for providing the RAG Mini Wikipedia dataset 