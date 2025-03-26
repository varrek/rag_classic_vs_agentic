# RAG Comparison Application

This repository contains a Retrieval-Augmented Generation (RAG) comparison application that demonstrates the differences between Classic RAG and Agentic RAG approaches.

## Features

- **Knowledge Base Management**: Create and manage a FAISS vector store of documents
- **Classic RAG Implementation**: Traditional retrieval followed by generation
- **Agentic RAG Implementation**: Advanced RAG with planning, self-critique, and specialized tools
  - **LangGraph Implementation**: Graph-based agentic RAG using LangGraph (Default)
  - **Original Implementation**: Custom-built agentic RAG with iterative retrieval (Legacy)
- **Side-by-Side Comparison**: View and compare results from both approaches
- **Interactive UI**: Streamlit-based interface with real-time processing logs
- **Semantic Evaluation**: Compare answers using semantic similarity metrics
- **Graph Visualization**: Visualize the LangGraph structure of the agentic RAG system
- **Shared Modules**: Centralized configuration, types, prompts, and utilities

## Project Structure

The project follows a modular structure for better organization and maintainability:

```
rags_presentation/
├── app.py                    # Main Streamlit application
├── modules/                  # Core modules directory
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Common utilities and helper functions
│   ├── evaluation.py         # Evaluation utilities
│   ├── retrieval.py          # Document retrieval functionality
│   ├── knowledge_base.py     # Knowledge base and vector store management
│   ├── rag/                  # RAG implementations
│   │   ├── __init__.py       # RAG package initialization
│   │   ├── config.py         # Shared configuration parameters
│   │   ├── types.py          # Shared type definitions
│   │   ├── prompts.py        # Shared prompt templates
│   │   ├── utils.py          # Shared utility functions
│   │   ├── factory.py        # Factory pattern for selecting implementations
│   │   ├── classic.py        # Classic RAG implementation
│   │   ├── agentic/          # Agentic RAG implementations
│   │   │   ├── __init__.py   # Agentic package initialization
│   │   │   └── original.py   # Original agentic implementation (refactored)
│   │   ├── langgraph/        # LangGraph implementations
│   │   │   ├── __init__.py   # LangGraph package initialization
│   │   │   └── implementation.py # LangGraph-based implementation (refactored)
├── scripts/                  # Utility scripts
│   ├── increase_inotify_watches.sh  # Fix Linux inotify limit issues
│   ├── rebuild_kb.py         # Rebuild knowledge base script
│   ├── restart_streamlit.sh  # Restart Streamlit application
│   ├── run_app.sh            # Run the application with optimized settings
│   └── setup_and_test.sh     # Setup environment and run tests
├── tests/                    # Test scripts
│   ├── test_rag.py           # RAG functionality tests
│   ├── test_streamlit_fixes.py # Streamlit compatibility tests
│   ├── simple_test.py        # Simple test application
│   └── check_faiss.py        # FAISS index checking utility
├── docs/                     # Documentation
│   ├── CHANGES.md            # Changelog
│   ├── TROUBLESHOOTING.md    # Troubleshooting guide
│   └── examples/             # Example usage patterns
├── data/                     # Data directory for knowledge base
├── index/                    # Vector store index directory
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT license
└── README.md                 # This file
```

## Dataset

The application uses a custom dataset consisting of question-answer pairs and their corresponding source documents. The data is organized into sets (S08, S09, S10) with the following structure:

```
data/
├── qa/                           # Question-Answer pairs directory
│   ├── S08_question_answer_pairs.txt
│   ├── S09_question_answer_pairs.txt
│   └── S10_question_answer_pairs.txt
├── S08_set[1-4]_topics.txt      # Topic files for each set
├── S09_set[1-5]_topics.txt
├── S10_set[1-6]_topics.txt
└── S[08-10]_set[1-6]_a[1-10].txt.clean  # Source documents

```

### Dataset Structure

1. **Question-Answer Pairs** (`data/qa/`):
   - Contains pairs of questions and their corresponding answers
   - Organized by year (S08, S09, S10)
   - Used for testing and evaluating RAG performance

2. **Topic Files** (`*_topics.txt`):
   - Define the topics covered in each set
   - Help organize and categorize the source documents

3. **Source Documents** (`*.txt.clean`):
   - Clean text files containing the source information
   - Named in the format: `S{year}_set{set_number}_a{article_number}.txt.clean`
   - Used as the knowledge base for RAG retrieval

## RAG Approaches Compared

The application compares three different RAG approaches using this custom dataset:

### Classic RAG
- Simple retrieval followed by generation using the retrieved context
- Single-step process: retrieve documents, then generate an answer
- Deterministic document retrieval based on vector similarity
- Best for straightforward queries with clear context needs

### Agentic RAG (Original Implementation)
- Iterative retrieval with planning and self-critique
- Multi-step process with query refinement
- Dynamic context building based on intermediate results
- Better for complex queries requiring multiple pieces of information
- Includes fallback strategies and synthetic data generation

### Agentic RAG (LangGraph Implementation)
- Graph-based execution flow using LangGraph
- State management with explicit decision nodes
- Iterative refinement with clear state transitions
- Optimal for queries requiring structured reasoning
- Provides better visibility into the reasoning process

Each approach is evaluated on:
- Answer accuracy and completeness
- Context relevance
- Processing time
- Number of retrieval iterations
- Resource usage

## Shared Modules

The application uses several shared modules to ensure consistency and reduce code duplication:

### Configuration (`config.py`)
- Common parameters like iteration limits and retrieval settings
- Model configuration (e.g., default model name)
- Feature flags for various capabilities
- API keys and external service configuration

### Types (`types.py`)
- Common type definitions using TypedDict
- State types for RAG implementations
- Result types for consistent return values
- Analysis and retrieval result types

### Prompts (`prompts.py`)
- Shared prompt templates for consistency
- Templates for context analysis
- Templates for answer generation
- Templates for self-critique
- Templates for query planning

### Utilities (`utils.py`)
- Shared utility functions
- LLM response processing
- Document processing and optimization
- Context management
- Specialized tool implementations

## Using the RAG Factory

The new modular design includes a factory pattern for easy switching between implementations:

```python
from modules.rag.factory import RAGFactory, run_rag

# Get available implementations
implementations = RAGFactory.list_available_implementations()
print(f"Available implementations: {implementations}")

# Run a specific implementation
result = run_rag(
    query="What is the capital of France?", 
    implementation="langgraph",  # or "agentic" for original implementation
    config={"max_iterations": 3}
)

# Compare multiple implementations
comparison = RAGFactory.compare_implementations(
    query="What is the capital of France?",
    implementations=["agentic", "langgraph"]
)
```

## Setup

1. **Environment Setup**:
   ```bash
   ./scripts/setup_and_test.sh
   ```
   This will:
   - Create a virtual environment (`venv_latest`)
   - Install all dependencies
   - Run tests to verify the setup

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
   Alternatively, you can provide it when prompted by the application.

3. **Run the Application**:
   ```bash
   ./scripts/restart_streamlit.sh
   ```
   This script will:
   - Kill any existing Streamlit processes
   - Start the Streamlit app with proper error handling

## Linux Users: Fixing inotify Watch Limits

If you encounter the error "OSError: [Errno 28] inotify watch limit reached", you can increase the system's inotify watch limit by running:

```bash
sudo bash scripts/increase_inotify_watches.sh
```

This script will:
- Display current inotify limits
- Increase the limits temporarily and permanently
- Apply the new settings

## Usage

### Knowledge Base

The application will prompt you to create a knowledge base if one doesn't exist. This process:
- Loads documents from the `data` directory
- Creates embeddings using OpenAI
- Stores the embeddings in a FAISS index

### Query Interface

The interface allows you to:
1. Enter a custom query or select from example questions
2. Process the query with both Classic and Agentic RAG
3. View real-time processing logs for both approaches
4. Compare the generated answers side-by-side
5. Toggle between different Agentic RAG implementations (default: LangGraph)

### Evaluation

The comparison tab provides:
1. Semantic similarity scores between answers
2. Interpretation of the similarity
3. Determination of which approach performed better

## Dependencies

- LangChain 0.3.21+
- LangChain OpenAI 0.3.9+
- LangGraph 0.0.38+
- Streamlit 2.0.0+
- FAISS for vector storage
- Sentence Transformers for semantic evaluation
- RAGAS 0.2.14+ for RAG evaluation metrics
- PyTorch (CPU mode for lower resource usage)
- ChromaDB 0.4.18+ for vector storage (optional)

## Troubleshooting

Common issues and solutions:

1. **inotify watch limit errors**: Run `scripts/increase_inotify_watches.sh` as described above
2. **UI not updating**: Use the Reset button to clear the application state
3. **Questions remain disabled after processing**: Check logs for errors, reset the application
4. **PyTorch errors**: The application is configured to use PyTorch in CPU mode with minimal resources
5. **ModuleNotFoundError**: Make sure all dependencies are installed with `pip install -r requirements.txt`

See `docs/TROUBLESHOOTING.md` for more detailed guidance.

## License

This project is open source and available under the MIT license.

## Acknowledgements

This project demonstrates RAG techniques described in research papers by:
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2023)
- Shao et al., "Enhancing RAG with Agent Techniques" (2023)
- LangChain & LangGraph Documentation and Examples (2023-2024) 