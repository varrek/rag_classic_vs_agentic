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

## RAG Approaches Compared

### Classic RAG
- Simple retrieval followed by generation using the retrieved context
- Single-step process: retrieve documents, then generate an answer
- Deterministic document retrieval based on vector similarity

### Agentic RAG
- Iterative retrieval with planning, self-critique, and specialized tools
- Multi-step process with query refinement and rewriting
- Dynamic context building based on intermediate results
- Self-critique and validation of generated answers

#### LangGraph Implementation (Default)
- Graph-based execution flow using LangGraph
- State management with BaseModel for better type safety
- Clearly defined nodes, edges, and state for easier visualization and debugging
- Explicit decision-making in the evaluation node for determining next actions
- More modular and maintainable code structure with separation of concerns

#### Original Implementation (Legacy)
- Procedural execution flow with iterative refinement
- Function-based approach with custom state management
- Multiple specialized retrieval strategies based on query type
- Synthetic data generation for handling edge cases

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