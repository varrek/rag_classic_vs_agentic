# RAG Comparison Application

This repository contains a Retrieval-Augmented Generation (RAG) comparison application that demonstrates the differences between Classic RAG and Agentic RAG approaches.

## Features

- **Knowledge Base Management**: Create and manage a FAISS vector store of documents
- **Classic RAG Implementation**: Traditional retrieval followed by generation
- **Agentic RAG Implementation**: Advanced RAG with planning, self-critique, and specialized tools
  - **Original Implementation**: Custom-built agentic RAG with iterative retrieval
  - **LangGraph Implementation**: Graph-based agentic RAG using LangGraph
- **Side-by-Side Comparison**: View and compare results from both approaches
- **Interactive UI**: Streamlit-based interface with real-time processing logs
- **Semantic Evaluation**: Compare answers using semantic similarity metrics
- **Graph Visualization**: Visualize the LangGraph structure of the agentic RAG system

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

#### LangGraph Implementation
- Graph-based execution flow using LangGraph
- State management with TypedDict for better type safety
- Clearly defined nodes and edges for easier visualization and debugging
- Explicit decision-making conditions for graph traversal
- More modular and maintainable code structure

## Project Structure

- `app.py` - Streamlit web application
- `knowledge_base.py` - Knowledge base creation and management
- `rag_classic.py` - Classic RAG implementation
- `rag_agentic.py` - Original agentic RAG implementation
- `rag_langgraph.py` - LangGraph-based agentic RAG implementation
- `evaluation.py` - Evaluation utilities and similarity metrics
- `test_rag.py` - Testing script
- `setup_and_test.sh` - Setup and test script
- `restart_streamlit.sh` - Streamlit restart script
- `increase_inotify_watches.sh` - Script to increase inotify watch limits

## Recent Improvements

The application has been updated with significant improvements:

- **LangGraph Integration**: Added a graph-based agentic RAG implementation using LangGraph
- **Implementation Toggle**: Switch between original and LangGraph implementations
- **Graph Visualization**: View the structure of the LangGraph implementation
- **Enhanced UI Stability**: Eliminated screen flickering and UI glitches during processing
- **Real-time Processing Logs**: Shows logs during RAG processing
- **Stable Button Behavior**: Improved button state management during processing
- **Error Handling**: Better error handling and recovery mechanisms
- **inotify Limit Fix**: Solution for Linux inotify watch limit issues
- **Session State Management**: Fixed issues with state persistence

## Setup

1. **Environment Setup**:
   ```bash
   ./setup_and_test.sh
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
   ./restart_streamlit.sh
   ```
   This script will:
   - Kill any existing Streamlit processes
   - Start the Streamlit app with proper error handling

## Linux Users: Fixing inotify Watch Limits

If you encounter the error "OSError: [Errno 28] inotify watch limit reached", you can increase the system's inotify watch limit by running:

```bash
sudo bash increase_inotify_watches.sh
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

### Evaluation

The comparison tab provides:
1. Semantic similarity scores between answers
2. Interpretation of the similarity
3. Determination of which approach performed better

## Dependencies

- LangChain 0.3.21
- LangChain OpenAI 0.3.9
- Streamlit 2.0.0+
- FAISS for vector storage
- Sentence Transformers for semantic evaluation
- RAGAS 0.2.14 for RAG evaluation metrics
- PyTorch (CPU mode for lower resource usage)

## Troubleshooting

Common issues and solutions:

1. **inotify watch limit errors**: Run `increase_inotify_watches.sh` as described above
2. **UI not updating**: Use the Reset button to clear the application state
3. **Questions remain disabled after processing**: Check logs for errors, reset the application
4. **PyTorch errors**: The application is configured to use PyTorch in CPU mode with minimal resources

## License

This project is open source and available under the MIT license.

## Acknowledgements

This project demonstrates RAG techniques described in research papers by:
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2023)
- Shao et al., "Enhancing RAG with Agent Techniques" (2023) 