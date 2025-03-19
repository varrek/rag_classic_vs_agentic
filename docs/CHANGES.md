# Changelog

## Version 1.0.2 (Current)

### UI and User Experience Improvements
- Fixed issue with questions being disabled after processing completes
- Added real-time processing logs for both Classic and Agentic RAG
- Improved UI stability during processing
- Enhanced session state management to prevent state loss
- Added sample questions in an expandable section

### Technical Fixes
- Addressed Linux inotify watch limit issues with a dedicated fix script
- Disabled file watching in Streamlit to reduce resource usage
- Fixed PyTorch compatibility issues by limiting threads and disabling gradients
- Added graceful error handling with detailed logging
- Implemented a reset functionality to clear application state

### RAG Improvements
- Enhanced Agentic RAG with self-critique capabilities
- Added comparison functionality between Classic and Agentic RAG results
- Improved document retrieval with multi-strategy approach
- Added semantic evaluation of answer quality

## Version 1.0.1

### Initial Features
- Basic Classic RAG implementation
- Simple Streamlit UI for query input
- Document retrieval from FAISS vector store
- Answer generation with OpenAI models
- Knowledge base creation functionality

### Known Issues (Fixed in 1.0.2)
- Questions would reload after processing
- Results and logs were not always visible
- UI would flicker during processing
- Occasional PyTorch memory issues
- Linux inotify watch limit errors

## Future Plans

### Version 1.1.0 (Planned)
- Add customizable RAG parameters
- Implement additional retrieval strategies
- Enhance evaluation metrics
- Add document highlighting in retrieved context
- Support for uploading custom documents
- Visualization of retrieval process 