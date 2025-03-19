"""
Retrieval Module

This module handles document retrieval functionality for the RAG system,
including vector search, hybrid search, and additional retrieval methods.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd
    pandas_available = True
except ImportError:
    logger.warning("pandas not available, some functionality will be limited")
    pandas_available = False

try:
    import chromadb
    chroma_available = True
except ImportError:
    logger.warning("chromadb not available, vector database functionality will be limited")
    chroma_available = False

try:
    import requests
    requests_available = True
except ImportError:
    logger.warning("requests not available, web search functionality will be limited")
    requests_available = False

# Constants
INDEX_DIR = os.path.join(os.getcwd(), "index")
DATA_DIR = os.path.join(os.getcwd(), "data")
QA_DATA_DIR = os.path.join(DATA_DIR, "qa")
DEFAULT_TOP_K = 5

# Initialize vector database client
_chroma_client = None

def get_chroma_client():
    """Get or initialize the Chroma client."""
    global _chroma_client
    
    if not chroma_available:
        raise ImportError("chromadb is required for vector database functionality")
        
    if _chroma_client is None:
        # Ensure the index directory exists
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # Initialize the persistent client
        try:
            _chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
            logger.info(f"Initialized Chroma client with persistence at {INDEX_DIR}")
        except Exception as e:
            logger.error(f"Error initializing Chroma client: {e}")
            # Fallback to in-memory client
            _chroma_client = chromadb.Client()
            logger.warning("Using in-memory Chroma client (data will not persist)")
            
    return _chroma_client

def get_collection(collection_name="documents"):
    """Get or create a collection from the vector database."""
    if not chroma_available:
        raise ImportError("chromadb is required for vector database functionality")
        
    client = get_chroma_client()
    
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document collection"}
        )
        return collection
    except Exception as e:
        logger.error(f"Error getting collection {collection_name}: {e}")
        raise

def retrieve_documents(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "documents",
    retrieval_method: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Retrieve documents relevant to the query.
    
    Args:
        query: The search query
        top_k: Number of documents to retrieve
        collection_name: Name of the vector collection
        retrieval_method: Method to use (semantic, keyword, hybrid)
        
    Returns:
        List of document dictionaries with content and metadata
    """
    logger.info(f"Retrieving documents for query: '{query}' using {retrieval_method} retrieval")
    
    if not query or not query.strip():
        logger.warning("Empty query provided to retrieve_documents")
        return []
    
    try:
        # Get vector collection
        collection = get_collection(collection_name)
        
        # Execute retrieval based on method
        if retrieval_method == "semantic":
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
        elif retrieval_method == "keyword":
            # Use where filter for keyword search if possible
            # This depends on how the documents were indexed
            results = collection.query(
                query_texts=[query],
                n_results=top_k * 2,  # Get more results for filtering
                where_document={"$contains": query}  # Simple keyword filter
            )
        else:  # hybrid (default)
            # For hybrid, we'll combine results from both methods
            semantic_results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Try keyword search with a simple filter
            keyword_results = collection.query(
                query_texts=[""],  # Empty query text to rely on filters
                n_results=top_k,
                where_document={"$contains": query}
            )
            
            # Combine and deduplicate results
            combined_ids = []
            combined_docs = []
            combined_metadatas = []
            combined_distances = []
            
            # Process semantic results first
            if 'ids' in semantic_results and semantic_results['ids']:
                for i, doc_id in enumerate(semantic_results['ids'][0]):
                    if doc_id not in combined_ids:
                        combined_ids.append(doc_id)
                        combined_docs.append(semantic_results['documents'][0][i])
                        combined_metadatas.append(semantic_results['metadatas'][0][i])
                        combined_distances.append(semantic_results.get('distances', [[0] * len(semantic_results['ids'][0])])[0][i])
            
            # Then add keyword results if they're not already included
            if 'ids' in keyword_results and keyword_results['ids']:
                for i, doc_id in enumerate(keyword_results['ids'][0]):
                    if doc_id not in combined_ids:
                        combined_ids.append(doc_id)
                        combined_docs.append(keyword_results['documents'][0][i])
                        combined_metadatas.append(keyword_results['metadatas'][0][i])
                        combined_distances.append(1.0)  # Default distance for keyword results
            
            # Create a combined results structure
            results = {
                'ids': [combined_ids[:top_k]],
                'documents': [combined_docs[:top_k]],
                'metadatas': [combined_metadatas[:top_k]],
                'distances': [combined_distances[:top_k]]
            }
        
        # Process results into a more usable format
        processed_results = []
        
        if 'ids' in results and results['ids'] and len(results['ids'][0]) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                doc = {
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'Unknown'),
                    'score': 1.0 - (results.get('distances', [[0] * len(results['ids'][0])])[0][i] / 2),  # Normalize to 0-1
                    'metadata': results['metadatas'][0][i]
                }
                processed_results.append(doc)
        
        logger.info(f"Retrieved {len(processed_results)} documents")
        return processed_results
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        # Return empty list on error
        return []

def web_search(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Perform a web search for additional information.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        List of document dictionaries with content from web search
    """
    if not requests_available:
        logger.error("Web search requested but requests module not available")
        return []
    
    # Check for Google Custom Search API key and CX
    api_key = os.environ.get("GOOGLE_API_KEY")
    search_engine_id = os.environ.get("GOOGLE_CSE_ID")
    
    if not api_key or not search_engine_id:
        logger.warning("Google API key or CSE ID not set, web search unavailable")
        return []
    
    try:
        # Google Custom Search API endpoint
        url = "https://www.googleapis.com/customsearch/v1"
        
        # Parameters for the search
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(num_results, 10)  # API limit is 10
        }
        
        # Make the request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for error status codes
        
        # Parse the response
        search_results = response.json()
        
        documents = []
        if "items" in search_results:
            for item in search_results["items"]:
                # Extract the information we need
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                
                # Format the content
                content = f"{title}\n\n{snippet}\n\nSource: {link}"
                
                # Create a document
                document = {
                    "id": f"web_{link.replace('://', '_').replace('/', '_').replace('.', '_')}",
                    "content": content,
                    "source": link,
                    "score": 0.7,  # Default score for web results
                    "metadata": {
                        "title": title,
                        "source": link,
                        "type": "web_search"
                    }
                }
                
                documents.append(document)
        
        logger.info(f"Web search retrieved {len(documents)} results")
        return documents
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return []

def load_qa_pairs(dataset_name: str = "default") -> List[Dict[str, str]]:
    """
    Load question-answer pairs from a JSON file.
    
    Args:
        dataset_name: Name of the dataset file (without extension)
        
    Returns:
        List of question-answer dictionaries
    """
    # Ensure the QA data directory exists
    os.makedirs(QA_DATA_DIR, exist_ok=True)
    
    # Path to the dataset file
    file_path = os.path.join(QA_DATA_DIR, f"{dataset_name}.json")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Ensure the data is in the expected format
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # Make sure each item has 'question' and 'answer' keys
                valid_data = [item for item in data if 'question' in item and 'answer' in item]
                
                if len(valid_data) != len(data):
                    logger.warning(f"Some items in {dataset_name}.json do not have both 'question' and 'answer' keys")
                
                logger.info(f"Loaded {len(valid_data)} QA pairs from {dataset_name}.json")
                return valid_data
            else:
                logger.error(f"Invalid format in {dataset_name}.json")
                return []
        else:
            logger.warning(f"QA dataset file {file_path} not found")
            return []
    except Exception as e:
        logger.error(f"Error loading QA pairs from {file_path}: {e}")
        return []

def save_qa_pairs(qa_pairs: List[Dict[str, str]], dataset_name: str = "default") -> bool:
    """
    Save question-answer pairs to a JSON file.
    
    Args:
        qa_pairs: List of question-answer dictionaries
        dataset_name: Name of the dataset file (without extension)
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure the QA data directory exists
    os.makedirs(QA_DATA_DIR, exist_ok=True)
    
    # Path to the dataset file
    file_path = os.path.join(QA_DATA_DIR, f"{dataset_name}.json")
    
    try:
        # Write the data to the file
        with open(file_path, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {dataset_name}.json")
        return True
    except Exception as e:
        logger.error(f"Error saving QA pairs to {file_path}: {e}")
        return False

def find_similar_questions(
    query: str, 
    dataset_name: str = "default", 
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Find similar questions in the QA dataset.
    
    Args:
        query: The question to find matches for
        dataset_name: Name of the dataset file (without extension)
        similarity_threshold: Minimum similarity score to include a match
        
    Returns:
        List of matches with similarity scores
    """
    # First, load the QA pairs
    qa_pairs = load_qa_pairs(dataset_name)
    
    if not qa_pairs:
        logger.warning(f"No QA pairs found for dataset {dataset_name}")
        return []
    
    try:
        # Import sentence_transformers for computing text similarity
        from sentence_transformers import SentenceTransformer
        import torch
        from torch.nn import CosineSimilarity
        
        # Load a sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Encode all questions
        questions = [item['question'] for item in qa_pairs]
        question_embeddings = model.encode(questions, convert_to_tensor=True)
        
        # Compute similarity scores
        cos = CosineSimilarity(dim=1)
        similarities = cos(query_embedding.unsqueeze(0), question_embeddings).squeeze().tolist()
        
        # Create matches with scores above the threshold
        matches = []
        for i, similarity in enumerate(similarities):
            if similarity >= similarity_threshold:
                matches.append({
                    "question": qa_pairs[i]['question'],
                    "answer": qa_pairs[i]['answer'],
                    "similarity": float(similarity)
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(matches)} similar questions with threshold {similarity_threshold}")
        return matches
    except ImportError:
        logger.error("sentence_transformers not available for computing similarity")
        
        # Fallback to simple keyword matching
        logger.info("Using keyword matching as fallback")
        matches = []
        query_words = set(query.lower().split())
        
        for item in qa_pairs:
            question = item['question']
            question_words = set(question.lower().split())
            
            # Calculate Jaccard similarity
            if not query_words or not question_words:
                continue
                
            intersection = len(query_words.intersection(question_words))
            union = len(query_words.union(question_words))
            
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard >= similarity_threshold:
                matches.append({
                    "question": question,
                    "answer": item['answer'],
                    "similarity": jaccard
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(matches)} similar questions using keyword matching")
        return matches
    except Exception as e:
        logger.error(f"Error finding similar questions: {e}")
        return []

def add_documents_to_index(
    documents: List[Dict[str, Any]],
    collection_name: str = "documents",
    batch_size: int = 100
) -> bool:
    """
    Add documents to the vector index.
    
    Args:
        documents: List of document dictionaries with 'content' and optional 'metadata'
        collection_name: Name of the vector collection
        batch_size: Number of documents to add in each batch
        
    Returns:
        True if successful, False otherwise
    """
    if not chroma_available:
        logger.error("chromadb not available, cannot add documents to index")
        return False
    
    if not documents:
        logger.warning("No documents provided to add_documents_to_index")
        return False
    
    try:
        # Get the collection
        collection = get_collection(collection_name)
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare batch data
            ids = []
            contents = []
            metadatas = []
            
            for doc in batch:
                # Generate ID if not provided
                doc_id = doc.get('id', f"doc_{len(ids)}_{hash(doc['content'])}")
                
                ids.append(doc_id)
                contents.append(doc['content'])
                
                # Prepare metadata
                metadata = doc.get('metadata', {})
                
                # Ensure source is included in metadata
                if 'source' in doc and 'source' not in metadata:
                    metadata['source'] = doc['source']
                
                metadatas.append(metadata)
            
            # Add the batch to the collection
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch of {len(batch)} documents to collection {collection_name}")
        
        logger.info(f"Successfully added {len(documents)} documents to index")
        return True
    except Exception as e:
        logger.error(f"Error adding documents to index: {e}")
        return False

def index_data_files(
    directory: str = DATA_DIR,
    collection_name: str = "documents",
    file_extensions: List[str] = [".txt", ".md", ".json"],
    recursive: bool = True
) -> int:
    """
    Index all compatible data files in a directory.
    
    Args:
        directory: Path to the data directory
        collection_name: Name of the vector collection
        file_extensions: List of file extensions to process
        recursive: Whether to recursively search subdirectories
        
    Returns:
        Number of documents indexed
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return 0
    
    try:
        # Track the documents
        documents = []
        
        # Walk through the directory
        if recursive:
            walker = os.walk(directory)
        else:
            walker = [(directory, [], [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])]
        
        for root, _, files in walker:
            for file in files:
                # Check file extension
                if not any(file.endswith(ext) for ext in file_extensions):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                
                try:
                    # Process the file based on its type
                    if file.endswith(".txt") or file.endswith(".md"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Create a document
                        document = {
                            "id": f"file_{relative_path.replace('/', '_')}",
                            "content": content,
                            "source": relative_path,
                            "metadata": {
                                "source": relative_path,
                                "type": "file",
                                "format": file.split(".")[-1]
                            }
                        }
                        
                        documents.append(document)
                        
                    elif file.endswith(".json"):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Handle different JSON formats
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                            # List of objects - treat each as a document if it has a content field
                            for i, item in enumerate(data):
                                if "content" in item:
                                    # Use the content field
                                    content = item["content"]
                                    
                                    # Create document ID
                                    doc_id = f"json_{relative_path}_{i}"
                                    
                                    # Create a document
                                    document = {
                                        "id": doc_id,
                                        "content": content,
                                        "source": f"{relative_path}[{i}]",
                                        "metadata": {
                                            "source": relative_path,
                                            "type": "json",
                                            "index": i,
                                            "original_metadata": {k: v for k, v in item.items() if k != "content"}
                                        }
                                    }
                                    
                                    documents.append(document)
                        elif isinstance(data, dict):
                            # Single object - treat as one document
                            if "content" in data:
                                content = data["content"]
                            else:
                                # If no content field, use the entire JSON
                                content = json.dumps(data, indent=2)
                            
                            # Create a document
                            document = {
                                "id": f"json_{relative_path}",
                                "content": content,
                                "source": relative_path,
                                "metadata": {
                                    "source": relative_path,
                                    "type": "json"
                                }
                            }
                            
                            documents.append(document)
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        # Add documents to the index
        if documents:
            success = add_documents_to_index(documents, collection_name)
            if success:
                logger.info(f"Indexed {len(documents)} documents from {directory}")
                return len(documents)
        
        logger.warning(f"No compatible documents found in {directory}")
        return 0
    except Exception as e:
        logger.error(f"Error indexing data files: {e}")
        return 0 