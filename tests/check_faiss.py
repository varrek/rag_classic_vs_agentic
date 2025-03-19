import os
import json
import pickle
from pathlib import Path

def check_faiss_store():
    """Check if there are any documents in the FAISS store and print information without OpenAI API key."""
    # Define paths
    INDEX_DIR = Path("index")
    FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
    DOCUMENT_METADATA_PATH = INDEX_DIR / "document_metadata.json"
    QA_PAIRS_PATH = INDEX_DIR / "qa_pairs.json"
    
    # Check if knowledge base files exist
    print("Checking if knowledge base files exist...")
    faiss_index_exists = (FAISS_INDEX_PATH / "index.faiss").exists() and (FAISS_INDEX_PATH / "index.pkl").exists()
    metadata_exists = DOCUMENT_METADATA_PATH.exists()
    qa_pairs_exists = QA_PAIRS_PATH.exists()
    
    print(f"FAISS index files exist: {faiss_index_exists}")
    print(f"Document metadata file exists: {metadata_exists}")
    print(f"QA pairs file exists: {qa_pairs_exists}")
    
    # Examine document metadata
    if metadata_exists:
        try:
            with open(DOCUMENT_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            print(f"\nDocument metadata: contains {len(metadata)} entries")
            if metadata:
                print(f"Sample metadata entry: {metadata[0]}")
            else:
                print("Document metadata is empty: []")
        except Exception as e:
            print(f"Error reading document metadata: {str(e)}")
    
    # Examine QA pairs
    if qa_pairs_exists:
        try:
            with open(QA_PAIRS_PATH, 'r') as f:
                qa_pairs = json.load(f)
            
            print(f"\nQA pairs: contains {len(qa_pairs)} entries")
            if qa_pairs:
                print(f"Sample QA pair: {qa_pairs[0]}")
            else:
                print("QA pairs file is empty: []")
        except Exception as e:
            print(f"Error reading QA pairs: {str(e)}")
    
    # Examine FAISS index
    if faiss_index_exists:
        try:
            # Try to load index.pkl to see docstore info
            with open(FAISS_INDEX_PATH / "index.pkl", 'rb') as f:
                # Load only the header to avoid full deserialization
                try:
                    pkl_data = pickle.load(f)
                    print("\nSuccessfully loaded index.pkl")
                    
                    if hasattr(pkl_data, 'docstore') and hasattr(pkl_data.docstore, '_dict'):
                        doc_count = len(pkl_data.docstore._dict)
                        print(f"Documents in docstore: {doc_count}")
                        
                        if doc_count > 0:
                            sample_key = list(pkl_data.docstore._dict.keys())[0]
                            print(f"Sample document ID: {sample_key}")
                        else:
                            print("No documents in docstore.")
                    else:
                        print("Could not access docstore information.")
                except Exception as e:
                    print(f"Could not fully deserialize index.pkl: {str(e)}")
            
            # Check FAISS index file size
            faiss_size = os.path.getsize(FAISS_INDEX_PATH / "index.faiss")
            print(f"\nFAISS index file size: {faiss_size} bytes")
            print("A very small index file (few KB) likely indicates an empty or dummy index.")
            
        except Exception as e:
            print(f"Error examining FAISS index: {str(e)}")

if __name__ == "__main__":
    check_faiss_store() 