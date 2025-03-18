import os
from pathlib import Path
import glob
import pandas as pd
import random
from typing import List, Dict, Any, Tuple, Callable, Optional
import json

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Constants
DATA_DIR = Path("data")
CLEAN_FILES_PATTERN = "*.clean"
QA_DIR = DATA_DIR / "qa"
INDEX_DIR = Path("index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index"
DOCUMENT_METADATA_PATH = INDEX_DIR / "document_metadata.json"
QA_PAIRS_PATH = INDEX_DIR / "qa_pairs.json"

# Create the index directory if it doesn't exist
INDEX_DIR.mkdir(exist_ok=True, parents=True)

def check_knowledge_base_exists() -> bool:
    """Check if the knowledge base has been created."""
    return FAISS_INDEX_PATH.exists() and DOCUMENT_METADATA_PATH.exists() and QA_PAIRS_PATH.exists()

def load_clean_text_files() -> List[Document]:
    """Load all .clean text files from the data directory."""
    clean_files = list(DATA_DIR.glob(CLEAN_FILES_PATTERN))
    documents = []
    
    for file_path in clean_files:
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            
            # Extract article title from filename
            filename = file_path.name
            article_title = filename.split('_a')[1].split('.')[0]
            set_name = filename.split('_set')[1].split('_')[0]
            article_name = f"S{filename.split('_S')[1].split('_')[0]}_set{set_name}_a{article_title}"
            
            # Add metadata to each document
            for doc in docs:
                doc.metadata["source"] = str(file_path)
                doc.metadata["article_title"] = article_name
            
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents

def load_qa_pairs() -> List[Dict[str, str]]:
    """Load all question-answer pairs from the qa directory."""
    qa_files = list(QA_DIR.glob("*.txt"))
    all_qa_pairs = []
    
    for file_path in qa_files:
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    qa_pair = {
                        "article_title": row["ArticleTitle"],
                        "question": row["Question"],
                        "answer": row["Answer"],
                        "article_file": row["ArticleFile"] if "ArticleFile" in row else None
                    }
                    all_qa_pairs.append(qa_pair)
                except Exception as e:
                    print(f"Error processing row in {file_path}: {e}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_qa_pairs

def create_knowledge_base(progress_callback: Callable[[float, str], None]) -> None:
    """Create the knowledge base from the raw text data."""
    # Step 1: Load all clean text files
    progress_callback(0.1, "Loading text documents...")
    documents = load_clean_text_files()
    progress_callback(0.3, f"Loaded {len(documents)} documents")
    
    # Create empty files as placeholders if no documents are found
    if not documents:
        progress_callback(0.4, "No documents were found. Creating empty knowledge base...")
        
        # Step 4: Store empty document metadata
        progress_callback(0.7, "Creating empty document metadata file...")
        with open(DOCUMENT_METADATA_PATH, "w") as f:
            json.dump([], f)
        
        # Step 5: Try to load QA pairs or create empty file
        progress_callback(0.8, "Attempting to load QA pairs...")
        try:
            qa_pairs = load_qa_pairs()
            progress_callback(0.9, f"Loaded {len(qa_pairs)} QA pairs")
        except Exception as e:
            progress_callback(0.9, f"Error loading QA pairs: {str(e)}. Creating empty QA pairs file.")
            qa_pairs = []
        
        with open(QA_PAIRS_PATH, "w") as f:
            json.dump(qa_pairs, f)
        
        # Create an empty FAISS index
        progress_callback(0.95, "Creating empty FAISS index...")
        embeddings = OpenAIEmbeddings()
        # Create a dummy document for indexing
        dummy_doc = Document(page_content="This is a placeholder document.", metadata={"source": "placeholder"})
        vectorstore = FAISS.from_documents([dummy_doc], embeddings)
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        
        progress_callback(1.0, "Empty knowledge base created successfully!")
        return
    
    # Step 2: Split documents into chunks
    progress_callback(0.4, "Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    progress_callback(0.5, f"Split into {len(chunks)} chunks")
    
    if not chunks:
        progress_callback(0.5, "No chunks were created. Creating an empty knowledge base...")
        # Create a dummy document for indexing
        dummy_doc = Document(page_content="This is a placeholder document.", metadata={"source": "placeholder"})
        chunks = [dummy_doc]
    
    # Step 3: Create embeddings and store in FAISS
    progress_callback(0.6, "Creating embeddings and FAISS index...")
    embeddings = OpenAIEmbeddings()
    
    # Add debug information before creating the index
    try:
        # Test embedding on a single document to verify it works
        progress_callback(0.65, "Testing embedding on a sample chunk...")
        sample_embedding = embeddings.embed_query(chunks[0].page_content)
        progress_callback(0.67, f"Sample embedding dimensions: {len(sample_embedding)}")
        
        # Create the FAISS index
        vectorstore = FAISS.from_documents(chunks, embeddings)
        progress_callback(0.7, "Saving FAISS index...")
        vectorstore.save_local(str(FAISS_INDEX_PATH))
    except Exception as e:
        progress_callback(0.6, f"Error creating embeddings: {str(e)}")
        raise
    
    # Step 4: Store document metadata
    progress_callback(0.8, "Storing document metadata...")
    document_metadata = [
        {
            "source": doc.metadata.get("source", ""),
            "article_title": doc.metadata.get("article_title", "")
        }
        for doc in documents
    ]
    with open(DOCUMENT_METADATA_PATH, "w") as f:
        json.dump(document_metadata, f)
    
    # Step 5: Load and store QA pairs
    progress_callback(0.9, "Loading and storing QA pairs...")
    qa_pairs = load_qa_pairs()
    # If no QA pairs were found, create an empty file
    if not qa_pairs:
        progress_callback(0.9, "No QA pairs found. Creating empty QA pairs file.")
    
    with open(QA_PAIRS_PATH, "w") as f:
        json.dump(qa_pairs, f)
    
    progress_callback(1.0, "Knowledge base created successfully!")

def get_document_store():
    """Get the document store for querying."""
    if not check_knowledge_base_exists():
        raise ValueError("Knowledge base not found. Please create it first.")
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def get_qa_pairs() -> List[Dict[str, str]]:
    """Get all QA pairs from the stored JSON file."""
    if not QA_PAIRS_PATH.exists():
        # Create an empty file if it doesn't exist
        with open(QA_PAIRS_PATH, "w") as f:
            json.dump([], f)
    
    with open(QA_PAIRS_PATH, "r") as f:
        qa_pairs = json.load(f)
    
    return qa_pairs

def get_random_example_questions(n: int = 10) -> List[Tuple[str, str]]:
    """Get n random example questions with their answers."""
    qa_pairs = get_qa_pairs()
    
    if not qa_pairs:
        # Return a default question if no QA pairs are available
        return [("What is RAG?", "RAG stands for Retrieval Augmented Generation, a technique that combines retrieval of information with text generation.")]
    
    if len(qa_pairs) <= n:
        return [(qa["question"], qa["answer"]) for qa in qa_pairs]
    
    # Get random sample of QA pairs
    random_pairs = random.sample(qa_pairs, n)
    return [(qa["question"], qa["answer"]) for qa in random_pairs] 