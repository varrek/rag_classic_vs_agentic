import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib to non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

# Initialize global variables
sentence_model = None
compute_similarity_enabled = True

# Import SentenceTransformer in a try-except block to handle errors
try:
    from sentence_transformers import SentenceTransformer
    import torch
    # Pre-check if CUDA is available to avoid runtime errors
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("CUDA is available for torch")
    else:
        logger.info("CUDA is not available, using CPU for torch")
except ImportError as e:
    logger.error(f"Error importing SentenceTransformer or torch: {e}")
    compute_similarity_enabled = False
except Exception as e:
    logger.error(f"Unexpected error with torch/sentence_transformers: {e}")
    compute_similarity_enabled = False

# Try to import RAGAS, but handle it gracefully if it fails
ragas_available = True
try:
    # RAGAS imports - updated for version 0.2.14
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
    from ragas.metrics import AnswerCorrectness, AnswerSimilarity
    from ragas import evaluate
except ImportError as e:
    logger.error(f"Error importing RAGAS: {e}")
    ragas_available = False
except Exception as e:
    logger.error(f"Unexpected error with RAGAS: {e}")
    ragas_available = False

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Error importing LangChain: {e}")

# Local imports
from knowledge_base import get_random_example_questions
from rag_classic import query_rag

# Load environment variable for OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_sentence_model():
    """Get or initialize a sentence transformer model for computing similarity."""
    global sentence_model
    if not compute_similarity_enabled:
        raise ValueError("Sentence transformers is not available")
        
    if sentence_model is None:
        # Load a sentence transformer model
        try:
            # Use a smaller model if resources are limited
            logger.info("Loading sentence transformer model: all-MiniLM-L6-v2")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            # Fallback to a very small model
            try:
                logger.info("Trying fallback model: paraphrase-MiniLM-L3-v2")
                sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {e2}")
                raise ValueError("Failed to load any sentence transformer model")
    return sentence_model

def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using sentence transformers."""
    if not compute_similarity_enabled:
        logger.warning("Compute similarity called but sentence transformers not available, returning default value")
        return 0.5  # Return a neutral similarity score as fallback
        
    try:
        model = get_sentence_model()
        
        # Convert inputs to strings
        text1 = str(text1)
        text2 = str(text2)
        
        # Ensure texts aren't too long - truncate if needed
        max_length = 5000  # Character limit to avoid OOM errors
        if len(text1) > max_length:
            text1 = text1[:max_length]
        if len(text2) > max_length:
            text2 = text2[:max_length]
            
        # Wrap tensor operations in try-except
        try:
            # Use with torch.no_grad() to reduce memory usage
            with torch.no_grad():
                embedding1 = model.encode(text1, convert_to_tensor=True)
                embedding2 = model.encode(text2, convert_to_tensor=True)
                
                # Compute cosine similarity
                from torch.nn import CosineSimilarity
                cos = CosineSimilarity(dim=0)
                similarity = cos(embedding1, embedding2).item()
        except RuntimeError as e:
            logger.error(f"Runtime error during tensor operations: {e}")
            # Fallback to non-tensor computation
            embedding1 = model.encode(text1, convert_to_tensor=False)
            embedding2 = model.encode(text2, convert_to_tensor=False)
            
            # Manual cosine similarity calculation
            from numpy import dot
            from numpy.linalg import norm
            similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        
        return similarity
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        traceback.print_exc()
        return 0.5  # Return a neutral similarity score on error

def compute_ragas_metrics(
    questions: List[str],
    answers: List[str],
    retrieved_contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute RAGAS metrics for a set of question-answering examples.
    
    Args:
        questions: List of questions
        answers: List of generated answers
        retrieved_contexts: List of lists of retrieved contexts for each question
        ground_truths: Optional list of ground truth answers
        
    Returns:
        Dictionary of metric names and their values
    """
    # Check if RAGAS is available
    if not ragas_available:
        logger.warning("RAGAS metrics called but RAGAS not available")
        return {"error": "RAGAS metrics not available"}
        
    try:
        # Create dataset for RAGAS evaluation
        data = {
            "question": questions,
            "answer": answers, 
            "contexts": retrieved_contexts
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        # Initialize the OpenAI model for RAGAS
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        # Choose metrics based on available data
        metrics = []
        if ground_truths:
            # Use all metrics when we have ground truth
            metrics = [
                AnswerCorrectness(llm=llm),
                Faithfulness(llm=llm),
                ContextRelevance(llm=llm),
                AnswerRelevancy(llm=llm)
            ]
        else:
            # Use only metrics that don't require ground truth
            metrics = [
                Faithfulness(llm=llm),
                ContextRelevance(llm=llm),
                AnswerRelevancy(llm=llm)
            ]
        
        # Run RAGAS evaluation
        result = evaluate(
            data,
            metrics=metrics
        )
        
        # Convert result to a dictionary
        metrics_dict = {}
        for metric_name, metric_value in result.items():
            metrics_dict[metric_name] = np.mean(metric_value) if hasattr(metric_value, "__len__") else metric_value
        
        return metrics_dict
        
    except Exception as e:
        logger.error(f"Error computing RAGAS metrics: {e}")
        traceback.print_exc()
        
        # Return empty dictionary if evaluation fails
        return {"error": str(e)}

def compare_answers(answer1: str, answer2: str) -> Tuple[float, str]:
    """
    Compare two answers and return similarity score with explanation.
    
    Args:
        answer1: First answer text
        answer2: Second answer text
        
    Returns:
        Tuple of (similarity_score, explanation)
    """
    similarity = compute_similarity(answer1, answer2)
    
    # Generate explanation based on similarity score
    if similarity > 0.8:
        explanation = "The answers are very similar in meaning."
    elif similarity > 0.6:
        explanation = "The answers share significant content but have some differences."
    elif similarity > 0.4:
        explanation = "The answers have moderate similarity but substantial differences."
    else:
        explanation = "The answers are substantially different."
    
    return similarity, explanation

# Alternative similarity function that doesn't use pytorch
# This will be used as a fallback if sentence_transformers has issues
def simple_similarity(text1: str, text2: str) -> float:
    """
    Compute a simple similarity score based on overlapping words.
    Not as good as neural embeddings but works without dependencies.
    """
    # Normalize and tokenize texts
    def normalize(text):
        text = str(text).lower()
        # Remove punctuation
        import re
        text = re.sub(r'[^\w\s]', '', text)
        # Split into words
        return set(text.split())
    
    words1 = normalize(text1)
    words2 = normalize(text2)
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0.0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    jaccard = intersection / union if union > 0 else 0
    
    # Scale to make it more like cosine similarity range
    # This is a very rough approximation
    scaled = 0.5 + (jaccard * 0.5)
    
    return scaled 