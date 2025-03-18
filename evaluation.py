import os
from typing import Dict, Any, List, Optional
import numpy as np

# Import RAGAS for evaluation
try:
    from ragas.metrics import answer_similarity
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not available")

def calculate_simple_similarity(generated_answer: str, ground_truth: str) -> float:
    """
    Calculate a simple similarity score between the generated answer and ground truth.
    This is used as a fallback when RAGAS fails.
    """
    # Ensure inputs are strings
    if not isinstance(generated_answer, str):
        generated_answer = str(generated_answer)
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    
    # Convert to lowercase and strip whitespace
    gen_lower = generated_answer.lower().strip()
    truth_lower = ground_truth.lower().strip()
    
    # Check for exact match first (after normalization)
    if gen_lower == truth_lower:
        return 1.0
    
    # For very short answers (1-2 words), check if the ground truth appears in the generated answer
    if len(truth_lower.split()) <= 2 and truth_lower in gen_lower:
        return 1.0
        
    # For longer answers, use a combination of exact word matches and word overlap
    gen_words = set(gen_lower.split())
    truth_words = set(truth_lower.split())
    
    if not gen_words or not truth_words:
        return 0.0
        
    # Calculate exact word matches
    exact_matches = len(gen_words.intersection(truth_words))
    
    # Calculate Jaccard similarity (intersection over union)
    union = len(gen_words.union(truth_words))
    jaccard = exact_matches / union if union > 0 else 0.0
    
    # For short ground truth answers, weight exact matches more heavily
    if len(truth_words) <= 3:
        # If all ground truth words are found in generated answer, score higher
        if truth_words.issubset(gen_words):
            return 0.9  # High but not perfect score
        # Weight exact matches more for short answers
        return min(1.0, (exact_matches / len(truth_words)) * 0.8 + jaccard * 0.2)
    
    # For longer answers, use standard Jaccard similarity
    return jaccard

def compare_answers(generated_answer: str, ground_truth: str) -> float:
    """
    Compare a generated answer with the ground truth using RAGAS.
    Falls back to a simple similarity metric if RAGAS fails.
    Returns a similarity score between 0 and 1.
    """
    # Ensure inputs are strings
    if not isinstance(generated_answer, str):
        generated_answer = str(generated_answer)
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    
    if not RAGAS_AVAILABLE:
        return calculate_simple_similarity(generated_answer, ground_truth)
        
    try:
        # Use RAGAS answer_similarity
        similarity = answer_similarity.score([generated_answer], [ground_truth])
        # Return the similarity score (convert from numpy array if needed)
        return float(similarity[0] if hasattr(similarity, '__iter__') else similarity)
    except Exception as e:
        print(f"Error using RAGAS for evaluation: {e}")
        print("Falling back to simple similarity calculation")
        return calculate_simple_similarity(generated_answer, ground_truth) 