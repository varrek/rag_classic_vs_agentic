import os
from typing import Dict, Any, List, Optional
import numpy as np
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Filter out LangChain deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Import RAGAS for evaluation - try multiple import strategies
RAGAS_AVAILABLE = False
_semantic_similarity = None
_answer_similarity = None
RAGAS_VERSION = None

# Try to determine RAGAS version
try:
    import ragas
    RAGAS_VERSION = getattr(ragas, "__version__", "unknown")
    print(f"Detected RAGAS version: {RAGAS_VERSION}")
except ImportError:
    print("RAGAS package not found")

# Try first import strategy - newer RAGAS
try:
    from ragas.metrics import SemanticSimilarity
    # Try different locations for OpenAIEmbeddings
    try:
        from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
        RAGAS_AVAILABLE = True
        _semantic_similarity = SemanticSimilarity
        print(f"Successfully imported newer RAGAS (v{RAGAS_VERSION}) with LangChain OpenAI embeddings")
    except ImportError:
        print("Failed to import LangChain OpenAI embeddings")
except ImportError:
    print("Failed to import SemanticSimilarity from RAGAS")

# Try second import strategy - older RAGAS
if not RAGAS_AVAILABLE:
    try:
        from ragas.metrics import answer_similarity
        RAGAS_AVAILABLE = True
        _answer_similarity = answer_similarity
        print("Successfully imported older RAGAS with answer_similarity")
    except ImportError:
        print("Failed to import answer_similarity from RAGAS")

if not RAGAS_AVAILABLE:
    print("WARNING: RAGAS not available. Will use simple similarity metric.")

# Cache the embeddings model to avoid reloading it each time
_langchain_embeddings = None

class EmbeddingsWrapper:
    """Wrapper for LangChain embeddings to make them compatible with RAGAS."""
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings
        
    async def embed_text(self, text):
        """RAGAS calls embed_text, but LangChain has embed_query"""
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                print(f"Warning: embed_text received non-string input of type {type(text)}, converting to string")
                text = str(text)
                
            # Handle empty text
            if not text.strip():
                print("Warning: embed_text received empty text")
                # Return a zero vector of appropriate dimension (typical embedding size is 1536 for OpenAI)
                return [0.0] * 1536
                
            if hasattr(self.langchain_embeddings, 'embed_query'):
                # Call embed_query synchronously since it's not an async method
                result = self.langchain_embeddings.embed_query(text)
                print(f"Successfully used embed_query, returned vector of length {len(result) if result else 0}")
                return result
            elif hasattr(self.langchain_embeddings, 'embed_documents'):
                # Call embed_documents synchronously
                result = self.langchain_embeddings.embed_documents([text])
                print(f"Successfully used embed_documents, returned vector of length {len(result[0]) if result and result[0] else 0}")
                return result[0] if result else []
            else:
                raise AttributeError("The embeddings object has neither embed_query nor embed_documents method")
        except Exception as e:
            print(f"Error in embed_text: {e}")
            print(f"Input type: {type(text)}, Input preview: {str(text)[:100]}...")
            # Fall back to sentence-transformers if the LangChain embedding fails
            try:
                from sentence_transformers import SentenceTransformer
                print("Falling back to SentenceTransformer for embeddings")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                result = model.encode(text).tolist()
                print(f"SentenceTransformer produced vector of length {len(result)}")
                return result
            except Exception as inner_e:
                print(f"Fallback embedding also failed: {inner_e}")
                # Return a zero vector as last resort
                print("Returning zero vector as last resort")
                return [0.0] * 1536

def get_langchain_embeddings():
    """Get or create the LangChain embeddings model."""
    global _langchain_embeddings
    if _langchain_embeddings is None:
        try:
            # Check if OpenAI API key is set
            import os
            import streamlit as st
            
            openai_key = None
            if 'OPENAI_API_KEY' in os.environ:
                openai_key = os.environ['OPENAI_API_KEY']
                print("Using OpenAI API key from environment variable")
            elif hasattr(st, 'secrets') and 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
                openai_key = st.secrets['openai']['api_key']
                print("Using OpenAI API key from Streamlit secrets")
            
            if not openai_key:
                print("WARNING: OpenAI API key not found. Falling back to SentenceTransformer embeddings.")
                try:
                    # Create a wrapper for SentenceTransformer that matches our API
                    class SentenceTransformerWrapper:
                        def __init__(self):
                            from sentence_transformers import SentenceTransformer
                            self.model = SentenceTransformer('all-MiniLM-L6-v2')
                            print("Initialized SentenceTransformer model")
                            
                        def embed_query(self, text):
                            return self.model.encode(text).tolist()
                    
                    _langchain_embeddings = EmbeddingsWrapper(SentenceTransformerWrapper())
                    print("Successfully initialized SentenceTransformer embeddings with wrapper")
                    return _langchain_embeddings
                except Exception as e:
                    print(f"Error initializing SentenceTransformer: {e}")
                    return None
            
            # Use LangChain's OpenAI embeddings
            from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
            raw_embeddings = LangchainOpenAIEmbeddings()
            # Test the embeddings with a simple query to make sure they work
            try:
                test_result = raw_embeddings.embed_query("Test embedding query")
                print(f"Embedding test successful, received vector of length {len(test_result)}")
            except Exception as test_e:
                print(f"Embedding test failed: {test_e}. Falling back to SentenceTransformer.")
                # Fall back to SentenceTransformer
                from sentence_transformers import SentenceTransformer
                class SentenceTransformerWrapper:
                    def __init__(self):
                        self.model = SentenceTransformer('all-MiniLM-L6-v2')
                        
                    def embed_query(self, text):
                        return self.model.encode(text).tolist()
                
                raw_embeddings = SentenceTransformerWrapper()
            
            _langchain_embeddings = EmbeddingsWrapper(raw_embeddings)
            print("Successfully initialized embeddings with wrapper")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            import traceback
            traceback.print_exc()
            _langchain_embeddings = None
    return _langchain_embeddings

def get_content_from_llm_response(response):
    """
    Extract string content from various types of LLM responses.
    Handles both string responses and LangChain AIMessage objects.
    
    Args:
        response: LLM response (string or AIMessage)
        
    Returns:
        String content from the response
    """
    # If it's already a string, return it
    if isinstance(response, str):
        return response
    
    # If it's an AIMessage, extract the content
    if hasattr(response, 'content'):
        return response.content
        
    # If it's any other object with string representation, convert to string
    return str(response)

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
    # Ensure inputs are strings by extracting content if needed
    generated_answer = get_content_from_llm_response(generated_answer)
    ground_truth = get_content_from_llm_response(ground_truth)
    
    # If either input is very short or empty, handle specially
    if not generated_answer.strip() or not ground_truth.strip():
        print("Empty input detected, returning similarity score of 0")
        return 0.0
    
    # Add a direct implementation of semantic similarity as top priority
    try:
        print("Trying direct implementation of semantic similarity with SentenceTransformer")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings for both texts
        gen_embedding = model.encode(generated_answer)
        truth_embedding = model.encode(ground_truth)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([gen_embedding], [truth_embedding])[0][0]
        print(f"Direct semantic similarity calculation: {similarity}")
        return float(similarity)
    except Exception as e:
        print(f"Direct semantic similarity failed: {e}, falling back to RAGAS")
    
    if not RAGAS_AVAILABLE:
        print("RAGAS not available, using simple similarity metric")
        return calculate_simple_similarity(generated_answer, ground_truth)
    
    try:
        # Before calling RAGAS, check if inputs are valid
        if not isinstance(generated_answer, str) or not isinstance(ground_truth, str):
            raise TypeError("Inputs must be strings")
            
        print(f"Comparing answers: Response length: {len(generated_answer)}, Ground truth length: {len(ground_truth)}")
            
        # Try the newer RAGAS API first with SemanticSimilarity
        if _semantic_similarity is not None:
            try:
                print(f"Using SemanticSimilarity from RAGAS (v{RAGAS_VERSION}) for evaluation")
                # Get embeddings from LangChain
                embeddings = get_langchain_embeddings()
                
                if embeddings is not None:
                    # Create data dictionary in format expected by RAGAS
                    # For RAGAS 0.1.x, use "reference", for RAGAS 0.0.x, might need "ground_truth"
                    data = {
                        "response": [generated_answer],
                        "reference": [ground_truth]
                    }
                    
                    # Print the actual data being sent to RAGAS
                    print(f"Sending data to RAGAS: {data}")
                    
                    # Initialize the semantic similarity metric with embeddings
                    semantic_sim = _semantic_similarity(embeddings=embeddings)
                    # Calculate the score
                    result = semantic_sim.score(data)
                    print(f"SemanticSimilarity evaluation completed successfully: {result}")
                    
                    # Extract the score from result - handle both dictionary and non-dictionary returns
                    if isinstance(result, dict) and "semantic_similarity" in result:
                        score = result["semantic_similarity"][0]
                    else:
                        score = result[0] if hasattr(result, "__getitem__") else result
                        
                    return float(score)
                else:
                    print("Unable to initialize embeddings, falling back to simple similarity")
            except Exception as e:
                print(f"Error with newer RAGAS API: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to the older API or simple similarity
        
        # Try the older RAGAS API with answer_similarity
        if _answer_similarity is not None:
            try:
                print("Using answer_similarity from older RAGAS version")
                # Use a try-except block with a simpler direct approach to avoid callback issues
                try:
                    # Completely avoid callbacks by using a direct import of sentence_transformers
                    from sentence_transformers import SentenceTransformer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    # Initialize the model
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Get embeddings for both texts
                    gen_embedding = model.encode(generated_answer)
                    truth_embedding = model.encode(ground_truth)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity([gen_embedding], [truth_embedding])[0][0]
                    print(f"Manual similarity calculation: {similarity}")
                    return float(similarity)
                except Exception as inner_e:
                    print(f"Manual similarity calculation failed: {inner_e}")
                    # Fall back to the RAGAS implementation
                    similarity = _answer_similarity.score([generated_answer], [ground_truth])
                    
                    # Handle different return types
                    if hasattr(similarity, '__iter__'):
                        first_value = similarity[0] if len(similarity) > 0 else 0.0
                        print(f"RAGAS answer_similarity score: {first_value}")
                        return float(first_value)
                    else:
                        print(f"RAGAS answer_similarity score: {similarity}")
                        return float(similarity)
            except Exception as e:
                print(f"Error with older RAGAS API: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to the simple similarity
        
        # If both RAGAS attempts failed, use simple similarity
        print("All RAGAS attempts failed, falling back to simple similarity calculation")
        return calculate_simple_similarity(generated_answer, ground_truth)
        
    except Exception as e:
        print(f"Unexpected error using RAGAS for evaluation: {e}")
        print("Falling back to simple similarity calculation")
        import traceback
        traceback.print_exc()
        return calculate_simple_similarity(generated_answer, ground_truth) 