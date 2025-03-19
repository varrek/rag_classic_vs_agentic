"""
RAG Factory Module

This module provides a factory pattern for selecting between different RAG implementations.
It allows for standardized access to various RAG approaches through a unified interface.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import available implementations
try:
    from modules.rag.agentic.original import run_agentic_rag
    agentic_available = True
except ImportError as e:
    logger.error(f"Error importing agentic RAG: {e}")
    agentic_available = False

try:
    from modules.rag.langgraph.implementation import run_langgraph_rag
    langgraph_available = True
except ImportError as e:
    logger.error(f"Error importing LangGraph RAG: {e}")
    langgraph_available = False

# Try to import classic implementation (optional)
try:
    from modules.rag.classic import run_classic_rag
    classic_available = True
except ImportError:
    logger.info("Classic RAG implementation not available")
    classic_available = False

class RAGFactory:
    """Factory class for managing and running different RAG implementations."""
    
    @staticmethod
    def list_available_implementations() -> List[str]:
        """
        List all available RAG implementations.
        
        Returns:
            List of implementation names
        """
        implementations = []
        
        if agentic_available:
            implementations.append("agentic")
        
        if langgraph_available:
            implementations.append("langgraph")
            
        if classic_available:
            implementations.append("classic")
            
        return implementations
    
    @staticmethod
    def run_rag(
        query: str,
        implementation: str = "agentic",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the specified RAG implementation with the given query.
        
        Args:
            query: The user query
            implementation: Which implementation to use ("agentic", "langgraph", "classic")
            config: Optional configuration parameters
            
        Returns:
            Dict containing the answer and metadata
            
        Raises:
            ValueError: If the specified implementation is not available
        """
        if not query or not query.strip():
            return {
                "query": query,
                "answer": "No question was provided.",
                "error": "Empty query",
                "implementation": implementation
            }
        
        if config is None:
            config = {}
            
        # Standardize and record basic metadata
        start_time = time.time()
        result = {
            "query": query,
            "implementation": implementation,
            "config": config
        }
        
        try:
            # Route to the appropriate implementation
            if implementation == "agentic":
                if not agentic_available:
                    raise ValueError("Agentic RAG implementation is not available")
                
                logger.info(f"Running agentic RAG implementation with query: {query}")
                rag_result = run_agentic_rag(query, config)
                
            elif implementation == "langgraph":
                if not langgraph_available:
                    raise ValueError("LangGraph RAG implementation is not available")
                
                logger.info(f"Running LangGraph RAG implementation with query: {query}")
                rag_result = run_langgraph_rag(query, config)
                
            elif implementation == "classic":
                if not classic_available:
                    raise ValueError("Classic RAG implementation is not available")
                
                logger.info(f"Running classic RAG implementation with query: {query}")
                rag_result = run_classic_rag(query, config)
                
            else:
                available_impls = RAGFactory.list_available_implementations()
                raise ValueError(f"Unknown implementation: {implementation}. Available implementations: {available_impls}")
            
            # Combine the implementation result with our metadata
            result.update(rag_result)
            
        except Exception as e:
            logger.error(f"Error running {implementation} RAG: {e}")
            import traceback
            
            # Record the error in the result
            result["error"] = str(e)
            result["error_traceback"] = traceback.format_exc()
            result["answer"] = f"I encountered an error while processing your question: {str(e)}"
        
        # Add timing information
        result["total_time"] = time.time() - start_time
        
        return result
    
    @staticmethod
    def compare_implementations(
        query: str,
        implementations: List[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run multiple RAG implementations and compare their results.
        
        Args:
            query: The user query
            implementations: List of implementations to compare (defaults to all available)
            config: Optional configuration parameters
            
        Returns:
            Dict containing the results from each implementation and comparison metrics
        """
        if implementations is None:
            implementations = RAGFactory.list_available_implementations()
            
        if not implementations:
            return {
                "query": query,
                "error": "No RAG implementations available",
                "results": {}
            }
            
        if config is None:
            config = {}
            
        start_time = time.time()
        
        # Run each implementation
        results = {}
        for impl in implementations:
            try:
                results[impl] = RAGFactory.run_rag(query, impl, config)
            except Exception as e:
                logger.error(f"Error running {impl} implementation: {e}")
                import traceback
                results[impl] = {
                    "query": query,
                    "implementation": impl,
                    "error": str(e),
                    "error_traceback": traceback.format_exc(),
                    "answer": f"Error with {impl} implementation: {str(e)}"
                }
        
        # Compare results if we have multiple implementations
        comparison = {}
        if len(results) > 1:
            try:
                # Import evaluation module for comparing answers
                from modules.evaluation import compute_similarity, compare_answers
                
                for i, impl1 in enumerate(implementations):
                    for impl2 in implementations[i+1:]:
                        if "answer" in results[impl1] and "answer" in results[impl2]:
                            answer1 = results[impl1]["answer"]
                            answer2 = results[impl2]["answer"]
                            
                            # Calculate similarity between answers
                            similarity, explanation = compare_answers(answer1, answer2)
                            
                            comparison[f"{impl1}_vs_{impl2}"] = {
                                "similarity": similarity,
                                "explanation": explanation
                            }
            except ImportError:
                logger.warning("Evaluation module not available, skipping answer comparison")
        
        return {
            "query": query,
            "results": results,
            "comparison": comparison,
            "total_time": time.time() - start_time
        }
        
# Convenience function for direct access
def run_rag(
    query: str,
    implementation: str = "agentic",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the specified RAG implementation with the given query.
    
    Args:
        query: The user query
        implementation: Which implementation to use
        config: Optional configuration parameters
        
    Returns:
        Dict containing the answer and metadata
    """
    return RAGFactory.run_rag(query, implementation, config) 