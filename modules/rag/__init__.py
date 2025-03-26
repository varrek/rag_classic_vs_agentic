"""
RAG Implementations Package

This package contains different implementations of Retrieval-Augmented Generation (RAG),
including agentic and LangGraph approaches.
"""

from .agentic.original import run_agentic_rag
from .langgraph.implementation import run_langgraph_rag
from .types import AgentState, RAGResult
from .config import (
    MAX_ITERATIONS,
    INITIAL_TOP_K,
    ENABLE_PLANNING,
    ENABLE_SELF_CRITIQUE
)

__all__ = [
    'run_agentic_rag',
    'run_langgraph_rag',
    'AgentState',
    'RAGResult',
    'MAX_ITERATIONS',
    'INITIAL_TOP_K',
    'ENABLE_PLANNING',
    'ENABLE_SELF_CRITIQUE'
]
