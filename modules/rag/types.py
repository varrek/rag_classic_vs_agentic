"""Shared type definitions for RAG implementations"""

from typing import TypedDict, List, Optional, Dict, Any, Union, Literal
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """State for the agentic RAG system"""
    messages: List[BaseMessage]
    documents: List[Document]
    current_iteration: int
    is_sufficient: bool
    refined_query: Optional[str]
    final_answer: Optional[str]

class RetrievalResult(TypedDict):
    """Result from a retrieval operation"""
    documents: List[Document]
    metadata: Dict[str, Any]

class AnalysisResult(TypedDict):
    """Result from context analysis"""
    is_sufficient: bool
    refined_query: Optional[str]
    full_analysis: str

class RAGResult(TypedDict):
    """Final result from RAG processing"""
    query: str
    answer: str
    refined_query: Optional[str]
    num_documents: int
    iterations: int
    time_taken: float
    metadata: Dict[str, Any]
    error: Optional[str] 