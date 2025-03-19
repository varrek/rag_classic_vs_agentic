"""
Utilities Module

This module contains common utility functions used across the RAG system.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def suppress_streamlit_thread_warnings():
    """
    Suppress the 'missing ScriptRunContext' warnings from ThreadPoolExecutor threads.
    These warnings are harmless when running in 'bare mode'.
    """
    class ThreadWarningFilter(logging.Filter):
        def filter(self, record):
            # Filter out messages containing both these patterns
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                if "ThreadPoolExecutor" in record.msg and "missing ScriptRunContext" in record.msg:
                    return False
            return True
    
    # Add the filter to the root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(ThreadWarningFilter())
    logger.info("Applied filter for Streamlit ThreadPoolExecutor warnings")

def check_api_key() -> bool:
    """
    Check if the OpenAI API key is set.
    
    Returns:
        bool: True if the key is set, False otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return bool(api_key)

def setup_directory_structure() -> None:
    """
    Create necessary directories for the application.
    """
    # Core directories
    directories = [
        "data",
        "data/qa",
        "index",
        "logs"
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            logger.info(f"Creating directory: {directory}")
            path.mkdir(parents=True, exist_ok=True)

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Set up logging to file
    log_file = Path("logs/rag_app.log")
    if not log_file.parent.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Map string level to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Filter out noisy logs
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('chroma').setLevel(logging.WARNING)
    logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Used by Chroma
    logging.getLogger('httpcore').setLevel(logging.WARNING)  # Used by Chroma
    
    # Filter RAGAS and other evaluation related logs
    logging.getLogger('ragas').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized at level {log_level}")

def disable_pytorch_warnings() -> None:
    """
    Disable PyTorch warnings and configure it for minimal resource usage.
    """
    try:
        import torch
        
        # Disable gradients to save memory
        torch.set_grad_enabled(False)
        
        # Set number of threads to 1 for lower resource usage
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)
        
        # Set environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        
        logger.info("PyTorch configured for minimal resource usage")
    except ImportError:
        logger.warning("PyTorch not available")
    except Exception as e:
        logger.error(f"Error configuring PyTorch: {e}")

def fix_torch_module_scanning() -> None:
    """
    Fix for PyTorch module scanning issue in Streamlit.
    This modifies sys.modules to prevent Streamlit from scanning torch.classes.
    """
    try:
        import sys
        import torch
        
        # Create a custom module to prevent _path access
        class CustomModule:
            def __init__(self):
                self.__path__ = None
                
        # Patch torch.classes to prevent Streamlit scanning errors
        if 'torch.classes' in sys.modules:
            sys.modules['torch.classes.__path__'] = CustomModule()
            logger.info("Applied torch.classes scanning fix for Streamlit")
    except ImportError:
        logger.warning("PyTorch not available, skipping module scanning fix")
    except Exception as e:
        logger.error(f"Error applying torch module scanning fix: {e}")

def initialize_app() -> None:
    """
    Initialize the application with all necessary setup.
    """
    # Create directory structure
    setup_directory_structure()
    
    # Set up logging
    setup_logging()
    
    # Apply PyTorch fixes
    disable_pytorch_warnings()
    fix_torch_module_scanning()
    
    # Suppress Streamlit thread warnings
    suppress_streamlit_thread_warnings()
    
    logger.info("Application initialization complete") 