"""Logger configuration for the RAG system.

Provides structured logging with timestamps and log levels.
"""
import logging


def get_logger(name: str):
    """Get or create a logger with standard formatting.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    return logging.getLogger(name)
