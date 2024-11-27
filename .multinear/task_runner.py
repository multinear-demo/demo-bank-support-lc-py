"""
Multinear platform integration module for the Bank Customer Support app.
This module serves as the entry point for the Multinear platform,
allowing testing and evaluation of the RAG system.

It provides:
- A singleton RAG engine instance for consistent testing
- Task execution interface (entry point) for the Multinear platform
- Integration with the main application's configuration and tracing
"""

import sys
from pathlib import Path
import asyncio
from dotenv import load_dotenv


# Add parent directory to Python path so we can import engine
sys.path.append(str(Path(__file__).parent.parent))
# flake8: noqa: E402
from engine import RAGEngine
from tracing import init_tracing

# Singleton instance
_rag_engine = None


def _get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        load_dotenv()
        init_tracing()
        _rag_engine = RAGEngine()
        _rag_engine.refresh_index()
    return _rag_engine


def run_task(input: str) -> dict:
    """
    Execute a task with the given input using the RAG engine.

    Args:
        input (str): The user query or input to process.

    Returns:
        dict: A dictionary containing:
            - 'output' (str): The AI's response.
            - 'details' (dict): Metadata (the model and temperature used).
    """
    engine = _get_rag_engine()
    response, _ = asyncio.run(engine.process_query([(input, True)]))

    return {
        'output': response,
        'details': {
            'model': engine.model,
            'temperature': engine.temperature,
        }
    }
