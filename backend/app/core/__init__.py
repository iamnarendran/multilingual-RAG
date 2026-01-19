"""
Core components for the Multilingual RAG System

This package contains the fundamental building blocks:
- Embeddings: Multilingual text embedding generation
- Vector Store: Qdrant integration for semantic search
- Document Processor: Extract and chunk documents
- Language Detector: Detect document language
- Reranker: Rerank search results
"""

from app.core.embeddings import MultilingualEmbedder, get_embedder
from app.core.vector_store import VectorStore, get_vector_store

__all__ = [
    "MultilingualEmbedder",
    "get_embedder",
    "VectorStore",
    "get_vector_store",
]
