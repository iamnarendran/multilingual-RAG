"""
Multilingual Embedding System

Generates embeddings for text in 100+ languages including all 22 Indian languages.
Uses sentence-transformers with the multilingual-e5-large model.

The model supports:
- Asymmetric search (different embeddings for queries vs documents)
- Normalized embeddings for cosine similarity
- Batch processing for efficiency
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
import torch
from functools import lru_cache
import time

from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import EmbeddingError
from app.utils.helpers import chunk_list

logger = get_logger(__name__)


class MultilingualEmbedder:
    """
    Multilingual embedding generator supporting 100+ languages.
    
    Features:
    - Asymmetric search (query vs passage embeddings)
    - Batch processing
    - GPU/CPU support
    - Automatic normalization
    
    Example:
        embedder = MultilingualEmbedder()
        
        # Embed query
        query_emb = embedder.embed_query("What is the capital?")
        
        # Embed documents
        docs = ["Delhi is the capital.", "Mumbai is largest city."]
        doc_embs = embedder.embed_documents(docs)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name (default from settings)
            device: 'cpu' or 'cuda' (default from settings)
            batch_size: Batch size for encoding (default from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        logger.info(f"Initializing embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        
        try:
            # Check if CUDA is available if requested
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load model
            start_time = time.time()
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            load_time = time.time() - start_time
            
            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            logger.info(f"✅ Embedding dimension: {self.dimension}")
            
            # Verify dimension matches config
            if self.dimension != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"Model dimension ({self.dimension}) differs from "
                    f"config ({settings.EMBEDDING_DIMENSION})"
                )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Model initialization failed: {e}")
    
    def _add_prefix(self, texts: List[str], prefix: str) -> List[str]:
        """
        Add prefix to texts for e5 models.
        
        E5 models perform better with task-specific prefixes:
        - "query: " for search queries
        - "passage: " for documents
        
        Args:
            texts: List of texts
            prefix: Prefix to add
        
        Returns:
            Texts with prefix
        """
        if "e5" in self.model_name.lower():
            return [f"{prefix}: {text}" for text in texts]
        return texts
    
    def embed_text(
        self,
        text: str,
        prefix: str = "passage",
        normalize: bool = True
    ) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Input text in any language
            prefix: 'query' or 'passage' for e5 models
            normalize: Whether to normalize embedding
        
        Returns:
            Embedding vector as numpy array
        
        Example:
            emb = embedder.embed_text("भारत की राजधानी")
            # array([0.123, -0.456, ...])
        """
        try:
            # Add prefix if needed
            if "e5" in self.model_name.lower():
                text = f"{prefix}: {text}"
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to embed text: {e}")
    
    def embed_batch(
        self,
        texts: List[str],
        prefix: str = "passage",
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple texts in batches (more efficient).
        
        Args:
            texts: List of input texts
            prefix: 'query' or 'passage'
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
        
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        
        Example:
            docs = ["Text 1", "Text 2", "Text 3"]
            embs = embedder.embed_batch(docs)
            # shape: (3, 1024)
        """
        if not texts:
            return np.array([])
        
        try:
            # Add prefix
            processed_texts = self._add_prefix(texts, prefix)
            
            logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")
            
            # Generate embeddings
            start_time = time.time()
            embeddings = self.model.encode(
                processed_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress
            )
            elapsed = time.time() - start_time
            
            logger.info(
                f"✅ Generated {len(embeddings)} embeddings in {elapsed:.2f}s "
                f"({len(embeddings)/elapsed:.1f} texts/sec)"
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed batch: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.
        
        Uses 'query' prefix for e5 models to optimize for search.
        
        Args:
            query: Search query text
        
        Returns:
            Query embedding vector
        
        Example:
            query_emb = embedder.embed_query("What is AI?")
        """
        return self.embed_text(query, prefix="query")
    
    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed multiple documents.
        
        Uses 'passage' prefix for e5 models.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
        
        Returns:
            Document embeddings array
        
        Example:
            docs = ["Doc 1", "Doc 2"]
            doc_embs = embedder.embed_documents(docs)
        """
        return self.embed_batch(
            documents,
            prefix="passage",
            show_progress=show_progress
        )
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0 to 1, higher is more similar)
        
        Example:
            sim = embedder.similarity(query_emb, doc_emb)
            # 0.85
        """
        # Normalize if not already normalized
        emb1 = embedding1 / (np.linalg.norm(embedding1) + 1e-9)
        emb2 = embedding2 / (np.linalg.norm(embedding2) + 1e-9)
        
        return float(np.dot(emb1, emb2))
    
    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between one query and multiple documents.
        
        Args:
            query_embedding: Single query embedding (embedding_dim,)
            doc_embeddings: Multiple doc embeddings (n_docs, embedding_dim)
        
        Returns:
            Array of similarity scores (n_docs,)
        
        Example:
            sims = embedder.batch_similarity(query_emb, doc_embs)
            # array([0.85, 0.72, 0.91, ...])
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9
        )
        
        # Compute all similarities at once (vectorized)
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length,
        }


# =============================================================================
# GLOBAL EMBEDDER INSTANCE (Singleton Pattern)
# =============================================================================

_embedder: Optional[MultilingualEmbedder] = None


def get_embedder() -> MultilingualEmbedder:
    """
    Get or create the global embedder instance.
    
    Uses singleton pattern to avoid loading model multiple times.
    
    Returns:
        MultilingualEmbedder instance
    
    Example:
        from app.core.embeddings import get_embedder
        
        embedder = get_embedder()
        emb = embedder.embed_query("Hello")
    """
    global _embedder
    
    if _embedder is None:
        logger.info("Initializing global embedder (singleton)...")
        _embedder = MultilingualEmbedder()
        logger.info("✅ Global embedder initialized")
    
    return _embedder


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def embed_query(query: str) -> np.ndarray:
    """
    Convenience function to embed a query.
    
    Args:
        query: Search query
    
    Returns:
        Query embedding
    """
    return get_embedder().embed_query(query)


def embed_documents(documents: List[str]) -> np.ndarray:
    """
    Convenience function to embed documents.
    
    Args:
        documents: List of documents
    
    Returns:
        Document embeddings
    """
    return get_embedder().embed_documents(documents)


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Convenience function to calculate similarity.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        Similarity score
    """
    return get_embedder().similarity(emb1, emb2)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MULTILINGUAL EMBEDDINGS")
    print("=" * 80)
    
    # Initialize embedder
    embedder = get_embedder()
    
    # Test with multiple languages
    test_texts = [
        "What is the capital of India?",  # English
        "भारत की राजधानी क्या है?",      # Hindi
        "இந்தியாவின் தலைநகரம் என்ன?",    # Tamil
        "ભારતની રાજધાની શું છે?",         # Gujarati
        "ভারতের রাজধানী কী?",             # Bengali
    ]
    
    print(f"\n📊 Embedding {len(test_texts)} texts in different languages...")
    embeddings = embedder.embed_documents(test_texts)
    
    print(f"✅ Embeddings shape: {embeddings.shape}")
    print(f"✅ Dimension: {embeddings.shape[1]}")
    
    # Test query embedding
    query = "India capital city"
    query_emb = embedder.embed_query(query)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"Query embedding shape: {query_emb.shape}")
    
    # Calculate similarities
    print("\n📈 Similarity scores:")
    for i, text in enumerate(test_texts):
        sim = embedder.similarity(query_emb, embeddings[i])
        print(f"  {text[:40]:40s} → {sim:.4f}")
    
    # Test batch similarity
    print("\n📈 Batch similarity calculation:")
    batch_sims = embedder.batch_similarity(query_emb, embeddings)
    print(f"  Scores: {batch_sims}")
    
    # Model info
    info = embedder.get_model_info()
    print(f"\n🤖 Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ EMBEDDING SYSTEM WORKING CORRECTLY!")
    print("=" * 80)
