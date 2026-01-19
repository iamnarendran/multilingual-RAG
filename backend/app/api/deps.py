"""
API Dependencies

Dependency injection functions for FastAPI endpoints.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Header

from app.config import settings, Settings
from app.agents.orchestrator import RAGOrchestrator
from app.core.vector_store import get_vector_store, VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# CONFIGURATION DEPENDENCY
# =============================================================================

def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings instance
    """
    return settings


# =============================================================================
# ORCHESTRATOR DEPENDENCY (Singleton)
# =============================================================================

_orchestrator: Optional[RAGOrchestrator] = None


def get_orchestrator() -> RAGOrchestrator:
    """
    Get RAG orchestrator instance (singleton).
    
    Returns:
        RAGOrchestrator instance
    """
    global _orchestrator
    
    if _orchestrator is None:
        logger.info("Initializing RAG Orchestrator...")
        _orchestrator = RAGOrchestrator()
        logger.info("✅ RAG Orchestrator ready")
    
    return _orchestrator


# =============================================================================
# VECTOR STORE DEPENDENCY
# =============================================================================

def get_vector_db() -> VectorStore:
    """
    Get vector store instance.
    
    Returns:
        VectorStore instance
    """
    return get_vector_store()


# =============================================================================
# AUTHENTICATION (Simple - Enhance for Production)
# =============================================================================

def get_current_user(
    x_user_id: Optional[str] = Header(None)
) -> str:
    """
    Get current user ID from header.
    
    In production, replace this with proper JWT authentication.
    
    Args:
        x_user_id: User ID from header
    
    Returns:
        User ID
    
    Raises:
        HTTPException: If authentication fails
    """
    if not x_user_id:
        # For now, use a default user if not provided
        # In production, this should raise an authentication error
        logger.warning("No user ID provided, using default")
        return "default_user"
    
    return x_user_id


# =============================================================================
# RATE LIMITING (Simple - Enhance for Production)
# =============================================================================

# In production, use a proper rate limiter like slowapi
# For now, this is a placeholder

def check_rate_limit(user_id: str = Depends(get_current_user)):
    """
    Check rate limit for user.
    
    In production, implement proper rate limiting with Redis.
    
    Args:
        user_id: User ID
    
    Raises:
        HTTPException: If rate limit exceeded
    """
    # TODO: Implement proper rate limiting
    pass
