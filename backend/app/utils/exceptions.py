"""
Custom Exception Classes

Define application-specific exceptions for better error handling.
"""

from typing import Any, Optional


class RAGException(Exception):
    """Base exception for RAG system"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Any] = None,
        status_code: int = 500
    ):
        self.message = message
        self.details = details
        self.status_code = status_code
        super().__init__(self.message)


class EmbeddingError(RAGException):
    """Exception raised when embedding generation fails"""
    
    def __init__(self, message: str = "Embedding generation failed", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class VectorStoreError(RAGException):
    """Exception raised for vector store operations"""
    
    def __init__(self, message: str = "Vector store operation failed", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class DocumentProcessingError(RAGException):
    """Exception raised when document processing fails"""
    
    def __init__(self, message: str = "Document processing failed", **kwargs):
        super().__init__(message, status_code=400, **kwargs)


class LanguageNotSupportedError(RAGException):
    """Exception raised when language is not supported"""
    
    def __init__(self, language: str, **kwargs):
        message = f"Language '{language}' is not supported"
        super().__init__(message, status_code=400, **kwargs)


class AgentError(RAGException):
    """Exception raised when agent execution fails"""
    
    def __init__(self, agent_name: str, message: str, **kwargs):
        full_message = f"Agent '{agent_name}' failed: {message}"
        super().__init__(full_message, status_code=500, **kwargs)


class LLMError(RAGException):
    """Exception raised when LLM API call fails"""
    
    def __init__(self, message: str = "LLM API call failed", **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class DocumentNotFoundError(RAGException):
    """Exception raised when document is not found"""
    
    def __init__(self, doc_id: str, **kwargs):
        message = f"Document '{doc_id}' not found"
        super().__init__(message, status_code=404, **kwargs)


class ValidationError(RAGException):
    """Exception raised when validation fails"""
    
    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(message, status_code=400, **kwargs)


class AuthenticationError(RAGException):
    """Exception raised for authentication failures"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(RAGException):
    """Exception raised for authorization failures"""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class RateLimitError(RAGException):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)
