"""
Document Management Endpoints

Upload, list, and delete documents.
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query
)
from typing import Optional, List
from datetime import datetime
import os

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    ErrorResponse
)
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import get_vector_store, VectorStore
from app.core.language_detector import detect_language
from app.api.deps import get_current_user, get_vector_db
from app.config import settings
from app.utils.logger import get_logger
from app.utils.exceptions import DocumentProcessingError, VectorStoreError
from app.utils.helpers import sanitize_filename, format_size

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload and process a document (PDF, DOCX, TXT, etc.)",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    metadata: Optional[str] = Form(None, description="Additional metadata (JSON string)"),
    user_id: str = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_db)
) -> DocumentUploadResponse:
    """
    Upload and process a document.
    
    **Supported Formats:**
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt, .md)
    - CSV (.csv)
    
    **Max Size:** 50MB per file
    
    **Processing:**
    1. Extract text from document
    2. Detect language
    3. Chunk into smaller pieces
    4. Generate embeddings
    5. Store in vector database
    
    Args:
        file: Document file
        metadata: Optional JSON metadata
        user_id: Current user ID
        vector_store: Vector store instance
    
    Returns:
        DocumentUploadResponse with document ID and stats
    
    Raises:
        HTTPException: If upload or processing fails
    
    Example:
```bash
        curl -X POST "http://localhost:8000/api/v1/documents/upload" \
          -H "X-User-Id: user123" \
          -F "file=@document.pdf"
```
    """
    logger.info(f"Upload request from user {user_id}: {file.filename}")
    
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1][1:].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '.{file_ext}' not supported. "
                       f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Read file
        file_bytes = await file.read()
        file_size = len(file_bytes)
        
        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({format_size(file_size)}). "
                       f"Max size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        logger.info(f"File size: {format_size(file_size)}")
        
        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        
        # Process document
        processor = DocumentProcessor()
        
        # Parse metadata if provided
        import json
        extra_metadata = {}
        if metadata:
            try:
                extra_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON, ignoring: {metadata}")
        
        chunks = processor.process_bytes(
            file_bytes=file_bytes,
            filename=safe_filename,
            user_id=user_id,
            additional_metadata=extra_metadata
        )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from document"
            )
        
        # Detect language from first chunk
        detected_language = detect_language(chunks[0].text)
        
        # Add chunks to vector store
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        vector_store.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=chunk_ids
        )
        
        # Get document ID (same for all chunks)
        document_id = chunks[0].metadata["document_id"]
        
        response = DocumentUploadResponse(
            document_id=document_id,
            filename=safe_filename,
            chunks_created=len(chunks),
            language=detected_language,
            message="Document uploaded and processed successfully"
        )
        
        logger.info(
            f"✅ Document uploaded: {document_id}, "
            f"{len(chunks)} chunks, language={detected_language}"
        )
        
        return response
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except VectorStoreError as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the document"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="List all documents for the current user"
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    user_id: str = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_db)
) -> DocumentListResponse:
    """
    List documents for the current user.
    
    Args:
        page: Page number
        page_size: Number of items per page
        user_id: Current user ID
        vector_store: Vector store instance
    
    Returns:
        DocumentListResponse with list of documents
    """
    logger.info(f"Listing documents for user {user_id}")
    
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # List documents from vector store
        docs = vector_store.list_documents(
            filters={"user_id": user_id},
            limit=page_size,
            offset=offset
        )
        
        # Group by document_id to get unique documents
        unique_docs = {}
        for doc in docs:
            doc_id = doc["metadata"].get("document_id")
            if doc_id and doc_id not in unique_docs:
                unique_docs[doc_id] = doc
        
        # Build document info list
        document_list = []
        for doc_id, doc in unique_docs.items():
            metadata = doc["metadata"]
            document_list.append(
                DocumentInfo(
                    document_id=doc_id,
                    filename=metadata.get("document_name", "Unknown"),
                    language=metadata.get("language", "unknown"),
                    chunks_count=metadata.get("total_chunks", 0),
                    uploaded_at=metadata.get("indexed_at", ""),
                    file_type=metadata.get("file_type", "unknown")
                )
            )
        
        response = DocumentListResponse(
            documents=document_list,
            total=len(document_list),
            page=page,
            page_size=page_size
        )
        
        logger.info(f"Found {len(document_list)} documents")
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete a document and all its chunks"
)
async def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
    vector_store: VectorStore = Depends(get_vector_db)
) -> dict:
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: Document ID to delete
        user_id: Current user ID
        vector_store: Vector store instance
    
    Returns:
        Success message
    
    Raises:
        HTTPException: If deletion fails
    """
    logger.info(f"Deleting document {
