"""
Query Endpoints

Main RAG query processing endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.agents.orchestrator import RAGOrchestrator
from app.api.deps import get_orchestrator, get_current_user, check_rate_limit
from app.utils.logger import get_logger
from app.utils.exceptions import RAGException

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Query",
    description="Process a natural language query and return answer with citations",
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def process_query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
    orchestrator: RAGOrchestrator = Depends(get_orchestrator),
    _: None = Depends(check_rate_limit)
) -> QueryResponse:
    """
    Process a query through the RAG pipeline.
    
    **Workflow:**
    1. Detect language
    2. Classify query type
    3. Generate search queries
    4. Retrieve relevant documents
    5. Analyze documents
    6. Synthesize answer
    7. Validate quality
    
    **Supported Languages:** 22+ Indian languages + English
    
    **Query Types:**
    - SIMPLE_QA: Direct factual questions
    - COMPARISON: Compare entities
    - SUMMARIZATION: Summarize documents
    - ANALYSIS: Deep analysis
    - EXTRACTION: Extract structured data
    - MULTI_HOP: Multi-step reasoning
    
    Args:
        request: Query request with query text and options
        user_id: Current user ID (from header)
        orchestrator: RAG orchestrator instance
    
    Returns:
        QueryResponse with answer, citations, and metadata
    
    Raises:
        HTTPException: If query processing fails
    
    Example:
```python
        POST /api/v1/query
        {
            "query": "What is the capital of India?",
            "language": "auto",
            "top_k": 5
        }
```
    """
    logger.info(f"Processing query from user {user_id}: '{request.query[:50]}...'")
    
    try:
        # Add user_id to filters
        filters = request.filters or {}
        filters["user_id"] = user_id
        
        # Process query through orchestrator
        result = orchestrator.process_query(
            query=request.query,
            user_id=user_id,
            filters=filters
        )
        
        # Build response
        response = QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            language=result["language"],
            query_type=result["query_type"],
            citations=result["citations"],
            sources=result["sources"] if request.include_sources else None,
            metadata=result["metadata"]
        )
        
        logger.info(
            f"Query processed successfully: "
            f"confidence={response.confidence:.2f}, "
            f"time={result['metadata']['processing_time']:.2f}s"
        )
        
        return response
        
    except RAGException as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": e.__class__.__name__,
                "message": e.message,
                "details": e.details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing your query",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get(
    "/stats",
    summary="Get Query Statistics",
    description="Get statistics about query processing"
)
async def get_query_stats(
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> dict:
    """
    Get statistics about query processing.
    
    Returns:
        Statistics for all agents
    """
    return orchestrator._get_all_stats()


@router.post(
    "/stats/reset",
    summary="Reset Statistics",
    description="Reset all agent statistics (admin only)"
)
async def reset_stats(
    orchestrator: RAGOrchestrator = Depends(get_orchestrator)
) -> dict:
    """
    Reset all agent statistics.
    
    Returns:
        Success message
    """
    orchestrator.reset_stats()
    return {"message": "Statistics reset successfully"}
