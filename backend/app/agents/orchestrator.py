"""
RAG Orchestrator using LangGraph

Coordinates all agents in a stateful workflow to process queries end-to-end.
Uses LangGraph for complex multi-agent orchestration.
"""

from typing import Dict, Any, TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END
import time

from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.analyzer import AnalyzerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.validator import ValidatorAgent
from app.core.language_detector import detect_with_confidence
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class RAGState(TypedDict):
    """
    State object passed between agents in the workflow.
    
    Each agent reads from and writes to this state.
    """
    # Input
    query: str
    user_id: str
    filters: Dict[str, Any]
    
    # Language detection
    language: str
    language_confidence: float
    
    # Router output
    query_type: str
    
    # Planner output
    search_queries: List[str]
    
    # Retriever output
    documents: List[Dict[str, Any]]
    total_retrieved: int
    
    # Analyzer output
    analysis: str
    sources_used: List[str]
    analysis_confidence: float
    
    # Synthesizer output
    answer: str
    citations: List[Dict[str, Any]]
    
    # Validator output
    validation_valid: bool
    validation_confidence: float
    validation_issues: List[str]
    
    # Metadata
    processing_time: float
    total_cost: float
    agent_stats: Dict[str, Any]
    errors: Annotated[List[str], operator.add]


# =============================================================================
# RAG ORCHESTRATOR
# =============================================================================

class RAGOrchestrator:
    """
    Main orchestrator for the RAG system.
    
    Coordinates all agents using LangGraph to process queries from start to finish.
    
    Workflow:
    1. Detect language
    2. Route query (classify type)
    3. Plan queries (generate search queries)
    4. Retrieve documents
    5. Analyze documents
    6. Synthesize answer
    7.
