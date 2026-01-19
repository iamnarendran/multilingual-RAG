"""
Multi-Agent System for Multilingual RAG

This package contains all agents and the orchestration logic:
- Router Agent: Classify query type
- Planner Agent: Generate search queries
- Retriever Agent: Execute search and rerank
- Analyzer Agent: Extract information from documents
- Synthesizer Agent: Combine information into final answer
- Validator Agent: Validate answer quality

Orchestrated using LangGraph for complex workflows.
"""

from app.agents.base import BaseAgent
from app.agents.router import RouterAgent
from app.agents.planner import PlannerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.analyzer import AnalyzerAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.validator import ValidatorAgent
from app.agents.orchestrator import RAGOrchestrator

__all__ = [
    "BaseAgent",
    "RouterAgent",
    "PlannerAgent",
    "RetrieverAgent",
    "AnalyzerAgent",
    "SynthesizerAgent",
    "ValidatorAgent",
    "RAGOrchestrator",
]
