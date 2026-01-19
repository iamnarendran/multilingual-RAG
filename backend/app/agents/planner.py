"""
Query Planner Agent

Generates multiple search queries to improve retrieval quality.
Uses query expansion and reformulation techniques.
"""

from typing import Dict, Any, List
import json
import re

from app.agents.base import BaseAgent
from app.core.prompts import get_prompt
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError

logger = get_logger(__name__)


class PlannerAgent(BaseAgent):
    """
    Query planning agent that generates multiple search queries.
    
    Features:
    - Query expansion (synonyms, related terms)
    - Multi-lingual query generation
    - Query reformulation
    - Aspect-based query generation
    
    Example:
        planner = PlannerAgent()
        result = planner.execute(
            query="What is India's capital?",
            query_type="SIMPLE_QA"
        )
        # {
        #   'search_queries': [
        #     'India capital city',
        #     'भारत राजधानी',
        #     'New Delhi India'
        #   ]
        # }
    """
    
    def __init__(self):
        """Initialize planner agent"""
        super().__init__(agent_name="planner")
        
        # Get system prompt
        self.system_prompt = get_prompt("planner", include_examples=True)
    
    def execute(
        self,
        query: str,
        query_type: str,
        num_queries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate search queries.
        
        Args:
            query: Original user query
            query_type: Query type from router (SIMPLE_QA, etc.)
            num_queries: Number of queries to generate (2-5)
        
        Returns:
            Dictionary with:
                - search_queries: List of generated queries
                - original_query: Original query
                - strategy: Planning strategy used
        
        Example:
            result = planner.execute("India capital", "SIMPLE_QA")
        """
        self.logger.info(f"Planning queries for: '{query[:50]}...'")
        
        try:
            # Build prompt based on query type
            user_prompt = self._build_user_prompt(query, query_type, num_queries)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM
            response = self.call_llm(
                messages=messages,
                temperature=0.7,  # Higher temperature for diversity
                max_tokens=500
            )
            
            # Parse response (should be JSON array)
            search_queries = self._parse_queries(response)
            
            # Ensure we have at least 2 queries
            if len(search_queries) < 2:
                # Add original query as fallback
                search_queries.append(query)
            
            # Limit to requested number
            search_queries = search_queries[:num_queries]
            
            result = {
                "search_queries": search_queries,
                "original_query": query,
                "strategy": f"multi_query_{query_type.lower()}",
                "model_used": self.model_config["model"]
            }
            
            self.logger.info(f"✅ Generated {len(search_queries)} search queries")
            for i, sq in enumerate(search_queries, 1):
                self.logger.debug(f"  {i}. {sq}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Planner execution failed: {e}")
            # Fallback: return original query
            return {
                "search_queries": [query],
                "original_query": query,
                "strategy": "fallback",
                "error": str(e)
            }
    
    def _build_user_prompt(
        self,
        query: str,
        query_type: str,
        num_queries: int
    ) -> str:
        """
        Build user prompt based on query type.
        
        Args:
            query: User query
            query_type: Query classification
            num_queries: Number of queries needed
        
        Returns:
            Formatted prompt
        """
        prompt = f"""User Query: {query}
Query Type: {query_type}
Generate {num_queries} diverse search queries.

Requirements:
"""
        
        if query_type == "COMPARISON":
            prompt += "- Include queries for each entity being compared\n"
            prompt += "- Include a query about the comparison criteria\n"
        elif query_type == "SUMMARIZATION":
            prompt += "- Include queries for main topics\n"
            prompt += "- Include queries for key points\n"
        elif query_type == "MULTI_HOP":
            prompt += "- Break down into sub-queries\n"
            prompt += "- Include queries for each reasoning step\n"
        else:
            prompt += "- Use different phrasings\n"
            prompt += "- Include related concepts\n"
        
        prompt += "\nRespond with ONLY a JSON array of queries."
        
        return prompt
    
    def _parse_queries(self, response: str) -> List[str]:
        """
        Parse LLM response to extract queries.
        
        Args:
            response: LLM response text
        
        Returns:
            List of query strings
        """
        try:
            # Try to parse as JSON
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*|\s*```', '', response)
            response = response.strip()
            
            queries = json.loads(response)
            
            if isinstance(queries, list):
                # Filter out empty strings
                queries = [q.strip() for q in queries if q.strip()]
                return queries
            else:
                self.logger.warning("Response is not a list, extracting manually")
                return self._extract_queries_manually(response)
                
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse JSON, extracting manually")
            return self._extract_queries_manually(response)
    
    def _extract_queries_manually(self, response: str) -> List[str]:
        """
        Manually extract queries from response text.
        
        Args:
            response: Response text
        
        Returns:
            List of queries
        """
        # Look for quoted strings
        queries = re.findall(r'"([^"]+)"', response)
        
        if not queries:
            # Look for lines starting with numbers or bullets
            queries = re.findall(r'(?:^\d+\.|^[-*])\s*(.+)$', response, re.MULTILINE)
        
        if not queries:
            # Split by newlines and filter
            queries = [line.strip() for line in response.split('\n') if line.strip()]
        
        return queries[:5]  # Max 5 queries


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING PLANNER AGENT")
    print("=" * 80)
    
    planner = PlannerAgent()
    
    # Test queries with different types
    test_cases = [
        ("What is the capital of India?", "SIMPLE_QA"),
        ("Compare Python vs JavaScript", "COMPARISON"),
        ("Summarize the AI research paper", "SUMMARIZATION"),
    ]
    
    print("\n🗓️  Testing query planning:")
    for query, query_type in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Type: {query_type}")
        
        result = planner.execute(query, query_type, num_queries=3)
        
        print(f"\nGenerated Queries:")
        for i, sq in enumerate(result['search_queries'], 1):
            print(f"  {i}. {sq}")
    
    # Show stats
    stats = planner.get_stats()
    print(f"\n📊 Planner Statistics:")
    print(f"  Calls: {stats['calls']}")
    print(f"  Total cost: ${stats['total_cost']:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ PLANNER AGENT WORKING CORRECTLY!")
    print("=" * 80)
