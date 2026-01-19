"""
Base Agent Class

Provides common functionality for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import openai

from app.config import settings, get_model_config
from app.utils.logger import get_logger
from app.utils.exceptions import AgentError, LLMError
from app.utils.helpers import calculate_cost

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides:
    - LLM client management
    - Error handling
    - Logging
    - Cost tracking
    - Retry logic
    
    Subclasses must implement:
    - execute() method
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent (router, planner, etc.)
        """
        self.agent_name = agent_name
        self.logger = get_logger(f"Agent.{agent_name}")
        
        # Get model configuration for this agent
        self.model_config = get_model_config(agent_name)
        
        # Initialize OpenAI client (works with OpenRouter)
        self.client = openai.OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["base_url"]
        )
        
        # Stats tracking
        self.stats = {
            "calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
            "errors": 0,
        }
        
        self.logger.info(
            f"Initialized {agent_name} agent with model {self.model_config['model']}"
        )
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Agent-specific arguments
        
        Returns:
            Dictionary with execution results
        """
        pass
    
    def call_llm(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Call LLM with retry logic and error handling.
        
        Args:
            messages: List of message dicts
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            max_retries: Number of retries on failure
        
        Returns:
            LLM response text
        
        Raises:
            LLMError: If all retries fail
        """
        temperature = temperature or self.model_config["temperature"]
        max_tokens = max_tokens or self.model_config.get("max_tokens", 1000)
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                
                elapsed_time = time.time() - start_time
                
                # Extract response
                result = response.choices[0].message.content
                
                # Track usage
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                # Calculate cost
                cost = calculate_cost(
                    input_tokens,
                    output_tokens,
                    self.model_config["model"]
                )
                
                # Update stats
                self.stats["calls"] += 1
                self.stats["total_tokens"] += total_tokens
                self.stats["total_cost"] += cost
                self.stats["total_time"] += elapsed_time
                
                self.logger.info(
                    f"LLM call successful: {total_tokens} tokens, "
                    f"${cost:.4f}, {elapsed_time:.2f}s"
                )
                
                return result
                
            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit hit, retrying... (attempt {attempt + 1})")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except openai.APIError as e:
                self.logger.error(f"API error: {e}")
                if attempt == max_retries - 1:
                    self.stats["errors"] += 1
                    raise LLMError(f"API error after {max_retries} retries: {e}")
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.stats["errors"] += 1
                raise LLMError(f"LLM call failed: {e}")
        
        # If we get here, all retries failed
        self.stats["errors"] += 1
        raise LLMError(f"LLM call failed after {max_retries} retries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "agent_name": self.agent_name,
            "model": self.model_config["model"],
            **self.stats
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            "calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
            "errors": 0,
        }
        self.logger.info(f"Stats reset for {self.agent_name}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test base agent (can't instantiate directly, but can test methods)
    print("=" * 80)
    print("BASE AGENT CLASS")
    print("=" * 80)
    
    print("\n✅ BaseAgent class defined successfully")
    print("✅ Provides LLM calling, error handling, and stats tracking")
    print("✅ All agents will inherit from this base class")
