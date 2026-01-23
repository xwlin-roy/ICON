"""
Cost tracking module for LLM API calls
Tracks API query count and token consumption for each sample
"""
from typing import Dict, Any, Optional


class CostTracker:
    """
    Track LLM API query count and token consumption for a single sample
    """
    
    def __init__(self):
        """Initialize cost tracker"""
        self.api_query_count = 0  # Total number of LLM API calls
        self.total_prompt_tokens = 0  # Total prompt tokens consumed
        self.total_completion_tokens = 0  # Total completion tokens consumed
        self.total_tokens = 0  # Total tokens consumed (prompt + completion)
        
        # Breakdown by component
        self.router_queries = 0
        self.router_tokens = 0
        
        self.generator_queries = 0
        self.generator_tokens = 0
        
        self.target_llm_queries = 0  # Target LLM (attacker) interactions
        self.target_llm_tokens = 0
        
        self.judge_queries = 0
        self.judge_tokens = 0
        
        self.reflector_queries = 0  # Inner loop (tactical) + Outer loop (strategic)
        self.reflector_tokens = 0
    
    def record_api_call(
        self,
        component: str,
        token_usage: Optional[Dict[str, Any]] = None
    ):
        """
        Record an API call and its token usage
        
        Args:
            component: Component name ('router', 'generator', 'target_llm', 'judge', 'reflector')
            token_usage: Token usage dictionary with keys: 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        # Increment API query count
        self.api_query_count += 1
        
        # Extract token counts
        if token_usage:
            prompt_tokens = token_usage.get('prompt_tokens', 0) or 0
            completion_tokens = token_usage.get('completion_tokens', 0) or 0
            total_tokens = token_usage.get('total_tokens', 0) or 0
            
            # If total_tokens is not provided, calculate it
            if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
                total_tokens = prompt_tokens + completion_tokens
            
            # Update total counts
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            
            # Update component-specific counts
            if component == 'router':
                self.router_queries += 1
                self.router_tokens += total_tokens
            elif component == 'generator':
                self.generator_queries += 1
                self.generator_tokens += total_tokens
            elif component == 'target_llm':
                self.target_llm_queries += 1
                self.target_llm_tokens += total_tokens
            elif component == 'judge':
                self.judge_queries += 1
                self.judge_tokens += total_tokens
            elif component == 'reflector':
                self.reflector_queries += 1
                self.reflector_tokens += total_tokens
        else:
            # No token usage info, but still count the API call
            if component == 'router':
                self.router_queries += 1
            elif component == 'generator':
                self.generator_queries += 1
            elif component == 'target_llm':
                self.target_llm_queries += 1
            elif component == 'judge':
                self.judge_queries += 1
            elif component == 'reflector':
                self.reflector_queries += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get cost summary for this sample
        
        Returns:
            Dictionary containing cost statistics
        """
        return {
            'api_query_count': self.api_query_count,
            'total_tokens': self.total_tokens,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'breakdown': {
                'router': {
                    'queries': self.router_queries,
                    'tokens': self.router_tokens
                },
                'generator': {
                    'queries': self.generator_queries,
                    'tokens': self.generator_tokens
                },
                'target_llm': {
                    'queries': self.target_llm_queries,
                    'tokens': self.target_llm_tokens
                },
                'judge': {
                    'queries': self.judge_queries,
                    'tokens': self.judge_tokens
                },
                'reflector': {
                    'queries': self.reflector_queries,
                    'tokens': self.reflector_tokens
                }
            }
        }
    
    def reset(self):
        """Reset all counters"""
        self.api_query_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        
        self.router_queries = 0
        self.router_tokens = 0
        
        self.generator_queries = 0
        self.generator_tokens = 0
        
        self.target_llm_queries = 0
        self.target_llm_tokens = 0
        
        self.judge_queries = 0
        self.judge_tokens = 0
        
        self.reflector_queries = 0
        self.reflector_tokens = 0

