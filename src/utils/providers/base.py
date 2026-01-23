"""
LLM Provider base interface
Defines interfaces that all LLM Providers must implement
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class BaseLLMProvider(ABC):
    """LLM Provider base abstract class, all concrete Providers need to inherit this class"""
    
    def __init__(self, api_key: str, api_base: Optional[str] = None, **kwargs):
        """
        Initialize Provider
        
        Args:
            api_key: API key
            api_base: API base URL (optional)
            **kwargs: Other Provider-specific parameters
        """
        # Clean API key: remove any whitespace, newlines, quotes, etc.
        if api_key and isinstance(api_key, str):
            api_key = api_key.strip()
            # Remove quotes if present (both single and double)
            if (api_key.startswith('"') and api_key.endswith('"')) or \
               (api_key.startswith("'") and api_key.endswith("'")):
                api_key = api_key[1:-1].strip()
            # Remove any remaining whitespace characters
            api_key = api_key.replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
        
        self.api_key = api_key
        self.api_base = api_base
        self._client = None
        self._initialize_client(**kwargs)
    
    @abstractmethod
    def _initialize_client(self, **kwargs):
        """
        Initialize specific API client
        
        Args:
            **kwargs: Provider-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text response
        
        Args:
            messages: Conversation message list, format: [{"role": "user/assistant/system", "content": "..."}]
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum token count
            **kwargs: Other generation parameters
        
        Returns:
            Dictionary containing response content and metadata, format:
            {
                "content": str,  # Response text content
                "usage": Optional[Dict[str, int]],  # Token usage (if available)
                "model": str  # Actually used model name
            }
        """
        pass
    
    def normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize message format (some Providers may need specific role names)
        
        Args:
            messages: Original message list
        
        Returns:
            Normalized message list
        """
        # Default implementation: return directly, subclasses can override this method
        return messages

