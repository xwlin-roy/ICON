"""
DeepSeek API Provider adapter
DeepSeek API supports calling via official API or Alibaba Cloud Bailian platform, using OpenAI compatible format
Supports calling DeepSeek series models via Alibaba Cloud Bailian API (e.g., deepseek-v3.2, deepseek-v3, etc.)
"""
import logging
from typing import Dict, Any, List
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class DeepSeekProvider(OpenAIProvider):
    """
    DeepSeek API Provider
    
    Supports calling DeepSeek series models via:
    1. Alibaba Cloud Bailian platform (recommended): Use Alibaba Cloud Bailian API key and base URL
    2. DeepSeek official API: Use DeepSeek official API key and base URL
    
    Uses OpenAI compatible format API, inherits from OpenAIProvider, supports all OpenAI compatible features, including:
    - Proxy support (HTTP/SOCKS5)
    - Automatic retry mechanism
    - Error handling
    - Token usage statistics
    
    Supported models (via Alibaba Cloud Bailian):
    - deepseek-v3.2: DeepSeek V3.2 version
    - deepseek-v3: DeepSeek V3 version
    - deepseek-chat: DeepSeek Chat version
    - deepseek-coder: DeepSeek Coder version
    - Other DeepSeek series models
    
    Configuration:
    - Alibaba Cloud Bailian API Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
    - DeepSeek official API Base URL: https://api.deepseek.com/v1
    - Need to configure api_key and api_base in config.json
    - If using Alibaba Cloud Bailian, api_base should be set to Alibaba Cloud Bailian URL
    """
    
    def _initialize_client(self, **kwargs):
        """
        Initialize DeepSeek client (using OpenAI compatible format)
        
        Args:
            **kwargs: Additional initialization parameters, including:
                - config_data: Configuration data dictionary, can be used to pass proxy configuration, etc.
        """
        import openai
        import httpx
        
        # If api_base is not specified, use DeepSeek official default endpoint
        # Note: If using Alibaba Cloud Bailian, should explicitly specify api_base in config.json
        if not self.api_base:
            self.api_base = "https://api.deepseek.com/v1"
            logger.info("DeepSeekProvider: Using default API base URL (DeepSeek official)")
        else:
            # Check if using Alibaba Cloud Bailian
            if "dashscope.aliyuncs.com" in self.api_base:
                logger.info("DeepSeekProvider: Using Alibaba Cloud Bailian API")
            else:
                logger.info("DeepSeekProvider: Using DeepSeek official API")
        
        # If using Alibaba Cloud Bailian API, do not use proxy, access directly via domestic network
        # If using DeepSeek official API, may need proxy, use parent class method
        if "dashscope.aliyuncs.com" in self.api_base:
            # Alibaba Cloud Bailian API: do not use proxy
            http_client = httpx.Client(
                proxy=None,  # Explicitly do not use proxy
                timeout=httpx.Timeout(300.0, connect=30.0)  # 5 minute read timeout, 30 second connect timeout
            )
            
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client
            )
            
            logger.info("DeepSeekProvider: Initialization complete (Alibaba Cloud Bailian, no proxy, direct access via domestic network)")
        else:
            # DeepSeek official API: call parent class method (may use proxy)
            super()._initialize_client(**kwargs)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DeepSeekProvider initialization complete: api_base={self.api_base}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call DeepSeek API to generate response with comprehensive error handling and retry mechanism
        
        Inherits from OpenAIProvider, already has:
        - Exponential backoff retry mechanism (default max 5 retries)
        - Automatic retry for network errors (connection/timeout errors)
        - Rate limit error retry (429)
        - Reasonable delay strategy: 1s, 2s, 4s, 8s, 16s, max 60s
        
        Args:
            messages: Conversation message list
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            max_retries: Maximum retry count (default 5, suitable for network fluctuation scenarios)
            initial_retry_delay: Initial retry delay (default 1 second)
            max_retry_delay: Maximum retry delay (default 60 seconds)
            **kwargs: Other parameters
            
        Returns:
            Dictionary containing response content
        """
        # Call parent class generate method, use inherited retry mechanism
        # Default parameters optimized: max_retries=5, initial_retry_delay=1.0, max_retry_delay=60.0
        return super().generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay,
            max_retry_delay=max_retry_delay,
            **kwargs
        )

