"""
Qwen API Provider adapter
Qwen API (Tongyi Qianwen) provided via Alibaba Cloud Bailian platform, using OpenAI compatible format
Supports calling Qwen series models via Alibaba Cloud Bailian API (e.g., qwen-max, qwen-plus, qwen-turbo, etc.)
"""
import logging
from typing import Dict, Any, List
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class QwenProvider(OpenAIProvider):
    """
    Qwen API Provider (Tongyi Qianwen)
    
    Calls Qwen series models via Alibaba Cloud Bailian platform, using OpenAI compatible format API.
    Inherits from OpenAIProvider, supports all OpenAI compatible features, including:
    - Proxy support (HTTP/SOCKS5)
    - Automatic retry mechanism
    - Error handling
    - Token usage statistics
    
    Supported models:
    - qwen-max: Tongyi Qianwen Max version
    - qwen-plus: Tongyi Qianwen Plus version
    - qwen-turbo: Tongyi Qianwen Turbo version
    - qwen-max-longcontext: Long context version
    - Other Qwen series models
    
    Configuration:
    - API Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
    - Need to configure api_key and api_base in config.json
    """
    
    def _initialize_client(self, **kwargs):
        """
        Initialize Qwen client (using OpenAI compatible format)
        
        Args:
            **kwargs: Additional initialization parameters, including:
                - config_data: Configuration data dictionary, can be used to pass proxy configuration, etc.
        """
        import openai
        import httpx
        
        # If api_base is not specified, use Alibaba Cloud Bailian default endpoint
        if not self.api_base:
            self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            logger.info("QwenProvider: Using default API base URL (Alibaba Cloud Bailian)")
        
        # Alibaba Cloud Bailian API does not need proxy, access directly via domestic network
        # Create httpx client without proxy
        http_client = httpx.Client(
            proxy=None,  # Explicitly do not use proxy
            timeout=httpx.Timeout(300.0, connect=30.0)  # 5 minute read timeout, 30 second connect timeout
        )
        
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client
        )
        
        logger.info("QwenProvider: Initialization complete (no proxy, direct access via domestic network)")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"QwenProvider initialization complete: api_base={self.api_base}")
    
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
        Call Qwen API to generate response with comprehensive error handling and retry mechanism
        
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

