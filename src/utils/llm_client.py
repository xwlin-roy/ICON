"""
LLM client wrapper module
Provides unified LLM API calling interface, supports multiple LLM service providers
"""
import json
import time
from typing import Optional, Dict, Any, List
from ..config import LLMConfig
from .logger import get_logger
from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    QwenProvider,
    DeepSeekProvider,
    YunwuGeminiProvider
)


class LLMClient:
    """LLM API client, encapsulates common methods for interacting with LLM services"""
    
    # Provider class mapping
    PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,  # Alias
        "google": GoogleProvider,
        "gemini": GoogleProvider,  # Alias (for official Google API)
        "yunwu_gemini": YunwuGeminiProvider,  # yunwu Gemini API
        "qwen": QwenProvider,
        "deepseek": DeepSeekProvider,
    }
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client
        
        Args:
            config: LLM configuration object, containing API key, model name, provider, etc.
        """
        self.config = config
        self._provider: Optional[BaseLLMProvider] = None
        self._last_response_time = None
        self._last_token_usage = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize specific LLM Provider based on configuration"""
        provider_name = self.config.provider.lower()
        
        # Get Provider class
        provider_class = self.PROVIDER_MAP.get(provider_name)
        if not provider_class:
            raise ValueError(
                f"Unsupported Provider: {provider_name}. "
                f"Supported Providers: {', '.join(self.PROVIDER_MAP.keys())}"
            )
        
        # Initialize Provider instance
        try:
            self._provider = provider_class(
                api_key=self.config.api_key,
                api_base=self.config.api_base
            )
        except ImportError as e:
            # Provide more friendly error message
            provider_packages = {
                "openai": "openai",
                "anthropic": "openai",  # Anthropic now uses OpenAI-compatible format via yunwu
                "google": "google-generativeai",
                "gemini": "google-generativeai",
                "qwen": "openai",  # Qwen uses OpenAI-compatible format
                "deepseek": "openai",  # DeepSeek uses OpenAI-compatible format
            }
            package = provider_packages.get(provider_name, "unknown")
            raise ImportError(
                f"Using {provider_name} Provider requires installing {package} library: "
                f"pip install {package}\nOriginal error: {str(e)}"
            )
    
    def generate(
        self, 
        prompt: str, 
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text response
        
        Args:
            prompt: Input prompt text (used if messages is None)
            messages: Conversation history message list, format: [{"role": "user/assistant", "content": "..."}]
            **kwargs: Additional generation parameters, such as temperature, max_tokens, etc.
        
        Returns:
            LLM-generated text response
        """
        # Get logger
        logger = get_logger()
        
        # If messages provided, use messages; otherwise use prompt
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Merge configuration parameters and additional parameters
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        
        # Log request
        if logger:
            logger.log_request(
                model_name=self.config.model_name,
                prompt=prompt if messages is None or len(messages) == 1 else None,
                messages=messages if messages and len(messages) > 1 else None,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=self.config.provider,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
        
        try:
            # Record start time
            start_time = time.time()
            
            # Call Provider to generate response
            response_dict = self._provider.generate(
                messages=messages,
                model=self.config.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response content
            response_content = response_dict.get("content", "").strip()
            
            # Extract token usage
            token_usage = response_dict.get("usage")
            
            # Save metadata from last call (for SampleLogger)
            self._last_response_time = response_time
            self._last_token_usage = token_usage
            
            # Log response (if request was logged, log response)
            if logger:
                # Use force_log=False, let log_response decide whether to log
                logger.log_response(
                    response=response_content,
                    response_time=response_time,
                    token_usage=token_usage,
                    force_log=False
                )
            
            return response_content
        except Exception as e:
            # Log error
            if logger:
                logger.log_error(e, context=f"Model: {self.config.model_name}, Provider: {self.config.provider}")
            raise RuntimeError(f"LLM API call failed: {str(e)}")
    
    def get_last_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from last generate call
        
        Returns:
            Dictionary containing response_time and token_usage
        """
        return {
            'response_time': self._last_response_time,
            'token_usage': self._last_token_usage
        }
    
    def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate JSON format response
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
        
        Returns:
            Parsed JSON dictionary
        """
        logger = get_logger()
        response_text = self.generate(prompt, **kwargs)
        try:
            # Try to parse JSON directly
            result = json.loads(response_text)
            if logger:
                logger.log_debug(f"Successfully parsed JSON response: {result}")
            return result
        except json.JSONDecodeError as e:
            # If direct parsing fails, try to extract JSON portion
            # Find content between first { and last }
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    result = json.loads(json_str)
                    if logger:
                        logger.log_debug(f"Successfully extracted and parsed JSON: {result}")
                    return result
                except json.JSONDecodeError:
                    # Try to fix common JSON format issues
                    # Remove trailing comma
                    json_str = json_str.rstrip().rstrip(',')
                    # Try parsing again
                    try:
                        result = json.loads(json_str)
                        if logger:
                            logger.log_debug(f"Successfully parsed JSON after cleanup: {result}")
                        return result
                    except json.JSONDecodeError:
                        if logger:
                            logger.log_error(e, context="Failed to parse JSON from response")
                        raise ValueError(f"Failed to extract JSON from response: {response_text[:500]}")
            else:
                if logger:
                    logger.log_error(e, context="No JSON structure found in response")
                raise ValueError(f"Failed to extract JSON from response: {response_text[:500]}")

