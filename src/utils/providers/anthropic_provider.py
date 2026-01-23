"""
Anthropic Claude API Provider adapter
Calls Claude models via yunwu's OpenAI compatible interface
"""
import os
import time
import logging
from typing import Optional, Dict, Any, List
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider - Calls via yunwu's OpenAI compatible interface"""
    
    def _initialize_client(self, **kwargs):
        """Initialize OpenAI client for yunwu API"""
        try:
            import openai
            import httpx
            
            # Check proxy settings with priority:
            # 1. Environment variables (HTTP_PROXY, HTTPS_PROXY)
            # 2. Config data passed via kwargs (if available)
            proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
            
            # If not in environment, check kwargs for config_data
            if not proxy_url and 'config_data' in kwargs:
                config_data = kwargs.get('config_data', {})
                proxy_config = config_data.get('proxy', {})
                if proxy_config:
                    proxy_url = proxy_config.get('url') or proxy_config.get('http_proxy') or proxy_config.get('https_proxy')
            
            # If proxy is set, create httpx client with proxy
            # Note: httpx 0.28.0+ uses proxy parameter (singular), not proxies
            # Support both HTTP and SOCKS5 proxies
            http_client = None
            if proxy_url:
                # Support SOCKS5 proxies (socks5:// or socks5h://)
                # If proxy URL doesn't have a scheme, assume HTTP
                if not proxy_url.startswith(('http://', 'https://', 'socks5://', 'socks5h://')):
                    proxy_url = f"http://{proxy_url}"
                
                http_client = httpx.Client(
                    proxy=proxy_url,  # Use proxy (singular), supports SOCKS5
                    timeout=httpx.Timeout(60.0, connect=10.0)  # Set timeout
                )
            else:
                # Even without proxy, set timeout
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0)
                )
            
            # Use OpenAI client with yunwu API base URL
            # api_base should be set to https://yunwu.ai/v1 via config
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client
            )
        except ImportError:
            raise ImportError("Please install openai and httpx libraries: pip install openai httpx httpx[socks]")
    
    def normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize message format (via yunwu's OpenAI compatible interface, using standard OpenAI format)
        Keep this method for compatibility, actually uses standard OpenAI format when calling via yunwu API
        """
        if not messages:
            raise ValueError("Message list cannot be empty")
        
        # Call via yunwu's OpenAI compatible interface, using standard OpenAI format
        # Only do basic validation, no format conversion
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Invalid message format: {msg}")
            
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            # Ensure content is string type
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = str(content)
            
            normalized.append({"role": role, "content": content})
        
        return normalized
    
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
        """Call Claude model via yunwu OpenAI-compatible API to generate response"""
        import openai
        
        # Normalize message format (basic validation only)
        normalized_messages = self.normalize_messages(messages)
        
        try:
            # Validate API key exists
            if not self.api_key:
                raise ValueError("API key not set, please configure api_key in config.json or set environment variable")
            
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": normalized_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ["max_tokens"]}
            }
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    response = self._client.chat.completions.create(**api_params)
                    
                    # Validate response structure
                    if not response.choices or len(response.choices) == 0:
                        raise RuntimeError(
                            f"API returned empty choices. "
                            f"Response: {response}"
                        )
                    
                    # Extract response content
                    message = response.choices[0].message
                    if message is None:
                        raise RuntimeError(
                            f"API returned None message. "
                            f"Response: {response}"
                        )
                    
                    # Handle None content
                    content = message.content
                    if content is None:
                        logger.warning(
                            f"API returned None content for model {model}. "
                            f"This may indicate the response was filtered or empty. "
                            f"Response finish_reason: {getattr(message, 'finish_reason', 'unknown')}"
                        )
                        content = ""
                    else:
                        content = content.strip()
                    
                    # Extract token usage
                    usage = None
                    if hasattr(response, 'usage') and response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    
                    # Extract finish_reason for debugging
                    finish_reason = getattr(response.choices[0], 'finish_reason', None)
                    
                    return {
                        "content": content,
                        "usage": usage,
                        "model": response.model if hasattr(response, 'model') else model,
                        "finish_reason": finish_reason
                    }
                    
                except openai.RateLimitError as e:
                    # Rate limit error - retry with backoff
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        logger.warning(
                            f"Rate limit error (429) on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.2f} seconds... Model: {model}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"API call failed after {max_retries + 1} attempts due to rate limiting (429). "
                            f"Last error: {str(e)}"
                        )
                    
                except openai.APIError as e:
                    # Other API errors - don't retry
                    raise RuntimeError(f"API call failed: {str(e)}")
                except Exception as e:
                    # For other exceptions, don't retry
                    raise RuntimeError(f"API call failed: {str(e)}")
            
            # Should not reach here, but just in case
            if last_exception:
                raise RuntimeError(
                    f"API call failed after {max_retries + 1} attempts. "
                    f"Last error: {str(last_exception)}"
                )
            else:
                raise RuntimeError("API call failed: Unknown error")
                
        except Exception as e:
            # Provide more detailed error information
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Extract status code if available
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            
            # Check if it's a permission error (403)
            if status_code == 403 or "403" in error_msg or "forbidden" in error_msg.lower() or "not allowed" in error_msg.lower():
                detailed_msg = (
                    f"Claude API call via yunwu failed (403 Forbidden)\n"
                    f"Error type: {error_type}\n"
                    f"Error message: {error_msg}\n"
                    f"Model: {model}\n"
                    f"\nPossible causes:\n"
                    f"  1. API key does not have permission to access model '{model}'\n"
                    f"  2. Model name '{model}' does not exist or format is incorrect\n"
                    f"  3. yunwu API does not support this model\n"
                    f"  4. API key has expired or is invalid\n"
                )
                raise RuntimeError(detailed_msg)
            
            # Check if it's an authentication error (401)
            if status_code == 401 or "401" in error_msg or "unauthorized" in error_msg.lower():
                detailed_msg = (
                    f"Claude API call via yunwu failed (401 Unauthorized)\n"
                    f"Error type: {error_type}\n"
                    f"Error message: {error_msg}\n"
                    f"\nPossible causes:\n"
                    f"  1. API key is invalid or expired\n"
                    f"  2. API key not configured correctly\n"
                    f"  3. Wrong API key used (should use yunwu OpenAI key)\n"
                    f"\nSuggestions:\n"
                    f"  - Check api_key configuration in config.json\n"
                    f"  - Confirm using yunwu OpenAI API key\n"
                )
                raise RuntimeError(detailed_msg)
            
            # Other errors
            raise RuntimeError(
                f"Claude API call via yunwu failed\n"
                f"Error type: {error_type}\n"
                f"Error message: {error_msg}\n"
                f"Model: {model}"
            )

