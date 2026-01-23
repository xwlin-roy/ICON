"""
OpenAI API Provider adapter
Supports OpenAI official API and OpenAI-compatible APIs (e.g., DeepSeek, Qwen, Llama, etc.)
Also supports yunwu API which provides OpenAI-compatible interface for various models including GPT, Llama, etc.
"""
import os
import time
import logging
from typing import Optional, Dict, Any, List
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider, also supports other APIs compatible with OpenAI format (e.g., yunwu, DeepSeek, Qwen, Llama, etc.)"""
    
    def _initialize_client(self, **kwargs):
        """Initialize OpenAI client"""
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
            # Increase timeout for long generation tasks (especially for models like deepseek-v3.2 with max_tokens=2000)
            # Read timeout: 300 seconds (5 minutes) to accommodate long text generation
            # Connect timeout: 30 seconds to handle proxy connection delays
            http_client = None
            if proxy_url:
                # Support SOCKS5 proxies (socks5:// or socks5h://)
                # If proxy URL doesn't have a scheme, assume HTTP
                if not proxy_url.startswith(('http://', 'https://', 'socks5://', 'socks5h://')):
                    proxy_url = f"http://{proxy_url}"
                
                http_client = httpx.Client(
                    proxy=proxy_url,  # Use proxy (singular), supports SOCKS5
                    timeout=httpx.Timeout(300.0, connect=30.0)  # 5 min read timeout, 30s connect timeout
                )
            else:
                # Even without proxy, set timeout with longer read timeout for long generation
                http_client = httpx.Client(
                    timeout=httpx.Timeout(300.0, connect=30.0)  # 5 min read timeout, 30s connect timeout
                )
            
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=http_client
            )
        except ImportError:
            raise ImportError("Please install openai and httpx libraries: pip install openai httpx httpx[socks]")
    
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
        Call OpenAI API to generate response with automatic retry for rate limit errors
        
        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts for rate limit errors (default: 5)
            initial_retry_delay: Initial delay in seconds before first retry (default: 1.0)
            max_retry_delay: Maximum delay in seconds between retries (default: 60.0)
            **kwargs: Additional parameters
        """
        import openai
        
        # Prepare API parameters
        # Remove max_tokens from kwargs to avoid conflicts
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **{k: v for k, v in kwargs.items() if k not in ["max_tokens"]}
        }
        
        # Use max_tokens from kwargs if provided, otherwise use max_tokens parameter
        api_params["max_tokens"] = kwargs.get("max_tokens", max_tokens)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(**api_params)
                
                # Validate response structure
                if not response.choices or len(response.choices) == 0:
                    raise RuntimeError(
                        f"OpenAI API returned empty choices. "
                        f"Response: {response}"
                    )
                
                # Extract response content
                message = response.choices[0].message
                if message is None:
                    raise RuntimeError(
                        f"OpenAI API returned None message. "
                        f"Response: {response}"
                    )
                
                # Extract finish_reason early (needed for retry logic)
                finish_reason = getattr(response.choices[0], 'finish_reason', None) or getattr(message, 'finish_reason', None)
                
                # Handle None content (some APIs may return None for content)
                content = message.content
                
                # Determine if we should retry based on finish_reason
                # content_filter: Content was filtered by safety system - retrying won't help
                # length: Max tokens reached - retrying with same params won't help
                # stop: Normal completion - shouldn't have None content, but if it does, might be transient
                # unknown/None: Unknown reason - might be transient, can retry
                should_retry = True
                if finish_reason == "content_filter":
                    # Content was filtered by safety system - retrying won't help
                    should_retry = False
                    # DEBUG: Log detailed information about the request that triggered content_filter
                    last_user_message = None
                    if messages and len(messages) > 0:
                        # Find the last user message
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                last_user_message = msg.get("content", "")
                                break
                    logger.warning(
                        f"[CONTENT_FILTER DEBUG] OpenAI API returned None content for model {model} due to content_filter. "
                        f"Content was filtered by safety system. Returning empty string (not retrying)."
                    )
                    if last_user_message:
                        # Truncate long messages for logging
                        msg_preview = last_user_message[:200] + "..." if len(last_user_message) > 200 else last_user_message
                        logger.warning(
                            f"[CONTENT_FILTER DEBUG] Last user message (first 200 chars): {msg_preview}"
                        )
                    logger.warning(
                        f"[CONTENT_FILTER DEBUG] Total messages in conversation: {len(messages)}"
                    )
                elif finish_reason == "length":
                    # Max tokens reached - retrying with same params won't help
                    should_retry = False
                    logger.warning(
                        f"OpenAI API returned None content for model {model} due to length limit. "
                        f"Max tokens reached. Returning empty string (not retrying)."
                    )
                
                if content is None:
                    if should_retry and attempt < max_retries:
                        # Retry for None content (might be transient issue)
                        logger.warning(
                            f"OpenAI API returned None content for model {model} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Response finish_reason: {finish_reason}. Retrying..."
                        )
                        # Calculate delay for retry
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        time.sleep(delay)
                        continue
                    else:
                        # Don't retry (content_filter/length) or last attempt - return empty string
                        if not should_retry:
                            logger.warning(
                                f"OpenAI API returned None content for model {model}. "
                                f"Response finish_reason: {finish_reason}. Returning empty string (not retrying)."
                            )
                        else:
                            logger.warning(
                                f"OpenAI API returned None content for model {model} after {max_retries + 1} attempts. "
                                f"Response finish_reason: {finish_reason}. Returning empty string."
                            )
                        content = ""  # Use empty string as fallback
                else:
                    content = content.strip()
                    
                    # Also check for empty content after stripping
                    # Only retry if should_retry is True
                    if not content and should_retry and attempt < max_retries:
                        # Empty content - retry (only if not content_filter/length)
                        logger.warning(
                            f"OpenAI API returned empty content for model {model} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Response finish_reason: {finish_reason}. Retrying..."
                        )
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        time.sleep(delay)
                        continue
                    elif not content and not should_retry:
                        # Empty content due to content_filter/length - don't retry
                        logger.warning(
                            f"OpenAI API returned empty content for model {model} due to {finish_reason}. "
                            f"Returning empty string (not retrying)."
                        )
                
                # Extract token usage
                usage = None
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                
                return {
                    "content": content,
                    "usage": usage,
                    "model": response.model if hasattr(response, 'model') else model,
                    "finish_reason": finish_reason  # Add finish_reason for debugging
                }
                
            except openai.RateLimitError as e:
                # Check if it's actually a quota error (insufficient_quota)
                # Sometimes quota errors are reported as RateLimitError
                error_str = str(e).lower()
                error_code = None
                error_type = None
                
                # Try to extract error code and type from error message
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        if isinstance(error_body, dict) and 'error' in error_body:
                            error_info = error_body['error']
                            error_code = error_info.get('code', '')
                            error_type = error_info.get('type', '')
                    except:
                        pass
                
                # Check if it's actually a quota error
                is_quota_error = (
                    "insufficient_quota" in error_str or
                    "insufficient_quota" in str(error_code).lower() or
                    "insufficient_quota" in str(error_type).lower() or
                    "exceeded your current quota" in error_str or
                    "quota" in error_str and "exceeded" in error_str
                )
                
                if is_quota_error:
                    # Quota exceeded - don't retry, fail immediately
                    logger.error(
                        f"API quota exceeded (insufficient_quota) on attempt {attempt + 1}. "
                        f"This is a billing/quota issue, not a rate limit. "
                        f"Please check your API plan and billing details. Model: {model}"
                    )
                    raise RuntimeError(
                        f"OpenAI API quota exceeded (insufficient_quota). "
                        f"Please check your API plan and billing details. "
                        f"Error: {str(e)}"
                    )
                
                # Real rate limit error - retry with backoff
                last_exception = e
                # Check if we have retries left
                if attempt < max_retries:
                    # Calculate exponential backoff delay
                    # Start with initial_retry_delay, double each time, cap at max_retry_delay
                    delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                    
                    # Try to extract retry-after from error if available
                    retry_after = None
                    if hasattr(e, 'response') and e.response is not None:
                        retry_after_header = e.response.headers.get('Retry-After')
                        if retry_after_header:
                            try:
                                retry_after = float(retry_after_header)
                                delay = min(retry_after, max_retry_delay)
                            except (ValueError, TypeError):
                                pass
                    
                    # Log retry attempt
                    logger.warning(
                        f"Rate limit error (429) on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f} seconds... Model: {model}"
                    )
                    
                    time.sleep(delay)
                    continue
                else:
                    # No more retries left
                    raise RuntimeError(
                        f"OpenAI API call failed after {max_retries + 1} attempts due to rate limiting (429). "
                        f"Last error: {str(e)}"
                    )
                    
            except openai.APIError as e:
                # Check error type to determine if we should retry
                error_str = str(e).lower()
                error_code = None
                error_type = None
                
                # Try to extract error code and type from error message
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_body = e.response.json()
                        if isinstance(error_body, dict) and 'error' in error_body:
                            error_info = error_body['error']
                            error_code = error_info.get('code', '')
                            error_type = error_info.get('type', '')
                    except:
                        pass
                
                # Check if it's a content filter error (data_inspection_failed) - should NOT retry, return empty string
                # This is common with Alibaba Cloud's DeepSeek API when content is filtered
                is_content_filter_error = (
                    "data_inspection_failed" in error_str or
                    "data_inspection_failed" in str(error_code).lower() or
                    "data_inspection_failed" in str(error_type).lower() or
                    "inappropriate content" in error_str
                )
                
                # Check if it's a quota error (insufficient_quota) - should NOT retry
                is_quota_error = (
                    "insufficient_quota" in error_str or
                    "insufficient_quota" in str(error_code).lower() or
                    "insufficient_quota" in str(error_type).lower() or
                    "exceeded your current quota" in error_str or
                    "quota" in error_str and "exceeded" in error_str
                )
                
                # Check if it's a rate limit error (429) - should retry
                is_rate_limit = (
                    "429" in error_str or
                    "rate limit" in error_str or
                    "rate_limit" in error_str
                ) and not is_quota_error
                
                if is_content_filter_error:
                    # Content filter error (e.g., data_inspection_failed from Alibaba Cloud) - return empty string, don't retry
                    logger.warning(
                        f"OpenAI API returned content filter error (data_inspection_failed) for model {model}. "
                        f"Content was filtered by safety system. Returning empty string (not retrying)."
                    )
                    return {
                        "content": "",
                        "usage": None,
                        "model": model,
                        "finish_reason": "content_filter"
                    }
                elif is_quota_error:
                    # Quota exceeded - don't retry, fail immediately
                    logger.error(
                        f"API quota exceeded (insufficient_quota) on attempt {attempt + 1}. "
                        f"This is a billing/quota issue, not a rate limit. "
                        f"Please check your API plan and billing details. Model: {model}"
                    )
                    raise RuntimeError(
                        f"OpenAI API quota exceeded (insufficient_quota). "
                        f"Please check your API plan and billing details. "
                        f"Error: {str(e)}"
                    )
                elif is_rate_limit:
                    # Rate limit - retry with backoff
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        logger.warning(
                            f"Rate limit error (429) detected on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.2f} seconds... Model: {model}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"OpenAI API call failed after {max_retries + 1} attempts due to rate limiting (429). "
                            f"Last error: {str(e)}"
                        )
                else:
                    # Other API errors - don't retry
                    raise RuntimeError(f"OpenAI API call failed: {str(e)}")
            
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                # Connection/timeout errors - should retry
                last_exception = e
                if attempt < max_retries:
                    delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                    logger.warning(
                        f"Connection/timeout error on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f} seconds... Model: {model}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(
                        f"OpenAI API call failed after {max_retries + 1} attempts due to connection/timeout errors. "
                        f"Last error: {str(e)}"
                    )
            
            except Exception as e:
                # Check if it's a connection-related error (from httpx or requests)
                error_str = str(e).lower()
                is_connection_error = (
                    "connection" in error_str or
                    "timeout" in error_str or
                    "connect" in error_str or
                    "network" in error_str or
                    "socket" in error_str
                )
                
                if is_connection_error:
                    # Connection-related errors - should retry
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.2f} seconds... Model: {model}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"OpenAI API call failed after {max_retries + 1} attempts due to connection errors. "
                            f"Last error: {str(e)}"
                        )
                else:
                    # For other exceptions, don't retry
                    raise RuntimeError(f"OpenAI API call failed: {str(e)}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise RuntimeError(
                f"OpenAI API call failed after {max_retries + 1} attempts. "
                f"Last error: {str(last_exception)}"
            )
        else:
            raise RuntimeError("OpenAI API call failed: Unknown error")

