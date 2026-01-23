"""
Yunwu Gemini API Provider
Supports calling Gemini models via yunwu.ai (using Gemini native API format)
"""
import os
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class YunwuGeminiProvider(BaseLLMProvider):
    """Yunwu Gemini API Provider - Uses Gemini native API format"""
    
    def _initialize_client(self, **kwargs):
        """Initialize yunwu Gemini client (no special initialization needed)"""
        # yunwu uses HTTP requests, no special client object needed
        self._client = None
    
    def normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert standard message format to Gemini native format
        
        Gemini API format:
        - contents: array, each element contains role and parts
        - role: "user" or "model"
        - parts: array, each element contains text
        """
        return messages
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 120.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call yunwu Gemini API to generate response
        
        Args:
            messages: Conversation message list
            model: Model name (e.g., gemini-3-pro-preview)
            temperature: Sampling temperature
            max_tokens: Maximum generated tokens
            max_retries: Maximum retry count for quota/rate limit errors (default: 5)
            initial_retry_delay: Initial delay before first retry (seconds, default: 1.0)
            max_retry_delay: Maximum delay between retries (seconds, default: 120.0)
            **kwargs: Other parameters (e.g., systemInstruction, generationConfig, etc.)
        """
        # Build API URL
        # yunwu Gemini API endpoint format: https://yunwu.ai/v1beta/models/{model}:generateContent
        if self.api_base:
            # If api_base is set, use it as base URL
            base_url = self.api_base.rstrip('/')
            # Remove possible /v1 suffix (if exists)
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]
            api_url = f"{base_url}/v1beta/models/{model}:generateContent"
        else:
            api_url = f"https://yunwu.ai/v1beta/models/{model}:generateContent"
        
        # Convert message format to Gemini native format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages need separate handling
                system_instruction = content
                continue
            
            # Gemini format: each message contains role and parts
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })
        
        # Build request body
        request_body = {
            "contents": contents
        }
        
        # If system instruction exists, add to request body
        if system_instruction:
            request_body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        # Build generation config
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        
        # If generationConfig exists in kwargs, merge it
        if "generationConfig" in kwargs:
            generation_config.update(kwargs["generationConfig"])
        
        request_body["generationConfig"] = generation_config
        
        # If other parameters exist in kwargs, add to request body
        for key in ["thinkingConfig"]:
            if key in kwargs:
                request_body[key] = kwargs[key]
        
        # Prepare request headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare query parameters (API key as query parameter)
        params = {
            "key": self.api_key
        }
        
        # Check proxy settings
        # requests library automatically reads proxy from environment variables HTTP_PROXY/HTTPS_PROXY
        # But if proxy cannot connect, we need to handle this case
        # Can disable proxy by setting environment variable YUNWU_GEMINI_NO_PROXY=1
        use_proxy = os.getenv("YUNWU_GEMINI_NO_PROXY") != "1"
        proxies = None
        
        if use_proxy:
            proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
            if proxy_url:
                # Check if it's a SOCKS5 proxy
                if proxy_url.startswith(('socks5://', 'socks5h://')):
                    # requests library needs PySocks to support SOCKS5 proxy
                    try:
                        import socks
                        # PySocks installed, requests will automatically handle SOCKS5
                        proxies = {
                            "http": proxy_url,
                            "https": proxy_url
                        }
                        logger.debug(f"Using SOCKS5 proxy: {proxy_url}")
                    except ImportError:
                        logger.warning(
                            f"SOCKS5 proxy detected but PySocks not installed. "
                            f"Install with: pip install requests[socks] or pip install PySocks"
                        )
                        # If SOCKS5 not supported, try converting to HTTP (usually won't work, but at least won't crash)
                        logger.warning(f"Falling back to HTTP proxy format (may not work)")
                        proxies = {
                            "http": proxy_url,
                            "https": proxy_url
                        }
                else:
                    # HTTP/HTTPS proxy
                    proxies = {
                        "http": proxy_url,
                        "https": proxy_url
                    }
                    logger.debug(f"Using HTTP/HTTPS proxy: {proxy_url}")
            else:
                # If no proxy is set, explicitly set to None to avoid using system default proxy
                proxies = {
                    "http": None,
                    "https": None
                }
        else:
            # Explicitly disable proxy
            proxies = {
                "http": None,
                "https": None
            }
            logger.debug("Proxy disabled for yunwu_gemini (YUNWU_GEMINI_NO_PROXY=1)")
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Send HTTP request
                response = requests.post(
                    api_url,
                    headers=headers,
                    params=params,
                    json=request_body,
                    proxies=proxies,  # Explicitly pass proxy settings
                    timeout=120  # 120 second timeout
                )
                
                # Check response status
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Extract response content
                    # Gemini API response format:
                    # {
                    #   "candidates": [
                    #     {
                    #       "content": {
                    #         "parts": [{"text": "..."}]
                    #       }
                    #     }
                    #   ],
                    #   "usageMetadata": {...}
                    # }
                    content = ""
                    if "candidates" in response_data and len(response_data["candidates"]) > 0:
                        candidate = response_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if parts and len(parts) > 0:
                                content = parts[0].get("text", "")
                    
                    # Extract token usage
                    usage = None
                    if "usageMetadata" in response_data:
                        usage_meta = response_data["usageMetadata"]
                        usage = {
                            "prompt_tokens": usage_meta.get("promptTokenCount"),
                            "completion_tokens": usage_meta.get("candidatesTokenCount"),
                            "total_tokens": usage_meta.get("totalTokenCount")
                        }
                    
                    return {
                        "content": content.strip(),
                        "usage": usage,
                        "model": model
                    }
                else:
                    # Handle error response
                    error_str = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        error_str = str(error_data)
                    except:
                        error_str = response.text or f"HTTP {response.status_code}"
                    
                    # Check if it's a quota/rate limit error
                    is_quota_error = (
                        response.status_code == 429 or
                        response.status_code == 503 or
                        "quota" in error_str.lower() or
                        "rate limit" in error_str.lower() or
                        "rate_limit" in error_str.lower() or
                        "exceeded" in error_str.lower() or
                        "no available channel" in error_str.lower()
                    )
                    
                    if is_quota_error and attempt < max_retries:
                        # Calculate delay
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        
                        logger.warning(
                            f"Quota/rate limit error ({response.status_code}) on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.2f} seconds... Model: {model}"
                        )
                        
                        time.sleep(delay)
                        continue
                    else:
                        # Not a quota error, or no retries left
                        raise RuntimeError(f"Yunwu Gemini API call failed: {error_str}")
                        
            except requests.exceptions.ProxyError as e:
                # Proxy connection error - this is a fatal error, should not retry
                error_str = str(e)
                logger.error(f"Proxy connection error: {error_str}")
                raise RuntimeError(
                    f"Yunwu Gemini API call failed: Unable to connect to proxy. "
                    f"Please check your proxy settings (HTTP_PROXY/HTTPS_PROXY) or disable proxy. "
                    f"Error: {error_str}"
                )
            
            except requests.exceptions.ConnectionError as e:
                # Connection error (may be proxy or network issue)
                last_exception = e
                error_str = str(e)
                
                # Check if it's a proxy-related connection error
                is_proxy_error = (
                    "proxy" in error_str.lower() or
                    "connection refused" in error_str.lower() or
                    "errno 111" in error_str.lower()
                )
                
                if is_proxy_error:
                    # Proxy connection error - this is a fatal error, should not retry
                    logger.error(f"Proxy connection error: {error_str}")
                    raise RuntimeError(
                        f"Yunwu Gemini API call failed: Unable to connect to proxy. "
                        f"Please check your proxy settings (HTTP_PROXY/HTTPS_PROXY) or disable proxy. "
                        f"Error: {error_str}"
                    )
                else:
                    # Other connection errors, may be network issues, can retry
                    if attempt < max_retries:
                        delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.2f} seconds... Model: {model}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f"Yunwu Gemini API call failed after {max_retries + 1} attempts: {error_str}")
            
            except requests.exceptions.RequestException as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's a quota/rate limit error
                is_quota_error = (
                    "429" in error_str or
                    "503" in error_str or
                    "quota" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "no available channel" in error_str.lower()
                )
                
                if is_quota_error and attempt < max_retries:
                    # Calculate delay
                    delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                    
                    logger.warning(
                        f"Quota/rate limit error on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f} seconds... Model: {model}"
                    )
                    
                    time.sleep(delay)
                    continue
                else:
                    # Not a quota error, or no retries left
                    if is_quota_error:
                        raise RuntimeError(
                            f"Yunwu Gemini API call failed after {max_retries + 1} attempts due to quota/rate limiting. "
                            f"Last error: {error_str}"
                        )
                    else:
                        raise RuntimeError(f"Yunwu Gemini API call failed: {error_str}")
            
            except Exception as e:
                last_exception = e
                error_str = str(e)
                raise RuntimeError(f"Yunwu Gemini API call failed: {error_str}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise RuntimeError(
                f"Yunwu Gemini API call failed after {max_retries + 1} attempts. "
                f"Last error: {str(last_exception)}"
            )
        else:
            raise RuntimeError("Yunwu Gemini API call failed: Unknown error")

