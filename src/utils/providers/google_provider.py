"""
Google Gemini API Provider adapter
"""
import time
import logging
import re
from typing import Optional, Dict, Any, List
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API Provider"""
    
    def _initialize_client(self, **kwargs):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
        except ImportError:
            raise ImportError("Please install google-generativeai library: pip install google-generativeai")
    
    def normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Gemini API message format:
        - Uses parts list
        - Role names are slightly different
        """
        # Gemini uses different message format, but we can pass text directly through generate_content
        # Return original messages here, process in generate method
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
        Call Google Gemini API to generate response with automatic retry for quota/rate limit errors
        
        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts for quota/rate limit errors (default: 5)
            initial_retry_delay: Initial delay in seconds before first retry (default: 1.0)
            max_retry_delay: Maximum delay in seconds between retries (default: 120.0)
            **kwargs: Additional parameters
        """
        import google.generativeai as genai
        
        # Get model
        gemini_model = genai.GenerativeModel(model)
        
        # Build conversation content
        # Gemini needs to convert conversation history to specific format
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages need special handling, usually as part of first user message
                continue
            
            conversation_parts.append({
                "role": role,
                "parts": [content]
            })
        
        # If system messages exist, add them to first user message
        system_content = None
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
                break
        
        if system_content and conversation_parts:
            if conversation_parts[0].get("role") == "user":
                conversation_parts[0]["parts"][0] = f"{system_content}\n\n{conversation_parts[0]['parts'][0]}"
        
        # Build generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Call API
                # Gemini API needs to convert history messages to specific format
                if len(conversation_parts) == 1:
                    # Single-turn conversation, use generate_content directly
                    response = gemini_model.generate_content(
                        conversation_parts[0]["parts"][0],
                        generation_config=generation_config
                    )
                else:
                    # Multi-turn conversation, need to build history
                    # History format: list, each element is a dictionary containing role and parts
                    history = conversation_parts[:-1]
                    last_message = conversation_parts[-1]["parts"][0]
                    
                    # Create or continue conversation
                    chat = gemini_model.start_chat(history=history)
                    response = chat.send_message(
                        last_message,
                        generation_config=generation_config
                    )
                
                # Extract response content
                content = response.text if hasattr(response, 'text') else str(response)
                
                # Gemini token usage
                usage = None
                if hasattr(response, 'usage_metadata'):
                    usage_meta = response.usage_metadata
                    usage = {
                        "prompt_tokens": usage_meta.prompt_token_count if hasattr(usage_meta, 'prompt_token_count') else None,
                        "completion_tokens": usage_meta.candidates_token_count if hasattr(usage_meta, 'candidates_token_count') else None,
                        "total_tokens": usage_meta.total_token_count if hasattr(usage_meta, 'total_token_count') else None
                    }
                
                return {
                    "content": content.strip(),
                    "usage": usage,
                    "model": model
                }
                
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's a quota/rate limit error (429)
                is_quota_error = (
                    "429" in error_str or
                    "quota" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "rate_limit" in error_str.lower() or
                    "ResourceExhausted" in error_str or
                    "exceeded" in error_str.lower()
                )
                
                if is_quota_error and attempt < max_retries:
                    # Calculate delay
                    delay = min(initial_retry_delay * (2 ** attempt), max_retry_delay)
                    
                    # Try to extract retry_delay from error message
                    # Google API error format: "Please retry in 23.309460155s."
                    retry_delay_match = re.search(r'retry in ([\d.]+)s?', error_str, re.IGNORECASE)
                    if retry_delay_match:
                        try:
                            extracted_delay = float(retry_delay_match.group(1))
                            # Add a small buffer (10%) to the extracted delay
                            delay = min(extracted_delay * 1.1, max_retry_delay)
                        except (ValueError, TypeError):
                            pass
                    
                    # Also try to extract from retry_delay { seconds: 23 } format
                    retry_delay_seconds_match = re.search(r'seconds[:\s]+([\d.]+)', error_str, re.IGNORECASE)
                    if retry_delay_seconds_match:
                        try:
                            extracted_delay = float(retry_delay_seconds_match.group(1))
                            delay = min(extracted_delay * 1.1, max_retry_delay)
                        except (ValueError, TypeError):
                            pass
                    
                    # Log retry attempt
                    logger.warning(
                        f"Quota/rate limit error (429) on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f} seconds... Model: {model}"
                    )
                    
                    time.sleep(delay)
                    continue
                else:
                    # Not a quota error, or no retries left
                    if is_quota_error:
                        raise RuntimeError(
                            f"Google Gemini API call failed after {max_retries + 1} attempts due to quota/rate limiting (429). "
                            f"Last error: {error_str}"
                        )
                    else:
                        raise RuntimeError(f"Google Gemini API call failed: {error_str}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise RuntimeError(
                f"Google Gemini API call failed after {max_retries + 1} attempts. "
                f"Last error: {str(last_exception)}"
            )
        else:
            raise RuntimeError("Google Gemini API call failed: Unknown error")

