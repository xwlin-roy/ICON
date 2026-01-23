"""
Logging module for LLM interactions
Used to record the interaction process with LLMs for debugging and analysis
"""
import logging
import os
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class LLMInteractionLogger:
    """LLM interaction logger, specifically for recording LLM API calls"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_sample_rate: float = 0.1,
        log_first_n: int = 3,
        target_llm_name: Optional[str] = None
    ):
        """
        Initialize the logger
        
        Args:
            log_dir: Base directory for log files
            log_level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_to_file: Whether to save to file
            log_to_console: Whether to output to console
            log_sample_rate: Log sampling rate (0.0-1.0), only record part of requests for efficiency, default 0.1 (10%)
            log_first_n: Record the first N complete LLM calls, default 3
            target_llm_name: Target LLM model name (used to create subdirectory)
        """
        # Create log directory based on target_llm if provided
        if target_llm_name:
            # Sanitize model name for filesystem
            safe_model_name = target_llm_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            self.log_dir = Path(log_dir) / safe_model_name
        else:
            self.log_dir = Path(log_dir)
        
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_sample_rate = max(0.0, min(1.0, log_sample_rate))  # Limit between 0-1
        self.log_first_n = max(0, log_first_n)
        
        # Thread-safe counter
        self._request_count = 0
        self._lock = threading.Lock()
        # Track whether the last request was logged
        self._last_request_logged = False
        
        # Create log directory
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        self.log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger("LLMInteraction")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler (using thread-safe FileHandler)
        if self.log_to_file:
            # Use date as filename
            log_filename = f"llm_interaction_{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = self.log_dir / log_filename
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def _should_log(self) -> bool:
        """
        Determine whether this request should be logged (sampling logic)
        
        Returns:
            Whether to log
        """
        with self._lock:
            self._request_count += 1
            current_count = self._request_count
        
        # Log first N requests
        should_log = False
        if current_count <= self.log_first_n:
            should_log = True
        # Then log according to sampling rate
        elif self.log_sample_rate > 0:
            should_log = random.random() < self.log_sample_rate
        
        # Save state for log_response to use
        with self._lock:
            self._last_request_logged = should_log
        
        return should_log
    
    def log_request(
        self,
        model_name: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Log LLM request (decided by sampling rate)
        
        Args:
            model_name: Model name
            prompt: Prompt text (if messages is None)
            messages: Conversation message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            **kwargs: Other parameters
        """
        # Decide whether to log based on sampling logic
        if not self._should_log():
            return
        
        self.logger.info("=" * 80)
        self.logger.info(f"LLM API Request - Model: {model_name}")
        self.logger.info("-" * 80)
        
        # Log parameters
        if temperature is not None:
            self.logger.info(f"Temperature: {temperature}")
        if max_tokens is not None:
            self.logger.info(f"Max Tokens: {max_tokens}")
        
        # Log input content
        if messages:
            self.logger.info("Messages:")
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Truncate long content for readability
                content_preview = content[:500] + "..." if len(content) > 500 else content
                self.logger.info(f"  [{i}] Role: {role}")
                self.logger.info(f"      Content: {content_preview}")
                if len(content) > 500:
                    self.logger.info(f"      (Content truncated, total length: {len(content)} chars)")
        elif prompt:
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            self.logger.info(f"Prompt: {prompt_preview}")
            if len(prompt) > 500:
                self.logger.info(f"(Prompt truncated, total length: {len(prompt)} chars)")
        
        # Log other parameters
        if kwargs:
            self.logger.info(f"Additional Parameters: {kwargs}")
        
        self.logger.info("-" * 80)
    
    def log_response(
        self,
        response: str,
        response_time: Optional[float] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        force_log: bool = False
    ):
        """
        Log LLM response (decided by sampling rate)
        
        Args:
            response: Response content
            response_time: Response time (seconds)
            token_usage: Token usage information
            force_log: Whether to force log (for errors, etc.)
        """
        # If force log (e.g., errors), always log
        if not force_log:
            # Check if the last request was logged
            with self._lock:
                should_log = self._last_request_logged
            
            # If request was not logged, don't log response either
            if not should_log:
                return
        
        self.logger.info("LLM API Response:")
        
        if response_time is not None:
            self.logger.info(f"Response Time: {response_time:.2f}s")
        
        if token_usage:
            self.logger.info(f"Token Usage: {token_usage}")
        
        # Log response content
        response_preview = response[:1000] + "..." if len(response) > 1000 else response
        self.logger.info(f"Response Content: {response_preview}")
        if len(response) > 1000:
            self.logger.info(f"(Response truncated, total length: {len(response)} chars)")
        
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """
        Log error information (errors are always logged, not affected by sampling rate)
        
        Args:
            error: Exception object
            context: Error context information
        """
        # Errors are always logged, not affected by sampling rate
        self.logger.error("=" * 80)
        self.logger.error("LLM API Error")
        self.logger.error("-" * 80)
        if context:
            self.logger.error(f"Context: {context}")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        self.logger.error("=" * 80)
        self.logger.error("")
    
    def log_info(self, message: str):
        """Log general information"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log debug information"""
        self.logger.debug(message)


# Global logger instance
_global_logger: Optional[LLMInteractionLogger] = None


def get_logger() -> Optional[LLMInteractionLogger]:
    """Get the global logger instance"""
    return _global_logger


def init_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_sample_rate: float = 0.1,
    log_first_n: int = 3,
    target_llm_name: Optional[str] = None
) -> LLMInteractionLogger:
    """
    Initialize the global logger
    
    Args:
        log_dir: Base directory for log files
        log_level: Log level
        log_to_file: Whether to save to file
        log_to_console: Whether to output to console
        log_sample_rate: Log sampling rate (0.0-1.0), only record part of requests for efficiency, default 0.1 (10%)
        log_first_n: Record the first N complete LLM calls, default 3
        target_llm_name: Target LLM model name (used to create subdirectory)
    
    Returns:
        Logger instance
    """
    global _global_logger
    _global_logger = LLMInteractionLogger(
        log_dir=log_dir,
        log_level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        log_sample_rate=log_sample_rate,
        log_first_n=log_first_n,
        target_llm_name=target_llm_name
    )
    return _global_logger

