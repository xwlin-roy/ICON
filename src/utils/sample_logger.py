"""
Sample-level logging module
Used to record the complete interaction process (prompt and response) for each sample with LLMs
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class SampleLogger:
    """Sample-level logger that creates independent log files for each sample"""
    
    def __init__(self, log_dir: str = "logs/samples"):
        """
        Initialize sample logger
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_sample_index: Optional[int] = None
        self.current_sample_query: Optional[str] = None
        self.interactions: list = []
    
    def start_sample(self, sample_index: int, harmful_query: str):
        """
        Start logging a new sample
        
        Args:
            sample_index: Sample index
            harmful_query: Harmful query text
        """
        self.current_sample_index = sample_index
        self.current_sample_query = harmful_query
        self.interactions = []
    
    def log_interaction(
        self,
        stage: str,
        model_name: str,
        prompt: str,
        response: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_time: Optional[float] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """
        Log one LLM interaction
        
        Args:
            stage: Interaction stage (e.g., "domain_analysis", "paper_generation", "single_turn_attack")
            model_name: Model name
            prompt: Prompt text
            response: Response content
            temperature: Temperature parameter
            max_tokens: Maximum token count
            response_time: Response time (seconds)
            token_usage: Token usage information
            error: Error message (if any)
        """
        interaction = {
            'stage': stage,
            'model_name': model_name,
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if temperature is not None:
            interaction['temperature'] = temperature
        if max_tokens is not None:
            interaction['max_tokens'] = max_tokens
        if response_time is not None:
            interaction['response_time'] = response_time
        if token_usage is not None:
            interaction['token_usage'] = token_usage
        if error is not None:
            interaction['error'] = error
        
        self.interactions.append(interaction)
    
    def save_sample_log(self):
        """
        Save the current sample's log to file (TXT format only)
        """
        if self.current_sample_index is None:
            return
        
        # Generate filename (using index and first 30 characters of query)
        safe_query = self.current_sample_query[:30].replace('/', '_').replace('\\', '_').replace(':', '_')
        safe_query = ''.join(c for c in safe_query if c.isalnum() or c in (' ', '-', '_')).strip()
        text_filename = f"sample_{self.current_sample_index:04d}_{safe_query}.txt"
        text_filepath = self.log_dir / text_filename
        
        # Save as human-readable text file
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Sample #{self.current_sample_index}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Harmful Query: {self.current_sample_query}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, interaction in enumerate(self.interactions, 1):
                f.write("-" * 80 + "\n")
                f.write(f"Interaction #{i}: {interaction['stage']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Model: {interaction['model_name']}\n")
                if 'temperature' in interaction:
                    f.write(f"Temperature: {interaction['temperature']}\n")
                if 'max_tokens' in interaction:
                    f.write(f"Max Tokens: {interaction['max_tokens']}\n")
                if 'response_time' in interaction:
                    f.write(f"Response Time: {interaction['response_time']:.2f}s\n")
                if 'token_usage' in interaction:
                    f.write(f"Token Usage: {interaction['token_usage']}\n")
                f.write("\n")
                f.write("Prompt:\n")
                f.write("-" * 40 + "\n")
                f.write(interaction['prompt'])
                f.write("\n\n")
                f.write("Response:\n")
                f.write("-" * 40 + "\n")
                f.write(interaction['response'])
                f.write("\n\n")
                if 'error' in interaction:
                    f.write(f"Error: {interaction['error']}\n\n")
            
            f.write("=" * 80 + "\n")
        
        return text_filepath

