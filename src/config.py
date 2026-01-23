"""
Configuration management module
Manages configuration information for the entire framework, including API keys, model parameters, etc.
Supports reading configuration from JSON config file, falls back to environment variables if config file does not exist
"""
import os
import json
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM API configuration class"""
    api_key: str
    api_base: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    provider: str = "openai"  # API provider: openai, anthropic, google, qwen, deepseek


class Config:
    """Global configuration manager, supports reading configuration from JSON config file or environment variables"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Configuration file path, if None then automatically search for config.json
        """
        # Determine configuration file path
        if config_file is None:
            # Default to search for config.json in project root directory (jailbreak_framework directory)
            project_root = Path(__file__).parent.parent
            config_file = str(project_root / "config.json")
        
        self.config_file = Path(config_file)
        self.config_data: Dict[str, Any] = {}
        
        # Try to load configuration file
        if self.config_file.exists():
            self._load_from_file()
        else:
            # If configuration file does not exist, use environment variables
            print(f"Note: Configuration file {self.config_file} does not exist, will use environment variables or defaults")
            print("Note: You can copy config.json.example to config.json and fill in the configuration")
        
        # Router LLM configuration (for determining Intent category and selecting context pattern)
        self.router_llm = self._load_llm_config(
            "router_llm",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.7,
            default_max_tokens=1000
        )
        
        # Various authoritative text generator LLM configurations
        self.paper_generator_llm = self._load_llm_config(
            "paper_generator",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.8,
            default_max_tokens=4000
        )
        
        self.script_generator_llm = self._load_llm_config(
            "script_generator",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.8,
            default_max_tokens=4000
        )
        
        self.case_study_generator_llm = self._load_llm_config(
            "case_study_generator",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.8,
            default_max_tokens=4000
        )
        
        self.cti_briefing_generator_llm = self._load_llm_config(
            "cti_briefing_generator",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.8,
            default_max_tokens=4000
        )
        
        self.rca_report_generator_llm = self._load_llm_config(
            "rca_report_generator",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.8,
            default_max_tokens=4000
        )
        
        # Target LLM configuration
        self.target_llm = self._load_llm_config(
            "target_llm",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.7,
            default_max_tokens=2000
        )
        
        # Judge LLM configuration
        self.judge_llm = self._load_llm_config(
            "judge_llm",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.3,  # Judge needs lower temperature to ensure judgment consistency
            default_max_tokens=500
        )
        
        # Reflector LLM configuration
        self.reflector_llm = self._load_llm_config(
            "reflector_llm",
            default_api_key="",
            default_api_base=None,
            default_model="gpt-4",
            default_temperature=0.7,
            default_max_tokens=1500
        )
        
        # Reflector optimization hyperparameters
        reflector_config = self.config_data.get("reflector", {})
        self.max_tactical_retries = reflector_config.get("max_tactical_retries", 3)
        self.max_strategic_retries = reflector_config.get("max_strategic_retries", 1)
        
        # Data file paths
        data_config = self.config_data.get("data", {})
        self.data_dir = data_config.get("data_dir", os.getenv("DATA_DIR", "data"))
        self.harmful_behaviors_csv = data_config.get(
            "harmful_behaviors_csv",
            os.path.join(self.data_dir, "harmful_behaviors.csv")
        )
        
        # Output directory
        output_config = self.config_data.get("output", {})
        self.output_dir = output_config.get("output_dir", os.getenv("OUTPUT_DIR", "output"))
        
        # Logging configuration
        log_config = self.config_data.get("logging", {})
        self.log_dir = log_config.get("log_dir", os.getenv("LOG_DIR", "logs"))
        self.log_level = log_config.get("log_level", os.getenv("LOG_LEVEL", "INFO"))
        self.log_to_file = log_config.get("log_to_file", os.getenv("LOG_TO_FILE", "true").lower() == "true")
        self.log_to_console = log_config.get("log_to_console", os.getenv("LOG_TO_CONSOLE", "true").lower() == "true")
        self.log_sample_rate = log_config.get("log_sample_rate", 0.1)  # Default 10% sampling rate
        self.log_first_n = log_config.get("log_first_n", 3)  # Default record first 3 times
    
    def _load_from_file(self):
        """Load configuration from JSON config file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            print(f"Configuration file loaded: {self.config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Configuration file format error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file: {str(e)}")
    
    def _get_api_key_for_provider(
        self,
        provider: str,
        model_name: str,
        config_key: str = ""
    ) -> Tuple[str, Optional[str]]:
        """
        Automatically get corresponding API key and api_base based on provider and model_name
        
        Args:
            provider: API provider name
            model_name: Model name
            config_key: Configuration key name (for environment variable fallback)
        
        Returns:
            (api_key, api_base) tuple
        """
        # Special handling: anthropic provider uses yunwu OpenAI API
        if provider.lower() == "anthropic":
            # Use yunwu OpenAI API key and base URL for Claude models
            api_keys_config = self.config_data.get("api_keys", {})
            openai_config = api_keys_config.get("openai", {})
            
            # Get yunwu OpenAI API key
            api_key = openai_config.get("default", "")
            
            # Get yunwu API base URL
            api_base = openai_config.get("api_base", "https://yunwu.ai/v1")
            
            # Clean API key: remove any whitespace, newlines, etc.
            if api_key and isinstance(api_key, str):
                api_key = api_key.strip().replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
            
            return api_key or "", api_base
        
        # Search in api_keys configuration for other providers
        api_keys_config = self.config_data.get("api_keys", {})
        provider_config = api_keys_config.get(provider, {})
        
        # Prioritize model-specific key
        api_key = None
        if isinstance(provider_config, dict):
            models_config = provider_config.get("models", {})
            if model_name in models_config:
                model_key = models_config[model_name]
                # Only use model-specific key if it's not empty
                if model_key and model_key.strip():
                    api_key = model_key
            
            # If no model-specific key, use default key
            if not api_key:
                api_key = provider_config.get("default")
            
            # Clean API key: remove any whitespace, newlines, etc.
            if api_key and isinstance(api_key, str):
                api_key = api_key.strip().replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
            
            # Get api_base (if present in provider config)
            api_base = provider_config.get("api_base")
        else:
            # If provider_config is a string, use it directly as default key
            api_key = provider_config if isinstance(provider_config, str) else None
            api_base = None
        
        # If not found in api_keys, try to get from environment variables
        if not api_key and config_key:
            env_key_mapping = {
                "domain_analyzer": "DOMAIN_ANALYZER",
                "paper_generator": "PAPER_GENERATOR",
                "target_llm": "TARGET_LLM",
                "judge_llm": "JUDGE_LLM"
            }
            env_prefix = env_key_mapping.get(config_key, "")
            api_key = os.getenv(f"{env_prefix}_API_KEY", "")
            if not api_base:
                api_base = os.getenv(f"{env_prefix}_API_BASE")
        
        return api_key or "", api_base
    
    def _load_llm_config(
        self,
        config_key: str,
        default_api_key: str = "",
        default_api_base: Optional[str] = None,
        default_model: str = "gpt-4",
        default_temperature: float = 0.7,
        default_max_tokens: int = 2000
    ) -> LLMConfig:
        """
        Load LLM configuration, prioritize reading from config file, fall back to environment variables if not present
        Supports automatically selecting corresponding key from api_keys configuration
        
        Args:
            config_key: Configuration key name (e.g., "domain_analyzer")
            default_api_key: Default API key
            default_api_base: Default API base URL
            default_model: Default model name
            default_temperature: Default temperature parameter
            default_max_tokens: Default maximum token count
        
        Returns:
            LLMConfig object
        """
        # Read from configuration file
        config = self.config_data.get(config_key, {})
        
        # Environment variable key name mapping
        env_key_mapping = {
            "router_llm": "ROUTER_LLM",
            "paper_generator": "PAPER_GENERATOR",
            "script_generator": "SCRIPT_GENERATOR",
            "case_study_generator": "CASE_STUDY_GENERATOR",
            "cti_briefing_generator": "CTI_BRIEFING_GENERATOR",
            "rca_report_generator": "RCA_REPORT_GENERATOR",
            "target_llm": "TARGET_LLM",
            "judge_llm": "JUDGE_LLM",
            "reflector_llm": "REFLECTOR_LLM"
        }
        env_prefix = env_key_mapping.get(config_key, "")
        
        # Get provider and model_name (for automatic key lookup)
        provider = config.get("provider", "openai")
        if not provider:
            provider = os.getenv(f"{env_prefix}_PROVIDER", "openai")
        
        model_name = config.get("model_name") or os.getenv(f"{env_prefix}_MODEL", default_model)
        
        # Prioritize api_key directly specified in config, otherwise automatically lookup from api_keys config
        api_key = config.get("api_key")
        api_base = config.get("api_base")
        
        if not api_key:
            # Automatically lookup from api_keys configuration
            auto_key, auto_base = self._get_api_key_for_provider(provider, model_name, config_key)
            api_key = auto_key or os.getenv(f"{env_prefix}_API_KEY", default_api_key)
            if not api_base:
                api_base = auto_base or os.getenv(f"{env_prefix}_API_BASE", default_api_base)
        else:
            # If api_key is specified in config but api_base is not, try to get from api_keys config
            if not api_base:
                _, auto_base = self._get_api_key_for_provider(provider, model_name, config_key)
                api_base = auto_base or os.getenv(f"{env_prefix}_API_BASE", default_api_base)
        
        # Temperature parameter needs to be converted to float
        temperature = config.get("temperature")
        if temperature is None:
            temp_str = os.getenv(f"{env_prefix}_TEMPERATURE")
            temperature = float(temp_str) if temp_str else default_temperature
        else:
            temperature = float(temperature)
        
        # max_tokens needs to be converted to int
        max_tokens = config.get("max_tokens")
        if max_tokens is None:
            tokens_str = os.getenv(f"{env_prefix}_MAX_TOKENS")
            max_tokens = int(tokens_str) if tokens_str else default_max_tokens
        else:
            max_tokens = int(max_tokens)
        
        return LLMConfig(
            api_key=api_key,
            api_base=api_base if api_base else None,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider
        )
        
    def validate(self) -> bool:
        """Validate if configuration is complete"""
        errors = []
        if not self.router_llm.api_key:
            errors.append("router_llm.api_key not set (please configure in config.json or set environment variable ROUTER_LLM_API_KEY)")
        if not self.paper_generator_llm.api_key:
            errors.append("paper_generator.api_key not set (please configure in config.json or set environment variable PAPER_GENERATOR_API_KEY)")
        if not self.script_generator_llm.api_key:
            errors.append("script_generator.api_key not set (please configure in config.json or set environment variable SCRIPT_GENERATOR_API_KEY)")
        if not self.case_study_generator_llm.api_key:
            errors.append("case_study_generator.api_key not set (please configure in config.json or set environment variable CASE_STUDY_GENERATOR_API_KEY)")
        if not self.cti_briefing_generator_llm.api_key:
            errors.append("cti_briefing_generator.api_key not set (please configure in config.json or set environment variable CTI_BRIEFING_GENERATOR_API_KEY)")
        if not self.rca_report_generator_llm.api_key:
            errors.append("rca_report_generator.api_key not set (please configure in config.json or set environment variable RCA_REPORT_GENERATOR_API_KEY)")
        if not self.target_llm.api_key:
            errors.append("target_llm.api_key not set (please configure in config.json or set environment variable TARGET_LLM_API_KEY)")
        if not self.judge_llm.api_key:
            errors.append("judge_llm.api_key not set (please configure in config.json or set environment variable JUDGE_LLM_API_KEY)")
        if not self.reflector_llm.api_key:
            errors.append("reflector_llm.api_key not set (please configure in config.json or set environment variable REFLECTOR_LLM_API_KEY)")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        return True

