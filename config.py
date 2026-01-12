"""
Configuration management for Writing Coach
Handles environment variables and API key management
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Writing Coach system"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        self.load_env_config()
    
    def load_env_config(self):
        """Load configuration from environment variables"""
        # OpenAI Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # Gemini Configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-pro')
        
        # Anthropic Configuration
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')
        
        # Azure OpenAI Configuration
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY', '')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
        self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self.azure_openai_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', '')
        
        # Ollama Configuration
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'mistral')
        
        # Hugging Face Configuration
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.huggingface_model = os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium')
        
        # Cohere Configuration
        self.cohere_api_key = os.getenv('COHERE_API_KEY', '')
        self.cohere_model = os.getenv('COHERE_MODEL', 'command')
        
        # Qwen Configuration
        self.qwen_api_key = os.getenv('QWEN_API_KEY', os.getenv('DASHSCOPE_API_KEY', ''))
        self.qwen_model = os.getenv('QWEN_MODEL', os.getenv('QWEN3_VL_PLUS_MODEL', 'qwen3-vl-plus'))
        self.qwen_base_url = os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        
        # DeepSeek Configuration
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', '')
        self.deepseek_model = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        self.deepseek_base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        
        # Application Settings
        self.default_llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'ollama').lower()
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.api_host = os.getenv('API_HOST', '127.0.0.1')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        
        # OCR Settings
        self.ocr_language = os.getenv('OCR_LANGUAGE', 'en')
        self.ocr_method = 'gemini_vision'  # Only use Gemini Vision
        
        # Development Settings
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.environment = os.getenv('ENVIRONMENT', 'development')
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get LLM configuration for specified provider
        
        Args:
            provider: LLM provider name (openai, gemini, anthropic, azure, ollama, huggingface, cohere)
                     If None, uses default provider
        
        Returns:
            Dictionary with LLM configuration
        """
        provider = provider or self.default_llm_provider
        
        configs = {
            'openai': {
                'api_key': self.openai_api_key,
                'model': self.openai_model,
                'api_base': self.openai_base_url,
                'api_type': 'openai'
            },
            'gemini': {
                'api_key': self.gemini_api_key,
                'model': self.gemini_model,
                'api_type': 'gemini'
            },
            'anthropic': {
                'api_key': self.anthropic_api_key,
                'model': self.anthropic_model,
                'api_type': 'anthropic'
            },
            'azure': {
                'api_key': self.azure_openai_api_key,
                'api_base': self.azure_openai_endpoint,
                'api_version': self.azure_openai_api_version,
                'model': self.azure_openai_deployment_name,
                'api_type': 'azure'
            },
            'ollama': {
                'api_key': 'NA',  # Ollama doesn't need API key
                'api_base': f"{self.ollama_base_url}/v1",
                'model': self.ollama_model,
                'api_type': 'open_ai'
            },
            'huggingface': {
                'api_key': self.huggingface_api_key,
                'model': self.huggingface_model,
                'api_type': 'huggingface'
            },
            'cohere': {
                'api_key': self.cohere_api_key,
                'model': self.cohere_model,
                'api_type': 'cohere'
            },
            'qwen': {
                'api_key': self.qwen_api_key,
                'model': self.qwen_model,
                'api_base': self.qwen_base_url,
                'api_type': 'openai'
            },
            'deepseek': {
                'api_key': self.deepseek_api_key,
                'model': self.deepseek_model,
                'api_base': self.deepseek_base_url,
                'api_type': 'openai'
            }
        }
        
        if provider not in configs:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        config = configs[provider]
        
        # Validate that required API keys are present (except for Ollama)
        if provider != 'ollama' and not config.get('api_key'):
            logger.warning(f"No API key found for {provider}. Please set the appropriate environment variable.")
        
        return config
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available LLM providers (those with API keys configured)
        
        Returns:
            List of provider names that have API keys configured
        """
        available = []
        
        # Ollama is always available (no API key needed)
        available.append('ollama')
        
        # Check other providers for API keys
        if self.openai_api_key:
            available.append('openai')
        if self.gemini_api_key:
            available.append('gemini')
        if self.anthropic_api_key:
            available.append('anthropic')
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            available.append('azure')
        if self.huggingface_api_key:
            available.append('huggingface')
        if self.cohere_api_key:
            available.append('cohere')
        if self.qwen_api_key:
            available.append('qwen')
        if self.deepseek_api_key:
            available.append('deepseek')
        
        return available
    
    def get_oai_config_list(self, providers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate OAI_CONFIG_LIST format configuration
        
        Args:
            providers: List of providers to include. If None, includes all available providers
        
        Returns:
            List of configuration dictionaries in OAI_CONFIG_LIST format
        """
        if providers is None:
            providers = self.get_available_providers()
        
        config_list = []
        for provider in providers:
            try:
                config = self.get_llm_config(provider)
                config_list.append(config)
            except ValueError as e:
                logger.warning(f"Skipping provider {provider}: {e}")
        
        return config_list
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'available_providers': self.get_available_providers()
        }
        
        # Check if at least one provider is available
        if not results['available_providers']:
            results['valid'] = False
            results['errors'].append("No LLM providers configured. Please set at least one API key.")
        
        # Check if default provider is available
        if self.default_llm_provider not in results['available_providers']:
            results['warnings'].append(
                f"Default provider '{self.default_llm_provider}' is not available. "
                f"Available providers: {', '.join(results['available_providers'])}"
            )
        
        return results


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def validate_environment() -> bool:
    """
    Validate environment configuration
    
    Returns:
        True if configuration is valid, False otherwise
    """
    validation = config.validate_config()
    
    if not validation['valid']:
        for error in validation['errors']:
            logger.error(error)
        return False
    
    for warning in validation['warnings']:
        logger.warning(warning)
    
    logger.info(f"Available LLM providers: {', '.join(validation['available_providers'])}")
    return True
