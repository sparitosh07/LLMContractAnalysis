"""Configuration management for Azure Function."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OpenAIConfig:
    """OpenAI service configuration."""
    api_key: str
    endpoint: str
    api_version: str = "2024-02-01"
    embedding_model: str = "text-embedding-ada-002"
    deployment_name: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create config from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_ENDPOINT")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not endpoint:
            raise ValueError("OPENAI_ENDPOINT environment variable is required")
            
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME")
        )


@dataclass
class SearchConfig:
    """Azure Cognitive Search configuration."""
    endpoint: str
    api_key: str
    index_name: str
    api_version: str = "2023-11-01"
    
    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Create config from environment variables."""
        endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
        api_key = os.getenv("SEARCH_SERVICE_KEY") 
        index_name = os.getenv("SEARCH_INDEX_NAME")
        
        if not endpoint:
            raise ValueError("SEARCH_SERVICE_ENDPOINT environment variable is required")
        if not api_key:
            raise ValueError("SEARCH_SERVICE_KEY environment variable is required")
        if not index_name:
            raise ValueError("SEARCH_INDEX_NAME environment variable is required")
            
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            index_name=index_name,
            api_version=os.getenv("SEARCH_API_VERSION", "2023-11-01")
        )


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    batch_size: int = 10
    use_rcts: bool = True
    use_nltk: bool = False
    encoding_name: str = "cl100k_base"
    timeout_seconds: int = 300
    
    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Create config from environment variables."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            use_rcts=os.getenv("USE_RCTS", "true").lower() in ("true", "1", "yes"),
            use_nltk=os.getenv("USE_NLTK", "false").lower() in ("true", "1", "yes"),
            encoding_name=os.getenv("ENCODING_NAME", "cl100k_base"),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "300"))
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    openai: OpenAIConfig
    search: SearchConfig
    processing: ProcessingConfig
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create complete config from environment variables."""
        try:
            return cls(
                openai=OpenAIConfig.from_env(),
                search=SearchConfig.from_env(),
                processing=ProcessingConfig.from_env(),
                debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
            )
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise


def validate_config(config: AppConfig) -> Dict[str, Any]:
    """Validate configuration and return validation results."""
    issues = []
    
    # Validate OpenAI config
    if not config.openai.api_key.strip():
        issues.append("OpenAI API key is empty")
    if not config.openai.endpoint.startswith("https://"):
        issues.append("OpenAI endpoint must be HTTPS URL")
    
    # Validate Search config  
    if not config.search.endpoint.startswith("https://"):
        issues.append("Search endpoint must be HTTPS URL")
    if not config.search.api_key.strip():
        issues.append("Search API key is empty")
    
    # Validate processing config
    if config.processing.chunk_size < 100:
        issues.append("Chunk size must be at least 100")
    if config.processing.chunk_overlap >= config.processing.chunk_size:
        issues.append("Chunk overlap must be less than chunk size")
    if config.processing.batch_size < 1:
        issues.append("Batch size must be at least 1")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def get_app_config() -> AppConfig:
    """Get validated application configuration."""
    config = AppConfig.from_env()
    validation = validate_config(config)
    
    if not validation["valid"]:
        error_msg = f"Configuration validation failed: {'; '.join(validation['issues'])}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration loaded and validated successfully")
    return config


# Environment variable documentation
ENV_VARS_HELP = {
    # Required
    "OPENAI_API_KEY": "Azure OpenAI API key",
    "OPENAI_ENDPOINT": "Azure OpenAI endpoint URL",
    "SEARCH_SERVICE_ENDPOINT": "Azure Cognitive Search endpoint URL", 
    "SEARCH_SERVICE_KEY": "Azure Cognitive Search API key",
    "SEARCH_INDEX_NAME": "Azure Cognitive Search index name",
    
    # Optional - OpenAI
    "OPENAI_API_VERSION": "OpenAI API version (default: 2024-02-01)",
    "OPENAI_EMBEDDING_MODEL": "Embedding model name (default: text-embedding-ada-002)",
    "OPENAI_DEPLOYMENT_NAME": "Optional deployment name for OpenAI",
    
    # Optional - Search
    "SEARCH_API_VERSION": "Search API version (default: 2023-11-01)",
    
    # Optional - Processing
    "CHUNK_SIZE": "Maximum chunk size in tokens (default: 1000)",
    "CHUNK_OVERLAP": "Chunk overlap in tokens (default: 200)", 
    "MAX_RETRIES": "Maximum retry attempts (default: 3)",
    "BATCH_SIZE": "Batch size for processing (default: 10)",
    "USE_RCTS": "Use recursive character text splitter (default: true)",
    "USE_NLTK": "Use NLTK for sentence splitting (default: false)",
    "ENCODING_NAME": "Token encoding name (default: cl100k_base)",
    "TIMEOUT_SECONDS": "Request timeout in seconds (default: 300)",
    
    # Optional - General
    "DEBUG": "Enable debug logging (default: false)"
}


def print_env_vars_help():
    """Print help for environment variables."""
    print("Required Environment Variables:")
    required_vars = [
        "OPENAI_API_KEY", "OPENAI_ENDPOINT", 
        "SEARCH_SERVICE_ENDPOINT", "SEARCH_SERVICE_KEY", "SEARCH_INDEX_NAME"
    ]
    
    for var in required_vars:
        print(f"  {var}: {ENV_VARS_HELP[var]}")
    
    print("\nOptional Environment Variables:")
    optional_vars = [k for k in ENV_VARS_HELP.keys() if k not in required_vars]
    
    for var in optional_vars:
        print(f"  {var}: {ENV_VARS_HELP[var]}")