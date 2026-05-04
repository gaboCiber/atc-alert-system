"""Factory for creating LLM clients with appropriate instructor wrapper.

Shared by Knowledge_Extractor and Alert_System.
"""
import instructor
from openai import OpenAI
from dataclasses import dataclass
from typing import Optional, Any, Tuple, Literal


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str = "llama3.2"
    provider: Literal["openai", "gemini", "anthropic"] = "openai"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120


def create_instructor_client(config: ModelConfig) -> Tuple[Any, instructor.Mode]:
    """
    Create an instructor client based on provider type.
    
    Args:
        config: Model configuration with provider, base_url, api_key, name
        
    Returns:
        Tuple of (client, mode) where mode is the instructor Mode to use
    """
    if config.provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini provider. "
                "Install with: pip install google-generativeai"
            )
        
        genai.configure(api_key=config.api_key)
        client = instructor.from_gemini(
            genai.GenerativeModel(model_name=config.name),
            mode=instructor.Mode.GEMINI_JSON,
        )
        return client, instructor.Mode.GEMINI_JSON
    
    elif config.provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic is required for Anthropic provider. "
                "Install with: pip install anthropic"
            )
        
        client = instructor.from_anthropic(
            Anthropic(api_key=config.api_key),
            mode=instructor.Mode.ANTHROPIC_JSON,
        )
        return client, instructor.Mode.ANTHROPIC_JSON
    
    else:  # openai (default) or ollama
        # Asegurar que el base_url tenga el formato correcto para Ollama
        base_url = config.base_url
        if config.provider == "ollama" and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        
        openai_client = OpenAI(
            base_url=base_url,
            api_key=config.api_key,
        )
        client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        return client, instructor.Mode.JSON_SCHEMA


def create_raw_client(config: ModelConfig) -> Optional[OpenAI]:
    """
    Create a raw OpenAI client for fallback (when structured extraction fails).
    
    For providers without OpenAI-compatible endpoints, returns None.
    
    Args:
        config: Model configuration
        
    Returns:
        OpenAI client or None if raw fallback not available
    """
    if config.provider == "gemini":
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=config.api_key,
        )
    
    elif config.provider == "anthropic":
        return None
    
    else:  # openai or ollama
        # Asegurar que el base_url tenga el formato correcto para Ollama
        base_url = config.base_url
        if config.provider == "ollama" and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        
        return OpenAI(
            base_url=base_url,
            api_key=config.api_key,
        )
