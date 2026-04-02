"""Factory for creating LLM clients with appropriate instructor wrapper."""
import instructor
from openai import OpenAI
from typing import Optional, Any, Tuple

from ..config.settings import ModelConfig


def create_instructor_client(config: ModelConfig) -> Tuple[Any, instructor.Mode]:
    """
    Create an instructor client based on provider type.
    
    Args:
        config: Model configuration with provider, base_url, api_key, name
        
    Returns:
        Tuple of (client, mode) where mode is the instructor Mode to use
    """
    if config.provider == "gemini":
        # Native Gemini client
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
        # Anthropic client
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
    
    else:  # openai (default)
        # OpenAI-compatible client (Ollama, OpenAI, etc.)
        openai_client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        client = instructor.from_openai(
            openai_client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        return client, instructor.Mode.JSON_SCHEMA


def create_raw_client(config: ModelConfig) -> Optional[OpenAI]:
    """
    Create a raw OpenAI client for fallback extraction (when structured fails).
    
    For providers without OpenAI-compatible endpoints, this returns None
    to indicate no raw fallback is available.
    
    Args:
        config: Model configuration
        
    Returns:
        OpenAI client or None if raw fallback not available for this provider
    """
    if config.provider == "gemini":
        # For Gemini, we use the OpenAI-compatible endpoint for raw fallback
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=config.api_key,
        )
    
    elif config.provider == "anthropic":
        # Anthropic doesn't have an OpenAI-compatible endpoint
        # Return None to indicate no raw fallback is available
        return None
    
    else:  # openai
        return OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
