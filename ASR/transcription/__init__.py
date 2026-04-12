"""
Módulo de transcripción ASR para ATC.

Este módulo proporciona clases para transcribir audio usando diferentes
modelos ASR (Whisper, HuggingFace, FasterWhisper) con una interfaz unificada.
"""

# Clase base
from .base import BaseASRModel, TranscriptionResult

# Modelos
from .models import WhisperModel, WhisperPromptedModel, HuggingFaceModel, WhisperATCModel, FasterWhisperModel

# Pipeline y output
from .pipeline import TranscriptionPipeline, MultiModelPipeline
from .output import OutputManager

# Prompts
from .config import (
    DEFAULT_ATC_PROMPT,
    MINIMAL_ATC_PROMPT,
    EXTENDED_ATC_PROMPT,
    get_prompt,
    create_custom_prompt,
    AVAILABLE_PROMPTS,
)

__all__ = [
    # Base
    "BaseASRModel",
    "TranscriptionResult",
    # Modelos
    "WhisperModel",
    "WhisperPromptedModel",
    "HuggingFaceModel",
    "WhisperATCModel",
    "FasterWhisperModel",
    # Pipeline
    "TranscriptionPipeline",
    "MultiModelPipeline",
    "OutputManager",
    # Prompts
    "DEFAULT_ATC_PROMPT",
    "MINIMAL_ATC_PROMPT",
    "EXTENDED_ATC_PROMPT",
    "get_prompt",
    "create_custom_prompt",
    "AVAILABLE_PROMPTS",
]
