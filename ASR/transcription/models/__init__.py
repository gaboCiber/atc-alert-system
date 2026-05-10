"""
Modelos ASR disponibles para transcripción.
"""

from .whisper import WhisperModel, WhisperPromptedModel
from .huggingface import HuggingFaceModel, WhisperATCModel
from .faster_whisper import FasterWhisperModel
from .whisper_cpp import WhisperCppModel

__all__ = [
    "WhisperModel",
    "WhisperPromptedModel",
    "HuggingFaceModel",
    "WhisperATCModel",
    "FasterWhisperModel",
    "WhisperCppModel",
]
