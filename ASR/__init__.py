"""
Módulo ASR (Automatic Speech Recognition) para ATC.
"""

from .normalization import ATCTextNormalizer, quick_normalize
from .evaluation import (
    ASREvaluator,
    ASREvaluationResult,
    BaseDataLoader,
    EcnaDataLoader,
    Atco2DataLoader,
)

from .transcription import WhisperModel, FasterWhisperModel, WhisperATCModel, HuggingFaceModel, TranscriptionPipeline, TranscriptionResult

__all__ = [
    'ATCTextNormalizer',
    'quick_normalize',
    'ASREvaluator',
    'ASREvaluationResult',
    'BaseDataLoader',
    'EcnaDataLoader',
    'Atco2DataLoader',
    'WhisperModel',
    'FasterWhisperModel',
    'WhisperATCModel',
    'HuggingFaceModel',
    'TranscriptionPipeline',
    'TranscriptionResult',
]
