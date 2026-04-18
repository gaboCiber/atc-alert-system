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

# Noise reduction (optional - requires separate Python 3.9 environment)
try:
    from .noise_reduction import DeepFilterNetWrapper, NoiseReductionError, TempAudioManager
    __noise_reduction_available = True
except ImportError:
    __noise_reduction_available = False
    DeepFilterNetWrapper = None
    NoiseReductionError = None
    TempAudioManager = None

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

# Añadir noise reduction al __all__ si está disponible
if __noise_reduction_available:
    __all__.extend(['DeepFilterNetWrapper', 'NoiseReductionError', 'TempAudioManager'])
