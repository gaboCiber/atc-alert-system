"""
Módulo ASR (Automatic Speech Recognition) para ATC.
"""

from .normalization import ATCTextNormalizer, quick_normalize
from .evaluation import (
    ASREvaluator,
    ASREvaluationResult,
    load_ground_truth,
    load_transcriptions,
)

__all__ = [
    'ATCTextNormalizer',
    'quick_normalize',
    'ASREvaluator',
    'ASREvaluationResult',
    'load_ground_truth',
    'load_transcriptions',
]
