"""
Módulo de evaluación ASR para ATC.
"""

from .evaluator import (
    ASREvaluator,
    ASREvaluationResult,
)

from .data_loaders import (
    BaseDataLoader,
    EcnaDataLoader,
    Atco2DataLoader,
    HuggingFaceDataLoader,
)

__all__ = [
    # Evaluator
    'ASREvaluator',
    'ASREvaluationResult',

    # Data loaders (new OOP approach)
    'BaseDataLoader',
    'EcnaDataLoader',
    'Atco2DataLoader',
    'HuggingFaceDataLoader',

]
