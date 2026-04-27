"""
Módulo de integración para conectar ASR y KEX con el Alert System.

Este módulo proporciona adaptadores que permiten usar los componentes
existentes de ASR y Knowledge Extractor dentro del pipeline de alertas.
"""

from .asr_adapter import ASRAdapter, TranscriptionContext
from .kex_adapter import KEXAdapter, KnowledgeContext
from .end_to_end_pipeline import EndToEndPipeline

__all__ = [
    # ASR Integration
    "ASRAdapter",
    "TranscriptionContext",
    # KEX Integration
    "KEXAdapter",
    "KnowledgeContext",
    # End-to-End
    "EndToEndPipeline",
]
