"""
Módulo de reducción de ruido para audio ATC.

Este módulo proporciona integración con DeepFilterNet para limpieza de audio
antes de la transcripción ASR. Como DeepFilterNet requiere Python 3.9,
se ejecuta en un entorno virtual separado mediante subprocess.

Ejemplo de uso:
    from ASR.noise_reduction import DeepFilterNetWrapper
    
    wrapper = DeepFilterNetWrapper(venv_path="./.venv-deepfilter")
    cleaned_path = wrapper.clean_audio("input.wav")
"""

from .deepfilter_wrapper import DeepFilterNetWrapper, NoiseReductionError
from .temp_audio import TempAudioManager

__all__ = [
    "DeepFilterNetWrapper",
    "NoiseReductionError", 
    "TempAudioManager",
]
