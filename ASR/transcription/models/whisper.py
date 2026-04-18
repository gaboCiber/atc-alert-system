"""
Implementación de OpenAI Whisper para transcripción ASR.
"""

from pathlib import Path
from typing import Optional, Union
import warnings

from ..base import BaseASRModel, TranscriptionResult


class WhisperModel(BaseASRModel):
    """
    Implementación de OpenAI Whisper.
    
    Basado en la API de whisper:
    - whisper.load_model() para cargar
    - model.transcribe() para transcribir
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        prompt: Optional[str] = None,
        fp16: bool = False,
        **kwargs
    ):
        """
        Inicializa Whisper.
        
        Args:
            model_name: Tamaño del modelo (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Dispositivo ("cpu", "cuda", "auto")
            prompt: Prompt inicial para contextualización ATC
            fp16: Usar FP16 (default False para compatibilidad CPU)
            **kwargs: Parámetros adicionales
        """
        super().__init__(model_name=model_name, device=device, prompt=prompt, **kwargs)
        self.fp16 = fp16
    
    def load(self) -> None:
        """Carga el modelo Whisper."""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "whisper no está instalado. "
                "Instala con: pip install git+https://github.com/openai/whisper.git"
            )
        
        # Determinar dispositivo
        if self.device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        # Cargar modelo
        self._model = whisper.load_model(self.model_name).to(device)
        self._is_loaded = True
    
    def unload(self) -> None:
        """Descarga el modelo de la memoria."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            
            # Limpiar caché de CUDA si está disponible
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            prompt: Prompt opcional para esta transcripción específica.
                   Si es None, usa el prompt por defecto del modelo (self.prompt).
            
        Returns:
            TranscriptionResult con el texto y metadata
        """
        if not self._is_loaded:
            self.load()
        
        audio_path = str(audio_path)
        
        # Preparar parámetros de transcripción
        transcribe_kwargs = {
            "fp16": self.fp16,
        }
        
        # Usar el prompt proporcionado o el del modelo
        effective_prompt = prompt if prompt is not None else self.prompt
        if effective_prompt:
            transcribe_kwargs["initial_prompt"] = effective_prompt
        
        # Agregar parámetros adicionales del config
        transcribe_kwargs.update(self.config)
        
        # Transcribir
        result = self._model.transcribe(audio_path, **transcribe_kwargs)
        
        # Extraer texto
        text = result.get("text", "").strip()
        
        # Extraer metadata si está disponible
        metadata = {}
        if "language" in result:
            metadata["language"] = result["language"]
        if "segments" in result:
            metadata["num_segments"] = len(result["segments"])
        
        # Crear resultado
        return TranscriptionResult(
            text=text,
            file_path=audio_path,
            model_name=self.model_name,
            metadata=metadata
        )


class WhisperPromptedModel(WhisperModel):
    """
    Whisper con prompt ATC por defecto.
    
    Es un wrapper conveniente para usar Whisper con el prompt
    de terminología ATC preconfigurado.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        fp16: bool = False,
        **kwargs
    ):
        """
        Inicializa Whisper con prompt ATC.
        
        Args:
            model_name: Tamaño del modelo
            device: Dispositivo
            fp16: Usar FP16
            **kwargs: Parámetros adicionales
        """
        from ..config.prompts import DEFAULT_ATC_PROMPT
        
        super().__init__(
            model_name=model_name,
            device=device,
            prompt=DEFAULT_ATC_PROMPT,
            fp16=fp16,
            **kwargs
        )
