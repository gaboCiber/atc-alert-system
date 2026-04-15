"""
Implementación de modelos HuggingFace para transcripción ASR.
Soporta WhisperATC y otros modelos de HuggingFace.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

from ..base import BaseASRModel, TranscriptionResult


class HuggingFaceModel(BaseASRModel):
    """
    Implementación de modelos HuggingFace usando transformers.
    
    Usa pipeline("automatic-speech-recognition") de transformers.
    Compatible con WhisperATC y otros modelos.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        prompt: Optional[str] = None,
        return_timestamps: bool = False,
        **kwargs
    ):
        """
        Inicializa un modelo HuggingFace.
        
        Args:
            model_name: ID del modelo en HuggingFace (e.g., "jlvdoorn/whisper-large-v2-atco2-asr")
            device: Dispositivo ("cpu", "cuda", "auto")
            prompt: Prompt opcional (algunos modelos HF lo ignoran)
            return_timestamps: Devolver timestamps en la transcripción
            **kwargs: Parámetros adicionales para el pipeline
        """
        super().__init__(model_name=model_name, device=device, prompt=prompt, **kwargs)
        self.return_timestamps = return_timestamps
        self._pipe = None
    
    def load(self) -> None:
        """Carga el pipeline de HuggingFace."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers no está instalado. "
                "Instala con: pip install transformers"
            )
        
        # Determinar dispositivo
        device_map = None
        if self.device == "auto":
            import torch
            device_map = 0 if torch.cuda.is_available() else -1
        elif self.device == "cuda":
            device_map = 0
        else:
            device_map = -1
        
        # Cargar pipeline
        pipe_kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
            "device": device_map,
        }
        
        # Agregar torch_dtype si está configurado
        if "torch_dtype" in self.config:
            pipe_kwargs["torch_dtype"] = self.config["torch_dtype"]
        
        self._pipe = pipeline(**pipe_kwargs)
        self._is_loaded = True
    
    def unload(self) -> None:
        """Descarga el modelo de la memoria."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._is_loaded = False
            
            # Limpiar caché de CUDA si está disponible
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """
        Transcribe un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            TranscriptionResult con el texto y metadata
        """
        if not self._is_loaded:
            self.load()
        
        audio_path = str(audio_path)
        
        # Preparar parámetros de inferencia
        inference_kwargs = {}
        
        if self.return_timestamps:
            inference_kwargs["return_timestamps"] = True
        
        # Agregar otros parámetros del config
        inference_kwargs.update({
            k: v for k, v in self.config.items()
            if k not in ["torch_dtype"]
        })
        
        # Transcribir
        result = self._pipe(audio_path, **inference_kwargs)
        
        # Extraer texto
        text = result.get("text", "").strip()
        
        # Extraer timestamps si están disponibles
        timestamps = None
        if self.return_timestamps and "chunks" in result:
            timestamps = [
                {
                    "start": chunk.get("timestamp", [0, 0])[0] if isinstance(chunk.get("timestamp"), list) else chunk.get("timestamp", 0),
                    "end": chunk.get("timestamp", [0, 0])[1] if isinstance(chunk.get("timestamp"), list) else None,
                    "text": chunk.get("text", "")
                }
                for chunk in result["chunks"]
            ]
        
        # Metadata
        metadata = {}
        if "language" in result:
            metadata["language"] = result["language"]
        
        return TranscriptionResult(
            text=text,
            file_path=audio_path,
            model_name=self.model_name,
            timestamps=timestamps,
            metadata=metadata
        )


class WhisperATCModel(HuggingFaceModel):
    """
    Wrapper específico para WhisperATC (jlvdoorn/whisper-large-vX-atco2-asr).
    
    Este modelo está optimizado para ATC y no requiere prompt.
    """
    
    def __init__(
        self,
        model_version: str = "v3",
        device: str = "auto",
        return_timestamps: bool = True,
        **kwargs
    ):
        """
        Inicializa WhisperATC.
        
        Args:
            model_version: Versión del modelo ("v2" o "v3")
            device: Dispositivo
            return_timestamps: Devolver timestamps
            **kwargs: Parámetros adicionales
        """
        model_name = f"jlvdoorn/whisper-large-{model_version}-atco2-asr"
        
        super().__init__(
            model_name=model_name,
            device=device,
            prompt=None,  # WhisperATC no necesita prompt
            return_timestamps=return_timestamps,
            **kwargs
        )
