"""
Implementación de FasterWhisper para transcripción ASR.
FasterWhisper es una implementación optimizada de Whisper usando CTranslate2.
"""

from pathlib import Path
from typing import Optional, Union

from ..base import BaseASRModel, TranscriptionResult


class FasterWhisperModel(BaseASRModel):
    """
    Implementación de FasterWhisper.
    
    Usa faster-whisper (CTranslate2) para transcripción más rápida.
    Compatible con los mismos modelos que Whisper original.
    """
    
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "auto",
        prompt: Optional[str] = None,
        compute_type: str = "int8",
        beam_size: int = 5,
        **kwargs
    ):
        """
        Inicializa FasterWhisper.
        
        Args:
            model_name: Tamaño del modelo o path al modelo (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Dispositivo ("cpu", "cuda", "auto")
            prompt: Prompt inicial para contextualización ATC
            compute_type: Tipo de computación ("int8", "int8_float16", "float16", "float32")
            beam_size: Tamaño del beam para búsqueda
            **kwargs: Parámetros adicionales
        """
        super().__init__(model_name=model_name, device=device, prompt=prompt, **kwargs)
        self.compute_type = compute_type
        self.beam_size = beam_size
    
    def load(self) -> None:
        """Carga el modelo FasterWhisper."""
        try:
            from faster_whisper import WhisperModel as FWModel
        except ImportError:
            raise ImportError(
                "faster-whisper no está instalado. "
                "Instala con: pip install faster-whisper"
            )
        
        # Determinar dispositivo
        if self.device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        # Cargar modelo
        self._model = FWModel(
            self.model_name,
            device=device,
            compute_type=self.compute_type
        )
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
        
        # Preparar parámetros
        transcribe_kwargs = {
            "beam_size": self.beam_size,
        }
        
        # Usar el prompt proporcionado o el del modelo
        effective_prompt = prompt if prompt is not None else self.prompt
        if effective_prompt:
            transcribe_kwargs["initial_prompt"] = effective_prompt
        
        # Agregar parámetros adicionales
        transcribe_kwargs.update(self.config)
        
        # Transcribir
        segments, info = self._model.transcribe(audio_path, **transcribe_kwargs)
        
        # Construir texto completo
        text_parts = []
        timestamps = []
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            timestamps.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        full_text = " ".join(text_parts)
        
        # Metadata
        metadata = {
            "language": info.language,
            "language_probability": info.language_probability,
        }
        
        return TranscriptionResult(
            text=full_text,
            file_path=audio_path,
            model_name=self.model_name,
            timestamps=timestamps if timestamps else None,
            metadata=metadata
        )
