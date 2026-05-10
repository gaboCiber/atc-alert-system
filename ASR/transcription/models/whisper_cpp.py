"""
Implementación de WhisperCpp usando pywhispercpp bindings.
Optimizado para rendimiento y precisión en transcripción ATC.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings
import os

from ..base import BaseASRModel, TranscriptionResult


class WhisperCppModel(BaseASRModel):
    """
    Implementación de Whisper usando pywhispercpp (whisper.cpp bindings).
    
    Optimizado para rendimiento manteniendo precisión en transcripción ATC.
    Usa modelos GGML para procesamiento eficiente en CPU/GPU.
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        prompt: Optional[str] = None,
        n_threads: Optional[int] = None,
        language: str = "auto",
        temperature: float = 0.0,
        beam_search: bool = False,
        suppress_blank: bool = True,
        **kwargs
    ):
        """
        Inicializa WhisperCppModel.
        
        Args:
            model_name: Nombre del modelo GGML (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Dispositivo ("auto", "cpu", "cuda", "vulkan")
            prompt: Prompt inicial para contextualización ATC
            n_threads: Número de threads (auto-detecta si None)
            language: Idioma ("auto", "en", "es", etc.)
            temperature: Temperatura para muestreo (0.0 = más determinista)
            beam_search: Usar beam search (mejor calidad, más lento)
            suppress_blank: Suprimir tokens en blanco
            **kwargs: Parámetros adicionales de whisper.cpp
        """
        super().__init__(model_name=model_name, device=device, prompt=prompt, **kwargs)
        
        self.n_threads = n_threads
        self.language = language
        self.temperature = temperature
        self.beam_search = beam_search
        self.suppress_blank = suppress_blank
        
        # Backend detection
        self._backend = self._detect_backend()
        
        # Model cache
        self._model = None
        self._prompt_tokens = None
        
    def _detect_backend(self) -> str:
        """Detecta el mejor backend disponible."""
        if self.device == "auto":
            # Prioridad: CUDA > Vulkan > CPU
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            
            # Verificar Vulkan (más complejo de detectar)
            # Por ahora, default a CPU
            return "cpu"
        
        return self.device.lower()
    
    def _prepare_prompt_tokens(self, prompt: Optional[str] = None) -> Optional[List[int]]:
        """
        Convierte prompt de texto a tokens para whisper.cpp.
        
        Args:
            prompt: Texto del prompt
            
        Returns:
            Lista de tokens o None si no hay prompt
        """
        if not prompt:
            return None
        
        try:
            # Importar aquí para evitar dependencias tempranas
            from pywhispercpp.utils import tokenize_text
            return tokenize_text(prompt)
        except (ImportError, AttributeError):
            # Si no hay tokenización disponible o falla, devolver None
            # whisper.cpp manejará la conversión o usará initial_prompt
            return None
    
    def load(self) -> None:
        """Carga el modelo whisper.cpp."""
        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise ImportError(
                "pywhispercpp no está instalado. "
                "Instala con: pip install pywhispercpp\n"
                "Para soporte CUDA: GGML_CUDA=1 pip install pywhispercpp"
            )
        
        try:
            # Configurar backend primero (antes de importar)
            if self._backend == "cuda":
                os.environ["GGML_CUDA"] = "1"
            elif self._backend == "vulkan":
                os.environ["GGML_VULKAN"] = "1"
            
            # Cargar modelo con parámetros básicos
            self._model = Model(self.model_name)
            
            # Los parámetros se configuran durante la transcripción, no durante la carga
            # La API nueva maneja esto diferente
            
            # Preparar prompt tokens si hay prompt
            if self.prompt:
                self._prompt_tokens = self._prepare_prompt_tokens(self.prompt)
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo whisper.cpp: {e}")
    
    def unload(self) -> None:
        """Descarga el modelo y libera recursos."""
        if self._model:
            # whisper.cpp no tiene método explícito de descarga
            # Simplemente eliminamos la referencia
            self._model = None
            self._prompt_tokens = None
            self._is_loaded = False
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe un archivo de audio usando whisper.cpp.
        
        Args:
            audio_path: Ruta al archivo de audio
            prompt: Prompt opcional para esta transcripción
            
        Returns:
            TranscriptionResult con el texto y metadata
        """
        if not self._is_loaded:
            self.load()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Archivo de audio no encontrado: {audio_path}")
        
        try:
            # Preparar prompt para esta transcripción
            current_prompt = prompt or self.prompt
            
            # Preparar parámetros para transcripción
            transcribe_params = {}
            
            # Configurar parámetros básicos
            transcribe_params["print_realtime"] = False
            transcribe_params["print_progress"] = False
            transcribe_params["print_special"] = False
            transcribe_params["print_timestamps"] = False
            transcribe_params["token_timestamps"] = False
            
            # Configurar idioma
            if self.language != "auto":
                transcribe_params["language"] = self.language
            
            transcribe_params["temperature"] = self.temperature
            transcribe_params["suppress_blank"] = self.suppress_blank
            
            # Configurar threads
            if self.n_threads:
                transcribe_params["n_threads"] = self.n_threads
            
            # Configurar estrategia de decodificación
            if self.beam_search:
                transcribe_params["beam_search"] = {"beam_size": 5, "patience": 1}
                transcribe_params["greedy"] = {"best_of": 1}
            else:
                transcribe_params["greedy"] = {"best_of": 5}
                transcribe_params["beam_search"] = {"beam_size": -1, "patience": -1}
            
            # Configurar prompt
            if current_prompt:
                try:
                    from pywhispercpp.utils import tokenize_text
                    prompt_tokens = tokenize_text(current_prompt)
                    transcribe_params["prompt_tokens"] = prompt_tokens
                    transcribe_params["prompt_n_tokens"] = len(prompt_tokens)
                except ImportError:
                    # Si no hay tokenización, usar como texto inicial
                    transcribe_params["initial_prompt"] = current_prompt
            
            # Realizar transcripción
            segments = self._model.transcribe(str(audio_path), **transcribe_params)
            
            # Extraer texto y timestamps
            text_parts = []
            timestamps = []
            
            for segment in segments:
                if hasattr(segment, 'text') and segment.text.strip():
                    text_parts.append(segment.text.strip())
                    
                    # Extraer timestamps si están disponibles
                    if hasattr(segment, 't0') and hasattr(segment, 't1'):
                        timestamps.append({
                            "start": segment.t0,
                            "end": segment.t1,
                            "text": segment.text.strip()
                        })
            
            full_text = " ".join(text_parts)
            
            # Crear resultado
            result = TranscriptionResult(
                text=full_text,
                file_path=str(audio_path),
                model_name=f"whisper-cpp-{self.model_name}",
                timestamps=timestamps if timestamps else None,
                metadata={
                    "backend": self._backend,
                    "language": self.language,
                    "temperature": self.temperature,
                    "beam_search": self.beam_search,
                    "n_threads": self.n_threads,
                    "prompt_used": bool(current_prompt),
                    "model_type": "whisper.cpp"
                }
            )
            
            return result
            
        except Exception as e:
            # Crear resultado con error
            return TranscriptionResult(
                text="",
                file_path=str(audio_path),
                model_name=f"whisper-cpp-{self.model_name}",
                metadata={
                    "error": str(e),
                    "backend": self._backend,
                    "model_type": "whisper.cpp"
                }
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información detallada del modelo.
        
        Returns:
            Diccionario con información del modelo y configuración
        """
        return {
            "model_name": self.model_name,
            "model_type": "whisper.cpp",
            "backend": self._backend,
            "device": self.device,
            "language": self.language,
            "temperature": self.temperature,
            "beam_search": self.beam_search,
            "n_threads": self.n_threads,
            "prompt": self.prompt,
            "is_loaded": self._is_loaded,
            "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],  # ffmpeg dependent
            "performance_focus": "speed_optimized"
        }
