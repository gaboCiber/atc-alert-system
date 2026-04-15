"""
Clase base abstracta para modelos ASR.
Define la interfaz común para todos los modelos de transcripción.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TranscriptionResult:
    """Resultado de una transcripción."""
    text: str
    file_path: str
    model_name: str
    timestamps: Optional[List[Dict[str, Any]]] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseASRModel(ABC):
    """
    Clase base abstracta para modelos ASR.
    
    Todas las implementaciones de modelos ASR deben heredar de esta clase
    e implementar los métodos abstractos.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Inicializa el modelo ASR.
        
        Args:
            model_name: Nombre/identificador del modelo
            device: Dispositivo ("cpu", "cuda", "auto")
            prompt: Prompt opcional para contextualización ATC
            **kwargs: Parámetros adicionales específicos del modelo
        """
        self.model_name = model_name
        self.device = device
        self.prompt = prompt
        self.config = kwargs
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """
        Carga el modelo en memoria.
        
        Este método debe ser llamado antes de transcribir.
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Descarga el modelo de la memoria.
        
        Libera recursos (GPU/CPU) después de terminar de usar el modelo.
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """
        Transcribe un archivo de audio.
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            TranscriptionResult con el texto y metadata
        """
        pass
    
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[TranscriptionResult]:
        """
        Transcribe múltiples archivos de audio.
        
        Args:
            audio_paths: Lista de rutas a archivos de audio
            show_progress: Mostrar barra de progreso
            
        Returns:
            Lista de TranscriptionResult
        """
        if not self._is_loaded:
            self.load()
        
        results = []
        
        # Importar tqdm aquí para no depender de él globalmente
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc=f"Transcribiendo con {self.model_name}")
            except ImportError:
                iterator = audio_paths
        else:
            iterator = audio_paths
        
        for audio_path in iterator:
            try:
                result = self.transcribe(audio_path)
                results.append(result)
            except Exception as e:
                # Crear resultado con error
                error_result = TranscriptionResult(
                    text="",
                    file_path=str(audio_path),
                    model_name=self.model_name,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado en memoria."""
        return self._is_loaded
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
