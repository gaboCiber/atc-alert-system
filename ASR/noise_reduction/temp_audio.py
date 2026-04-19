"""
Utilidades para manejo de archivos de audio temporales.
"""

import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import atexit


class TempAudioManager:
    """
    Manejador de archivos de audio temporales.
    
    Asegura que los archivos temporales sean limpiados al finalizar,
    incluso si hay excepciones.
    
    Ejemplo:
        with TempAudioManager() as manager:
            temp_path = manager.create_temp(suffix=".wav")
            # Usar temp_path...
        # Al salir del contexto, el archivo se elimina automáticamente
    """
    
    def __init__(self, prefix: str = "asr_noise_reduction_"):
        self.prefix = prefix
        self._temp_files: list[Path] = []
        self._registered_cleanup = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - limpia todos los archivos temporales."""
        self.cleanup()
        return False
    
    def create_temp(
        self,
        suffix: str = ".wav",
        dir: Optional[Union[str, Path]] = None,
        basename: Optional[str] = None,
        deterministic: bool = False
    ) -> Path:
        """
        Crea un archivo temporal y lo registra para limpieza.
        
        Args:
            suffix: Extensión del archivo (default: .wav)
            dir: Directorio donde crear el archivo (default: sistema temp)
            basename: Nombre base del archivo original para preservar en el temporal
            deterministic: Si es True, usa nombre determinístico (sin random suffix)
        
        Returns:
            Ruta al archivo temporal creado
        """
        # Si se especificó directorio, asegurar que exista
        if dir is not None:
            dir_path = Path(dir)
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            dir_path = Path(tempfile.gettempdir())
        
        # Construir prefijo: usar basename si se proporciona, sino el prefijo por defecto
        if basename:
            temp_prefix = f"{self.prefix}{basename}_"
        else:
            temp_prefix = self.prefix
        
        if deterministic:
            # Nombre determinístico: /tmp/{prefix}{basename}_cleaned.wav
            temp_filename = f"{temp_prefix}cleaned{suffix}"
            temp_path = dir_path / temp_filename
            # Si existe, eliminarlo primero
            if temp_path.exists():
                temp_path.unlink()
            # Crear archivo vacío para reservar el nombre
            temp_path.touch()
        else:
            # Crear archivo temporal con random suffix (comportamiento original)
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=temp_prefix,
                dir=str(dir_path) if dir_path else None
            )
            os.close(fd)
            temp_path = Path(temp_path)
        
        self._temp_files.append(temp_path)
        
        # Registrar cleanup al exit si no está registrado
        if not self._registered_cleanup:
            atexit.register(self._cleanup_at_exit)
            self._registered_cleanup = True
        
        return temp_path
    
    def register_file(self, path: Union[str, Path]) -> Path:
        """
        Registra un archivo existente para limpieza posterior.
        
        Args:
            path: Ruta al archivo a registrar
        
        Returns:
            Path del archivo registrado
        """
        path = Path(path)
        if path not in self._temp_files:
            self._temp_files.append(path)
        
        if not self._registered_cleanup:
            atexit.register(self._cleanup_at_exit)
            self._registered_cleanup = True
        
        return path
    
    def remove_file(self, path: Union[str, Path]) -> bool:
        """
        Elimina un archivo específico y lo remueve del registro.
        
        Args:
            path: Ruta al archivo a eliminar
        
        Returns:
            True si se eliminó, False si no existía
        """
        path = Path(path)
        
        try:
            if path.exists():
                path.unlink()
            
            # Remover del registro si está ahí
            if path in self._temp_files:
                self._temp_files.remove(path)
            
            return True
        except (OSError, IOError):
            return False
    
    def cleanup(self):
        """Elimina todos los archivos temporales registrados."""
        for path in self._temp_files[:]:
            self.remove_file(path)
        
        self._temp_files.clear()
    
    def _cleanup_at_exit(self):
        """Limpieza al salir del programa (via atexit)."""
        self.cleanup()
    
    def __del__(self):
        """Destructor - intenta limpiar si quedan archivos."""
        if self._temp_files:
            self.cleanup()


def quick_temp_copy(
    source: Union[str, Path],
    suffix: str = ".wav",
    temp_manager: Optional[TempAudioManager] = None
) -> Path:
    """
    Crea una copia temporal de un archivo de audio preservando el nombre base.
    
    Args:
        source: Ruta al archivo original
        suffix: Extensión para el archivo temporal
        temp_manager: Manager opcional para registrar el archivo
    
    Returns:
        Ruta al archivo temporal
    """
    import shutil
    
    source = Path(source)
    
    # Extraer nombre base del archivo original (sin extensión)
    basename = source.stem
    
    if temp_manager:
        temp_path = temp_manager.create_temp(suffix=suffix, dir=source.parent, basename=basename)
    else:
        fd, temp_path = tempfile.mkstemp(
            suffix=suffix,
            prefix=f"{basename}_",
            dir=source.parent
        )
        os.close(fd)
        temp_path = Path(temp_path)
    
    shutil.copy2(source, temp_path)
    return temp_path
