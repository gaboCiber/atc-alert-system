"""
Pipeline de transcripción ASR.
Orquesta la transcripción de múltiples archivos con un modelo.
"""

from pathlib import Path
from typing import List, Union, Optional
import os

from .base import BaseASRModel, TranscriptionResult
from .output import OutputManager


class TranscriptionPipeline:
    """
    Pipeline para transcripción de audio con ASR.
    
    Orquesta el proceso de transcripción, manejo de resultados
    y guardado en diferentes formatos.
    """
    
    def __init__(
        self,
        model: BaseASRModel,
        output_format: str = "csv",
        show_progress: bool = True,
        append_mode: bool = False
    ):
        """
        Inicializa el pipeline.
        
        Args:
            model: Instancia de un modelo ASR (WhisperModel, HuggingFaceModel, etc.)
            output_format: Formato de salida ("csv" o "json")
            show_progress: Mostrar barra de progreso durante transcripción
            append_mode: Si es True, agrega resultados a archivo existente (solo CSV)
        """
        self.model = model
        self.output_manager = OutputManager(format=output_format)
        self.show_progress = show_progress
        self.append_mode = append_mode
    
    def run(
        self,
        audio_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        model_column: str = "model"
    ) -> List[TranscriptionResult]:
        """
        Ejecuta el pipeline de transcripción.
        
        Args:
            audio_files: Lista de rutas a archivos de audio
            output_path: Ruta al archivo de salida
            model_column: Nombre de columna para modelo (CSV)
            
        Returns:
            Lista de resultados de transcripción
        """
        # Verificar archivos
        valid_files = self._validate_files(audio_files)
        
        if not valid_files:
            raise ValueError("No se encontraron archivos de audio válidos")
        
        # Cargar modelo si es necesario
        if not self.model.is_loaded():
            print(f"Cargando modelo: {self.model.model_name}...")
            self.model.load()
        
        # Transcribir
        print(f"Transcribiendo {len(valid_files)} archivos...")
        results = self.model.transcribe_batch(
            valid_files,
            show_progress=self.show_progress
        )
        
        # Guardar resultados
        print(f"Guardando resultados en: {output_path}")
        if self.append_mode:
            self.output_manager.append(results, output_path, model_column)
        else:
            self.output_manager.save(results, output_path, model_column)
        
        # Reportar estadísticas
        success_count = sum(1 for r in results if not r.metadata.get("error"))
        error_count = len(results) - success_count
        
        print(f"\nTranscripción completada:")
        print(f"  - Exitosos: {success_count}/{len(results)}")
        if error_count > 0:
            print(f"  - Errores: {error_count}")
        
        return results
    
    def run_directory(
        self,
        directory: Union[str, Path],
        output_path: Union[str, Path],
        extensions: tuple = (".mp3", ".wav", ".flac", ".m4a", ".mkv", ".ogg"),
        recursive: bool = True,
        model_column: str = "model"
    ) -> List[TranscriptionResult]:
        """
        Ejecuta el pipeline en todos los archivos de audio de un directorio.
        
        Args:
            directory: Directorio a buscar
            output_path: Ruta al archivo de salida
            extensions: Extensiones de archivo a buscar
            recursive: Buscar recursivamente en subdirectorios
            model_column: Nombre de columna para modelo (CSV)
            
        Returns:
            Lista de resultados de transcripción
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"Directorio no encontrado: {directory}")
        
        # Buscar archivos
        audio_files = []
        
        if recursive:
            for ext in extensions:
                audio_files.extend(directory.rglob(f"*{ext}"))
                audio_files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                audio_files.extend(directory.glob(f"*{ext}"))
                audio_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Eliminar duplicados y ordenar
        audio_files = sorted(set(audio_files))
        
        print(f"Encontrados {len(audio_files)} archivos de audio en {directory}")
        
        return self.run(audio_files, output_path, model_column)
    
    def _validate_files(
        self,
        files: List[Union[str, Path]]
    ) -> List[str]:
        """
        Valida y filtra archivos existentes.
        
        Args:
            files: Lista de rutas de archivos
            
        Returns:
            Lista de rutas válidas como strings
        """
        valid = []
        
        for f in files:
            path = Path(f)
            if path.exists():
                valid.append(str(path))
            else:
                print(f"⚠️  Archivo no encontrado: {f}")
        
        return valid


class MultiModelPipeline:
    """
    Pipeline que ejecuta múltiples modelos en los mismos archivos.
    
    Útil para comparar diferentes modelos en el mismo conjunto de datos.
    """
    
    def __init__(
        self,
        models: List[BaseASRModel],
        output_format: str = "csv",
        show_progress: bool = True
    ):
        """
        Inicializa el pipeline multi-modelo.
        
        Args:
            models: Lista de modelos ASR a ejecutar
            output_format: Formato de salida
            show_progress: Mostrar progreso
        """
        self.models = models
        self.output_manager = OutputManager(format=output_format)
        self.show_progress = show_progress
    
    def run(
        self,
        audio_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        model_column: str = "model"
    ) -> List[TranscriptionResult]:
        """
        Ejecuta todos los modelos en los archivos.
        
        Args:
            audio_files: Lista de archivos de audio
            output_path: Ruta de salida
            model_column: Nombre de columna para modelo (CSV)
            
        Returns:
            Lista combinada de todos los resultados
        """
        all_results = []
        
        for model in self.models:
            print(f"\n{'='*60}")
            print(f"Ejecutando modelo: {model.model_name}")
            print(f"{'='*60}")
            
            pipeline = TranscriptionPipeline(
                model=model,
                output_format="json",  # Guardado temporal en JSON
                show_progress=self.show_progress
            )
            
            results = pipeline.run(audio_files, "/tmp/temp_results.json")
            all_results.extend(results)
        
        # Guardar todos los resultados juntos
        print(f"\nGuardando resultados combinados en: {output_path}")
        self.output_manager.save(all_results, output_path, model_column)
        
        return all_results
