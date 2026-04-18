"""
Pipeline de transcripción ASR.
Orquesta la transcripción de múltiples archivos con un modelo.
"""

from pathlib import Path
from typing import List, Union, Optional, Dict, Callable
import os

from .base import BaseASRModel, TranscriptionResult
from .output import OutputManager

# Import noise reduction if available
try:
    from ..noise_reduction import DeepFilterNetWrapper, TempAudioManager, NoiseReductionError
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False


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
        append_mode: bool = False,
        checkpoint_path: Optional[Path] = None,
        prompt_provider: Optional[Callable[[str], Optional[str]]] = None,
        noise_reduction: bool = False,
        deepfilter_venv: Optional[Union[str, Path]] = None,
        noise_reduction_device: Optional[str] = None
    ):
        """
        Inicializa el pipeline.
        
        Args:
            model: Instancia de un modelo ASR (WhisperModel, HuggingFaceModel, etc.)
            output_format: Formato de salida ("csv" o "json")
            show_progress: Mostrar barra de progreso durante transcripción
            append_mode: Si es True, agrega resultados a archivo existente (solo CSV)
            checkpoint_path: Ruta al archivo de checkpoint para resumir transcripciones
            prompt_provider: Callback opcional que recibe audio_path y retorna prompt
                            (o None para usar el prompt por defecto del modelo)
            noise_reduction: Si es True, aplica DeepFilterNet antes de transcribir
            deepfilter_venv: Ruta al entorno virtual Python 3.9 con DeepFilterNet
            noise_reduction_device: Dispositivo para DeepFilterNet ("cpu", "cuda", o None)
        """
        self.model = model
        self.output_manager = OutputManager(format=output_format)
        self.show_progress = show_progress
        self.append_mode = append_mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.prompt_provider = prompt_provider
        self._checkpoint_dict: Dict[str, TranscriptionResult] = {}
        
        # Inicializar noise reduction si está habilitado
        self._noise_reduction = noise_reduction
        self._deepfilter_wrapper = None
        self._temp_manager = None
        
        if noise_reduction:
            if not NOISE_REDUCTION_AVAILABLE:
                raise ImportError(
                    "El módulo de noise reduction no está disponible. "
                    "Asegúrate de que ASR.noise_reduction esté instalado correctamente."
                )
            
            if not deepfilter_venv:
                raise ValueError(
                    "Debes especificar 'deepfilter_venv' para usar noise_reduction. "
                    "Ejemplo: deepfilter_venv='./.venv-deepfilter'"
                )
            
            self._deepfilter_wrapper = DeepFilterNetWrapper(
                venv_path=deepfilter_venv,
                device=noise_reduction_device
            )
            self._temp_manager = TempAudioManager(prefix="asr_noise_reduction_")
        
        # Cargar checkpoint si existe
        if self.checkpoint_path:
            self._load_checkpoint()
    
    def _load_checkpoint(self) -> None:
        """Carga checkpoint existente si está disponible."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            print(f"Cargando checkpoint desde: {self.checkpoint_path}")
            checkpoint_results = self.output_manager.load_checkpoint(self.checkpoint_path)
            if checkpoint_results:
                for result in checkpoint_results:
                    self._checkpoint_dict[result.file_path] = result
                print(f"Checkpoint cargado: {len(self._checkpoint_dict)} transcripciones")
    
    def _apply_noise_reduction(self, audio_path: Union[str, Path]) -> Path:
        """
        Aplica reducción de ruido al audio si está habilitada.
        
        Args:
            audio_path: Ruta al archivo de audio original
        
        Returns:
            Ruta al archivo limpio (o original si falló)
        """
        if not self._noise_reduction or not self._deepfilter_wrapper:
            return Path(audio_path)
        
        try:
            # Crear archivo temporal para el audio limpio
            temp_output = self._temp_manager.create_temp(
                suffix="_cleaned.wav",
                dir=Path(audio_path).parent
            )
            
            # Aplicar DeepFilterNet
            cleaned_path = self._deepfilter_wrapper.clean_audio(
                input_path=audio_path,
                output_path=temp_output
            )
            
            return cleaned_path
            
        except Exception as e:
            # Si falla, usar el audio original y mostrar warning
            if self.show_progress:
                print(f"⚠️  Noise reduction falló para {audio_path}: {e}")
            return Path(audio_path)
    
    def _filter_files_with_checkpoint(
        self,
        audio_files: List[Union[str, Path]]
    ) -> List[str]:
        """
        Filtra archivos basado en el checkpoint.
        
        Args:
            audio_files: Lista de rutas de archivos
            
        Returns:
            Lista de archivos a procesar (nuevos o fallidos)
        """
        if not self._checkpoint_dict:
            return [str(f) for f in audio_files]
        
        filtered = []
        for file in audio_files:
            file_str = str(file)
            if file_str in self._checkpoint_dict:
                result = self._checkpoint_dict[file_str]
                # Reintentar si falló, saltar si fue exitoso
                if result.metadata.get("error"):
                    filtered.append(file_str)
            else:
                # Archivo nuevo, procesar
                filtered.append(file_str)
        
        return filtered
    
    def _save_checkpoint_incremental(self, result: TranscriptionResult) -> None:
        """
        Guarda checkpoint incrementalmente después de cada transcripción.
        
        Args:
            result: Resultado de transcripción a agregar
        """
        # Actualizar dict (siempre, independientemente de checkpoint_path)
        self._checkpoint_dict[result.file_path] = result
        
        # Guardar checkpoint completo solo si hay ruta
        if self.checkpoint_path:
            all_results = list(self._checkpoint_dict.values())
            self.output_manager.save_checkpoint(all_results, self.checkpoint_path)
    
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
        
        # Filtrar archivos según checkpoint
        files_to_process = self._filter_files_with_checkpoint(valid_files)
        
        if self._checkpoint_dict:
            skipped = len(valid_files) - len(files_to_process)
            print(f"Checkpoint: {skipped} archivos ya procesados, {len(files_to_process)} pendientes")
        
        if not files_to_process:
            print("Todos los archivos ya están procesados según el checkpoint")
            return list(self._checkpoint_dict.values())
        
        # Cargar modelo si es necesario
        if not self.model.is_loaded():
            print(f"Cargando modelo: {self.model.model_name}...")
            self.model.load()
        
        # Transcribir con checkpoint incremental
        print(f"Transcribiendo {len(files_to_process)} archivos...")
        results = []
        
        # Importar tqdm aquí para no depender de él globalmente
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(files_to_process, desc=f"Transcribiendo con {self.model.model_name}")
            except ImportError:
                iterator = files_to_process
        else:
            iterator = files_to_process
        
        for audio_path in iterator:
            cleaned_audio_path = None
            try:
                # Aplicar noise reduction si está habilitado
                if self._noise_reduction:
                    cleaned_audio_path = self._apply_noise_reduction(audio_path)
                    audio_to_transcribe = cleaned_audio_path
                else:
                    audio_to_transcribe = audio_path
                
                # Obtener prompt dinámico si hay un provider configurado
                prompt = None
                if self.prompt_provider:
                    prompt = self.prompt_provider(audio_path)
                
                result = self.model.transcribe(audio_to_transcribe, prompt=prompt)
                results.append(result)
                # Guardar checkpoint incrementalmente
                self._save_checkpoint_incremental(result)
            except Exception as e:
                # Crear resultado con error
                error_result = TranscriptionResult(
                    text="",
                    file_path=str(audio_path),
                    model_name=self.model.model_name,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
                # Guardar checkpoint incluso si falló
                self._save_checkpoint_incremental(error_result)
        
        # Combinar con checkpoint existente
        all_results = list(self._checkpoint_dict.values())
        
        # Guardar resultados finales
        print(f"Guardando resultados en: {output_path}")
        if self.append_mode:
            self.output_manager.append(all_results, output_path, model_column)
        else:
            self.output_manager.save(all_results, output_path, model_column)
        
        # Reportar estadísticas
        success_count = sum(1 for r in all_results if not r.metadata.get("error"))
        error_count = len(all_results) - success_count
        
        print(f"\nTranscripción completada:")
        print(f"  - Exitosos: {success_count}/{len(all_results)}")
        if error_count > 0:
            print(f"  - Errores: {error_count}")
        
        return all_results
    
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
