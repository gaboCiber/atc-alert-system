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
    # Try absolute import first (works when running as module or with proper PYTHONPATH)
    from ASR.noise_reduction import DeepFilterNetWrapper, TempAudioManager, NoiseReductionError
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    # Fall back to relative import (works in some execution contexts)
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
                
                # Reemplazar el path del archivo limpio con el original en el resultado
                if self._noise_reduction:
                    result.file_path = str(audio_path)

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
    Soporta checkpoint, append mode, noise reduction global y prompts dinámicos.
    """
    
    def __init__(
        self,
        models: List[BaseASRModel],
        output_format: str = "csv",
        show_progress: bool = True,
        append_mode: bool = False,
        checkpoint_path: Optional[Union[str, Path]] = None,
        noise_reduction: bool = False,
        deepfilter_venv: Optional[Union[str, Path]] = None,
        noise_reduction_device: Optional[str] = None,
        prompt_provider: Optional[Callable[[str], Optional[str]]] = None
    ):
        """
        Inicializa el pipeline multi-modelo.
        
        Args:
            models: Lista de modelos ASR a ejecutar
            output_format: Formato de salida ("csv" o "json")
            show_progress: Mostrar barra de progreso
            append_mode: Si True, agrega resultados a archivo CSV existente
            checkpoint_path: Ruta al archivo de checkpoint para resumir
            noise_reduction: Si True, aplica DeepFilterNet antes de transcribir (global)
            deepfilter_venv: Ruta al entorno virtual Python 3.9 con DeepFilterNet
            noise_reduction_device: Dispositivo para DeepFilterNet ("cpu", "cuda", None)
            prompt_provider: Callback opcional que recibe audio_path y retorna prompt
        """
        self.models = models
        self.output_format = output_format
        self.output_manager = OutputManager(format=output_format)
        self.show_progress = show_progress
        self.append_mode = append_mode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.prompt_provider = prompt_provider
        self._checkpoint_dict: Dict[str, List[TranscriptionResult]] = {}  # model_name -> results
        
        # Inicializar noise reduction si está habilitado
        self._noise_reduction = noise_reduction
        self._deepfilter_wrapper = None
        self._temp_manager = None
        self._cleaned_files: List[Path] = []
        
        if noise_reduction:
            if not NOISE_REDUCTION_AVAILABLE:
                raise ImportError(
                    "El módulo de noise reduction no está disponible. "
                    "Asegúrate de que ASR.noise_reduction esté instalado correctamente."
                )
            
            if not deepfilter_venv:
                raise ValueError(
                    "Debes especificar 'deepfilter_venv' para usar noise_reduction."
                )
            
            self._deepfilter_wrapper = DeepFilterNetWrapper(
                venv_path=deepfilter_venv,
                device=noise_reduction_device
            )
            self._temp_manager = TempAudioManager()
        
        # Cargar checkpoint si existe
        if self.checkpoint_path:
            self._load_checkpoint()
    
    def _get_model_checkpoint_path(self, model_name: str) -> Optional[Path]:
        """
        Genera la ruta de checkpoint individual para un modelo.
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            Path al checkpoint individual o None si no hay checkpoint_path principal
        """
        if not self.checkpoint_path:
            return None
        
        # Crear nombre basado en el checkpoint principal: {basename}_{model_name}.json
        base_checkpoint = self.checkpoint_path
        parent_dir = base_checkpoint.parent
        basename = base_checkpoint.stem  # sin extensión
        
        # Limpiar nombre del modelo para usar en filename (quitar caracteres inválidos)
        safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        model_checkpoint_name = f"{basename}_{safe_model_name}.json"
        return parent_dir / model_checkpoint_name
    
    def _load_checkpoint(self) -> None:
        """Carga checkpoints individuales de cada modelo si están disponibles."""
        loaded_models = 0
        
        for model in self.models:
            model_name = model.model_name
            model_checkpoint = self._get_model_checkpoint_path(model_name)
            
            if model_checkpoint and model_checkpoint.exists():
                print(f"Cargando checkpoint para '{model_name}': {model_checkpoint}")
                checkpoint_results = self.output_manager.load_checkpoint(model_checkpoint)
                if checkpoint_results:
                    self._checkpoint_dict[model_name] = checkpoint_results
                    loaded_models += 1
        
        if loaded_models > 0:
            print(f"✅ Checkpoints cargados: {loaded_models}/{len(self.models)} modelos")
        elif self.checkpoint_path:
            # Intentar cargar checkpoint antiguo (formato consolidado) si existe
            if self.checkpoint_path.exists():
                print(f"Cargando checkpoint consolidado: {self.checkpoint_path}")
                checkpoint_results = self.output_manager.load_checkpoint(self.checkpoint_path)
                if checkpoint_results:
                    for result in checkpoint_results:
                        model = result.model_name
                        if model not in self._checkpoint_dict:
                            self._checkpoint_dict[model] = []
                        self._checkpoint_dict[model].append(result)
                    print(f"✅ Checkpoint consolidado cargado: {len(self._checkpoint_dict)} modelos")
    
    def _save_master_checkpoint(self) -> None:
        """Guarda checkpoint maestro consolidado con todos los modelos."""
        if self.checkpoint_path:
            all_results = []
            for model_results in self._checkpoint_dict.values():
                all_results.extend(model_results)
            self.output_manager.save_checkpoint(all_results, self.checkpoint_path)
            print(f"💾 Checkpoint maestro actualizado: {self.checkpoint_path}")
    
    def _apply_noise_reduction_global(
        self,
        audio_files: List[Union[str, Path]]
    ) -> List[Path]:
        """
        Aplica noise reduction a todos los archivos una sola vez (global).
        
        Args:
            audio_files: Lista de archivos originales
            
        Returns:
            Lista de rutas a archivos limpios
        """
        if not self._noise_reduction or not self._deepfilter_wrapper:
            return [Path(f) for f in audio_files]
        
        print(f"\nAplicando noise reduction a {len(audio_files)} archivos...")
        cleaned_files = []
        
        from tqdm import tqdm
        iterator = tqdm(audio_files, desc="DeepFilterNet") if self.show_progress else audio_files
        
        for audio_path in iterator:
            try:
                # Crear archivo temporal para el audio limpio en /tmp
                base_name = Path(audio_path).stem
                temp_output = self._temp_manager.create_temp(
                    suffix=".wav",
                    dir=None,  # Usar directorio temporal del sistema (/tmp)
                    basename=f"{base_name}_cleaned",
                    deterministic=True  # Nombre determinístico para checkpoint consistency
                )
                
                cleaned_path = self._deepfilter_wrapper.clean_audio(
                    input_path=audio_path,
                    output_path=temp_output
                )
                cleaned_files.append(cleaned_path)
                self._cleaned_files.append(cleaned_path)
                
            except Exception as e:
                print(f"⚠️  Noise reduction falló para {audio_path}: {e}")
                # Fallback al original
                cleaned_files.append(Path(audio_path))
        
        print(f"✅ Noise reduction completado: {len(cleaned_files)} archivos")
        return cleaned_files
    
    def run(
        self,
        audio_files: List[Union[str, Path]],
        output_path: Union[str, Path],
        model_column: str = "model"
    ) -> List[TranscriptionResult]:
        """
        Ejecuta todos los modelos en los archivos.
        
        Si noise_reduction está habilitado, se aplica una sola vez globalmente
        antes de transcribir con todos los modelos.
        
        Args:
            audio_files: Lista de archivos de audio
            output_path: Ruta de salida
            model_column: Nombre de columna para modelo (CSV)
            
        Returns:
            Lista combinada de todos los resultados
        """
        output_path = Path(output_path)
        
        if not audio_files:
            raise ValueError("No se proporcionaron archivos de audio")
        
        # Validar archivos
        valid_files = []
        for f in audio_files:
            path = Path(f)
            if path.exists():
                valid_files.append(str(path))
            else:
                print(f"⚠️  Archivo no encontrado: {f}")
        
        if not valid_files:
            raise ValueError("No se encontraron archivos de audio válidos")
        
        # Aplicar noise reduction globalmente una sola vez si está habilitado
        files_to_transcribe = self._apply_noise_reduction_global(valid_files)
        
        all_results: List[TranscriptionResult] = []
        
        try:
            # Procesar cada modelo
            print(f"\n🔍 DEBUG: Procesando {len(self.models)} modelos: {[m.model_name for m in self.models]}")
            for idx, model in enumerate(self.models):
                model_name = model.model_name
                print(f"\n🔍 DEBUG: Modelo {idx+1}/{len(self.models)}: {model_name}")
                
                # Obtener checkpoint individual para este modelo
                model_checkpoint = self._get_model_checkpoint_path(model_name)
                
                # Verificar si el checkpoint está completo (tiene todos los archivos)
                checkpoint_complete = False
                print(f"🔍 DEBUG: checkpoint_complete inicializado = {checkpoint_complete}")
                print(f"🔍 DEBUG: {model_name} en checkpoint_dict? {model_name in self._checkpoint_dict}")
                if model_name in self._checkpoint_dict:
                    checkpoint_results = self._checkpoint_dict[model_name]
                    checkpoint_files = {r.file_path for r in checkpoint_results if not r.metadata.get("error")}
                    # Usar valid_files (paths originales) para comparación, no files_to_transcribe (paths limpios)
                    required_files = set(files_to_transcribe)
                    
                    # Verificar si todos los archivos requeridos están en el checkpoint sin errores
                    if checkpoint_files >= required_files:
                        checkpoint_complete = True
                        print(f"\n⏭️  Modelo '{model_name}' completado en checkpoint ({len(checkpoint_results)} archivos), saltando...")
                        all_results.extend(checkpoint_results)
                        continue
                    else:
                        # Checkpoint incompleto - continuar desde donde quedó
                        missing_files = required_files - checkpoint_files
                        print(f"\n⏯️  Modelo '{model_name}' checkpoint incompleto ({len(checkpoint_files)}/{len(required_files)}), reanudando...")
            
                print(f"🔍 DEBUG: Antes de if not checkpoint_complete: {checkpoint_complete}")
                if not checkpoint_complete:
                    print(f"\n{'='*60}")
                    print(f"Ejecutando modelo: {model_name}")
                    print(f"{'='*60}")
                    
                    # Crear pipeline individual con checkpoint propio y prompt provider compartido
                    pipeline = TranscriptionPipeline(
                        model=model,
                        output_format="json",  # Guardado temporal
                        show_progress=self.show_progress,
                        prompt_provider=self.prompt_provider,  # Compartido
                        checkpoint_path=model_checkpoint  # Checkpoint individual por modelo
                    )
                    
                    # Transcribir (usando archivos limpios si se aplicó NR)
                    # El archivo temporal se guarda junto al checkpoint individual
                    temp_output = model_checkpoint.parent / f"temp_{model_name.replace('/', '_')}.json" if model_checkpoint else "/tmp/temp_results.json"
                    results = pipeline.run(files_to_transcribe, temp_output)
                    
                    # Reemplazar paths de archivos limpios con originales en los resultados
                    if self._noise_reduction:
                        for i, result in enumerate(results):
                            if i < len(valid_files):
                                result.file_path = valid_files[i]
                    
                    all_results.extend(results)
                    
                    # El checkpoint ya fue guardado por TranscriptionPipeline
                    # Solo actualizar el dict interno
                    self._checkpoint_dict[model_name] = results
                    
                    # Opcional: Guardar checkpoint consolidado maestro
                    if self.checkpoint_path:
                        self._save_master_checkpoint()
        
        finally:
            # Limpiar archivos temporales de audio (noise reduction)
            self._cleanup_temp_files()
        
        # Guardar resultados finales
        print(f"\n{'='*60}")
        print(f"Guardando resultados combinados en: {output_path}")
        print(f"{'='*60}")
        
        if self.append_mode and output_path.exists():
            self.output_manager.append(all_results, output_path, model_column)
            print(f"✅ Resultados añadidos a CSV existente")
        else:
            self.output_manager.save(all_results, output_path, model_column)
            print(f"✅ Resultados guardados en nuevo archivo")
        
        # Reportar estadísticas
        success_count = sum(1 for r in all_results if not r.metadata.get("error"))
        error_count = len(all_results) - success_count
        
        print(f"\nResumen:")
        print(f"  - Modelos: {len(self.models)}")
        print(f"  - Archivos por modelo: {len(valid_files)}")
        print(f"  - Total transcripciones: {len(all_results)}")
        print(f"  - Exitosos: {success_count}")
        if error_count > 0:
            print(f"  - Errores: {error_count}")
        
        return all_results
    
    def _cleanup_temp_files(self) -> None:
        """
        Limpia archivos temporales de audio creados por noise reduction.
        Se ejecuta automáticamente en finally block.
        """
        if not self._cleaned_files:
            return
        
        cleaned_count = 0
        for temp_file in self._cleaned_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️  No se pudo eliminar temporal {temp_file}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Limpiados {cleaned_count} archivos temporales de audio")
    
    def cleanup(self) -> None:
        """
        Limpia todos los recursos temporales manualmente.
        Útil para llamar desde código cliente si es necesario.
        """
        self._cleanup_temp_files()
        if self._temp_manager:
            self._temp_manager.cleanup_all()
    
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
            Lista combinada de todos los resultados
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
