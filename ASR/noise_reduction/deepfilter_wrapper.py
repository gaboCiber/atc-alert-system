"""
Wrapper para DeepFilterNet que maneja la ejecución en subprocess con Python 3.9.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union
import shutil


class NoiseReductionError(Exception):
    """Error específico para fallos en la reducción de ruido."""
    pass


class DeepFilterNetWrapper:
    """
    Wrapper para ejecutar DeepFilterNet en un entorno virtual Python 3.9.
    
    Este wrapper maneja la comunicación con el subprocess que ejecuta DeepFilterNet,
    ya que el paquete requiere Python 3.9 y el proyecto principal usa 3.11+.
    
    Args:
        venv_path: Ruta al entorno virtual con Python 3.9 y DeepFilterNet instalado
        device: Dispositivo a usar ("cpu", "cuda", o None para auto-detectar)
        timeout: Timeout en segundos para la ejecución del subprocess
    """
    
    def __init__(
        self,
        venv_path: Union[str, Path],
        device: Optional[str] = None,
        timeout: int = 300
    ):
        self.venv_path = Path(venv_path)
        self.device = device
        self.timeout = timeout
        self._python_exe = self._find_python_executable()
        self._script_path = self._find_subprocess_script()
    
    def _find_python_executable(self) -> Path:
        """Encuentra el ejecutable de Python en el venv."""
        # Posibles ubicaciones del ejecutable de Python
        possible_paths = [
            self.venv_path / "bin" / "python",
            self.venv_path / "bin" / "python3",
            self.venv_path / "Scripts" / "python.exe",
            self.venv_path / "Scripts" / "python3.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise NoiseReductionError(
            f"No se encontró ejecutable de Python en el venv: {self.venv_path}\n"
            f"Buscado en: {[str(p) for p in possible_paths]}"
        )
    
    def _find_subprocess_script(self) -> Path:
        """Encuentra el script de subprocess para ejecutar DeepFilterNet."""
        # El script debe estar en el mismo directorio que este módulo
        module_dir = Path(__file__).parent
        script_path = module_dir / "scripts" / "run_deepfilter.py"
        
        if not script_path.exists():
            raise NoiseReductionError(
                f"No se encontró el script de subprocess: {script_path}\n"
                "Asegúrate de que el archivo run_deepfilter.py exista en ASR/noise_reduction/scripts/"
            )
        
        return script_path
    
    def is_available(self) -> bool:
        """Verifica si el entorno virtual y el script están configurados correctamente."""
        try:
            # Verificar que el Python del venv existe y puede ejecutar código
            result = subprocess.run(
                [str(self._python_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def clean_audio(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sample_rate: Optional[int] = None
    ) -> Path:
        """
        Limpia el audio usando DeepFilterNet.
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_path: Ruta al archivo de salida (opcional). Si no se proporciona,
                        se crea un archivo temporal.
            sample_rate: Sample rate de salida (opcional). DeepFilterNet usa 48kHz
                        internamente por defecto.
        
        Returns:
            Ruta al archivo de audio limpio
        
        Raises:
            NoiseReductionError: Si falla la limpieza del audio
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise NoiseReductionError(f"Archivo de entrada no encontrado: {input_path}")
        
        # Crear archivo temporal si no se especificó salida
        if output_path is None:
            # Usar el nombre base del archivo original para identificación
            base_name = input_path.stem  # nombre sin extensión
            # Usar directorio temporal del sistema (/tmp) para evitar mezclar con archivos originales
            temp_dir = Path(tempfile.gettempdir())
            # Nombre determinístico: /tmp/asr_nr_{basename}_cleaned.wav
            output_path = temp_dir / f"asr_nr_{base_name}_cleaned.wav"
            # Si existe, eliminarlo primero
            if output_path.exists():
                output_path.unlink()
        else:
            output_path = Path(output_path)
            # Asegurar que el directorio de salida existe
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar argumentos para el subprocess
        cmd = [
            str(self._python_exe),
            str(self._script_path),
            "--input", str(input_path),
            "--output", str(output_path),
        ]
        
        if self.device:
            cmd.extend(["--device", self.device])
        
        if sample_rate:
            cmd.extend(["--sample-rate", str(sample_rate)])
        
        try:
            # Ejecutar el subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                # Limpiar archivo temporal si falló
                if output_path.exists():
                    output_path.unlink()
                
                error_msg = result.stderr if result.stderr else "Error desconocido"
                raise NoiseReductionError(
                    f"DeepFilterNet falló con código {result.returncode}:\n{error_msg}"
                )
            
            # Verificar que el archivo de salida se creó
            if not output_path.exists():
                raise NoiseReductionError(
                    "DeepFilterNet no generó el archivo de salida"
                )
            
            return output_path
            
        except subprocess.TimeoutExpired:
            # Limpiar archivo temporal si timeout
            if output_path.exists():
                output_path.unlink()
            raise NoiseReductionError(
                f"Timeout después de {self.timeout}s ejecutando DeepFilterNet"
            )
        
        except Exception as e:
            # Limpiar archivo temporal si hay error
            if output_path.exists():
                output_path.unlink()
            raise NoiseReductionError(f"Error ejecutando DeepFilterNet: {e}")
    
    def clean_audio_with_fallback(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ) -> Path:
        """
        Limpia el audio con fallback al original si falla.
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_path: Ruta al archivo de salida (opcional)
            verbose: Si True, imprime warnings cuando hay fallback
        
        Returns:
            Ruta al archivo limpio, o al original si falló
        """
        try:
            return self.clean_audio(input_path, output_path)
        except NoiseReductionError as e:
            if verbose:
                print(f"⚠️  Noise reduction falló, usando audio original: {e}")
            return Path(input_path)

    def __repr__(self) -> str:
        return f"DeepFilterNetWrapper(venv='{self.venv_path}', device='{self.device}')"
