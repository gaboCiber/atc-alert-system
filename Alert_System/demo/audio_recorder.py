"""
Módulo de grabación de audio para comandos ATC en el demo CLI.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional
import threading
import time
import logging

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Grabador de audio para comandos ATC."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 max_duration: float = 30.0):
        """
        Inicializa el grabador de audio.
        
        Args:
            sample_rate: Tasa de muestreo (16kHz para ASR)
            channels: Número de canales (1 para mono)
            max_duration: Duración máxima en segundos
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_duration = max_duration
        self.is_recording = False
        self.audio_data = []
        
        # Verificar que sounddevice esté disponible
        try:
            sd.default.samplerate = sample_rate
            sd.default.channels = channels
        except Exception as e:
            logger.warning(f"Error configurando sounddevice: {e}")
    
    def record(self, duration: Optional[float] = None) -> str:
        """
        Graba audio y retorna path al archivo temporal.
        
        Args:
            duration: Duración en segundos (None para manual)
            
        Returns:
            Path al archivo .wav grabado
        """
        try:
            if duration is None:
                return self._record_manual()
            else:
                return self._record_timed(duration)
        except Exception as e:
            logger.error(f"Error en grabación: {e}")
            raise RuntimeError(f"No se pudo grabar audio: {e}")
    
    def _record_manual(self) -> str:
        """Grabación manual con Enter para detener."""
        print("🎤 Grabando... Presione Enter para detener")
        self.is_recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            """Callback para capturar audio."""
            if status:
                logger.warning(f"Status en callback: {status}")
            if self.is_recording:
                self.audio_data.append(indata.copy())
        
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=self.channels,
                              callback=callback):
                input()  # Espera a que el usuario presione Enter
                self.is_recording = False
                
        except Exception as e:
            self.is_recording = False
            raise RuntimeError(f"Error en grabación de audio: {e}")
        
        # Guardar archivo
        return self._save_recording()
    
    def _record_timed(self, duration: float) -> str:
        """Grabación temporizada."""
        print(f"🎤 Grabando {duration} segundos...")
        
        try:
            # Validar duración máxima
            if duration > self.max_duration:
                duration = self.max_duration
                print(f"⚠️ Duración limitada a {self.max_duration} segundos")
            
            recording = sd.rec(int(duration * self.sample_rate),
                              samplerate=self.sample_rate,
                              channels=self.channels)
            
            # Mostrar progreso
            start_time = time.time()
            while sd.get_stream().active:
                elapsed = time.time() - start_time
                if elapsed < duration:
                    remaining = duration - elapsed
                    print(f"\r⏱️ Tiempo restante: {remaining:.1f}s", end="", flush=True)
                    time.sleep(0.1)
                else:
                    break
            
            sd.wait()  # Espera a que termine la grabación
            print()  # Nueva línea después del contador
            
            self.audio_data = [recording]
            return self._save_recording()
            
        except Exception as e:
            raise RuntimeError(f"Error en grabación temporizada: {e}")
    
    def _save_recording(self) -> str:
        """Guarda la grabación en archivo temporal .wav."""
        try:
            # Concatenar todos los chunks
            if self.audio_data:
                full_recording = np.concatenate(self.audio_data, axis=0)
            else:
                full_recording = np.array([])
            
            # Validar que tengamos datos
            if len(full_recording) == 0:
                raise RuntimeError("No se capturó audio")
            
            # Crear archivo temporal
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                prefix='atc_recording_',
                delete=False
            )
            
            # Guardar como WAV
            sf.write(temp_file.name, full_recording, self.sample_rate)
            
            # Verificar archivo
            file_size = os.path.getsize(temp_file.name)
            if file_size == 0:
                raise RuntimeError("Archivo de audio vacío")
            
            print(f"✅ Audio guardado: {os.path.basename(temp_file.name)} ({file_size} bytes)")
            return temp_file.name
            
        except Exception as e:
            raise RuntimeError(f"Error guardando audio: {e}")
    
    @staticmethod
    def cleanup(file_path: str):
        """Elimina archivo temporal."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Archivo temporal eliminado: {file_path}")
        except OSError as e:
            logger.warning(f"No se pudo eliminar archivo temporal {file_path}: {e}")
    
    @staticmethod
    def check_audio_devices():
        """Verifica dispositivos de audio disponibles."""
        try:
            devices = sd.query_devices()
            print("\n🎧 Dispositivos de audio disponibles:")
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    status = "✅" if dev['max_input_channels'] > 0 else "❌"
                    print(f"  {i}: {dev['name']} ({dev['max_input_channels']} canales) {status}")
            
            # Mostrar dispositivo por defecto
            default_input = sd.default.device[0]
            if default_input >= 0 and default_input < len(devices):
                print(f"\n📍 Dispositivo por defecto: {devices[default_input]['name']}")
                
        except Exception as e:
            print(f"❌ Error verificando dispositivos: {e}")
    
    @staticmethod
    def test_recording() -> bool:
        """Prueba rápida de grabación."""
        try:
            print("🧪 Probando grabación de 2 segundos...")
            recorder = AudioRecorder()
            audio_file = recorder.record(duration=2.0)
            
            # Verificar archivo
            if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                print("✅ Test de grabación exitoso")
                AudioRecorder.cleanup(audio_file)
                return True
            else:
                print("❌ Test de grabación falló")
                return False
                
        except Exception as e:
            print(f"❌ Error en test de grabación: {e}")
            return False
