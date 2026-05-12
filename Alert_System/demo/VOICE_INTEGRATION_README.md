# Integración de Voz en Demo ATC - Guía de Uso

## Overview

Se ha integrado funcionalidad de dictado de voz en el demo CLI del ATC Alert System utilizando una API REST ASR corriendo en Docker.

## Arquitectura

```
Demo CLI (host) → Graba audio → API REST (Docker) → Faster Whisper → Transcripción → Demo CLI
```

## Requisitos

### Dependencias Python
```bash
pip install sounddevice>=0.4.6 soundfile>=0.13.1 requests>=2.31.0 fastapi>=0.104.0 uvicorn[standard]>=0.24.0 python-multipart>=0.0.6
```

### Dependencias del Sistema (Linux)
```bash
sudo apt-get install libasound2-dev portaudio19-dev
```

### Docker
- Docker y Docker Compose instalados
- Contenedor `tesis_asr` disponible

## Configuración

### 1. Iniciar el Contenedor ASR

```bash
# Iniciar el servicio con API REST
docker-compose up -d

# Verificar que esté corriendo
docker ps
curl http://localhost:8000/health
```

### 2. Verificar Instalación

```bash
# Ejecutar script de prueba
python test_voice_integration.py
```

### 3. Iniciar el Demo con Voz

```bash
# Cargar estado inicial
python -m Alert_System.demo.demo_cli --state Alert_System/demo/config/initial_state.json
```

## Comandos de Voz

### Grabación Manual
```bash
atc-demo> dict
🎤 Grabando... Presione Enter para detener
[Habla: "AAL123 descend to 4000"]
✅ Audio guardado: atc_recording_XYZ.wav (32000 bytes)
🎯 Transcripción: 'AAL123 descend to 4000'
[i] Callsign: AAL123
[i] Tipo: descent
[i] Parámetros: {'target_altitude': 4000}
[>] Ejecutando pipeline...
--------------------------------------------------
  ESTADO PROYECTADO:
    AAL123: ALT=4000 HDG=90 SPD=250
--------------------------------------------------
  ALERTAS GENERADAS: 1
    🔴 [CRITICAL] altitude_violation
       Altitude 4000ft below MSA 5000ft
  Decisión automática del sistema: ROLLBACK
  Tiempo de ejecución: 15.2 ms
```

### Grabación Temporizada
```bash
atc-demo> dict timed 5
🎤 Grabando 5 segundos...
⏱️ Tiempo restante: 2.1s
✅ Audio guardado: atc_recording_ABC.wav (80000 bytes)
[...procesamiento normal...]
```

### Configuración de API
```bash
# Verificar URL actual
atc-demo> set asr_url
URL actual: http://localhost:8000

# Cambiar URL
atc-demo> set asr_url http://192.168.1.100:8000
✅ URL de API ASR configurada: http://192.168.1.100:8000
```

### Tests y Diagnóstico
```bash
# Probar grabación de audio
atc-demo> test audio
🧪 Probando grabación de audio...
✅ Test de audio exitoso

# Probar conexión con API
atc-demo> test asr
🧪 Probando conexión con API ASR...
✅ API ASR respondiendo correctamente
```

## Comandos Disponibles

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `dict` | Grabación manual con Enter para detener | `dict` |
| `dict timed <seg>` | Grabación temporizada | `dict timed 5` |
| `set asr_url <url>` | Configurar URL de API ASR | `set asr_url http://localhost:8000` |
| `test audio` | Probar grabación de audio | `test audio` |
| `test asr` | Probar conexión con API | `test asr` |

## Flujo de Procesamiento

1. **Grabación**: El comando `dict` activa el grabador de audio
2. **Captura**: Audio se graba en formato WAV (16kHz, mono)
3. **API Request**: Archivo se envía a `/transcribe` endpoint
4. **Transcripción**: WhisperATC procesa el audio
5. **Parsing**: El resultado se parsea con SimpleATCParser
6. **Pipeline**: Se ejecuta el pipeline completo de alertas
7. **Resultados**: Se muestran alertas y decisión automática

## Formatos de Audio Soportados

- **WAV** (preferido para grabación)
- **MP3**
- **FLAC**
- **M4A**
- **OGG**
- **WebM**

## Configuración de Audio

### Dispositivos de Audio
```bash
# Ver dispositivos disponibles
atc-demo> help
# (busca "dispositivos de audio" en la ayuda completa)

# Ver dispositivos directamente
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Problemas Comunes

**"No se pudo inicializar grabador de audio"**
```bash
# Instalar dependencias del sistema
sudo apt-get install libasound2-dev portaudio19-dev

# Verificar permisos de audio
groups $USER | grep audio
# Si no está en el grupo audio:
sudo usermod -a -G audio $USER
# Reiniciar sesión
```

**"Error conectando con API ASR"**
```bash
# Verificar contenedor
docker ps | grep tesis_asr

# Verificar logs
docker logs tesis_asr

# Reiniciar si es necesario
docker-compose restart
```

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Transcripción
```bash
POST http://localhost:8000/transcribe
Content-Type: multipart/form-data

file: <audio_file>
```

### Models Info
```bash
GET http://localhost:8000/models
```

## Performance

### Tiempos Típicos
- **Grabación**: 2-5 segundos (depende de duración)
- **Transcripción**: 1-3 segundos (WhisperATC optimizado)
- **Pipeline**: 10-50ms (evaluación de reglas)

### Optimizaciones
- **Modelo**: WhisperATC v3 optimizado para ATC
- **Cache**: Embeddings cache para reglas genéricas
- **Timeouts**: 30 segundos para transcripción
- **Formato**: WAV 16kHz mono (óptimo para ASR)

## Troubleshooting

### Error: "Grabador de audio no disponible"
```bash
# Verificar instalación de sounddevice
pip show sounddevice

# Reinstalar si es necesario
pip uninstall sounddevice
pip install sounddevice
```

### Error: "API ASR no responde"
```bash
# Verificar que el contenedor esté corriendo
curl http://localhost:8000/health

# Si no responde:
docker-compose down
docker-compose up -d
```

### Error: "No se pudo transcribir el audio"
```bash
# Verificar logs del contenedor
docker logs tesis_asr

# Probar con archivo conocido
curl -X POST -F "file=@test.wav" http://localhost:8000/transcribe
```

## Tips de Uso

### Para Mejor Calidad de Transcripción
1. **Habla claro y a volumen normal**
2. **Usa terminología ATC estándar**
3. **Evita ruido de fondo**
4. **Mantén el micrófono cerca (20-30cm)**

### Comandos ATC que Funcionan Bien
- `AAL123 climb to 5000`
- `UAL456 descend to FL240`
- `DLH789 heading 090`
- `BA123 speed 250`
- `AAL123 cleared for takeoff runway 09L`

### Flujo de Trabajo Recomendado
1. **Cargar estado**: `load config/initial_state.json`
2. **Probar audio**: `test audio`
3. **Probar API**: `test asr`
4. **Dictar comando**: `dict`
5. **Revisar resultados**: Alertas y decisión automática

## Próximos Mejoras

1. **Streaming**: Transcripción en tiempo real
2. **VAD**: Detección automática de voz
3. **Noise Reduction**: Integración con DeepFilterNet
4. **Multiple Models**: Selección de modelo ASR
5. **Confidence Scores**: Mostrar confianza de transcripción

## Soporte

Para problemas o preguntas:
1. Revisa este README
2. Ejecuta `test_voice_integration.py`
3. Verifica logs de Docker
4. Revisa la configuración de audio del sistema
