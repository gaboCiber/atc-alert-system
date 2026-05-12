"""
API Server ASR para transcripción de voz ATC usando FastAPI.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from pathlib import Path
import logging

from ASR.transcription import WhisperATCModel, FasterWhisperModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="ATC ASR API", 
    version="1.0.0",
    description="API de transcripción de voz optimizada para Control de Tráfico Aéreo"
)

# Modelo ASR cargado globalmente
try:
    logger.info("Cargando modelo FasterWhisper...")
    model = FasterWhisperModel(model_name="turbo")
    logger.info("✅ Modelo ASR cargado exitosamente")
except Exception as e:
    logger.error(f"❌ Error cargando modelo ASR: {e}")
    model = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ATC ASR API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return JSONResponse(
            content={"status": "unhealthy", "error": "Model not loaded"},
            status_code=503
        )
    
    return {
        "status": "healthy", 
        "model": model.model_version if hasattr(model, 'model_version') else "whisperatc_v3",
        "service": "ATC ASR Transcription API"
    }

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe archivo de audio usando ASR optimizado para ATC.
    
    Args:
        file: Archivo de audio a transcribir (.wav, .mp3, .flac, .m4a)
    
    Returns:
        JSON con transcripción y metadatos
    """
    if model is None:
        return JSONResponse(content={
            "success": False,
            "error": "ASR service not available - model not loaded"
        }, status_code=503)
    
    try:
        # Validar formato de archivo
        filename = file.filename.lower()
        supported_formats = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm')
        
        if not any(filename.endswith(fmt) for fmt in supported_formats):
            raise HTTPException(
                status_code=400, 
                detail=f"Formato no soportado. Formatos soportados: {', '.join(supported_formats)}"
            )
        
        # Leer contenido del archivo
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Archivo vacío")
        
        # Validar tamaño máximo (50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"Archivo demasiado grande. Máximo: {max_size // (1024*1024)}MB"
            )
        
        # Guardar archivo temporal
        file_suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"Transcribiendo archivo: {file.filename} ({len(content)} bytes)")
        
        # Transcribir usando el modelo ASR directamente
        result = model.transcribe(tmp_path)
        
        # Limpiar archivo temporal
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning(f"No se pudo eliminar archivo temporal: {tmp_path}")
        
        # Construir respuesta
        response_data = {
            "success": True,
            "text": result.text,
            "model": getattr(result, 'model_name', 'whisperatc_v3'),
            "duration": getattr(result, 'duration', None),
            "filename": file.filename
        }
        
        logger.info(f"✅ Transcripción exitosa: '{result.text[:50]}...'")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en transcripción: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"Error en transcripción: {str(e)}"
        }, status_code=500)

@app.get("/models")
async def list_models():
    """Lista modelos ASR disponibles."""
    return {
        "current_model": {
            "name": getattr(model, 'model_name', 'faster-whisper-turbo') if model else None,
            "type": "Faster Whisper",
            "status": "loaded" if model else "not_loaded"
        },
        "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"],
        "max_file_size": "50MB"
    }

if __name__ == "__main__":
    # Ejecutar servidor directamente para desarrollo
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
