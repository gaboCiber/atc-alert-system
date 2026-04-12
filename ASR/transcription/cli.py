"""
CLI para el pipeline de transcripción ASR.
"""

import argparse
import sys
from pathlib import Path

# Añadir el project root al path para importar ASR
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ASR.transcription import (
    WhisperModel,
    HuggingFaceModel,
    FasterWhisperModel,
    WhisperATCModel,
    TranscriptionPipeline,
    MultiModelPipeline,
    get_prompt,
)


def create_parser():
    """Crea el parser de argumentos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de transcripción ASR para ATC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Whisper con prompt por defecto
  python -m ASR.transcription.cli --model whisper --model-size large-v3 --input ./audio --output resultados.csv
  
  # WhisperATC (no requiere prompt)
  python -m ASR.transcription.cli --model whisperatc --version v3 --input ./audio --output resultados.json
  
  # HuggingFace con modelo personalizado
  python -m ASR.transcription.cli --model huggingface --hf-model "jlvdoorn/whisper-large-v3-atco2-asr" --input ./audio --output resultados.csv
        """
    )
    
    # Modelo
    parser.add_argument(
        "--model", "-m",
        choices=["whisper", "whisperatc", "huggingface", "faster-whisper"],
        default="whisper",
        help="Tipo de modelo a usar (default: whisper)"
    )
    
    # Opciones de Whisper
    parser.add_argument(
        "--model-size", "-s",
        default="base",
        choices=["tiny", "base", "small", "medium", "turbo", "large-v1", "large-v2", "large-v3"],
        help="Tamaño del modelo Whisper (default: base)"
    )
    
    # Opciones de HuggingFace
    parser.add_argument(
        "--hf-model",
        help="ID del modelo en HuggingFace (ej: jlvdoorn/whisper-large-v3-atco2-asr)"
    )
    
    # Opciones de WhisperATC
    parser.add_argument(
        "--version", "-v",
        default="v3",
        choices=["v2", "v3"],
        help="Versión de WhisperATC (default: v3)"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Directorio o archivo(s) de audio a transcribir"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Ruta al archivo de salida (.csv o .json)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        help="Formato de salida (default: inferir de la extensión)"
    )
    
    # Opciones de transcripción
    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo para inferencia (default: auto)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        choices=["default", "minimal", "extended", "none"],
        default="default",
        help="Prompt ATC a usar (default: default)"
    )
    
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Incluir timestamps en la salida (JSON)"
    )
    
    # Opciones adicionales
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="No mostrar barra de progreso"
    )
    
    parser.add_argument(
        "--extensions",
        default=".mp3,.wav,.flac,.m4a,.ogg,.mkv",
        help="Extensiones de audio a buscar (default: .mp3,.wav,.flac,.m4a,.ogg,.mkv)"
    )
    
    return parser


def create_model(args):
    """Crea el modelo según los argumentos."""
    prompt = get_prompt(args.prompt) if args.model != "whisperatc" else None
    
    if args.model == "whisper":
        model = WhisperModel(
            model_name=args.model_size,
            device=args.device,
            prompt=prompt,
            fp16=False
        )
    
    elif args.model == "whisperatc":
        model = WhisperATCModel(
            model_version=args.version,
            device=args.device,
            return_timestamps=args.timestamps
        )
    
    elif args.model == "huggingface":
        if not args.hf_model:
            raise ValueError("--hf-model es requerido para HuggingFace")
        
        model = HuggingFaceModel(
            model_name=args.hf_model,
            device=args.device,
            prompt=prompt,
            return_timestamps=args.timestamps
        )
    
    elif args.model == "faster-whisper":
        model = FasterWhisperModel(
            model_name=args.model_size,
            device=args.device,
            prompt=prompt,
            compute_type="int8"
        )
    
    else:
        raise ValueError(f"Modelo no soportado: {args.model}")
    
    return model


def main():
    """Función principal."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determinar formato de salida
    output_format = args.format
    if not output_format:
        output_path = Path(args.output)
        output_format = output_path.suffix.lower().replace(".", "")
        if output_format not in ["csv", "json"]:
            output_format = "csv"
    
    # Parsear extensiones
    extensions = tuple(args.extensions.split(","))
    
    # Crear modelo
    print(f"Creando modelo: {args.model}")
    model = create_model(args)
    
    # Crear pipeline
    pipeline = TranscriptionPipeline(
        model=model,
        output_format=output_format,
        show_progress=not args.no_progress
    )
    
    # Ejecutar
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Procesar directorio
        results = pipeline.run_directory(
            directory=input_path,
            output_path=args.output,
            extensions=extensions,
            recursive=True
        )
    else:
        # Procesar archivos individuales (input puede ser lista separada por comas)
        files = [f.strip() for f in args.input.split(",")]
        results = pipeline.run(
            audio_files=files,
            output_path=args.output
        )
    
    print(f"\n✅ Transcripción completada: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
