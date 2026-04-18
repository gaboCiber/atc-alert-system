#!/usr/bin/env python3
"""
Script para ejecutar DeepFilterNet en un entorno Python 3.9.

Este script es llamado por DeepFilterNetWrapper mediante subprocess.
Debe ejecutarse con el Python del entorno virtual que tiene DeepFilterNet instalado.

Uso:
    python run_deepfilter.py --input audio.wav --output cleaned.wav
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Limpia audio usando DeepFilterNet"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Ruta al archivo de audio de entrada"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Ruta al archivo de audio limpio de salida"
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "cuda"],
        default=None,
        help="Dispositivo a usar (default: auto)"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=None,
        help="Sample rate de salida (default: 48kHz de DeepFilterNet)"
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(args.input):
        print(f"ERROR: Archivo de entrada no encontrado: {args.input}", file=sys.stderr)
        return 1
    
    try:
        import torch
        import torchaudio
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError as e:
        print(f"ERROR: No se pudieron importar las dependencias de DeepFilterNet: {e}", file=sys.stderr)
        print("Asegúrate de ejecutar este script con el Python del entorno virtual correcto.", file=sys.stderr)
        return 1
    
    try:
        # Inicializar el modelo (descarga los pesos automáticamente la primera vez)
        # Por defecto usa DeepFilterNet3
        model, df_state, _ = init_df()
        
        # Cargar el audio
        # DeepFilterNet trabaja internamente a 48kHz, load_audio se encarga del resampling
        audio, _ = load_audio(args.input, sr=df_state.sr())
        
        # Mover a GPU si está disponible y se solicitó
        if args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            audio = audio.to(device)
            print("Procesando audio en CUDA...")
        elif args.device == "cpu":
            device = torch.device("cpu")
            model = model.to(device)
            audio = audio.to(device)
            print("Procesando audio en CPU...")
        else:
            # Auto-detectar
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            audio = audio.to(device)
            print(f"Procesando audio en {device}...")
        
        # Ejecutar la mejora (Denoising)
        enhanced_audio = enhance(model, df_state, audio)
        
        # Guardar el resultado
        save_audio(args.output, enhanced_audio, df_state.sr())
        print(f"Audio guardado en: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR procesando audio: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
