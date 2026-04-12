"""
Script principal de evaluación ASR.
Compara transcripciones de Whisper contra ground truth y calcula WER.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Añadir el directorio padre de ASR al path (project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from ASR.evaluation import (
    load_ground_truth,
    load_transcriptions_by_timestamp,
    ASREvaluator,
    print_evaluation_report,
    compare_models,
)
from ASR.normalization import ATCTextNormalizer


def run_evaluation(
    ground_truth_path: str = "ASR/Hora UTC.docx",
    transcriptions_path: str = "ASR/Recordings/recording3.csv",
    normalize: bool = True,
    show_samples: bool = False,
) -> Dict:
    """
    Ejecuta la evaluación completa.
    
    Args:
        ground_truth_path: Ruta al archivo DOCX con ground truth
        transcriptions_path: Ruta al archivo CSV con transcripciones
        normalize: Si aplicar normalización de texto
        show_samples: Si mostrar ejemplos individuales
        
    Returns:
        Dict con resultados de evaluación
    """
    print("=" * 80)
    print("EVALUACIÓN ASR - Transcripciones vs Ground Truth")
    print("=" * 80)
    
    # 1. Cargar ground truth
    print(f"\n📄 Cargando ground truth: {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"   ✓ {len(ground_truth)} timestamps cargados")
    
    # 2. Cargar transcripciones
    print(f"\n📄 Cargando transcripciones: {transcriptions_path}")
    transcriptions = load_transcriptions_by_timestamp(transcriptions_path)
    print(f"   ✓ {len(transcriptions)} modelos encontrados")
    for model in transcriptions.keys():
        print(f"     - {model}")
    
    # 3. Normalizar textos (si se solicita)
    if normalize:
        print("\n🔧 Normalizando textos...")
        normalizer = ATCTextNormalizer(
            expand_callsigns=True,
            expand_numbers=True,
            expand_icao=True,
            normalize_terminology=True,
            remove_punctuation=True,
            lowercase=True,
        )
        
        # Normalizar ground truth
        ground_truth_norm = {}
        print("============GROUD TRUTH============")
        for ts, text in ground_truth.items():
            ground_truth_norm[ts] = normalizer.normalize(text)
            print(f"\n   Timestamp: {ts}")
            print(f"   Original:  {text}")
            print(f"   Normalizado: {ground_truth_norm[ts]}")
        
        # Normalizar transcripciones
        transcriptions_norm = {}
        print("============TRANSCRIPTIONS============")
        for model_name, model_data in transcriptions.items():
            transcriptions_norm[model_name] = {}
            print(f"----------{model_name}----------")
            for ts, text in model_data.items():
                transcriptions_norm[model_name][ts] = normalizer.normalize(text)
                print(f"\n   Timestamp: {ts}")
                print(f"   Original:  {text}")
                print(f"   Normalizado: {transcriptions_norm[model_name][ts]}")
                print()
        ground_truth = ground_truth_norm
        transcriptions = transcriptions_norm
    else:
        print("\n⚠️  Normalización desactivada")
    
    # 4. Evaluar
    print("\n📊 Evaluando modelos...")
    evaluator = ASREvaluator(use_jiwer=True)
    results = evaluator.evaluate_all_models(ground_truth, transcriptions, detailed=True)
    
    # 5. Imprimir reporte
    print_evaluation_report(results, show_samples=show_samples)
    
    # 6. Ranking de modelos
    print("\n🏆 RANKING DE MODELOS (menor WER es mejor):")
    ranking = compare_models(results)
    for i, (model, wer) in enumerate(ranking, 1):
        print(f"   {i}. {model}: {wer:.2%}")
    
    return {
        'ground_truth': ground_truth,
        'transcriptions': transcriptions,
        'results': results,
        'ranking': ranking,
    }


def quick_test():
    """
    Prueba rápida con ejemplos de normalización.
    """
    from ASR.normalization import quick_normalize
    
    print("\n" + "=" * 80)
    print("PRUEBA DE NORMALIZACIÓN")
    print("=" * 80)
    
    test_cases = [
        "31236, Havana radar contact, 20 miles south of BMO, maintain FL360. 360, Havana 31236."
        "8-0-0",
        "W1676, Havana. Go ahead.",
        "JBU1676 Habana, confirm your flight level sir",
        "NKS236 maintain FL340 due traffic",
        "Turn right heading 270",
        "Contact frequency 133.85",
        "Direct BORDO maintain FL360",
    ]
    
    for text in test_cases:
        normalized = quick_normalize(text)
        print(f"\nOriginal:   {text}")
        print(f"Normalizado: {normalized}")


def main():
    """
    Punto de entrada principal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluación ASR para transcripciones ATC"
    )
    
    parser.add_argument(
        "--ground-truth", "-g",
        default="ASR/Hora UTC.docx",
        help="Ruta al archivo DOCX con ground truth"
    )
    
    parser.add_argument(
        "--transcriptions", "-t",
        default="ASR/Recordings/recording3.csv",
        help="Ruta al archivo CSV con transcripciones"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Deshabilitar normalización de texto"
    )
    
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Mostrar muestras individuales"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecutar prueba de normalización y salir"
    )
    
    args = parser.parse_args()
        
    if args.test:
        quick_test()
        return
    
    # Ejecutar evaluación
    results = run_evaluation(
        ground_truth_path=args.ground_truth,
        transcriptions_path=args.transcriptions,
        normalize=not args.no_normalize,
        show_samples=args.show_samples,
    )
    
    print("\n✅ Evaluación completada")


if __name__ == "__main__":
    main()
