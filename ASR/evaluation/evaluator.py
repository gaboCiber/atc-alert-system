"""
Evaluador de métricas ASR para transcripciones ATC.
Calcula WER (Word Error Rate) y otras métricas entre ground truth y predicciones.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import pandas as pd

from .data_loaders import BaseDataLoader
from ..normalization import ATCTextNormalizer

# Intentar importar jiwer, sino usar implementación básica
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    warnings.warn("jiwer no está instalado. Usando implementación básica de WER.")


@dataclass
class ASREvaluationResult:
    """
    Resultado de evaluación ASR.
    
    Attributes:
        model_name: Nombre del modelo evaluado
        timestamp: Timestamp de la conversación
        reference: Texto de referencia (ground truth normalizado)
        hypothesis: Texto de hipótesis (predicción normalizada)
        wer: Word Error Rate (0-1)
        mer: Match Error Rate (opcional)
        wil: Word Information Lost (opcional)
        wip: Word Information Preserved (opcional)
        cer: Character Error Rate (opcional)
        num_ref_words: Número de palabras en referencia
        num_hyp_words: Número de palabras en hipótesis
        errors: Detalle de errores (substitutions, insertions, deletions)
    """
    model_name: str
    timestamp: str
    reference: str
    hypothesis: str
    wer: float
    mer: Optional[float] = None
    wil: Optional[float] = None
    wip: Optional[float] = None
    cer: Optional[float] = None
    num_ref_words: int = 0
    num_hyp_words: int = 0
    errors: Optional[Dict[str, int]] = None


class ASREvaluator:
    """
    Evaluador de transcripciones ASR.
    
    Calcula métricas estándar como WER entre ground truth y predicciones.
    
    Args:
        use_jiwer: Si usar jiwer (recomendado) o implementación básica
        normalize_words: Si aplicar transformaciones adicionales a palabras
        use_atc_normalizer: Si usar ATCTextNormalizer para normalización ATC
    """
    
    def __init__(
        self,
        use_jiwer: bool = True,
        normalize_words: bool = True,
        use_atc_normalizer: bool = True
    ):
        self.use_jiwer = use_jiwer and JIWER_AVAILABLE
        self.normalize_words = normalize_words
        self.use_atc_normalizer = use_atc_normalizer
        
        # Crear normalizador ATC si está habilitado
        if self.use_atc_normalizer:
            self._normalizer = ATCTextNormalizer(
                expand_callsigns=True,
                expand_numbers=True,
                expand_icao=False,
                normalize_terminology=True,
                remove_punctuation=True,
                lowercase=True
            )
        else:
            self._normalizer = None
    
    def calculate_wer(
        self, 
        reference: str, 
        hypothesis: str,
        detailed: bool = False
    ) -> Dict[str, any]:
        """
        Calcula WER entre referencia e hipótesis.
        
        Args:
            reference: Texto de referencia (ground truth)
            hypothesis: Texto de hipótesis (predicción)
            detailed: Si incluir detalle de errores
            
        Returns:
            Diccionario con métricas
        """
        # Preprocesar
        ref_words = self._preprocess_text(reference)
        hyp_words = self._preprocess_text(hypothesis)
        
        if not ref_words:
            return {'wer': 1.0, 'num_ref_words': 0, 'num_hyp_words': len(hyp_words)}
        
        # Calcular WER
        if self.use_jiwer:
            return self._calculate_wer_jiwer(ref_words, hyp_words, detailed)
        else:
            return self._calculate_wer_basic(ref_words, hyp_words, detailed)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa texto para comparación.
        
        Si use_atc_normalizer es True, usa ATCTextNormalizer para normalizar
        el texto antes de dividir en palabras.
        """
        if not text or pd.isna(text):
            return []
        
        # Usar normalizador ATC si está habilitado
        if self.use_atc_normalizer and self._normalizer:
            normalized_text = self._normalizer.normalize(text)
        else:
            normalized_text = text
        
        # Convertir a minúsculas y split por espacios
        words = normalized_text.lower().split()
        
        return words
    
    def _calculate_wer_jiwer(
        self, 
        reference: List[str], 
        hypothesis: List[str],
        detailed: bool = False
    ) -> Dict[str, any]:
        """
        Calcula WER usando jiwer.
        """
        ref_str = ' '.join(reference)
        hyp_str = ' '.join(hypothesis)
        
        if detailed:
            # API moderna de jiwer
            wer = jiwer.wer(ref_str, hyp_str)
            mer = jiwer.mer(ref_str, hyp_str)
            wil = jiwer.wil(ref_str, hyp_str)
            wip = jiwer.wip(ref_str, hyp_str)
            
            # Para el desglose de errores, usar procesamiento manual
            # ya que jiwer moderno no expone substitutions/insertions/deletions directamente
            # en compute_measures (que ya no existe)
            basic_metrics = self._calculate_wer_basic(reference, hypothesis, detailed=True)
            
            return {
                'wer': wer,
                'mer': mer,
                'wil': wil,
                'wip': wip,
                'num_ref_words': len(reference),
                'num_hyp_words': len(hypothesis),
                'errors': basic_metrics.get('errors', {
                    'substitutions': 0,
                    'insertions': 0,
                    'deletions': 0,
                    'hits': 0,
                })
            }
        else:
            wer = jiwer.wer(ref_str, hyp_str)
            return {
                'wer': wer,
                'num_ref_words': len(reference),
                'num_hyp_words': len(hypothesis),
            }
    
    def _calculate_wer_basic(
        self, 
        reference: List[str], 
        hypothesis: List[str],
        detailed: bool = False
    ) -> Dict[str, any]:
        """
        Implementación básica de WER usando distancia de Levenshtein.
        """
        # Calcular matriz de Levenshtein
        m, n = len(reference), len(hypothesis)
        
        if m == 0:
            return {'wer': 1.0, 'num_ref_words': 0, 'num_hyp_words': n}
        
        # Matriz de distancias
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Inicializar
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Llenar matriz
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )
        
        # WER = distancia / longitud de referencia
        distance = dp[m][n]
        wer = distance / m
        
        result = {
            'wer': wer,
            'num_ref_words': m,
            'num_hyp_words': n,
        }
        
        if detailed:
            # Backtrack para contar errores
            i, j = m, n
            subs, ins, dels = 0, 0, 0
            
            while i > 0 or j > 0:
                if i == 0:
                    ins += 1
                    j -= 1
                elif j == 0:
                    dels += 1
                    i -= 1
                elif reference[i - 1] == hypothesis[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    # Determinar qué operación fue
                    sub_cost = dp[i - 1][j - 1] if i > 0 and j > 0 else float('inf')
                    del_cost = dp[i - 1][j] if i > 0 else float('inf')
                    ins_cost = dp[i][j - 1] if j > 0 else float('inf')
                    
                    min_cost = min(sub_cost, del_cost, ins_cost)
                    
                    if min_cost == sub_cost:
                        subs += 1
                        i -= 1
                        j -= 1
                    elif min_cost == del_cost:
                        dels += 1
                        i -= 1
                    else:
                        ins += 1
                        j -= 1
            
            result['errors'] = {
                'substitutions': subs,
                'insertions': ins,
                'deletions': dels,
                'hits': m - subs - dels,
            }
        
        return result
    
    def evaluate_model_with_loader(
        self,
        model_name: str,
        data_loader: BaseDataLoader,
        ground_truth_path: str,
        predictions_path: str,
        detailed: bool = True
    ) -> List[ASREvaluationResult]:
        """
        Evalúa un modelo usando un data loader.
        
        Args:
            model_name: Nombre del modelo
            data_loader: Instancia de BaseDataLoader (EcnaDataLoader o Atco2DataLoader)
            ground_truth_path: Ruta al ground truth (DOCX o directorio)
            predictions_path: Ruta al CSV de transcripciones
            detailed: Si incluir métricas detalladas
            
        Returns:
            Lista de ASREvaluationResult por ID
        """
        ground_truth = data_loader.load_ground_truth(ground_truth_path)
        all_predictions = data_loader.load_transcriptions(predictions_path)
        
        if model_name not in all_predictions:
            raise ValueError(f"Model '{model_name}' not found in transcriptions")
        
        predictions = all_predictions[model_name]
        
        results = []
        
        # Unir todos los IDs
        all_ids = set(ground_truth.keys()) & set(predictions.keys())
        
        for data_id in sorted(all_ids):
            ref = ground_truth.get(data_id, "")
            hyp = predictions.get(data_id, "")
            
            metrics = self.calculate_wer(ref, hyp, detailed=detailed)
            
            result = ASREvaluationResult(
                model_name=model_name,
                timestamp=data_id,  # Using timestamp field for the data ID
                reference=ref,
                hypothesis=hyp,
                wer=metrics['wer'],
                mer=metrics.get('mer'),
                wil=metrics.get('wil'),
                wip=metrics.get('wip'),
                num_ref_words=metrics['num_ref_words'],
                num_hyp_words=metrics['num_hyp_words'],
                errors=metrics.get('errors'),
            )
            
            results.append(result)
        
        return results
    
    def evaluate_all_models_with_loader(
        self,
        data_loader: BaseDataLoader,
        ground_truth_path: str,
        transcriptions_path: str,
        detailed: bool = True
    ) -> Dict[str, List[ASREvaluationResult]]:
        """
        Evalúa todos los modelos usando un data loader.
        
        Args:
            data_loader: Instancia de BaseDataLoader (EcnaDataLoader o Atco2DataLoader)
            ground_truth_path: Ruta al ground truth (DOCX o directorio)
            transcriptions_path: Ruta al CSV de transcripciones
            detailed: Si incluir métricas detalladas
            
        Returns:
            Dict {modelo: [ASREvaluationResult]}
        """
        ground_truth = data_loader.load_ground_truth(ground_truth_path)
        all_predictions = data_loader.load_transcriptions(transcriptions_path)
        
        results = {}
        
        for model_name, predictions in all_predictions.items():
            results[model_name] = self.evaluate_model_with_loader(
                model_name, data_loader, ground_truth_path, transcriptions_path, detailed
            )
        
        return results
    
    def aggregate_metrics(
        self, 
        results: List[ASREvaluationResult]
    ) -> Dict[str, float]:
        """
        Agrega métricas de múltiples evaluaciones.
        
        Returns:
            Dict con métricas promedio y totales
        """
        if not results:
            return {}
        
        total_ref = sum(r.num_ref_words for r in results)
        total_hyp = sum(r.num_hyp_words for r in results)
        
        # WER promedio ponderado por longitud
        total_errors = sum(r.wer * r.num_ref_words for r in results)
        avg_wer = total_errors / total_ref if total_ref > 0 else 0
        
        aggregated = {
            'average_wer': avg_wer,
            'total_ref_words': total_ref,
            'total_hyp_words': total_hyp,
            'num_samples': len(results),
        }
        
        # Otras métricas si están disponibles
        mers = [r.mer for r in results if r.mer is not None]
        if mers:
            aggregated['average_mer'] = sum(mers) / len(mers)
        
        wils = [r.wil for r in results if r.wil is not None]
        if wils:
            aggregated['average_wil'] = sum(wils) / len(wils)
        
        # Errores agregados
        all_errors = [r.errors for r in results if r.errors]
        if all_errors:
            aggregated['total_substitutions'] = sum(e['substitutions'] for e in all_errors)
            aggregated['total_insertions'] = sum(e['insertions'] for e in all_errors)
            aggregated['total_deletions'] = sum(e['deletions'] for e in all_errors)
        
        return aggregated
    
    def print_evaluation_report(
        self,
        results: Dict[str, List[ASREvaluationResult]],
        show_samples: bool = False
    ):
        """
        Imprime un reporte de evaluación formateado.
        
        Args:
            results: Dict {modelo: [ASREvaluationResult]}
            show_samples: Si mostrar ejemplos individuales
        """
        print("=" * 80)
        print("REPORTE DE EVALUACIÓN ASR")
        print("=" * 80)
        
        for model_name, model_results in results.items():
            print(f"\n{'=' * 80}")
            print(f"Modelo: {model_name}")
            print(f"{'=' * 80}")
            
            # Métricas agregadas
            agg = self.aggregate_metrics(model_results)
            print(f"  WER Promedio:     {agg['average_wer']:.2%}")
            print(f"  Total palabras ref: {agg['total_ref_words']}")
            print(f"  Total palabras hyp: {agg['total_hyp_words']}")
            print(f"  Muestras:         {agg['num_samples']}")
            
            if 'total_substitutions' in agg:
                print(f"\n  Desglose de errores:")
                print(f"    Sustituciones:  {agg['total_substitutions']}")
                print(f"    Inserciones:   {agg['total_insertions']}")
                print(f"    Eliminaciones:  {agg['total_deletions']}")
            
            if show_samples:
                print(f"\n  Muestras individuales:")
                for r in model_results:
                    print(f"\n    Timestamp: {r.timestamp}")
                    print(f"    WER: {r.wer:.2%}")
                    print(f"    Ref: {r.reference}")
                    print(f"    Hyp: {r.hypothesis}")
                    print(f"    Ref (Normalized): {' '.join(self._preprocess_text(r.reference))}")
                    print(f"    Hyp (Normalized): {' '.join(self._preprocess_text(r.hypothesis))}")
    
        print(f"\n{'=' * 80}")
    
    def compare_models(
        self,
        results: Dict[str, List[ASREvaluationResult]]
    ) -> List[Tuple[str, float]]:
        """
        Compara modelos por WER promedio.
        
        Returns:
            Lista de (modelo, wer_promedio) ordenada por WER
        """
        comparison = []
        for model_name, model_results in results.items():
            agg = self.aggregate_metrics(model_results)
            comparison.append((model_name, agg['average_wer']))
        
        # Ordenar por WER (menor es mejor)
        return sorted(comparison, key=lambda x: x[1])
