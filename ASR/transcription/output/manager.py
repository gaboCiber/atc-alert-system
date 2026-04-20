"""
Manejador de salidas para resultados de transcripción.
Soporta formatos CSV y JSON.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

import pandas as pd

from ..base import TranscriptionResult


class OutputManager:
    """
    Maneja el guardado de resultados de transcripción en diferentes formatos.
    """
    
    SUPPORTED_FORMATS = ["csv", "json"]
    
    def __init__(self, format: str = "csv"):
        """
        Inicializa el OutputManager.
        
        Args:
            format: Formato de salida ("csv" o "json")
        """
        self.format = format.lower()
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Formato no soportado: {format}. "
                f"Use uno de: {self.SUPPORTED_FORMATS}"
            )
    
    def save(
        self,
        results: List[TranscriptionResult],
        output_path: Union[str, Path],
        model_column: str = "model"
    ) -> None:
        """
        Guarda los resultados en el formato especificado.
        
        Args:
            results: Lista de resultados de transcripción
            output_path: Ruta al archivo de salida
            model_column: Nombre de la columna para el modelo (solo CSV)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.format == "csv":
            self._save_csv(results, output_path, model_column, mode='w')
        elif self.format == "json":
            self._save_json(results, output_path)
    
    def append(
        self,
        results: List[TranscriptionResult],
        output_path: Union[str, Path],
        model_column: str = "model"
    ) -> None:
        """
        Agrega resultados a un archivo existente.
        
        Si el archivo no existe, lo crea. Si existe, combina los nuevos resultados
        con los existentes (útil para acumular transcripciones de múltiples modelos).
        
        Args:
            results: Lista de resultados de transcripción
            output_path: Ruta al archivo de salida
            model_column: Nombre de la columna para el modelo (solo CSV)
        """
        if self.format != "csv":
            raise ValueError("Append mode is only supported for CSV format")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._save_csv(results, output_path, model_column, mode='a')
    
    def _save_csv(
        self,
        results: List[TranscriptionResult],
        output_path: Path,
        model_column: str,
        mode: str = 'w'
    ) -> None:
        """
        Guarda los resultados en formato CSV usando pandas.
        
        Formato: Una fila por modelo, columnas son archivos de audio.
        
        Args:
            results: Lista de resultados de transcripción
            output_path: Ruta al archivo de salida
            model_column: Nombre de la columna para el modelo
            mode: Modo de escritura ('w' para sobrescribir, 'a' para append)
        """
        # Agrupar resultados por modelo
        model_results: Dict[str, Dict[str, str]] = {}
        all_files: set = set()
        
        for result in results:
            model = result.model_name
            file = result.file_path
            all_files.add(file)
            
            if model not in model_results:
                model_results[model] = {}
            
            model_results[model][file] = result.text
        
        # Ordenar archivos para consistencia
        sorted_files = sorted(all_files)
        
        # Crear DataFrame
        data = []
        for model in sorted(model_results.keys()):
            row = {model_column: model}
            for file in sorted_files:
                row[file] = model_results[model].get(file, "")
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Reordenar columnas: model_column primero, luego archivos ordenados
        columns = [model_column] + sorted_files
        df = df.reindex(columns=columns)
        
        # Guardar CSV
        if mode == 'a' and output_path.exists():
            # Append: leer existente y combinar
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(output_path, index=False)
        else:
            # Sobrescribir o crear nuevo
            df.to_csv(output_path, index=False)
    
    def _save_json(
        self,
        results: List[TranscriptionResult],
        output_path: Path
    ) -> None:
        """
        Guarda los resultados en formato JSON.
        
        Incluye texto completo, timestamps y metadata.
        """
        data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "format": "json",
                "num_results": len(results)
            },
            "results": []
        }
        
        for result in results:
            entry = {
                "file_path": result.file_path,
                "model_name": result.model_name,
                "text": result.text,
            }
            
            # Agregar campos opcionales si existen
            if result.timestamps:
                entry["timestamps"] = result.timestamps
            if result.confidence is not None:
                entry["confidence"] = result.confidence
            if result.duration is not None:
                entry["duration"] = result.duration
            if result.metadata:
                entry["metadata"] = result.metadata
            
            data["results"].append(entry)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_checkpoint(
        self,
        results: List[TranscriptionResult],
        checkpoint_path: Union[str, Path]
    ) -> None:
        """
        Guarda checkpoint con estado actual de transcripciones.
        
        Usa el mismo formato que JSON normal para permitir resumen.
        Sobrescribe el archivo completo con el estado actual.
        
        Args:
            results: Lista de resultados de transcripción
            checkpoint_path: Ruta al archivo de checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Usar el mismo formato que _save_json
        data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "format": "checkpoint",
                "num_results": len(results)
            },
            "results": []
        }
        
        for result in results:
            entry = {
                "file_path": result.file_path,
                "model_name": result.model_name,
                "text": result.text,
            }
            
            # Agregar campos opcionales si existen
            if result.timestamps:
                entry["timestamps"] = result.timestamps
            if result.confidence is not None:
                entry["confidence"] = result.confidence
            if result.duration is not None:
                entry["duration"] = result.duration
            if result.metadata:
                entry["metadata"] = result.metadata
            
            data["results"].append(entry)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Optional[List[TranscriptionResult]]:
        """
        Carga checkpoint existente.
        
        Args:
            checkpoint_path: Ruta al archivo de checkpoint
            
        Returns:
            Lista de TranscriptionResult si el archivo existe, None si no
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return None
        
        data = self.load_json(checkpoint_path)
        
        results = []
        for entry in data.get("results", []):
            result = TranscriptionResult(
                text=entry.get("text", ""),
                file_path=entry.get("file_path", ""),
                model_name=entry.get("model_name", ""),
                timestamps=entry.get("timestamps"),
                confidence=entry.get("confidence"),
                duration=entry.get("duration"),
                metadata=entry.get("metadata", {})
            )
            results.append(result)
        
        return results
    
    @staticmethod
    def load_csv(input_path: Union[str, Path]) -> List[Dict[str, str]]:
        """
        Carga resultados de un archivo CSV.
        
        Args:
            input_path: Ruta al archivo CSV
            
        Returns:
            Lista de diccionarios con los datos
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    @staticmethod
    def load_json(input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Carga resultados de un archivo JSON.
        
        Args:
            input_path: Ruta al archivo JSON
            
        Returns:
            Diccionario con los datos
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def checkpoint_to_csv(
        checkpoint_path: Union[str, Path],
        csv_path: Union[str, Path],
        model_column: str = "model",
        append: bool = True
    ) -> None:
        """
        Convierte un archivo checkpoint JSON a formato CSV de tabla.
        
        Lee un checkpoint (formato JSON con lista de resultados) y lo convierte
        al formato de tabla CSV (una fila por modelo, columnas son archivos de audio).
        
        Args:
            checkpoint_path: Ruta al archivo checkpoint JSON
            csv_path: Ruta al archivo CSV de salida
            model_column: Nombre de la columna para el modelo
            append: Si True, añade al CSV existente; si False, sobrescribe
        """
        checkpoint_path = Path(checkpoint_path)
        csv_path = Path(csv_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        
        # Cargar checkpoint
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("results", [])
        if not results:
            print(f"⚠️  No hay resultados en {checkpoint_path}")
            return
        
        # Convertir a TranscriptionResult
        from ..base import TranscriptionResult
        transcription_results = []
        for entry in results:
            result = TranscriptionResult(
                text=entry.get("text", ""),
                file_path=entry.get("file_path", ""),
                model_name=entry.get("model_name", ""),
                timestamps=entry.get("timestamps"),
                confidence=entry.get("confidence"),
                duration=entry.get("duration"),
                metadata=entry.get("metadata", {})
            )
            transcription_results.append(result)
        
        # Usar OutputManager para guardar en formato CSV
        manager = OutputManager(format="csv")
        
        if append and csv_path.exists():
            manager.append(transcription_results, csv_path, model_column)
            print(f"✅ Añadidos {len(transcription_results)} resultados a {csv_path}")
        else:
            manager.save(transcription_results, csv_path, model_column)
            mode_str = "actualizado" if append else "creado"
            print(f"✅ CSV {mode_str} con {len(transcription_results)} resultos: {csv_path}")
        
        # Mostrar resumen
        models = set(r.model_name for r in transcription_results)
        files = len(set(r.file_path for r in transcription_results))
        print(f"   Modelos: {', '.join(sorted(models))}")
        print(f"   Archivos: {files}")
    
    @staticmethod
    def merge_checkpoints_to_csv(
        checkpoint_dir: Union[str, Path],
        csv_path: Union[str, Path],
        pattern: str = "checkpoint_*.json",
        model_column: str = "model",
        append: bool = False
    ) -> None:
        """
        Combina múltiples archivos checkpoint en un único CSV.
        
        Busca todos los checkpoints que coincidan con el patrón en el directorio
        especificado, los carga y genera una tabla CSV consolidada.
        
        Args:
            checkpoint_dir: Directorio donde buscar los checkpoints
            csv_path: Ruta al archivo CSV de salida
            pattern: Patrón glob para filtrar checkpoints (default: "checkpoint_*.json")
            model_column: Nombre de la columna para el modelo
            append: Si True, añade al CSV existente; si False, sobrescribe
        """
        checkpoint_dir = Path(checkpoint_dir)
        csv_path = Path(csv_path)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {checkpoint_dir}")
        
        # Encontrar todos los checkpoints
        checkpoint_files = list(checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            print(f"⚠️  No se encontraron checkpoints con patrón '{pattern}' en {checkpoint_dir}")
            return
        
        print(f"📁 Encontrados {len(checkpoint_files)} checkpoints en {checkpoint_dir}")
        
        # Cargar y combinar todos los resultados
        from ..base import TranscriptionResult
        all_results: List[TranscriptionResult] = []
        
        for checkpoint_path in sorted(checkpoint_files):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get("results", [])
                if not results:
                    print(f"   ⚠️  {checkpoint_path.name}: sin resultados")
                    continue
                
                for entry in results:
                    result = TranscriptionResult(
                        text=entry.get("text", ""),
                        file_path=entry.get("file_path", ""),
                        model_name=entry.get("model_name", ""),
                        timestamps=entry.get("timestamps"),
                        confidence=entry.get("confidence"),
                        duration=entry.get("duration"),
                        metadata=entry.get("metadata", {})
                    )
                    all_results.append(result)
                
                print(f"   ✅ {checkpoint_path.name}: {len(results)} resultados")
                
            except Exception as e:
                print(f"   ❌ {checkpoint_path.name}: error - {e}")
                continue
        
        if not all_results:
            print("⚠️  No hay resultados para exportar")
            return
        
        # Guardar en CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        manager = OutputManager(format="csv")
        
        if append and csv_path.exists():
            manager.append(all_results, csv_path, model_column)
            print(f"\n✅ Añadidos {len(all_results)} resultados a {csv_path}")
        else:
            manager.save(all_results, csv_path, model_column)
            mode_str = "actualizado" if append else "creado"
            print(f"\n✅ CSV {mode_str}: {csv_path}")
        
        # Mostrar resumen
        models = set(r.model_name for r in all_results)
        files = len(set(r.file_path for r in all_results))
        print(f"   Total modelos: {len(models)}")
        print(f"   Total archivos: {files}")
        print(f"   Total resultados: {len(all_results)}")
        print(f"   Modelos: {', '.join(sorted(models))}")
