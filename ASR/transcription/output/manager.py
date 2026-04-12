"""
Manejador de salidas para resultados de transcripción.
Soporta formatos CSV y JSON.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime

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
            self._save_csv(results, output_path, model_column)
        elif self.format == "json":
            self._save_json(results, output_path)
    
    def _save_csv(
        self,
        results: List[TranscriptionResult],
        output_path: Path,
        model_column: str
    ) -> None:
        """
        Guarda los resultados en formato CSV.
        
        Formato: Una fila por modelo, columnas son archivos de audio.
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
        
        # Escribir CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezado
            header = [model_column] + sorted_files
            writer.writerow(header)
            
            # Datos
            for model in sorted(model_results.keys()):
                row = [model]
                for file in sorted_files:
                    row.append(model_results[model].get(file, ""))
                writer.writerow(row)
    
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
