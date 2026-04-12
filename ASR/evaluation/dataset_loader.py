"""
Cargador de datasets para evaluación ASR.
Carga ground truth (DOCX) y transcripciones (CSV) con alineación por timestamp.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from docx import Document


def load_ground_truth(docx_path: str) -> Dict[str, str]:
    """
    Carga ground truth desde archivo DOCX.
    
    Lee tablas de 3 columnas (Hora, Hablante, Texto) y agrupa
    por timestamp como en el notebook original.
    
    Args:
        docx_path: Ruta al archivo .docx
        
    Returns:
        Diccionario {timestamp: texto_completo}
        
    Example:
        >>> gt = load_ground_truth("ASR/Hora UTC.docx")
        >>> gt['22.31.28']
        'JBU1676 Habana. Go ahead. JBU1676 confirm your flight level sir?...'
    """
    doc = Document(docx_path)
    
    # Lista temporal para almacenar todas las filas
    all_rows = []
    
    for tabla in doc.tables:
        # Verificar que tenga 3 columnas
        if len(tabla.columns) != 3:
            continue
        
        for fila in tabla.rows:
            celdas = [celda.text.strip() for celda in fila.cells]
            if len(celdas) == 3:
                all_rows.append({
                    'hora': celdas[0],
                    'hablante': celdas[1],
                    'texto': celdas[2]
                })
    
    # Agrupar por timestamp (misma lógica que el notebook)
    conversations = {}
    current_timestamp = None
    
    for row in all_rows:
        hora = row['hora']
        texto = row['texto']
        
        # Saltar headers
        if hora == 'Hora UTC' or not row['hablante']:
            continue
        
        # Nuevo timestamp
        if hora:
            # Normalizar formato de hora
            current_timestamp = _normalize_timestamp(hora)
            conversations[current_timestamp] = ""
        
        # Agregar texto al timestamp actual
        if current_timestamp and texto:
            conversations[current_timestamp] += texto + " "
    
    # Limpiar espacios extra
    return {ts: txt.strip() for ts, txt in conversations.items()}


def _normalize_timestamp(ts: str) -> str:
    """
    Normaliza formato de timestamp.
    Convierte formatos como '22.20.21' o '22:28:56' a un formato consistente.
    """
    # Reemplazar puntos por dos puntos
    ts = ts.replace('.', ':')
    # Asegurar formato HH:MM:SS
    return ts


def load_transcriptions(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Carga transcripciones desde CSV de Whisper.
    
    El CSV tiene formato:
    - Columna 0: índice
    - Columna 1: nombre del modelo
    - Columnas 2+: transcripciones para cada archivo de audio
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Diccionario anidado {modelo: {archivo_audio: transcripcion}}
        
    Example:
        >>> trans = load_transcriptions("ASR/Recordings/recording3.csv")
        >>> trans['large-v2']['2023_12_3_22_31_28_0_1_0_ch139.mp3']
        'W1676, Havana. Go ahead...'
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    results = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Leer primera fila (headers con nombres de archivos)
        try:
            headers = next(reader)
        except StopIteration:
            return results
        
        # Extraer nombres de archivos de audio (columnas 2 en adelante)
        # Formato: /path/to/Recordings/2023_12_3_22_31_28_0_1_0_ch139.mp3
        audio_files = []
        for h in headers[2:]:
            # Extraer solo el nombre del archivo
            if h:
                filename = Path(h).name
                audio_files.append(filename)
            else:
                audio_files.append(None)
        
        # Leer filas de datos
        for row in reader:
            if not row or len(row) < 2:
                continue
            
            # Columna 1: nombre del modelo
            model_name = row[1] if len(row) > 1 else "unknown"
            
            if model_name not in results:
                results[model_name] = {}
            
            # Columnas 2 en adelante: transcripciones
            for i, transcription in enumerate(row[2:], start=0):
                if i < len(audio_files) and audio_files[i]:
                    audio_file = audio_files[i]
                    results[model_name][audio_file] = transcription.strip() if transcription else ""
    
    return results


def load_transcriptions_by_timestamp(
    csv_path: str, 
    timestamp_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Carga transcripciones agrupadas por timestamp.
    
    Similar a load_transcriptions pero agrupa todas las transcripciones
    de archivos relacionados bajo un mismo timestamp.
    
    Args:
        csv_path: Ruta al archivo CSV
        timestamp_mapping: Opcional, mapeo de {nombre_archivo: timestamp}
            Si no se proporciona, extrae timestamp del nombre del archivo
            
    Returns:
        Diccionario {modelo: {timestamp: texto_completo}}
    """
    raw_data = load_transcriptions(csv_path)
    
    results = {}
    
    for model_name, files in raw_data.items():
        results[model_name] = {}
        
        # Agrupar por timestamp
        timestamp_groups = {}
        
        for filename, transcription in files.items():
            # Extraer timestamp del nombre de archivo
            if timestamp_mapping and filename in timestamp_mapping:
                ts = timestamp_mapping[filename]
            else:
                ts = _extract_timestamp_from_filename(filename)
            
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            
            timestamp_groups[ts].append(transcription)
        
        # Concatenar transcripciones del mismo timestamp
        for ts, texts in timestamp_groups.items():
            results[model_name][ts] = " ".join(texts).strip()
    
    return results


def _extract_timestamp_from_filename(filename: str) -> str:
    """
    Extrae timestamp del nombre de archivo de audio.
    
    Formatos soportados:
    - 2023_12_3_22_31_28_0_1_0_ch139.mp3 → 22:31:28
    - 2024_3_24_14_37_21_0_0_34_ch176.mp3 → 14:37:21
    - 2025-05-13T11:44:09Z_2025-05-13T11:59:09Z.mkv → 11:44:09
    """
    # Intentar formato con guiones (ISO-like)
    iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', filename)
    if iso_match:
        return f"{iso_match.group(4)}:{iso_match.group(5)}:{iso_match.group(6)}"
    
    # Intentar formato con underscores
    # 2023_12_3_22_31_28_0_1_0_ch139.mp3
    parts_match = re.search(r'\d{4}_\d{1,2}_\d{1,2}_(\d{2})_(\d{2})_(\d{2})', filename)
    if parts_match:
        return f"{parts_match.group(1)}:{parts_match.group(2)}:{parts_match.group(3)}"
    
    # Fallback: devolver nombre de archivo sin extensión
    return Path(filename).stem


def align_data(
    ground_truth: Dict[str, str],
    transcriptions: Dict[str, str],
    skip_missing: bool = True
) -> List[Tuple[str, str, str]]:
    """
    Alinea ground truth con transcripciones por timestamp.
    
    Args:
        ground_truth: Diccionario {timestamp: texto}
        transcriptions: Diccionario {timestamp: texto}
        skip_missing: Si True, omite timestamps sin par
        
    Returns:
        Lista de tuplas (timestamp, reference, hypothesis)
    """
    aligned = []
    
    all_timestamps = set(ground_truth.keys()) | set(transcriptions.keys())
    
    for ts in sorted(all_timestamps):
        ref = ground_truth.get(ts, "")
        hyp = transcriptions.get(ts, "")
        
        if skip_missing and (not ref or not hyp):
            continue
        
        aligned.append((ts, ref, hyp))
    
    return aligned


def get_available_models(csv_path: str) -> List[str]:
    """
    Obtiene lista de modelos disponibles en el CSV.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Lista de nombres de modelos
    """
    data = load_transcriptions(csv_path)
    return list(data.keys())


def get_available_audio_files(csv_path: str) -> List[str]:
    """
    Obtiene lista de archivos de audio en el CSV.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Lista de nombres de archivos
    """
    csv_path = Path(csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return []
        
        # Extraer nombres de archivos de columnas 2 en adelante
        files = []
        for h in headers[2:]:
            if h:
                files.append(Path(h).name)
        
        return files
