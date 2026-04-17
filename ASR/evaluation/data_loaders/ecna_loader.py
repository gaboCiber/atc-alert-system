"""
Data loader for ECNA dataset.
Handles DOCX ground truth files and CSV transcription files with timestamp-based audio mapping.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Optional
from docx import Document

from .base_loader import BaseDataLoader


class EcnaDataLoader(BaseDataLoader):
    """
    Data loader for ECNA dataset.
    
    Ground truth: DOCX files with tables (Hour, Speaker, Text)
    Transcriptions: CSV files with model rows and audio file columns
    Audio mapping: Timestamp embedded in audio filename
    """
    
    def id(self) -> str:
        return "ecna"
    
    def load_ground_truth(self, data_path: str) -> Dict[str, str]:
        """
        Load ground truth from DOCX file(s).
        
        Args:
            data_path: Path to .docx file or directory containing .docx files
            
        Returns:
            Dictionary mapping {docx_name:timestamp: text}
            
        Example:
            >>> loader = EcnaDataLoader()
            >>> gt = loader.load_ground_truth("./ECNA")
            >>> gt["2023-12-03.docx:22:20:21"]
            "JBU1676 Havana. Go ahead..."
        """
        path = Path(data_path)
        
        if path.is_file() and path.suffix == ".docx":
            # Load single DOCX file
            return self._load_single_docx(path)
        elif path.is_dir():
            # Load all DOCX files in directory
            ground_truth = {}
            for docx_file in path.glob("*.docx"):
                file_gt = self._load_single_docx(docx_file)
                ground_truth.update(file_gt)
            return ground_truth
        else:
            raise ValueError(f"Invalid path: {data_path}. Must be .docx file or directory.")
    
    def _load_single_docx(self, docx_path: Path) -> Dict[str, str]:
        """
        Load ground truth from a single DOCX file.
        
        Returns:
            Dictionary mapping {docx_name:timestamp: text}
        """
        doc = Document(docx_path)
        docx_name = docx_path.name
        
        # Collect all rows from all tables
        all_rows = []
        
        for table in doc.tables:
            if len(table.columns) != 3:
                continue
            
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if len(cells) == 3:
                    all_rows.append({
                        'hora': cells[0],
                        'hablante': cells[1],
                        'texto': cells[2]
                    })
        
        # Group by timestamp
        conversations = {}
        current_timestamp = None
        
        for row in all_rows:
            hora = row['hora']
            texto = row['texto']
            
            # Skip headers
            if hora == 'Hora UTC' or not row['hablante']:
                continue
            
            # New timestamp
            if hora:
                current_timestamp = self._normalize_timestamp(hora)
                compound_id = f"{docx_name}:{current_timestamp}"
                conversations[compound_id] = ""
            
            # Add text to current timestamp
            if current_timestamp and texto:
                compound_id = f"{docx_name}:{current_timestamp}"
                conversations[compound_id] += texto + " "
        
        # Clean extra spaces
        return {ts: txt.strip() for ts, txt in conversations.items()}
    
    def _normalize_timestamp(self, ts: str) -> str:
        """
        Normalize timestamp format.
        Converts formats like '22.20.21' or '22:28:56' to consistent format.
        """
        ts = ts.replace('.', ':')
        return ts
    
    def _load_transcriptions(self, csv_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load transcriptions from CSV file.
        
        CSV format:
        - Column 0: index
        - Column 1: model name
        - Columns 2+: transcriptions for each audio file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary {model: {audio_filename: transcription}}
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        results = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            try:
                headers = next(reader)
            except StopIteration:
                return results
            
            # Extract audio filenames from headers (columns 2+)
            audio_files = []
            for h in headers[2:]:
                if h:
                    filename = Path(h).name
                    audio_files.append(filename)
                else:
                    audio_files.append(None)
            
            # Read data rows
            for row in reader:
                if not row or len(row) < 2:
                    continue
                
                model_name = row[1] if len(row) > 1 else "unknown"
                
                if model_name not in results:
                    results[model_name] = {}
                
                for i, transcription in enumerate(row[2:], start=0):
                    if i < len(audio_files) and audio_files[i]:
                        audio_file = audio_files[i]
                        results[model_name][audio_file] = transcription.strip() if transcription else ""
        
        return results
    
    def get_audio_path(self, ground_truth_id: str, audio_dir: str) -> Optional[str]:
        """
        Map ground truth ID to audio file path.
        
        Args:
            ground_truth_id: Compound ID like "2023-12-03.docx:22:20:21"
            audio_dir: Directory containing audio files
            
        Returns:
            Full path to audio file, or None if not found
        """
        # Extract timestamp from compound ID
        if ":" in ground_truth_id:
            timestamp = ground_truth_id.split(":")[-1]  # Get last part (timestamp)
        else:
            timestamp = ground_truth_id
        
        audio_dir = Path(audio_dir)
        
        # Search for audio file containing timestamp in name
        # Format: 2023_12_3_22_20_21_0_0_24_ch139.mp3 contains "22:20:21"
        timestamp_pattern = timestamp.replace(":", "_")
        
        for audio_file in audio_dir.glob("*"):
            if timestamp_pattern in audio_file.name:
                return str(audio_file)
        
        return None
