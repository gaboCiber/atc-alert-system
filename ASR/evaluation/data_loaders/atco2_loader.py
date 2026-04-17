"""
Data loader for ATCO2 dataset.
Handles XML ground truth files with segment filtering based on quality tags.
"""

import csv
import xml.etree.ElementTree as ET
import regex as re
from pathlib import Path
from typing import Dict, List, Optional

from .base_loader import BaseDataLoader


class Atco2DataLoader(BaseDataLoader):
    """
    Data loader for ATCO2 dataset.
    
    Ground truth: XML files with segments and quality tags
    Transcriptions: CSV files (same format as ECNA)
    Audio mapping: XML filename matches WAV filename
    """
    
    def id(self) -> str:
        return "atco2"
    
    def load_ground_truth(self, data_dir: str) -> Dict[str, str]:
        """
        Load ground truth from ATCO2 XML files (valid transcripts only).
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory with XML files
            
        Returns:
            Dictionary mapping XML filename to concatenated text of all valid segments
        """
        return self.load_ground_truth_valid(data_dir)
    
    def load_ground_truth_valid(self, data_dir: str) -> Dict[str, str]:
        """
        Load ground truth with only valid transcripts (correct_transcript=1).
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory
            
        Returns:
            Dictionary {xml_filename: concatenated_text}
        """
        return self._load_xmls_with_filter(data_dir, filter_valid=True, filter_non_english=False)
    
    def load_ground_truth_invalid(self, data_dir: str) -> Dict[str, str]:
        """
        Load ground truth with only invalid transcripts (correct_transcript=0).
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory
            
        Returns:
            Dictionary {xml_filename: concatenated_text}
        """
        return self._load_xmls_with_filter(data_dir, filter_valid=False, filter_non_english=False, only_invalid=True)
    
    def load_ground_truth_non_english(self, data_dir: str) -> Dict[str, str]:
        """
        Load ground truth with only non-English segments (non_english=1).
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory
            
        Returns:
            Dictionary {xml_filename: concatenated_text}
        """
        return self._load_xmls_with_filter(data_dir, filter_valid=False, filter_non_english=True)
    
    def load_ground_truth_all(self, data_dir: str) -> Dict[str, str]:
        """
        Load all ground truth segments regardless of quality tags.
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory
            
        Returns:
            Dictionary {xml_filename: concatenated_text}
        """
        return self._load_xmls_with_filter(data_dir, filter_valid=False, filter_non_english=False)
    
    def _load_xmls_with_filter(
        self,
        data_dir: str,
        filter_valid: bool = True,
        filter_non_english: bool = False,
        only_invalid: bool = False
    ) -> Dict[str, str]:
        """
        Load XML files with segment filtering.
        
        Args:
            data_dir: Path to directory containing DATA/ subdirectory
            filter_valid: If True, only include segments with correct_transcript=1
            filter_non_english: If True, only include segments with non_english=1
            only_invalid: If True, only include segments with correct_transcript=0
            
        Returns:
            Dictionary {xml_filename: concatenated_text}
        """
        data_path = Path(data_dir) / "DATA"
        
        if not data_path.exists():
            raise FileNotFoundError(f"DATA directory not found: {data_path}")
        
        ground_truth = {}
        
        for xml_file in sorted(data_path.glob("*.xml")):
            xml_name = xml_file.stem  # Full filename without path
            segments_text = self._parse_xml_file(
                xml_file,
                filter_valid=filter_valid,
                filter_non_english=filter_non_english,
                only_invalid=only_invalid
            )
            
            if segments_text:
                ground_truth[xml_name] = segments_text
        
        return ground_truth
    
    def _parse_xml_file(
        self,
        xml_path: Path,
        filter_valid: bool = True,
        filter_non_english: bool = False,
        only_invalid: bool = False
    ) -> str:
        """
        Parse single XML file and concatenate filtered segments.
        
        Args:
            xml_path: Path to XML file
            filter_valid: If True, only include correct_transcript=1
            filter_non_english: If True, only include non_english=1
            only_invalid: If True, only include correct_transcript=0
            
        Returns:
            Concatenated text of all matching segments
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        segments = []
        
        for segment in root.findall('segment'):
            # Check tags
            tags = segment.find('tags')
            if tags is None:
                continue
            
            correct_transcript_elem = tags.find('correct_transcript')
            non_english_elem = tags.find('non_english')
            
            correct_transcript = int(correct_transcript_elem.text) if correct_transcript_elem is not None else 1
            non_english = int(non_english_elem.text) if non_english_elem is not None else 0
            
            # Apply filters
            if filter_valid and correct_transcript != 1:
                continue
            
            if only_invalid and correct_transcript != 0:
                continue
            
            if filter_non_english and non_english != 1:
                continue
            
            # Extract text
            text_elem = segment.find('text')
            if text_elem is not None and text_elem.text:
                text = re.sub(r"\[[^\]]*\]", "", text_elem.text.strip())
                segments.append(text)
        
        return " ".join(segments)
    
    def _load_transcriptions(self, csv_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load transcriptions from CSV file (same format as ECNA).
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary {model: {xml_filename: transcription}}
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
            
            # Extract audio/XML filenames from headers (columns 2+)
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
    
    def get_audio_path(self, ground_truth_id: str, data_dir: str) -> Optional[str]:
        """
        Map ground truth ID to WAV file path.
        
        Args:
            ground_truth_id: XML filename (e.g., LKPR_RUZYNE_Radar_120_520MHz_20201025_091112)
            data_dir: Path to directory containing DATA/ subdirectory
            
        Returns:
            Full path to WAV file, or None if not found
        """
        data_path = Path(data_dir) / "DATA"
        
        # XML filename -> WAV filename (same basename, different extension)
        xml_stem = Path(ground_truth_id).stem  # Remove .xml if present
        wav_path = data_path / f"{xml_stem}.wav"
        
        if wav_path.exists():
            return str(wav_path)
        
        return None
