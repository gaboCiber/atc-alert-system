"""
Base abstract class for ASR data loaders.
Provides common interface for loading ground truth and transcriptions from different data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd
from pathlib import Path

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Each data source (ECNA, ATCO2, etc.) should implement this interface
    to provide a consistent way to load ground truth and transcriptions.
    """
    
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this loader"""
        pass
    
    @abstractmethod
    def load_ground_truth(self, data_path: str) -> Dict[str, str]:
        """
        Load ground truth data.
        
        Args:
            data_path: Path to ground truth data (file or directory)
            
        Returns:
            Dictionary mapping unique IDs to ground truth text
        """
        pass
    
    def load_transcriptions(self, csv_path: str) -> Dict[str, Dict[str, str]]:
        """
        Load transcription data from CSV.
        
        Args:
            csv_path: Path to CSV file with transcriptions
            
        Returns:
            Dictionary mapping model names to {audio_id: transcription} dictionaries
        """
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        results = {}

        data = pd.read_csv(csv_path, index_col=0)
        data.columns = pd.Index(pd.Series(data.columns).apply(lambda x: Path(x).stem))

        for model in data.index:
            results[model] = dict(data.loc[model])

        return results 




    
    @abstractmethod
    def get_audio_path(self, ground_truth_id: str, audio_dir: str) -> Optional[str]:
        """
        Map ground truth ID to audio file path.
        
        Args:
            ground_truth_id: Unique ID from ground truth
            audio_dir: Directory containing audio files
            
        Returns:
            Full path to audio file, or None if not found
        """
        pass
