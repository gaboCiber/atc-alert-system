"""
File utilities for output management.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any


class FileUtils:
    """Utilities for file operations and output management."""
    
    @staticmethod
    def configure_output(doc_path: str, output_dir: str, model_name: str) -> str:
        """
        Configure and create output directory structure.
        
        Args:
            doc_path: Path to source document.
            output_dir: Base output directory.
            model_name: Model name for subdirectory.
            
        Returns:
            Path to document-specific output directory.
        """
        # Create base output dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract doc name
        doc_name = Path(doc_path).stem
        doc_dir_name = f"{doc_name}({model_name})"
        doc_dir = Path(output_dir) / doc_dir_name
        
        # Create doc-specific dir
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        return str(doc_dir)
    
    @staticmethod
    def save_page_result(
        output_dir: str,
        page_number: int,
        data: Dict[str, Any]
    ):
        """
        Save extraction results for a page.
        
        Args:
            output_dir: Output directory.
            page_number: Page number.
            data: Data to save.
        """
        filepath = Path(output_dir) / f"pagina_{page_number}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def save_page_chunks(
        output_dir: str,
        page_number: int,
        chunks: List[str],
        granularity: str
    ):
        """
        Save only the text chunks for a page (without NER results).
        
        Args:
            output_dir: Output directory.
            page_number: Page number.
            chunks: List of text chunks.
            granularity: Segmentation granularity used.
        """
        data = {
            "page_number": page_number,
            "granularity": granularity,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_index": i,
                    "text": chunk,
                    "char_count": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        filepath = Path(output_dir) / f"pagina_{page_number}_chunks.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def extract_accumulated_entities(
        folder_path: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract all unique entities from a folder of page results.
        
        Args:
            folder_path: Path to folder with pagina_*.json files.
            
        Returns:
            Dict mapping normalized text to entity info.
        """
        folder = Path(folder_path)
        accumulated = {}
        
        for json_file in sorted(folder.glob("pagina_*.json")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle both page and sentence granularity
                chunks = data.get('sentence_results') or [data]
                
                for chunk in chunks:
                    ner_data = chunk.get('ner', {})
                    if not isinstance(ner_data, dict):
                        continue
                    
                    entities = ner_data.get('entities', [])
                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        
                        text = entity.get('text', '')
                        normalized = text.lower().strip()
                        
                        if normalized and normalized not in accumulated:
                            accumulated[normalized] = {
                                'text': text,
                                'label': entity.get('label', 'Unknown'),
                                'context': entity.get('context', ''),
                                'source_file': json_file.name,
                            }
                            
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
        
        return accumulated
