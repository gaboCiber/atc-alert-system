"""
File utilities for output management.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / f"pagina_{page_number}.json"
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
    def load_page_chunks(chunks_dir: str, page_number: int) -> Optional[List[str]]:
        """
        Load pre-generated chunks for a specific page.
        
        Args:
            chunks_dir: Directory containing chunk JSON files.
            page_number: Page number to load chunks for.
            
        Returns:
            List of chunk texts if file exists, None otherwise.
        """
        filepath = Path(chunks_dir) / f"pagina_{page_number}_chunks.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract chunk texts from the JSON structure
            chunks_data = data.get("chunks", [])
            chunks = [chunk.get("text", "") for chunk in chunks_data]
            
            return chunks if chunks else None
        except Exception as e:
            print(f"⚠️ Error loading chunks from {filepath}: {e}")
            return None
    
    @staticmethod
    def get_available_chunk_pages(chunks_dir: str) -> set:
        """
        Get set of page numbers that have chunk files in the source directory.
        
        Args:
            chunks_dir: Directory containing chunk JSON files.
            
        Returns:
            Set of page numbers (integers) that have chunk files.
        """
        import re
        available_pages = set()
        chunks_path = Path(chunks_dir)
        
        if not chunks_path.exists():
            return available_pages
        
        for chunk_file in chunks_path.glob("pagina_*_chunks.json"):
            # Extract page number from filename like "pagina_5_chunks.json"
            match = re.match(r"pagina_(\d+)_chunks\.json", chunk_file.name)
            if match:
                available_pages.add(int(match.group(1)))
        
        return available_pages

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
