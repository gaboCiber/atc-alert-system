"""
Pipeline orchestrator - coordinates the full extraction workflow.
"""
import json
import glob
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..config.settings import PipelineConfig
from ..core.document_processor import DocumentProcessor, Page
from ..core.context_manager import ContextManager
from ..core.text_segmenter import TextSegmenter
from ..core.sentence_extractor import SentenceExtractor
from ..extractors.kex_extractor import KEXExtractor
from ..extractors.json_parser import JSONParser
from ..utils.id_manager import IDManager
from ..utils.file_utils import FileUtils
from .state import PipelineState


@dataclass
class ExtractionResult:
    """Result of extracting a single chunk."""
    page_number: int
    chunk_index: int
    chunk_text: str
    extraction: Optional[Dict[str, Any]]
    raw_llm_output: Optional[str]
    context_entities: List[Dict[str, Any]]
    last_ids: Dict[str, Optional[str]]
    error: Optional[str] = None


class KnowledgeExtractionPipeline:
    """Orchestrates the full knowledge extraction workflow."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()
        self.state = PipelineState()
        
        # Initialize components
        self.doc_processor = DocumentProcessor(margins=self.config.margins)
        self.text_segmenter = TextSegmenter(language=self.config.tokenizer_language)
        self.sentence_extractor = SentenceExtractor(config=self.config.model)
        self.context_manager = ContextManager(
            model_name=self.config.embedding.model_name,
            top_k=self.config.embedding.top_k,
            threshold=self.config.embedding.threshold,
            max_chars=self.config.embedding.max_chars,
        )
        self.kex_extractor = KEXExtractor(config=self.config.model)
        self.id_manager = IDManager()
        self.file_utils = FileUtils()
    
    def _load_previous_state(self, doc_dir: str):
        """
        Load entities and IDs from previously processed pages.
        
        Args:
            doc_dir: Directory with previous extraction results.
        """
        if not self.config.resume.load_previous_entities:
            return
        
        # Use specified previous dir or current output dir
        load_dir = self.config.resume.previous_output_dir or doc_dir
        load_path = Path(load_dir)
        
        if not load_path.exists():
            print(f"⚠️ Previous output directory not found: {load_dir}")
            return
        
        # Find all pagina_*.json files
        page_files = sorted(load_path.glob("pagina_*.json"))
        
        if not page_files:
            print(f"ℹ️ No previous page results found in {load_dir}")
            return
        
        print(f"📂 Loading previous state from {len(page_files)} pages...")
        
        loaded_entities = 0
        for page_file in page_files:
            try:
                with open(page_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                page_num = int(page_file.stem.split('_')[1])
                
                # Skip pages after start_page - 1
                if page_num >= self.config.resume.start_page:
                    continue
                
                # Load entities from sentence_results or directly from ner
                if "sentence_results" in data:
                    for chunk_data in data["sentence_results"]:
                        ner = chunk_data.get("ner", {})
                        if ner and isinstance(ner, dict):
                            entities = ner.get("entities", [])
                            self.context_manager.add_entities(entities)
                            loaded_entities += len(entities)
                            
                            # Update ID manager
                            self.id_manager.update_from_extraction(ner)
                
                # Load last IDs from last_ids_summary if available
                if "last_ids_summary" in data:
                    for category, last_id in data["last_ids_summary"].items():
                        if last_id:
                            self.id_manager.last_ids[category] = last_id
                            
            except Exception as e:
                print(f"⚠️ Error loading {page_file.name}: {e}")
        
        total_entities = self.context_manager.get_entity_count()
        print(f"✅ Loaded {loaded_entities} entities from {self.config.resume.start_page - 1} pages")
        print(f"📊 Total accumulated entities: {total_entities}")
        print(f"🔢 Last IDs: {self.id_manager.get_all_ids()}")
    
    def process(self, pdf_path: str, output_dir: Optional[str] = None) -> List[ExtractionResult]:
        """
        Process a PDF document end-to-end.
        
        Args:
            pdf_path: Path to PDF file.
            output_dir: Output directory (uses config default if not provided).
            
        Returns:
            List of extraction results.
        """
        output_dir = output_dir or self.config.output_dir
        
        # Configure output
        doc_dir = self.file_utils.configure_output(
            pdf_path, output_dir, self.config.model.name
        )
        
        # Load previous state if resuming
        if self.config.resume.start_page > 1 or self.config.resume.load_previous_entities:
            self._load_previous_state(doc_dir)
        
        # Extract text from PDF
        pages = self.doc_processor.extract_text(pdf_path)
        
        results = []
        
        try:
            for page in pages:
                # Skip pages before start_page
                if page.number < self.config.resume.start_page:
                    print(f"⏭️  Skipping page {page.number} (before start_page {self.config.resume.start_page})")
                    continue
                    
                page_results = self._process_page(page, doc_dir)
                results.extend(page_results)
                self.state.processed_pages += 1
        finally:
            # Always cleanup, even if there was an error
            self.cleanup()
        
        return results
    
    def _process_page(self, page: Page, output_dir: str) -> List[ExtractionResult]:
        """Process a single page."""
        results = []
        
        # Segment into chunks based on granularity
        if self.config.granularity == "sentence":
            sentences = self.text_segmenter.segment(page.text)
            chunks = sentences
        elif self.config.granularity == "chunk":
            # LLM segmentation
            sentences = page.text.split("\n") #self.text_segmenter.segment(page.text)
            if sentences:
                segmentation = self.sentence_extractor.segment(sentences)
                chunks = self.sentence_extractor.create_chunks(sentences, segmentation)
            else:
                chunks = []
        else:  # page
            chunks = [page.text]
        
        page_data = {
            "texto_original": page.text,
            "granularity": self.config.granularity,
            "tokenizer_language": self.config.tokenizer_language,
            "margins": self.config.margins,
            "sentence_results": [],
        }
        
        for chunk_idx, chunk_text in enumerate(chunks):
            result = self._process_chunk(
                page.number, chunk_idx, chunk_text
            )
            results.append(result)
            
            # Update state
            if result.extraction:
                # Add entities to context manager
                entities = result.extraction.get("entities", [])
                self.context_manager.add_entities(entities)
                
                # Update ID manager
                self.id_manager.update_from_extraction(result.extraction)
            
            # Build result data
            chunk_data = {
                "chunk_text": chunk_text,
                "ner": result.extraction,
                "llm_output": result.raw_llm_output,
                "context": {
                    "embedding_model": self.config.embedding.model_name,
                    "top_k": self.config.embedding.top_k,
                    "threshold": self.config.embedding.threshold,
                    "contexto_entidades_usadas": len(result.context_entities),
                    "contexto_entidades_seleccionadas": [
                        {"text": e.get("text"), "label": e.get("label")} 
                        for e in result.context_entities
                    ],
                    "entidades_acumuladas_total": self.context_manager.get_entity_count(),
                    "last_ids": self.id_manager.get_all_ids(),
                },
            }
            page_data["sentence_results"].append(chunk_data)
            self.state.processed_chunks += 1
        
        # Add final IDs summary
        page_data["last_ids_summary"] = self.id_manager.get_all_ids()
        
        # Save page results
        self.file_utils.save_page_result(output_dir, page.number, page_data)
        
        return results
    
    def _process_chunk(
        self, 
        page_number: int, 
        chunk_index: int, 
        chunk_text: str
    ) -> ExtractionResult:
        """Process a single text chunk."""
        # Select context entities
        context_entities = self.context_manager.select_context(chunk_text)
        
        # Get last IDs
        last_ids = self.id_manager.get_all_ids()
        
        try:
            # Extract with KEX
            extraction, raw_output = self.kex_extractor.extract(
                text=chunk_text,
                context_entities=context_entities,
                last_ids=last_ids,
            )
            
            # Convert to dict
            extraction_dict = extraction.model_dump()
            
            return ExtractionResult(
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                extraction=extraction_dict,
                raw_llm_output=raw_output,
                context_entities=context_entities,
                last_ids=last_ids,
            )
            
        except Exception as e:
            # Return error result
            return ExtractionResult(
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                extraction=None,
                raw_llm_output=None,
                context_entities=context_entities,
                last_ids=last_ids,
                error=str(e),
            )
    
    def cleanup(self):
        """
        Release Ollama models from memory.
        Called automatically at the end of processing.
        """
        try:
            # Get the base URL without /v1 suffix
            base_url = self.config.model.base_url.replace("/v1", "")
            model_name = self.config.model.name
            
            # Send request to unload model (keep_alive: 0)
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "keep_alive": 0
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✅ Model '{model_name}' unloaded from memory")
            else:
                print(f"⚠️ Could not unload model (status: {response.status_code})")
                
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
