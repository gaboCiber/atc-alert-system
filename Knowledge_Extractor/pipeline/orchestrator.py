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
    # All context types
    context_entities: List[Dict[str, Any]]
    context_definitions: List[Dict[str, Any]]
    context_rules: List[Dict[str, Any]]
    context_relationships: List[Dict[str, Any]]
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
        self.context_manager = ContextManager(config=self.config.embedding)
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
                
                # Load all context types from sentence_results
                if "sentence_results" in data:
                    for chunk_data in data["sentence_results"]:
                        ner = chunk_data.get("ner", {})
                        if ner and isinstance(ner, dict):
                            entities = ner.get("entities", [])
                            definitions = ner.get("definitions", [])
                            rules = ner.get("rules", [])
                            relationships = ner.get("relationships", [])
                            
                            self.context_manager.add_entities(entities)
                            self.context_manager.add_definitions(definitions)
                            self.context_manager.add_rules(rules)
                            self.context_manager.add_relationships(relationships)
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
        print(f"\n📄 Processing PDF: {pdf_path}")
        print(f"📁 Output directory: {doc_dir}")
        print(f"🤖 Model: {self.config.model.name}")
        print(f"📐 Granularity: {self.config.granularity}")
        
        # Load previous state if resuming
        if self.config.resume.start_page > 1 or self.config.resume.load_previous_entities:
            self._load_previous_state(doc_dir)
        
        # Check which pages have external chunks (for optimization)
        external_chunk_pages = set()
        if self.config.chunks_source_dir:
            print(f"🔍 Checking chunks source: {self.config.chunks_source_dir}")
            external_chunk_pages = self.file_utils.get_available_chunk_pages(
                self.config.chunks_source_dir
            )
            if external_chunk_pages:
                print(f"📂 Found external chunks for pages: {sorted(external_chunk_pages)[:10]}{'...' if len(external_chunk_pages) > 10 else ''}")
            else:
                print(f"⚠️ No external chunks found in: {self.config.chunks_source_dir}")
        
        # Get total pages from PDF without extracting text
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        
        # Determine which pages need PDF text extraction
        pages_needing_extraction = [
            i + 1 for i in range(total_pages)
            if (i + 1) not in external_chunk_pages and (i + 1) >= self.config.resume.start_page
        ]
        
        # Extract text only for pages that need it
        extracted_texts = {}
        if pages_needing_extraction:
            print(f"📖 Extracting text from {len(pages_needing_extraction)} pages...")
            doc = fitz.open(pdf_path)
            for page_num in pages_needing_extraction:
                page = doc[page_num - 1]
                text = self.doc_processor._extract_page_text(page)
                extracted_texts[page_num] = text
            doc.close()
        
        # Build list of Page objects
        pages = []
        for i in range(total_pages):
            page_num = i + 1
            if page_num in external_chunk_pages:
                # Page has external chunks - no PDF text needed
                pages.append(Page(
                    number=page_num,
                    text="",  # Empty text - will use external chunks
                    metadata={"external_chunks": True}
                ))
            else:
                # Page needs PDF text
                pages.append(Page(
                    number=page_num,
                    text=extracted_texts.get(page_num, ""),
                    metadata={}
                ))
        
        print(f"📊 Total pages in document: {total_pages}")
        print(f"🚀 Starting extraction...\n")
        
        results = []
        
        try:
            for page in pages:
                # Skip pages before start_page
                if page.number < self.config.resume.start_page:
                    continue
                
                is_external = page.metadata.get("external_chunks", False)
                if is_external:
                    print(f"\n{'='*60}")
                    print(f"📄 Processing page {page.number}/{total_pages} (using external chunks)")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"📄 Processing page {page.number}/{total_pages}")
                    print(f"{'='*60}")
                    
                page_results = self._process_page(page, doc_dir, total_pages)
                results.extend(page_results)
                self.state.processed_pages += 1
                
                # Progress summary after each page
                print(f"\n✅ Page {page.number} complete - {len(page_results)} chunks processed")
                print(f"📊 Total entities accumulated: {self.context_manager.get_entity_count()}")
                print(f"🔢 Current IDs: {self.id_manager.get_all_ids()}")
                
        finally:
            print(f"\n{'='*60}")
            print(f"🏁 Processing complete!")
            print(f"📄 Pages processed: {self.state.processed_pages}/{total_pages}")
            print(f"🔧 Total chunks: {self.state.processed_chunks}")
            print(f"📦 Total entities: {self.context_manager.get_entity_count()}")
            print(f"{'='*60}\n")
            
            # Always cleanup, even if there was an error
            self.cleanup()
        
        return results
    
    def _process_page(self, page: Page, output_dir: str, total_pages: int) -> List[ExtractionResult]:
        """Process a single page."""
        results = []
        chunks = None
        is_generated = False
        chunks_source = self.config.chunks_source_dir
        
        # Try to load chunks from external source first
        if chunks_source:
            chunks = self.file_utils.load_page_chunks(chunks_source, page.number)
            if chunks:
                print(f"  📂 Loaded {len(chunks)} chunks from external source")
        
        # If no external chunks, generate from PDF
        if chunks is None:
            is_generated = True
            # Segment into chunks based on granularity
            print(f"  🔍 Segmenting page {page.number}...")
            
            if self.config.granularity == "sentence":
                sentences = self.text_segmenter.segment(page.text)
                chunks = sentences
                print(f"     └─ Split into {len(chunks)} sentences")
            elif self.config.granularity == "chunk":
                # Try LLM segmentation first, fallback to NLTK on failure
                sentences = page.text.split("\n")
                print(f"     └─ Found {len(sentences)} lines, attempting LLM segmentation...")
                
                if sentences:
                    try:
                        segmentation = self.sentence_extractor.segment(sentences)
                        chunks = self.sentence_extractor.create_chunks(sentences, segmentation)
                        print(f"     └─ ✓ LLM created {len(chunks)} logical chunks")
                    except Exception as e:
                        # Fallback to NLTK sentence segmentation
                        print(f"     ⚠️  LLM segmentation failed: {str(e)[:60]}...")
                        print(f"     🔄 Falling back to NLTK sentence segmentation...")
                        chunks = self.text_segmenter.segment(page.text)
                        print(f"     └─ ✓ NLTK created {len(chunks)} sentence chunks")
                else:
                    chunks = []
                    print(f"     └─ No content to process")
            else:  # page
                chunks = [page.text]
                print(f"     └─ Processing as single page unit")
        
        # Always save chunks to output directory (whether loaded or generated)
        self.file_utils.save_page_chunks(
            output_dir, page.number, chunks, self.config.granularity
        )
        source_type = "external" if (chunks_source and not is_generated) else "generated"
        print(f"  💾 Saved {len(chunks)} {source_type} chunks to pagina_{page.number}_chunks.json")
        
        page_data = {
            "texto_original": page.text if chunks is None or not chunks_source else f"[Chunks from {source_type} source]",
            "granularity": self.config.granularity,
            "tokenizer_language": self.config.tokenizer_language,
            "margins": self.config.margins,
            "chunks_source": source_type,
            "sentence_results": [],
        }
        
        for chunk_idx, chunk_text in enumerate(chunks):
            # Progress indicator for chunks
            progress = f"[{chunk_idx+1}/{len(chunks)}]"
            print(f"    {progress} Processing chunk {chunk_idx+1}...", end=" ")
            
            result = self._process_chunk(
                page.number, chunk_idx, chunk_text
            )
            results.append(result)
            
            # Update state with all context types
            if result.extraction:
                # Add all types to context manager
                entities = result.extraction.get("entities", [])
                definitions = result.extraction.get("definitions", [])
                rules = result.extraction.get("rules", [])
                relationships = result.extraction.get("relationships", [])
                events = result.extraction.get("events", [])
                procedures = result.extraction.get("procedures", [])
                
                self.context_manager.add_entities(entities)
                self.context_manager.add_definitions(definitions)
                self.context_manager.add_rules(rules)
                self.context_manager.add_relationships(relationships)
                
                # Update ID manager
                self.id_manager.update_from_extraction(result.extraction)
                
                # Log extraction results
                total_extracted = len(entities) + len(relationships) + len(rules) + len(events) + len(procedures) + len(definitions)
                print(f"✓ Extracted: {len(entities)}E, {len(definitions)}D, {len(rules)}RULE, {len(relationships)}R, {len(events)}EV, {len(procedures)}P")
            else:
                error_msg = result.error or "Unknown error"
                print(f"✗ Failed: {error_msg[:50]}..." if len(error_msg) > 50 else f"✗ Failed: {error_msg}")
            
            # Build result data with all context types
            chunk_data = {
                "chunk_text": chunk_text,
                "ner": result.extraction,
                "llm_output": result.raw_llm_output,
                "context": {
                    "embedding_model": self.config.embedding.model_name,
                    "threshold": self.config.embedding.threshold,
                    # Entity context
                    "contexto_entidades_usadas": len(result.context_entities),
                    "contexto_entidades_seleccionadas": [
                        {"text": e.get("text"), "label": e.get("label")} 
                        for e in result.context_entities
                    ],
                    # Definition context
                    "contexto_definiciones_usadas": len(result.context_definitions),
                    "contexto_definiciones_seleccionadas": [
                        {"term": d.get("term"), "id": d.get("id")}
                        for d in result.context_definitions
                    ],
                    # Rule context
                    "contexto_reglas_usadas": len(result.context_rules),
                    "contexto_reglas_seleccionadas": [
                        {"type": r.get("rule_type"), "modality": r.get("modality"), "id": r.get("id")}
                        for r in result.context_rules
                    ],
                    # Relationship context
                    "contexto_relaciones_usadas": len(result.context_relationships),
                    "contexto_relaciones_seleccionadas": [
                        {"subject": rel.get("subject_text"), "predicate": rel.get("predicate"), "object": rel.get("object_text")}
                        for rel in result.context_relationships
                    ],
                    # Totals
                    "entidades_acumuladas_total": self.context_manager.get_entity_count(),
                    "definiciones_acumuladas_total": self.context_manager.get_definition_count(),
                    "reglas_acumuladas_total": self.context_manager.get_rule_count(),
                    "relaciones_acumuladas_total": self.context_manager.get_relationship_count(),
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
        # Select all context types based on config
        selected_context = self.context_manager.select_context(
            chunk_text,
            include_entities=True,
            include_definitions=self.config.embedding.include_definitions,
            include_rules=self.config.embedding.include_rules,
            include_relationships=self.config.embedding.include_relationships
        )
        
        context_entities = selected_context["entities"]
        context_definitions = selected_context["definitions"]
        context_rules = selected_context["rules"]
        context_relationships = selected_context["relationships"]
        
        # Get last IDs
        last_ids = self.id_manager.get_all_ids()
        
        # Log context info
        total_context = len(context_entities) + len(context_definitions) + len(context_rules) + len(context_relationships)
        if total_context > 0:
            parts = []
            if context_entities:
                parts.append(f"{len(context_entities)}E")
            if context_definitions:
                parts.append(f"{len(context_definitions)}D")
            if context_rules:
                parts.append(f"{len(context_rules)}RULE")
            if context_relationships:
                parts.append(f"{len(context_relationships)}R")
            print(f"(context: {', '.join(parts)})", end=" ")
        
        # Extract with KEX (always returns, even on failure)
        extraction, raw_output = self.kex_extractor.extract(
            text=chunk_text,
            context_entities=context_entities,
            context_definitions=context_definitions,
            context_rules=context_rules,
            context_relationships=context_relationships,
            include_definitions=self.config.embedding.include_definitions,
            include_rules=self.config.embedding.include_rules,
            include_relationships=self.config.embedding.include_relationships,
            last_ids=last_ids,
        )
        
        # Check if extraction succeeded
        if extraction is None:
            # Extraction failed after all retries, but we have raw_output for debugging
            error_msg = "KEX extraction failed after all retries (see raw_llm_output for details)"
            print(f"✗ Failed: {error_msg[:60]}...")
            
            return ExtractionResult(
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                extraction=None,
                raw_llm_output=raw_output,  # Always save for debugging
                context_entities=context_entities,
                context_definitions=context_definitions,
                context_rules=context_rules,
                context_relationships=context_relationships,
                last_ids=last_ids,
                error=error_msg,
            )
        
        # Successful extraction
        # Convert to dict
        extraction_dict = extraction.model_dump()
        
        return ExtractionResult(
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            extraction=extraction_dict,
            raw_llm_output=raw_output,
            context_entities=context_entities,
            context_definitions=context_definitions,
            context_rules=context_rules,
            context_relationships=context_relationships,
            last_ids=last_ids,
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
