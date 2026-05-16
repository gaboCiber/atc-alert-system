"""
Sentence segmenter using LLM for logical chunking.
"""
import json
from typing import Dict, List, Optional

from ..schemas.sentence_schemas import SegmentationOutput, LogicalChunk
from common.llm_client_factory import ModelConfig, create_instructor_client
from ..config.prompts import SENTENCE_SEGMENTATION_PROMPT, SENTENCE_SEGMENTATION_PROMPT_WITH_CONTEXT


class SentenceExtractor:
    """Segments sentences into logical chunks using LLM."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize sentence extractor.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        
        # Initialize appropriate instructor client based on provider
        self.client, self.mode = create_instructor_client(self.config)
    
    def segment_with_context(
        self,
        sentences: List[str],
        previous_page_chunk: Optional[str] = None
    ) -> SegmentationOutput:
        """
        Segmenta oraciones con contexto opcional de página anterior.
        
        Args:
            sentences: Lista de oraciones actuales (índices 0, 1, 2, ...)
            previous_page_chunk: Último chunk lógico de página anterior (índice -1)
        
        Returns:
            SegmentationOutput con chunks que pueden incluir -1
        """
        # Construir dict indexado
        indexed_sentences = {}
        has_context = previous_page_chunk is not None
        if has_context:
            indexed_sentences[-1] = previous_page_chunk
        indexed_sentences.update({i: s for i, s in enumerate(sentences)})
        
        # Total incluye contexto
        total_indices = len(indexed_sentences)
        first_actual_index = 0
        last_actual_index = len(sentences) - 1
        
        # Seleccionar prompt apropiado
        if has_context:
            prompt = SENTENCE_SEGMENTATION_PROMPT_WITH_CONTEXT.format(
                total_sentences=total_indices,
                last_index=last_actual_index
            )
        else:
            prompt = SENTENCE_SEGMENTATION_PROMPT.format(
                total_sentences=len(sentences),
                last_index=last_actual_index
            )
        
        # Construir input para LLM
        sentences_str = self._build_input(indexed_sentences)
        
        # Llamar al LLM
        response = self.client.chat.completions.create(
            **self.config.completion_kwargs(),
            response_model=SegmentationOutput,
            max_retries=self.config.max_retries,
            validation_context={
                "total_sentences": total_indices,
                "first_actual_index": first_actual_index,
                "last_actual_index": last_actual_index,
                "has_context": has_context,
            },
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sentences_str},
            ],
        )
        
        return response
    
    def segment(self, sentences: List[str]) -> SegmentationOutput:
        """Wrapper backward-compatible - sin contexto."""
        return self.segment_with_context(sentences, previous_page_chunk=None)
    
    def _build_input(self, sentences: Dict[int, str]) -> str:
        """Build the input format for the LLM from indexed sentences."""
        json_input = {str(index): sentence for index, sentence in sentences.items()}
        return json.dumps(json_input, indent=2, ensure_ascii=False)
    
    def create_chunks(
        self,
        sentences: List[str],
        segmentation: SegmentationOutput,
        previous_page_chunk: Optional[str] = None
    ) -> List[str]:
        """
        Crea chunks de texto, manejando índice -1.
        
        Args:
            sentences: Oraciones originales (sin el contexto de -1)
            segmentation: Output del LLM
            previous_page_chunk: Texto del chunk -1 para reconstruir
        
        Returns:
            List[str] con chunks finales
        """
        if not segmentation.chunks:
            raise ValueError("No chunks returned by LLM segmentation")
        
        # Sort chunks by start index
        sorted_chunks = sorted(segmentation.chunks, key=lambda x: x.indices[0])

        chunks = []
        for chunk in sorted_chunks:
            start, end = chunk.indices
            
            # Manejar índice -1
            if start == -1:
                if previous_page_chunk is None:
                    raise ValueError("Received chunk with -1 index but no previous_page_chunk was provided")
                # Incluir el contexto de la página anterior
                chunk_texts = [previous_page_chunk]
                # Agregar oraciones actuales desde 0 hasta end
                actual_start = max(0, start + 1)  # Convertir -1 a 0
                chunk_texts.extend(sentences[actual_start:end+1])
            else:
                # Chunk normal sin contexto
                chunk_texts = sentences[start:end+1]
            
            chunk_text = " ".join(chunk_texts)
            chunks.append(chunk_text)
        
        return chunks