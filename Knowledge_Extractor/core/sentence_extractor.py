"""
Sentence segmenter using LLM for logical chunking.
"""
import instructor
from openai import OpenAI
from typing import List, Optional

from ..schemas.sentence_schemas import SegmentationOutput, LogicalChunk
from ..config.settings import ModelConfig
from ..config.prompts import SENTENCE_SEGMENTATION_PROMPT


class SentenceExtractor:
    """Segments sentences into logical chunks using LLM."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize sentence extractor.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        
        # Initialize Instructor client
        self.client = instructor.from_openai(
            OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            ),
            mode=instructor.Mode.JSON_SCHEMA,
        )
    
    def segment(
        self,
        sentences: List[str]
    ) -> SegmentationOutput:
        """
        Group sentences into logical chunks.
        
        Args:
            sentences: List of sentences to group.
            
        Returns:
            SegmentationOutput with LogicalChunk ranges.
        """
        # Build input format
        sentences_str = self._build_input(sentences)
        
        # Get total for validation context
        total = len(sentences)
        
        # Call LLM with Instructor
        response = self.client.chat.completions.create(
            model=self.config.name,
            response_model=SegmentationOutput,
            max_retries=self.config.max_retries,
            validation_context={"total_sentences": total},
            messages=[
                {"role": "system", "content": SENTENCE_SEGMENTATION_PROMPT.format(total_sentences=total, last_index=total-1)},
                {"role": "user", "content": sentences_str},
            ],
        )
        
        return response
    
    def _build_input(self, sentences: List[str]) -> str:
        """Build the input format for the LLM."""
        lines = ["{"]
        for i, sentence in enumerate(sentences):
            normalized = sentence.replace('\n', ' ').replace('"', '\\"')
            lines.append(f'{i}: "{normalized}"\n')
        lines.append("}")
        return "\n".join(lines)
    
    def create_chunks(
        self,
        sentences: List[str],
        segmentation: SegmentationOutput
    ) -> List[str]:
        """
        Group sentences into text chunks based on segmentation.
        
        Args:
            sentences: Original sentences.
            segmentation: LLM segmentation output (already validated by Pydantic).
            
        Returns:
            List of concatenated text chunks.
        """
        if not segmentation.chunks:
            raise ValueError("No chunks returned by LLM segmentation")
        
        # Sort chunks by start index
        sorted_chunks = sorted(segmentation.chunks, key=lambda x: x.indices[0])

        chunks = []
        for chunk in sorted_chunks:
            start, end = chunk.indices
            
            # Validate range (should be valid given prior checks, but safety first)
            if start < 0 or end >= len(sentences) or start > end:
                raise ValueError(f"Invalid chunk range: [{start}, {end}] for {len(sentences)} sentences")
            
            # Concatenate sentences
            chunk_sentences = sentences[start:end+1]
            chunk_text = " ".join(chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks
