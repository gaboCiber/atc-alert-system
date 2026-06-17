"""
Centralized configuration for Knowledge Extractor.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union

from common.llm_client_factory import ModelConfig


@dataclass
class KEXModelConfig(ModelConfig):
    """KEX-specific model config with extraction mode."""
    extraction_mode: Literal["joint", "sequential"] = "joint"


@dataclass
class EmbeddingConfig:
    """Configuration for sentence embeddings and context selection."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 50  # entities (backward compatible)
    threshold: float = 0.1
    max_chars: int = 4000
    
    # Context type limits (conservative defaults)
    rule_top_k: int = 5
    relationship_top_k: int = 10
    event_top_k: int = 5
    procedure_top_k: int = 5
    
    # Enable/disable flags
    include_rules: bool = True
    include_relationships: bool = True
    include_events: bool = True
    include_procedures: bool = True


@dataclass
class ResumeConfig:
    """Configuration for resuming extraction from previous state."""
    start_page: float = 1.0  # Page to start from (1-indexed, supports sub-pages like 5.2)
    final_page: Optional[float] = None  # Page to end at (inclusive), None = process all
    load_previous_entities: bool = True  # Load entities from previous runs
    previous_output_dir: Optional[str] = None  # Directory with previous results


@dataclass
class PipelineConfig:
    """Main configuration for the knowledge extraction pipeline."""
    # Model configuration
    model: KEXModelConfig = field(default_factory=KEXModelConfig)
    
    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Processing configuration
    granularity: Literal["page", "sentence", "chunk"] = "sentence"
    tokenizer_language: str = "english"
    
    # PDF extraction
    margins: Optional[Tuple[float, float, float, float]] = None
    # margins = (left, bottom, right, top) in points
    
    # Output configuration
    output_dir: str = "output"
    
    # Resume configuration
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    
    # Source directory for pre-generated chunks (optional)
    chunks_source_dir: Optional[str] = None
    
    # Chunk-only mode (skip KEX extraction, only extract and save chunks)
    chunk_only: bool = False
    
    # Validation configuration
    strict_validation: bool = True  # Reject items with invalid cross-references
    
    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = KEXModelConfig(**self.model)
        if isinstance(self.embedding, dict):
            self.embedding = EmbeddingConfig(**self.embedding)
        if isinstance(self.resume, dict):
            self.resume = ResumeConfig(**self.resume)
