"""
Centralized configuration for Knowledge Extractor.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    max_retries: int = 3
    timeout: int = 120


@dataclass
class EmbeddingConfig:
    """Configuration for sentence embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 50
    threshold: float = 0.1
    max_chars: int = 4000


@dataclass
class ResumeConfig:
    """Configuration for resuming extraction from previous state."""
    start_page: int = 1  # Page to start from (1-indexed)
    load_previous_entities: bool = True  # Load entities from previous runs
    previous_output_dir: Optional[str] = None  # Directory with previous results


@dataclass
class PipelineConfig:
    """Main configuration for the knowledge extraction pipeline."""
    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    
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
    
    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.embedding, dict):
            self.embedding = EmbeddingConfig(**self.embedding)
        if isinstance(self.resume, dict):
            self.resume = ResumeConfig(**self.resume)
