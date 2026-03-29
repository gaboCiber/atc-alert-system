"""
Pipeline state management.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class PipelineState:
    """Maintains state across pipeline execution."""
    
    # Accumulated entities
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Last IDs for sequential numbering
    last_ids: Dict[str, Optional[str]] = field(default_factory=dict)
    
    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    processed_pages: int = 0
    processed_chunks: int = 0
    
    def __post_init__(self):
        # Initialize default last_ids if empty
        if not self.last_ids:
            self.last_ids = {
                "entities": None,
                "relationships": None,
                "events": None,
                "rules": None,
                "procedures": None,
                "definitions": None,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "entities": self.entities,
            "last_ids": self.last_ids,
            "start_time": self.start_time.isoformat(),
            "processed_pages": self.processed_pages,
            "processed_chunks": self.processed_chunks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Restore state from dictionary."""
        state = cls()
        state.entities = data.get("entities", [])
        state.last_ids = data.get("last_ids", {})
        state.processed_pages = data.get("processed_pages", 0)
        state.processed_chunks = data.get("processed_chunks", 0)
        
        # Parse datetime
        start_time_str = data.get("start_time")
        if start_time_str:
            state.start_time = datetime.fromisoformat(start_time_str)
        
        return state
