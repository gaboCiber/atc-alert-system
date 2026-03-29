"""
Context manager for entity embeddings and similarity-based selection.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class ContextManager:
    """Manages accumulated entities and provides semantic context selection."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 50,
        threshold: float = 0.1,
        max_chars: int = 4000
    ):
        """
        Initialize context manager.
        
        Args:
            model_name: Name of the sentence transformer model.
            top_k: Maximum number of context entities to select.
            threshold: Minimum similarity score for context selection.
            max_chars: Maximum characters to use for embedding generation.
        """
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.threshold = threshold
        self.max_chars = max_chars
        
        # Storage
        self.entities_by_norm: Dict[str, Dict[str, Any]] = {}
        self.entity_order: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication."""
        return (text or "").lower().strip()
    
    def _build_embed_text(self, entity: Dict[str, Any]) -> str:
        """Build rich text representation for embedding."""
        parts = []
        
        text = entity.get("text", "")
        label = entity.get("label", "")
        subtype = entity.get("subtype", "")
        context = entity.get("context", "")
        
        if text:
            parts.append(text)
        if label:
            parts.append(f"type: {label}")
        if subtype:
            parts.append(f"subtype: {subtype}")
        if context:
            parts.append(f"context: {context}")
        
        # Add attributes if present
        attrs = entity.get("attributes", {})
        if isinstance(attrs, dict):
            for key in ["status", "phase", "role"]:
                if attrs.get(key):
                    parts.append(f"{key}: {attrs[key]}")
        
        # Add aliases
        aliases = entity.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            parts.append(f"aliases: {' '.join(aliases)}")
        
        return " | ".join(parts).strip()
    
    def add_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add new entities, avoiding duplicates.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            List of actually added entities (new ones only).
        """
        new_entities = []
        
        for entity in entities:
            norm = self._normalize_text(entity.get("text", ""))
            if not norm:
                continue
            
            if norm not in self.entities_by_norm:
                self.entities_by_norm[norm] = entity
                self.entity_order.append(norm)
                new_entities.append(entity)
        
        # Update embeddings for new entities
        if new_entities:
            self._update_embeddings(new_entities)
        
        return new_entities
    
    def _update_embeddings(self, new_entities: List[Dict[str, Any]]):
        """Update embeddings matrix with new entities."""
        texts = [self._build_embed_text(e) for e in new_entities]
        new_embeddings = self.model.encode(texts, normalize_embeddings=False)
        new_embeddings = np.atleast_2d(new_embeddings)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def select_context(self, text: str) -> List[Dict[str, Any]]:
        """
        Select context entities based on semantic similarity.
        
        Args:
            text: Text to find relevant context for.
            
        Returns:
            List of relevant entities, sorted by similarity.
        """
        if not self.entity_order or self.embeddings is None:
            return []
        
        # Truncate text if too long
        if len(text) > self.max_chars:
            text = text[:self.max_chars]
        
        # Generate embedding for input text
        text_embedding = self.model.encode(text, normalize_embeddings=False)
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(text_embedding, self.embeddings)
        
        # Select top-k above threshold
        selected = []
        sorted_indices = np.argsort(-similarities)
        
        for idx in sorted_indices[:self.top_k]:
            sim = float(similarities[idx])
            if sim < self.threshold:
                break
            norm = self.entity_order[idx]
            selected.append(self.entities_by_norm[norm])
        
        return selected
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vector a and matrix b."""
        norm_a = np.linalg.norm(a)
        norms_b = np.linalg.norm(b, axis=1)
        
        # Avoid division by zero
        denom = norm_a * norms_b
        denom = np.where(denom == 0, 1e-12, denom)
        
        return (b @ a) / denom
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all accumulated entities in order."""
        return [self.entities_by_norm[norm] for norm in self.entity_order]
    
    def get_entity_count(self) -> int:
        """Get total number of accumulated entities."""
        return len(self.entity_order)
    
    def reset(self):
        """Clear all accumulated state."""
        self.entities_by_norm.clear()
        self.entity_order.clear()
        self.embeddings = None
