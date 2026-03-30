"""
Generic vector store for semantic similarity-based retrieval.
Provides DRY implementation for managing embeddings and similarity search.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Generic vector store for semantic similarity-based retrieval.
    
    Manages items with embeddings, deduplication, and similarity-based selection.
    """
    
    def __init__(
        self,
        model_name: str,
        embed_builder: Callable[[Any], str]
    ):
        """
        Initialize vector store.
        
        Args:
            model_name: Name of the sentence transformer model.
            embed_builder: Function to build embedding text from an item.
        """
        self.model = SentenceTransformer(model_name)
        self.embed_builder = embed_builder
        
        # Storage
        self.items_by_key: Dict[str, Any] = {}
        self.item_order: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key for deduplication."""
        return (key or "").lower().strip()
    
    def add_items(
        self,
        items: List[Any],
        key_fn: Callable[[Any], str]
    ) -> List[Any]:
        """
        Add items, avoiding duplicates.
        
        Args:
            items: List of items to add.
            key_fn: Function to extract unique key from an item.
            
        Returns:
            List of actually added items (new ones only).
        """
        new_items = []
        
        for item in items:
            key = self._normalize_key(key_fn(item))
            if not key:
                continue
            
            if key not in self.items_by_key:
                self.items_by_key[key] = item
                self.item_order.append(key)
                new_items.append(item)
        
        # Update embeddings for new items
        if new_items:
            self._update_embeddings(new_items)
        
        return new_items
    
    def _update_embeddings(self, new_items: List[Any]):
        """Update embeddings matrix with new items."""
        texts = [self.embed_builder(item) for item in new_items]
        new_embeddings = self.model.encode(texts, normalize_embeddings=False)
        new_embeddings = np.atleast_2d(new_embeddings)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def select_relevant(
        self,
        text: str,
        top_k: int,
        threshold: float,
        max_chars: int = 4000
    ) -> List[Any]:
        """
        Select top-k most similar items to the given text.
        
        Args:
            text: Text to find relevant items for.
            top_k: Maximum number of items to select.
            threshold: Minimum similarity score.
            max_chars: Maximum characters to use for embedding.
            
        Returns:
            List of relevant items, sorted by similarity (highest first).
        """
        if not self.item_order or self.embeddings is None:
            return []
        
        # Truncate text if too long
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Generate embedding for input text
        text_embedding = self.model.encode(text, normalize_embeddings=False)
        
        # Calculate cosine similarities
        similarities = self._cosine_similarity(text_embedding, self.embeddings)
        
        # Select top-k above threshold
        selected = []
        sorted_indices = np.argsort(-similarities)
        
        for idx in sorted_indices[:top_k]:
            sim = float(similarities[idx])
            if sim < threshold:
                break
            key = self.item_order[idx]
            selected.append(self.items_by_key[key])
        
        return selected
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vector a and matrix b."""
        norm_a = np.linalg.norm(a)
        norms_b = np.linalg.norm(b, axis=1)
        
        # Avoid division by zero
        denom = norm_a * norms_b
        denom = np.where(denom == 0, 1e-12, denom)
        
        return (b @ a) / denom
    
    def get_all(self) -> List[Any]:
        """Get all stored items in order."""
        return [self.items_by_key[key] for key in self.item_order]
    
    def get_count(self) -> int:
        """Get total number of stored items."""
        return len(self.item_order)
    
    def reset(self):
        """Clear all stored state."""
        self.items_by_key.clear()
        self.item_order.clear()
        self.embeddings = None
