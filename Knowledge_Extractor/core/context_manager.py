"""
Context manager for multi-type embeddings and similarity-based selection.
Manages entities, definitions, rules, and relationships with semantic retrieval.
"""
from typing import List, Dict, Any, Optional
from ..config.settings import EmbeddingConfig
from .context_store import VectorStore


class ContextManager:
    """Manages accumulated context items and provides semantic similarity-based selection."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize context manager.
        
        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        model_name = self.config.model_name
        
        # Initialize separate stores for each context type
        self.entity_store = VectorStore(model_name, self._build_entity_embed_text)
        self.definition_store = VectorStore(model_name, self._build_definition_embed_text)
        self.rule_store = VectorStore(model_name, self._build_rule_embed_text)
        self.relationship_store = VectorStore(model_name, self._build_relationship_embed_text)
        
        # Store limits from config
        self.threshold = self.config.threshold
        self.max_chars = self.config.max_chars
    
    # ==========================================
    # Embedding Text Builders
    # ==========================================
    
    def _build_entity_embed_text(self, entity: Dict[str, Any]) -> str:
        """Build rich text representation for entity embedding."""
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
        
        # Add aliases
        aliases = entity.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            parts.append(f"aliases: {' '.join(aliases)}")
        
        return " | ".join(parts).strip()
    
    def _build_definition_embed_text(self, definition: Dict[str, Any]) -> str:
        """Build text representation for definition embedding."""
        parts = []
        
        term = definition.get("term", "")
        def_text = definition.get("definition", "")
        scope = definition.get("scope", "")
        
        if term:
            parts.append(f"term: {term}")
        if def_text:
            parts.append(f"definition: {def_text}")
        if scope:
            parts.append(f"scope: {scope}")
        
        return " | ".join(parts).strip()
    
    def _build_rule_embed_text(self, rule: Dict[str, Any]) -> str:
        """Build text representation for rule embedding."""
        parts = []
        
        rule_type = rule.get("rule_type", "")
        modality = rule.get("modality", "")
        explainability = rule.get("explainability", "")
        
        # Extract trigger and constraint info
        trigger = rule.get("trigger", {})
        constraint = rule.get("constraint", {})
        trigger_desc = trigger.get("description", "") if isinstance(trigger, dict) else ""
        constraint_desc = constraint.get("description", "") if isinstance(constraint, dict) else ""
        
        if rule_type:
            parts.append(f"type: {rule_type}")
        if modality:
            parts.append(f"modality: {modality}")
        if trigger_desc:
            parts.append(f"trigger: {trigger_desc}")
        if constraint_desc:
            parts.append(f"constraint: {constraint_desc}")
        if explainability:
            parts.append(f"purpose: {explainability}")
        
        return " | ".join(parts).strip()
    
    def _build_relationship_embed_text(self, relationship: Dict[str, Any]) -> str:
        """Build text representation for relationship embedding."""
        parts = []
        
        subject = relationship.get("subject_text", "")
        predicate = relationship.get("predicate", "")
        obj = relationship.get("object_text", "")
        rel_type = relationship.get("relation_type", "")
        
        if subject and predicate and obj:
            parts.append(f"{subject} {predicate} {obj}")
        if rel_type:
            parts.append(f"type: {rel_type}")
        
        return " | ".join(parts).strip()
    
    # ==========================================
    # Add Methods
    # ==========================================
    
    def add_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add entities, avoiding duplicates."""
        return self.entity_store.add_items(entities, lambda e: e.get("text", ""))
    
    def add_definitions(self, definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add definitions, avoiding duplicates."""
        return self.definition_store.add_items(definitions, lambda d: d.get("term", ""))
    
    def add_rules(self, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add rules, avoiding duplicates."""
        # Use combination of rule_type and trigger description as key
        def get_rule_key(rule):
            trigger = rule.get("trigger", {})
            trigger_desc = trigger.get("description", "") if isinstance(trigger, dict) else ""
            return f"{rule.get('rule_type', '')}:{trigger_desc[:50]}"
        return self.rule_store.add_items(rules, get_rule_key)
    
    def add_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add relationships, avoiding duplicates."""
        # Use subject+predicate+object as key
        def get_rel_key(rel):
            subj = rel.get("subject_text", "")
            pred = rel.get("predicate", "")
            obj = rel.get("object_text", "")
            return f"{subj}:{pred}:{obj}"
        return self.relationship_store.add_items(relationships, get_rel_key)
    
    # ==========================================
    # Select Methods
    # ==========================================
    
    def select_context(
        self,
        text: str,
        include_entities: bool = True,
        include_definitions: bool = True,
        include_rules: bool = True,
        include_relationships: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Select relevant context items based on semantic similarity.
        
        Args:
            text: Text to find relevant context for.
            include_entities: Whether to include entities.
            include_definitions: Whether to include definitions.
            include_rules: Whether to include rules.
            include_relationships: Whether to include relationships.
            
        Returns:
            Dictionary with selected items by type:
            {
                "entities": [...],
                "definitions": [...],
                "rules": [...],
                "relationships": [...]
            }
        """
        result = {
            "entities": [],
            "definitions": [],
            "rules": [],
            "relationships": []
        }
        
        if include_entities:
            result["entities"] = self.entity_store.select_relevant(
                text, self.config.top_k, self.threshold, self.max_chars
            )
        
        if include_definitions:
            result["definitions"] = self.definition_store.select_relevant(
                text, self.config.definition_top_k, self.threshold, self.max_chars
            )
        
        if include_rules:
            result["rules"] = self.rule_store.select_relevant(
                text, self.config.rule_top_k, self.threshold, self.max_chars
            )
        
        if include_relationships:
            result["relationships"] = self.relationship_store.select_relevant(
                text, self.config.relationship_top_k, self.threshold, self.max_chars
            )
        
        return result
    
    # ==========================================
    # Legacy Compatibility Methods
    # ==========================================
    
    def select_entities(self, text: str) -> List[Dict[str, Any]]:
        """Legacy method: select only entities (for backward compatibility)."""
        return self.entity_store.select_relevant(
            text, self.config.top_k, self.threshold, self.max_chars
        )
    
    # ==========================================
    # Getters
    # ==========================================
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all accumulated entities."""
        return self.entity_store.get_all()
    
    def get_all_definitions(self) -> List[Dict[str, Any]]:
        """Get all accumulated definitions."""
        return self.definition_store.get_all()
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all accumulated rules."""
        return self.rule_store.get_all()
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all accumulated relationships."""
        return self.relationship_store.get_all()
    
    def get_entity_count(self) -> int:
        """Get total number of accumulated entities."""
        return self.entity_store.get_count()
    
    def get_definition_count(self) -> int:
        """Get total number of accumulated definitions."""
        return self.definition_store.get_count()
    
    def get_rule_count(self) -> int:
        """Get total number of accumulated rules."""
        return self.rule_store.get_count()
    
    def get_relationship_count(self) -> int:
        """Get total number of accumulated relationships."""
        return self.relationship_store.get_count()
    
    # ==========================================
    # Utility
    # ==========================================
    
    def reset(self):
        """Clear all accumulated state."""
        self.entity_store.reset()
        self.definition_store.reset()
        self.rule_store.reset()
        self.relationship_store.reset()
