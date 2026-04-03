"""
ID Manager for tracking sequential IDs across extraction categories.
"""
import re
from typing import Dict, Optional, List, Any


class IDManager:
    """Manages sequential IDs for entities, relationships, events, rules, procedures."""
    
    # ID prefixes for each category
    PREFIXES = {
        "entities": "E",
        "relationships": "R",
        "events": "EV",
        "rules": "RULE",
        "procedures": "P",
    }
    
    def __init__(self):
        """Initialize ID manager with empty state."""
        self.last_ids: Dict[str, Optional[str]] = {
            category: None for category in self.PREFIXES.keys()
        }
    
    @staticmethod
    def increment_id(last_id: Optional[str]) -> Optional[str]:
        """
        Increment an alphanumeric ID.
        
        Args:
            last_id: Previous ID (e.g., "E001")
            
        Returns:
            Next ID (e.g., "E002")
        """
        if not last_id:
            return None
        
        match = re.match(r'^([A-Z]+)(\d+)$', last_id)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            return f"{prefix}{number + 1:03d}"
        
        return last_id
    
    @staticmethod
    def extract_last_id(items: List[Any], prefix: str) -> Optional[str]:
        """
        Extract the highest ID from a list of items.
        
        Args:
            items: List of dicts with 'id' field.
            prefix: ID prefix to look for.
            
        Returns:
            Highest ID found or None.
        """
        if not items:
            return None
        
        max_num = 0
        for item in items:
            if isinstance(item, dict) and "id" in item:
                item_id = item["id"]
                if item_id.startswith(prefix):
                    match = re.match(rf'^{re.escape(prefix)}(\d+)$', item_id)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)
        
        return f"{prefix}{max_num:03d}" if max_num > 0 else None
    
    def update_from_extraction(self, extraction: Dict[str, Any]):
        """
        Update last IDs based on extraction results.
        
        Args:
            extraction: Dict with keys like 'entities', 'relationships', etc.
        """
        for category, prefix in self.PREFIXES.items():
            if category in extraction and isinstance(extraction[category], list):
                last_id = self.extract_last_id(extraction[category], prefix)
                if last_id:
                    self.last_ids[category] = last_id
    
    def get_next_id_hint(self, category: str) -> Optional[str]:
        """
        Get hint for next ID in sequence.
        
        Args:
            category: One of entities, relationships, events, rules, procedures, definitions.
            
        Returns:
            Next expected ID or None.
        """
        last_id = self.last_ids.get(category)
        return self.increment_id(last_id)
    
    def get_all_ids(self) -> Dict[str, Optional[str]]:
        """Get all last IDs."""
        return self.last_ids.copy()
    
    def reset(self):
        """Reset all IDs to None."""
        self.last_ids = {category: None for category in self.PREFIXES.keys()}
