"""
Post-processor for extraction results.
Detects and logs errors, deduplicates items, filters corrupt entities.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PostProcessingResult:
    """Result of post-processing an extraction."""
    cleaned_extraction: Dict[str, Any]
    errors: List[Dict[str, Any]]
    was_modified: bool


class ExtractionPostProcessor:
    """Post-processor that detects errors and cleans extraction results."""
    
    def __init__(self):
        self.error_log = []
    
    def process_extraction(
        self,
        extraction: Dict[str, Any],
        page_number: int,
        chunk_index: int,
        chunk_text: str 
    ) -> PostProcessingResult:
        """
        Process extraction: detect errors, deduplicate, clean.
        
        Args:
            extraction: Raw extraction dict from LLM
            page_number: Page number for error logging
            chunk_index: Chunk index for error logging
            chunk_text: Original text for context
            
        Returns:
            PostProcessingResult with cleaned extraction and error log
        """
        errors = []
        cleaned = {
            "entities": [],
            "relationships": [],
            "events": [],
            "rules": [],
            "procedures": []
        }
        
        # Track seen IDs
        seen_ids = set()
        
        # Define processors for each type
        processors = [
            ("entities", self._process_entities),
            ("relationships", self._process_relationships),
            ("events", self._process_events),
            ("rules", self._process_rules),
            ("procedures", self._process_procedures),
        ]
        
        # Process each type using DRY pattern
        for item_type, processor in processors:
            items = extraction.get(item_type, [])
            cleaned[item_type], type_errors = processor(
                items, seen_ids, page_number, chunk_index
            )
            errors.extend(type_errors)
            
            # Update seen IDs with valid items
            for item in cleaned[item_type]:
                seen_ids.add(item.get("id"))
        
        was_modified = len(errors) > 0
        
        # Add context to errors
        for error in errors:
            error["page_number"] = page_number
            error["chunk_index"] = chunk_index
            error["chunk_text"] = chunk_text if chunk_text else ""
        
        return PostProcessingResult(
            cleaned_extraction=cleaned,
            errors=errors,
            was_modified=was_modified
        )
    
    def _process_entities(
        self,
        entities: List[Dict],
        seen_ids: set,
        page: int,
        chunk: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process entities: filter corrupt, deduplicate."""
        result = []
        errors = []
        
        for idx, entity in enumerate(entities):
            entity_id = entity.get("id", "")
            text = entity.get("text", "")
            context = entity.get("context", "")
            
            # Check 1: Missing ID
            if not entity_id:
                errors.append({
                    "type": "missing_id",
                    "item_type": "entity",
                    "item_index": idx,
                    "original_item": entity,
                    "message": f"Entity at index {idx} has no ID"
                })
                continue
            
            # Check 2: Wrong prefix
            if not entity_id.startswith("E"):
                errors.append({
                    "type": "wrong_prefix",
                    "item_type": "entity",
                    "item_index": idx,
                    "item_id": entity_id,
                    "original_item": entity,
                    "message": f"Entity ID '{entity_id}' does not start with 'E'"
                })
                # Try to fix by adding prefix if it's just a number
                if entity_id.isdigit():
                    entity_id = f"E{entity_id}"
                    entity["id"] = entity_id
                else:
                    continue
            
            # Check 3: Duplicate ID
            if entity_id in seen_ids:
                errors.append({
                    "type": "duplicate_id",
                    "item_type": "entity",
                    "item_index": idx,
                    "item_id": entity_id,
                    "original_item": entity,
                    "message": f"Duplicate entity ID '{entity_id}'"
                })
                continue
            
            # Check 4: Corrupt entity (text == id)
            if text == entity_id:
                errors.append({
                    "type": "corrupt_entity",
                    "item_type": "entity",
                    "item_index": idx,
                    "item_id": entity_id,
                    "original_item": entity,
                    "message": f"Entity '{entity_id}' has text equal to ID (likely hallucination)"
                })
                continue
            
            # Check 5: Empty/N/A context
            if not context or context == "N/A" or context == "None":
                errors.append({
                    "type": "empty_context",
                    "item_type": "entity",
                    "item_index": idx,
                    "item_id": entity_id,
                    "original_item": entity,
                    "message": f"Entity '{entity_id}' has empty or N/A context"
                })
            
            result.append(entity)
            seen_ids.add(entity_id)
        
        return result, errors
    
    def _process_relationships(
        self,
        relationships: List[Dict],
        seen_ids: set,
        page: int,
        chunk: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process relationships: deduplicate, validate prefixes."""
        result = []
        errors = []
        
        for idx, rel in enumerate(relationships):
            rel_id = rel.get("id", "")
            
            if not rel_id:
                continue
            
            if not rel_id.startswith("R"):
                errors.append({
                    "type": "wrong_prefix",
                    "item_type": "relationship",
                    "item_index": idx,
                    "item_id": rel_id,
                    "original_item": rel,
                    "message": f"Relationship ID '{rel_id}' does not start with 'R'"
                })
                continue
            
            if rel_id in seen_ids:
                errors.append({
                    "type": "duplicate_id",
                    "item_type": "relationship",
                    "item_index": idx,
                    "item_id": rel_id,
                    "original_item": rel,
                    "message": f"Duplicate relationship ID '{rel_id}'"
                })
                continue
            
            result.append(rel)
            seen_ids.add(rel_id)
        
        return result, errors
    
    def _process_events(
        self,
        events: List[Dict],
        seen_ids: set,
        page: int,
        chunk: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process events: deduplicate, validate prefixes."""
        result = []
        errors = []
        
        for idx, event in enumerate(events):
            event_id = event.get("id", "")
            
            if not event_id:
                continue
            
            if not event_id.startswith("EV"):
                errors.append({
                    "type": "wrong_prefix",
                    "item_type": "event",
                    "item_index": idx,
                    "item_id": event_id,
                    "original_item": event,
                    "message": f"Event ID '{event_id}' does not start with 'EV'"
                })
                continue
            
            if event_id in seen_ids:
                errors.append({
                    "type": "duplicate_id",
                    "item_type": "event",
                    "item_index": idx,
                    "item_id": event_id,
                    "original_item": event,
                    "message": f"Duplicate event ID '{event_id}'"
                })
                continue
            
            result.append(event)
            seen_ids.add(event_id)
        
        return result, errors
    
    def _process_rules(
        self,
        rules: List[Dict],
        seen_ids: set,
        page: int,
        chunk: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process rules: deduplicate, validate prefixes."""
        result = []
        errors = []
        
        for idx, rule in enumerate(rules):
            rule_id = rule.get("id", "")
            
            if not rule_id:
                continue
            
            if not rule_id.startswith("RULE"):
                errors.append({
                    "type": "wrong_prefix",
                    "item_type": "rule",
                    "item_index": idx,
                    "item_id": rule_id,
                    "original_item": rule,
                    "message": f"Rule ID '{rule_id}' does not start with 'RULE'"
                })
                continue
            
            if rule_id in seen_ids:
                errors.append({
                    "type": "duplicate_id",
                    "item_type": "rule",
                    "item_index": idx,
                    "item_id": rule_id,
                    "original_item": rule,
                    "message": f"Duplicate rule ID '{rule_id}'"
                })
                continue
            
            result.append(rule)
            seen_ids.add(rule_id)
        
        return result, errors
    
    def _process_procedures(
        self,
        procedures: List[Dict],
        seen_ids: set,
        page: int,
        chunk: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process procedures: deduplicate, validate prefixes."""
        result = []
        errors = []
        
        for idx, proc in enumerate(procedures):
            proc_id = proc.get("id", "")
            
            if not proc_id:
                continue
            
            if not proc_id.startswith("P"):
                errors.append({
                    "type": "wrong_prefix",
                    "item_type": "procedure",
                    "item_index": idx,
                    "item_id": proc_id,
                    "original_item": proc,
                    "message": f"Procedure ID '{proc_id}' does not start with 'P'"
                })
                continue
            
            if proc_id in seen_ids:
                errors.append({
                    "type": "duplicate_id",
                    "item_type": "procedure",
                    "item_index": idx,
                    "item_id": proc_id,
                    "original_item": proc,
                    "message": f"Duplicate procedure ID '{proc_id}'"
                })
                continue
            
            # Check for RULE* IDs in procedures (contamination)
            if proc_id.startswith("RULE"):
                errors.append({
                    "type": "type_contamination",
                    "item_type": "procedure",
                    "item_index": idx,
                    "item_id": proc_id,
                    "original_item": proc,
                    "message": f"Procedure '{proc_id}' has RULE prefix (likely misclassified)"
                })
                continue
            
            result.append(proc)
            seen_ids.add(proc_id)
        
        return result, errors


def save_errors_to_file(
    errors: List[Dict[str, Any]],
    output_dir: str,
    page_number: int
) -> str:
    """
    Save errors to a JSON file for later analysis.
    
    Args:
        errors: List of error dictionaries
        output_dir: Directory to save the file
        page_number: Page number for filename
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    error_file = output_path / f"pagina_{page_number}_errors.json"
    
    error_data = {
        "page_number": page_number,
        "error_count": len(errors),
        "errors": errors,
        "summary": _generate_error_summary(errors)
    }
    
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    return str(error_file)


def _generate_error_summary(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of errors by type."""
    summary = {
        "total": len(errors),
        "by_type": {},
        "by_item_type": {}
    }
    
    for error in errors:
        error_type = error.get("type", "unknown")
        item_type = error.get("item_type", "unknown")
        
        summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
        summary["by_item_type"][item_type] = summary["by_item_type"].get(item_type, 0) + 1
    
    return summary
