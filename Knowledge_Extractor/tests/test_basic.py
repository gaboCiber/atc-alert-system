"""
Basic tests for Knowledge Extractor.
"""
import pytest
from pathlib import Path

from Knowledge_Extractor.config.settings import PipelineConfig, ModelConfig
from Knowledge_Extractor.utils.id_manager import IDManager
from Knowledge_Extractor.schemas.kex_schemas import Entity, Relationship


def test_id_manager():
    """Test ID tracking functionality."""
    manager = IDManager()
    
    # Test initial state
    assert manager.get_all_ids()["entities"] is None
    
    # Test update from extraction
    extraction = {
        "entities": [
            {"id": "E001", "text": "Test", "label": "Test"},
            {"id": "E002", "text": "Test2", "label": "Test"},
        ],
        "relationships": [{"id": "R001", "subject_id": "E001", "predicate": "test", "object_id": "E002"}]
    }
    
    manager.update_from_extraction(extraction)
    
    assert manager.get_all_ids()["entities"] == "E002"
    assert manager.get_all_ids()["relationships"] == "R001"
    
    # Test increment
    next_id = manager.increment_id("E002")
    assert next_id == "E003"


def test_config():
    """Test configuration."""
    config = PipelineConfig()
    
    assert config.model.name == "llama3.2"
    assert config.granularity in ["page", "sentence"]
    assert config.embedding.top_k == 50


def test_entity_schema():
    """Test Pydantic schema validation."""
    entity = Entity(
        id="E001",
        text="Runway 09",
        label="Runway",
        context="Active runway"
    )
    
    assert entity.id == "E001"
    assert entity.text == "Runway 09"
    assert entity.label == "Runway"


if __name__ == "__main__":
    test_id_manager()
    test_config()
    test_entity_schema()
    print("All tests passed!")
