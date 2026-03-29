"""
KEX Extractor using Instructor for structured extraction.
"""
import instructor
from openai import OpenAI
from typing import Optional, List, Dict, Any, Tuple

from ..schemas.kex_schemas import AeronauticalExtraction
from ..config.settings import ModelConfig
from ..config.prompts import build_kex_prompt


class KEXExtractor:
    """Extract aeronautical knowledge using structured LLM calls."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize KEX extractor.
        
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
    
    def extract(
        self,
        text: str,
        context_entities: Optional[List[Dict[str, Any]]] = None,
        last_ids: Optional[Dict[str, str]] = None
    ) -> Tuple[AeronauticalExtraction, str]:
        """
        Extract knowledge from text.
        
        Args:
            text: Text to analyze.
            context_entities: Previously extracted entities for context.
            last_ids: Last used IDs for sequential numbering.
            
        Returns:
            Tuple of (AeronauticalExtraction, raw_llm_output).
        """
        # Build prompts
        system_prompt, user_prompt = build_kex_prompt(
            text=text,
            context_entities=context_entities,
            last_ids=last_ids
        )
        
        # Call LLM with Instructor
        response = self.client.chat.completions.create(
            model=self.config.name,
            response_model=AeronauticalExtraction,
            max_retries=self.config.max_retries,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # Get raw output for debugging (Instructor doesn't expose this directly,
        # but we can use the model's dump)
        raw_output = response.model_dump_json(indent=2)
        
        return response, raw_output
