"""
KEX Extractor using Instructor for structured extraction.
"""
import json
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
        
        # Raw OpenAI client for fallback raw extraction
        self.raw_client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
    
    def extract(
        self,
        text: str,
        context_entities: Optional[List[Dict[str, Any]]] = None,
        context_definitions: Optional[List[Dict[str, Any]]] = None,
        context_rules: Optional[List[Dict[str, Any]]] = None,
        context_relationships: Optional[List[Dict[str, Any]]] = None,
        include_definitions: bool = True,
        include_rules: bool = True,
        include_relationships: bool = True,
        last_ids: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[AeronauticalExtraction], str]:
        """
        Extract knowledge from text.
        
        Args:
            text: Text to analyze.
            context_entities: Previously extracted entities for context.
            context_definitions: Previously extracted definitions for context.
            context_rules: Previously extracted rules for context.
            context_relationships: Previously extracted relationships for context.
            include_definitions: Whether to include definitions in prompt.
            include_rules: Whether to include rules in prompt.
            include_relationships: Whether to include relationships in prompt.
            last_ids: Last used IDs for sequential numbering.
            
        Returns:
            Tuple of (AeronauticalExtraction or None, raw_llm_output).
            Always returns raw output even on failure for debugging.
        """
        # Build prompts with all context types
        system_prompt, user_prompt = build_kex_prompt(
            text=text,
            context_entities=context_entities,
            context_definitions=context_definitions,
            context_rules=context_rules,
            context_relationships=context_relationships,
            include_definitions=include_definitions,
            include_rules=include_rules,
            include_relationships=include_relationships,
            last_ids=last_ids
        )
        
        try:
            # Call LLM with Instructor (structured)
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=AeronauticalExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            # Get raw output for debugging
            raw_output = response.model_dump_json(indent=2)
            
            return response, raw_output
            
        except Exception as e:
            # Structured extraction failed after all retries
            # Get raw output for debugging
            try:
                raw_response = self.raw_client.chat.completions.create(
                    model=self.config.name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                raw_output = raw_response.choices[0].message.content or "No content in response"
            except Exception as raw_e:
                raw_output = f"Failed to get raw output: {str(raw_e)}"
            
            # Return None extraction but include raw output and error
            return None, f"EXTRACTION_FAILED: {str(e)}\n\nRAW_OUTPUT:\n{raw_output}"
