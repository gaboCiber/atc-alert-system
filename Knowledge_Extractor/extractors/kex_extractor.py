"""
KEX Extractor using Instructor for structured extraction.
"""
import json
import instructor
from openai import OpenAI
from typing import Optional, List, Dict, Any, Tuple

from ..schemas.kex_schemas import (
    AeronauticalExtraction,
    EntityExtraction,
    RelationshipExtraction,
    EventExtraction,
    RuleExtraction,
    ProcedureExtraction,
)
from ..config.settings import ModelConfig
from ..config.prompts import (
    build_kex_prompt,
    build_entity_prompt,
    build_relationship_prompt,
    build_event_prompt,
    build_rule_prompt,
    build_procedure_prompt,
)
from .llm_client_factory import create_instructor_client, create_raw_client

class KEXExtractor:
    """Extract aeronautical knowledge using structured LLM calls."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize KEX extractor.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        self.config = config or ModelConfig()
        
        # Initialize appropriate instructor client based on provider
        self.client, self.mode = create_instructor_client(self.config)
        
        # Raw client for fallback extraction
        self.raw_client = create_raw_client(self.config)

        # Registramos el hook globalmente para este cliente
        # Cada vez que falle un parseo, se ejecutará esta función
        self.client.on("parse:error", self._log_validation_error)
        self.reply_number = 1

    def _log_validation_error(self, error: Exception):
        """Callback que se ejecuta inmediatamente al fallar una validación."""
        print(f"\n     └─ ⚠️ [Validación {self.reply_number} Fallida]. Instructor intentará corregir esto automáticamente...")
        self.reply_number += 1
    
    def extract(
        self,
        text: str,
        context_entities: Optional[List[Dict[str, Any]]] = None,
        context_rules: Optional[List[Dict[str, Any]]] = None,
        context_relationships: Optional[List[Dict[str, Any]]] = None,
        include_rules: bool = True,
        include_relationships: bool = True,
        last_ids: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[AeronauticalExtraction], str]:
        """
        Extract knowledge from text.
        
        Args:
            text: Text to analyze.
            context_entities: Previously extracted entities for context.
            context_rules: Previously extracted rules for context.
            context_relationships: Previously extracted relationships for context.
            include_rules: Whether to include rules in prompt.
            include_relationships: Whether to include relationships in prompt.
            last_ids: Last used IDs for sequential numbering.
            
        Returns:
            Tuple of (AeronauticalExtraction or None, raw_llm_output).
            Always returns raw output even on failure for debugging.
        """
        # Build prompts with context types
        system_prompt, user_prompt = build_kex_prompt(
            text=text,
            context_entities=context_entities,
            context_rules=context_rules,
            context_relationships=context_relationships,
            include_rules=include_rules,
            include_relationships=include_relationships,
            last_ids=last_ids
        )

        self.reply_number = 1
        
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
                # Check if raw fallback is available
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    raw_output = raw_response.choices[0].message.content or "No content in response"
                else:
                    raw_output = "Raw fallback not available for this provider"
            except Exception as raw_e:
                raw_output = f"Failed to get raw output: {str(raw_e)}"
            
            # Return None extraction but include raw output and error
            return None, f"EXTRACTION_FAILED: {str(e)}\n\nRAW_OUTPUT:\n{raw_output}"

    def extract_sequential(
        self,
        text: str,
        accumulated_entities: List[Dict[str, Any]],
        accumulated_relationships: List[Dict[str, Any]],
        accumulated_events: List[Dict[str, Any]],
        accumulated_rules: List[Dict[str, Any]],
        accumulated_procedures: List[Dict[str, Any]],
        last_ids: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[AeronauticalExtraction], str]:
        """
        Extract knowledge sequentially in 5 steps:
        1. Entities (no dependencies)
        2. Relationships (needs entities)
        3. Events (needs entities)
        4. Rules (needs entities + relationships)
        5. Procedures (needs entities + rules + events)
        
        Args:
            text: Text to analyze
            accumulated_*: Previously extracted items from previous pages (for deduplication)
            last_ids: Last used IDs for sequential numbering
            
        Returns:
            Tuple of (AeronauticalExtraction or None, combined raw output)
        """
        all_raw_outputs = []
        current_last_ids = dict(last_ids) if last_ids else {}

        # Step 1: Extract Entities (no dependencies)
        entities, entities_raw, success = self._run_extraction_step(
            step_num=1, total_steps=5, step_name="entities",
            extract_fn=lambda: self._extract_entities_step(text, accumulated_entities, current_last_ids),
            id_key="entities", id_default="E000", current_last_ids=current_last_ids,
            all_raw_outputs=all_raw_outputs
        )
        if not success:
            return None, "\n\n".join(all_raw_outputs)
        available_entities = accumulated_entities + entities

        # Step 2: Extract Relationships (needs entities)
        relationships, rels_raw, success = self._run_extraction_step(
            step_num=2, total_steps=5, step_name="relationships",
            extract_fn=lambda: self._extract_relationships_step(
                text, available_entities, accumulated_relationships, current_last_ids
            ),
            id_key="relationships", id_default="R000", current_last_ids=current_last_ids,
            all_raw_outputs=all_raw_outputs
        )
        if not success:
            return None, "\n\n".join(all_raw_outputs)
        available_relationships = accumulated_relationships + relationships

        # Step 3: Extract Events (needs entities)
        events, events_raw, success = self._run_extraction_step(
            step_num=3, total_steps=5, step_name="events",
            extract_fn=lambda: self._extract_events_step(
                text, available_entities, accumulated_events, current_last_ids
            ),
            id_key="events", id_default="EV000", current_last_ids=current_last_ids,
            all_raw_outputs=all_raw_outputs
        )
        if not success:
            return None, "\n\n".join(all_raw_outputs)
        available_events = accumulated_events + events

        # Step 4: Extract Rules (needs entities + relationships)
        rules, rules_raw, success = self._run_extraction_step(
            step_num=4, total_steps=5, step_name="rules",
            extract_fn=lambda: self._extract_rules_step(
                text, available_entities, available_relationships, accumulated_rules, current_last_ids
            ),
            id_key="rules", id_default="RULE000", current_last_ids=current_last_ids,
            all_raw_outputs=all_raw_outputs
        )
        if not success:
            return None, "\n\n".join(all_raw_outputs)
        available_rules = accumulated_rules + rules

        # Step 5: Extract Procedures (needs entities + rules + events)
        procedures, procs_raw, success = self._run_extraction_step(
            step_num=5, total_steps=5, step_name="procedures",
            extract_fn=lambda: self._extract_procedures_step(
                text, available_entities, available_rules, available_events, accumulated_procedures, current_last_ids
            ),
            id_key="procedures", id_default="P000", current_last_ids=current_last_ids,
            all_raw_outputs=all_raw_outputs
        )
        if not success:
            return None, "\n\n".join(all_raw_outputs)

        # Combine all results into AeronauticalExtraction
        extraction = AeronauticalExtraction(
            entities=entities,
            relationships=relationships,
            events=events,
            rules=rules,
            procedures=procedures
        )

        combined_raw = "\n\n".join(all_raw_outputs)
        return extraction, combined_raw

    def _run_extraction_step(
        self,
        step_num: int,
        total_steps: int,
        step_name: str,
        extract_fn,
        id_key: str,
        id_default: str,
        current_last_ids: Dict[str, str],
        all_raw_outputs: List[str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """
        Run a single extraction step with logging and error handling.

        Args:
            step_num: Current step number (1-based)
            total_steps: Total number of steps
            step_name: Name of the step for logging (e.g., "entities")
            extract_fn: Function that performs the extraction, returns (items, raw_output, success)
            id_key: Key for updating last_ids (e.g., "entities")
            id_default: Default ID prefix if no items (e.g., "E000")
            current_last_ids: Dict to update with last ID
            all_raw_outputs: List to append raw output to

        Returns:
            Tuple of (items, raw_output, success)
        """
        print(f"\n     └─ [Sequential] Step {step_num}/{total_steps}: Extracting {step_name}...", end=" ")
        items, raw_output, success = extract_fn()

        separator = "\n" if all_raw_outputs else ""
        all_raw_outputs.append(f"{separator}=== STEP {step_num}: {step_name.upper()} ===\n{raw_output}")

        if not success:
            print(f"\n     └─ ✗ Failed")
            return items, raw_output, False

        print(f"\n     └─ ✓ ({len(items)} {step_name})")

        # Update last_ids
        if items:
            last_item = items[-1]
            current_last_ids[id_key] = last_item.get("id", current_last_ids.get(id_key, id_default))

        return items, raw_output, True

    def _extract_entities_step(
        self,
        text: str,
        previous_entities: List[Dict[str, Any]],
        last_ids: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Step 1: Extract entities only."""
        system_prompt, user_prompt = build_entity_prompt(
            text=text,
            previous_entities=previous_entities,
            last_ids=last_ids
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=EntityExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.model_dump_json(indent=2)
            entities = [e.model_dump() for e in response.entities]
            return entities, raw_output, True
            
        except Exception as e:
            error_output = f"EXTRACTION_FAILED: {str(e)}"
            try:
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    error_output += f"\n\nRAW_OUTPUT:\n{raw_response.choices[0].message.content}"
            except:
                pass
            return [], error_output, False

    def _extract_relationships_step(
        self,
        text: str,
        available_entities: List[Dict[str, Any]],
        previous_relationships: List[Dict[str, Any]],
        last_ids: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Step 2: Extract relationships using available entities."""
        system_prompt, user_prompt = build_relationship_prompt(
            text=text,
            available_entities=available_entities,
            previous_relationships=previous_relationships,
            last_ids=last_ids
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=RelationshipExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.model_dump_json(indent=2)
            relationships = [r.model_dump() for r in response.relationships]
            return relationships, raw_output, True
            
        except Exception as e:
            error_output = f"EXTRACTION_FAILED: {str(e)}"
            try:
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    error_output += f"\n\nRAW_OUTPUT:\n{raw_response.choices[0].message.content}"
            except:
                pass
            return [], error_output, False

    def _extract_events_step(
        self,
        text: str,
        available_entities: List[Dict[str, Any]],
        previous_events: List[Dict[str, Any]],
        last_ids: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Step 3: Extract events using available entities."""
        system_prompt, user_prompt = build_event_prompt(
            text=text,
            available_entities=available_entities,
            previous_events=previous_events,
            last_ids=last_ids
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=EventExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.model_dump_json(indent=2)
            events = [e.model_dump() for e in response.events]
            return events, raw_output, True
            
        except Exception as e:
            error_output = f"EXTRACTION_FAILED: {str(e)}"
            try:
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    error_output += f"\n\nRAW_OUTPUT:\n{raw_response.choices[0].message.content}"
            except:
                pass
            return [], error_output, False

    def _extract_rules_step(
        self,
        text: str,
        available_entities: List[Dict[str, Any]],
        available_relationships: List[Dict[str, Any]],
        previous_rules: List[Dict[str, Any]],
        last_ids: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Step 4: Extract rules using available entities and relationships."""
        system_prompt, user_prompt = build_rule_prompt(
            text=text,
            available_entities=available_entities,
            available_relationships=available_relationships,
            previous_rules=previous_rules,
            last_ids=last_ids
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=RuleExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.model_dump_json(indent=2)
            rules = [r.model_dump(by_alias=True) for r in response.rules]
            return rules, raw_output, True
            
        except Exception as e:
            error_output = f"EXTRACTION_FAILED: {str(e)}"
            try:
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    error_output += f"\n\nRAW_OUTPUT:\n{raw_response.choices[0].message.content}"
            except:
                pass
            return [], error_output, False

    def _extract_procedures_step(
        self,
        text: str,
        available_entities: List[Dict[str, Any]],
        available_rules: List[Dict[str, Any]],
        available_events: List[Dict[str, Any]],
        previous_procedures: List[Dict[str, Any]],
        last_ids: Dict[str, str]
    ) -> Tuple[List[Dict[str, Any]], str, bool]:
        """Step 5: Extract procedures using available entities, rules, and events."""
        system_prompt, user_prompt = build_procedure_prompt(
            text=text,
            available_entities=available_entities,
            available_rules=available_rules,
            available_events=available_events,
            previous_procedures=previous_procedures,
            last_ids=last_ids
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                response_model=ProcedureExtraction,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.model_dump_json(indent=2)
            procedures = [p.model_dump() for p in response.procedures]
            return procedures, raw_output, True
            
        except Exception as e:
            error_output = f"EXTRACTION_FAILED: {str(e)}"
            try:
                if self.raw_client:
                    raw_response = self.raw_client.chat.completions.create(
                        model=self.config.name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    error_output += f"\n\nRAW_OUTPUT:\n{raw_response.choices[0].message.content}"
            except:
                pass
            return [], error_output, False
