"""
Prompts for LLM extraction.
"""
from typing import Optional, Tuple, List, Dict, Any

KEX_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor.
Extract ALL instances of aeronautical entities and their operational relationships.

[Specify Constraints]:
1. STRICT ID CROSS-REFERENCING: Every item must have a unique ID (E001, R001, EV001, RULE001, P001).
   Use these exact IDs when referencing entities inside relationships or rules.
2. ONTOLOGICAL SEPARATION:
   - "entities": Physical objects, roles, or abstract concepts. For formal definitions provided by the document, use the formal_definition field.
   - "relationships": Static or structural connections.
   - "events": Dynamic actions happening in time.
   - "procedures": Step-by-step ordered workflows.
   - "rules": Deontic constraints and obligations.
3. RULE EXTRACTION: Map modal verbs (shall, must, may) to correct modality.
   Separate trigger_condition from action_constraint.
4. FORMAL LOGIC: Generate pseudo-code for formal_if_then (e.g., "aircraft_visible == true").
5. DOMAIN FOCUS: Restrict to ATC operations, flight constraints, and safety procedures.
6. DYNAMIC DISCOVERY: Extract any domain-specific entity impacting flight operations.

FORMAL DEFINITION FIELD:
- Only populate entity.formal_definition when the document EXPLICITLY defines the term.
- Example: "Taxi clearance means an authorization to taxi..." → entity.text="Taxi clearance", formal_definition="an authorization to taxi..."
- If the term is just mentioned/used, leave formal_definition as null (NOT "N/A", NOT "None", NOT empty string).
- The formal_definition should be the EXACT or NEAR-EXACT text from the document.
- CRITICAL: NEVER use "N/A", "None", or "" for formal_definition. Use null/None (JSON null) instead.

CRITICAL ID RULES (MUST FOLLOW - Option 4):
1. NEVER duplicate an ID within the same extraction output. Each ID must be unique across ALL types.
2. Entity IDs MUST start with "E" (E001, E002, E003...). NEVER use D001 or RULE001 for entities.
3. Relationship IDs MUST start with "R" (R001, R002...).
4. Event IDs MUST start with "EV" (EV001, EV002...).
5. Rule IDs MUST start with "RULE" (RULE001, RULE002...). NEVER use E001 or P001 for rules.
6. Procedure IDs MUST start with "P" (P001, P002...). NEVER use RULE001 for procedures.
7. BEFORE assigning any ID, check if it already exists in your output. If it does, assign the NEXT sequential ID instead.
8. If Last Used IDs are provided, continue from those exact numbers (e.g., if entities=E003, the next entity must be E004).
9. VALIDATION: Double-check that no ID appears twice before finishing your response.

ENTITY CONTEXT RULE:
- Every entity MUST have a non-empty, meaningful context field describing its semantic meaning.
- Do NOT use empty strings, "N/A", "None", or "null" for context.
- Example: entity.text="taxiway", context="A defined path for aircraft to taxi on the ground"
- The context must provide enough information to understand the entity's role in ATC operations.
"""

KEX_USER_PROMPT_TEMPLATE = """Extract entities, relationships, events, rules, and procedures from the following text.

{context_section}

{last_ids_section}

Text to analyze:
{text}

Return valid JSON matching the AeronauticalExtraction schema.
"""

SENTENCE_SEGMENTATION_PROMPT = """
You are given a list of {total_sentences} sentences extracted from an aeronautical document. Each sentence has an index (starting from 0 up to {last_index}). Your task is to group these sentences into logical blocks that represent complete operational concepts (e.g., a complete rule, a procedure, a list, a warning, etc.).

**CRITICAL RULES (MUST FOLLOW):**
1. **FULL COVERAGE REQUIRED:** The first chunk MUST start at index 0. The last chunk MUST end at exactly {last_index}. NO EXCEPTIONS.
2. **NO GAPS:** Every single sentence from 0 to {last_index} must be included in exactly one chunk. If you leave any sentence out, the output is INVALID.
3. **CONTIGUOUS RANGES:** Each chunk is [start, end] where start <= end. No negative indices.
4. **NO OVERLAPS:** Chunks cannot share indices. If chunk 1 is [0, 4], chunk 2 must start at 5.
5. **ORDERED:** Chunks must be in increasing order by start index.

**VALIDATION CHECK:** After generating chunks, verify that:
- First chunk starts at 0
- Last chunk ends at {last_index} (the last sentence index)
- No gaps between chunks
- Total sentences covered: {total_sentences}

**Output format:** A JSON object with a "chunks" array containing objects with "indices": [start, end].

**Example (sentences=6, indices 0-5):**
Input:
{{
  0: "In all cases a conditional clearance shall be given in the following order and consist of:"}},
  1: "1. Identification;",
  2: "2. The condition",
  3: "3. The clearance; and",
  4: "4. Brief reiteration of the condition",
  5: "Conditional phrases shall not be used for movements affecting the active runway(s)."
}}

Output (covers 0-5 completely):
{{
  "chunks": [
    {{"indices": [0, 4]}},
    {{"indices": [5, 5]}}
  ]
}}

**REMINDER:** You have {total_sentences} sentences total (indices 0 to {last_index}). Your output MUST cover exactly this range.

Now process the following sentences. Return only valid JSON covering ALL sentences from first (0) to last ({last_index}).

"""


def build_kex_prompt(
    text: str,
    context_entities: Optional[list] = None,
    context_rules: Optional[list] = None,
    context_relationships: Optional[list] = None,
    include_rules: bool = True,
    include_relationships: bool = True,
    last_ids: Optional[dict] = None
) -> Tuple[str, str]:
    """Build the complete prompt for KEX extraction with context types."""
    
    # Build context section with types
    context_parts = []
    
    if context_entities:
        entity_lines = ["[Previously Extracted Entities]:"]
        for ent in context_entities:
            if isinstance(ent, dict):
                ent_id = ent.get("id", "")
                ent_text = ent.get("text", "")
                ent_label = ent.get("label", "Unknown")
                entity_lines.append(f"- ID: {ent_id}, Text: {ent_text}, Label: {ent_label}")
        context_parts.append("\n".join(entity_lines))
    
    if include_rules and context_rules:
        rule_lines = ["[Previously Extracted Rules]:"]
        for r in context_rules:
            if isinstance(r, dict):
                r_id = r.get("id", "")
                rule_type = r.get("rule_type", "")
                modality = r.get("modality", "")
                trigger = r.get("trigger", {})
                trigger_desc = trigger.get("description", "")[:80] if isinstance(trigger, dict) else ""
                rule_lines.append(f"- ID: {r_id}, Type: {rule_type}, Modality: {modality}, Trigger: {trigger_desc}...")
        context_parts.append("\n".join(rule_lines))
    
    if include_relationships and context_relationships:
        rel_lines = ["[Previously Extracted Relationships]:"]
        for rel in context_relationships:
            if isinstance(rel, dict):
                rel_id = rel.get("id", "")
                subj = rel.get("subject_text", "")
                pred = rel.get("predicate", "")
                obj = rel.get("object_text", "")
                rel_type = rel.get("relation_type", "")
                rel_lines.append(f"- ID: {rel_id}, {subj} {pred} {obj} (Type: {rel_type})")
        context_parts.append("\n".join(rel_lines))
    
    context_section = "\n\n".join(context_parts) if context_parts else ""
    
    # Build last IDs section
    last_ids_section = ""
    if last_ids:
        last_ids_section = "[Last Used IDs]:\nContinue the ID sequence from:\n"
        for category, last_id in last_ids.items():
            if last_id:
                last_ids_section += f"- {category}: {last_id}\n"
        last_ids_section += "\nUse sequential numbering continuing from these IDs."
    
    user_prompt = KEX_USER_PROMPT_TEMPLATE.format(
        context_section=context_section,
        last_ids_section=last_ids_section,
        text=text
    )
    
    return KEX_SYSTEM_PROMPT, user_prompt


# ==========================================
# PROMPTS PARA EXTRACCIÓN SECUENCIAL
# ==========================================

SEQUENTIAL_ENTITY_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor specializing in ENTITY extraction.
Extract ALL aeronautical entities from the provided text.

[Entity Definition]:
- Physical objects: runways, taxiways, aircraft, vehicles
- Roles and concepts: controllers, pilots, clearances, procedures
- Abstract concepts: flight phases, weather conditions, zones
- Locations: airports, gates, holds

[Extraction Rules]:
1. EVERY entity MUST have a unique ID starting with "E" (E001, E002, etc.)
2. IDs must be sequential. If Last Used IDs are provided, continue from there.
3. The 'text' field must contain the EXACT text as it appears in the document.
4. The 'context' field MUST be a meaningful semantic description (NOT empty, NOT "N/A").
5. Use 'formal_definition' ONLY when the document explicitly defines the term (e.g., "Term X means Y...").
6. Check [Previously Extracted Entities] to avoid duplicates - do NOT extract entities that already exist.

[Duplicates Prevention]:
- Before extracting any entity, check if it already exists in [Previously Extracted Entities].
- If an entity with the same text or similar meaning exists, do NOT create a duplicate.
- Reference existing entities by their IDs when needed by other extraction steps.
"""

SEQUENTIAL_RELATIONSHIP_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor specializing in RELATIONSHIP extraction.
Extract ALL structural and operational relationships between entities.

[Relationship Definition]:
- Structural: "is a", "is part of", "belongs to"
- Spatial: "is located at", "is adjacent to", "is north of"
- Procedural: "is used for", "leads to", "precedes"
- Operational: "controls", "monitors", "manages"
- Communication: "requests", "receives", "acknowledges"

[Extraction Rules]:
1. EVERY relationship MUST have a unique ID starting with "R" (R001, R002, etc.)
2. subject_id and object_id MUST reference entities from [Available Entities Context].
3. Use the exact entity IDs provided - do not invent new ones.
4. The predicate should be a clear verb connecting subject and object.
5. Include the entity text in subject_text and object_text for clarity.
6. Check [Previously Extracted Relationships] to avoid duplicates.

[Cross-Reference Validation]:
- Verify that subject_id and object_id exist in [Available Entities Context].
- If an entity is not in the context, you cannot create a relationship involving it.
"""

SEQUENTIAL_EVENT_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor specializing in EVENT extraction.
Extract ALL dynamic actions and events happening in time.

[Event Definition]:
- Actions: aircraft crossing runway, issuing clearance, vacating runway
- State changes: runway becoming active, aircraft entering hold
- Communications: pilot requesting, controller instructing
- Dynamic occurrences that happen at a specific point in time

[Extraction Rules]:
1. EVERY event MUST have a unique ID starting with "EV" (EV001, EV002, etc.)
2. actors and targets MUST be entity IDs from [Available Entities Context].
3. trigger_text must be the exact phrase indicating the event.
4. phase must be one of the defined FlightPhase values.
5. Include temporal context (when this occurs) if specified.
6. Check [Previously Extracted Events] to avoid duplicates.

[Cross-Reference Validation]:
- All entity references in actors[] and targets[] must exist in [Available Entities Context].
"""

SEQUENTIAL_RULE_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor specializing in RULE extraction.
Extract ALL deontic constraints, obligations, prohibitions, and permissions.

[Rule Definition]:
- Prohibitions: "shall not", "must not", "is prohibited"
- Obligations: "shall", "must", "is required"
- Permissions: "may", "is authorized"
- Recommendations: "should", "is recommended"
- Safety constraints and operational requirements

[Extraction Rules]:
1. EVERY rule MUST have a unique ID starting with "RULE" (RULE001, RULE002, etc.)
2. Map modal verbs correctly to modality: SHALL, SHALL_NOT, MUST, MUST_NOT, MAY, SHOULD, etc.
3. trigger.trigger_entities and constraint.action_entities MUST reference entity IDs from [Available Entities Context].
4. linked_relationships MUST reference relationship IDs from [Available Relationships Context].
5. formal_if_then must contain logical pseudo-code (e.g., "if aircraft_on_runway then clear_runway").
6. explainability must describe the safety rationale.
7. Check [Previously Extracted Rules] to avoid duplicates.

[Cross-Reference Validation]:
- All entity IDs in trigger.trigger_entities, constraint.action_entities, linked_entities must exist in [Available Entities Context].
- All relationship IDs in linked_relationships must exist in [Available Relationships Context].
"""

SEQUENTIAL_PROCEDURE_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor specializing in PROCEDURE extraction.
Extract ALL step-by-step operational workflows and procedures.

[Procedure Definition]:
- Ordered sequences of actions with mandatory_order
- Workflows with preconditions and exceptions
- Multi-step operations with required entities and events

[Extraction Rules]:
1. EVERY procedure MUST have a unique ID starting with "P" (P001, P002, etc.)
2. Steps must be numbered sequentially (1, 2, 3...) in step_no field.
3. Each step's required_entities[] must reference entity IDs from [Available Entities Context].
4. Each step's required_events[] must reference event IDs from [Available Events Context].
5. linked_rules[] must reference rule IDs from [Available Rules Context].
6. mandatory_order indicates if steps must be followed exactly in sequence.
7. Check [Previously Extracted Procedures] to avoid duplicates.

[Cross-Reference Validation]:
- All entity IDs in steps[].required_entities[] must exist in [Available Entities Context].
- All event IDs in steps[].required_events[] must exist in [Available Events Context].
- All rule IDs in linked_rules[] must exist in [Available Rules Context].
"""


def build_entity_prompt(
    text: str,
    previous_entities: Optional[List[Dict[str, Any]]] = None,
    last_ids: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """Build prompt for Step 1: Entity extraction."""
    context_parts = []
    
    if previous_entities:
        lines = ["[Previously Extracted Entities from Previous Pages]:",
                 "DO NOT extract these again. Reference them by their IDs if needed:"]
        for ent in previous_entities:
            if isinstance(ent, dict):
                lines.append(f"- {ent.get('id', '')}: {ent.get('text', '')} ({ent.get('label', '')})")
        context_parts.append("\n".join(lines))
    
    context_section = "\n\n".join(context_parts) if context_parts else "No previous entities."
    
    last_ids_section = ""
    if last_ids and last_ids.get("entities"):
        last_ids_section = f"\n\n[Last Used ID]:\nContinue from: {last_ids['entities']}\nNext entity should be: E{int(last_ids['entities'][1:]) + 1:03d}"
    
    user_prompt = f"""{context_section}{last_ids_section}

Text to analyze:
{text}

Extract ALL entities from this text. Remember:
- Use sequential IDs (E001, E002...)
- Do NOT duplicate entities from [Previously Extracted Entities]
- Every entity MUST have a meaningful context description"""
    
    return SEQUENTIAL_ENTITY_SYSTEM_PROMPT, user_prompt


def build_relationship_prompt(
    text: str,
    available_entities: List[Dict[str, Any]],
    previous_relationships: Optional[List[Dict[str, Any]]] = None,
    last_ids: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """Build prompt for Step 2: Relationship extraction."""
    # Available entities (from current chunk entities + previous pages)
    entity_lines = ["[Available Entities Context]:",
                    "MUST use these exact IDs for subject_id and object_id:"]
    for ent in available_entities:
        if isinstance(ent, dict):
            entity_lines.append(f"- {ent.get('id', '')}: {ent.get('text', '')}")
    
    context_parts = ["\n".join(entity_lines)]
    
    if previous_relationships:
        lines = ["\n[Previously Extracted Relationships from Previous Pages]:",
                 "DO NOT extract these again:"]
        for rel in previous_relationships:
            if isinstance(rel, dict):
                lines.append(f"- {rel.get('id', '')}: {rel.get('subject_text', '')} {rel.get('predicate', '')} {rel.get('object_text', '')}")
        context_parts.append("\n".join(lines))
    
    context_section = "\n".join(context_parts)
    
    last_ids_section = ""
    if last_ids and last_ids.get("relationships"):
        last_ids_section = f"\n\n[Last Used ID]:\nContinue from: {last_ids['relationships']}"
    
    user_prompt = f"""{context_section}{last_ids_section}

Text to analyze:
{text}

Extract ALL relationships between the available entities. Remember:
- subject_id and object_id MUST be from [Available Entities Context]
- Use sequential IDs (R001, R002...)
- Do NOT duplicate existing relationships"""
    
    return SEQUENTIAL_RELATIONSHIP_SYSTEM_PROMPT, user_prompt


def build_event_prompt(
    text: str,
    available_entities: List[Dict[str, Any]],
    previous_events: Optional[List[Dict[str, Any]]] = None,
    last_ids: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """Build prompt for Step 3: Event extraction."""
    entity_lines = ["[Available Entities Context]:",
                    "MUST use these exact IDs for actors and targets:"]
    for ent in available_entities:
        if isinstance(ent, dict):
            entity_lines.append(f"- {ent.get('id', '')}: {ent.get('text', '')}")
    
    context_parts = ["\n".join(entity_lines)]
    
    if previous_events:
        lines = ["\n[Previously Extracted Events from Previous Pages]:",
                 "DO NOT extract these again:"]
        for evt in previous_events:
            if isinstance(evt, dict):
                lines.append(f"- {evt.get('id', '')}: {evt.get('event_type', '')} - {evt.get('trigger_text', '')}")
        context_parts.append("\n".join(lines))
    
    context_section = "\n".join(context_parts)
    
    last_ids_section = ""
    if last_ids and last_ids.get("events"):
        last_ids_section = f"\n\n[Last Used ID]:\nContinue from: {last_ids['events']}"
    
    user_prompt = f"""{context_section}{last_ids_section}

Text to analyze:
{text}

Extract ALL events involving the available entities. Remember:
- actors[] and targets[] MUST be from [Available Entities Context]
- Use sequential IDs (EV001, EV002...)
- Do NOT duplicate existing events"""
    
    return SEQUENTIAL_EVENT_SYSTEM_PROMPT, user_prompt


def build_rule_prompt(
    text: str,
    available_entities: List[Dict[str, Any]],
    available_relationships: List[Dict[str, Any]],
    previous_rules: Optional[List[Dict[str, Any]]] = None,
    last_ids: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """Build prompt for Step 4: Rule extraction."""
    entity_lines = ["[Available Entities Context]:",
                    "MUST use these IDs for trigger.trigger_entities, constraint.action_entities, linked_entities:"]
    for ent in available_entities:
        if isinstance(ent, dict):
            entity_lines.append(f"- {ent.get('id', '')}: {ent.get('text', '')}")
    
    rel_lines = ["\n[Available Relationships Context]:",
                 "MUST use these IDs for linked_relationships:"]
    for rel in available_relationships:
        if isinstance(rel, dict):
            rel_lines.append(f"- {rel.get('id', '')}: {rel.get('subject_text', '')} {rel.get('predicate', '')} {rel.get('object_text', '')}")
    
    context_parts = ["\n".join(entity_lines), "\n".join(rel_lines)]
    
    if previous_rules:
        lines = ["\n[Previously Extracted Rules from Previous Pages]:",
                 "DO NOT extract these again:"]
        for rule in previous_rules:
            if isinstance(rule, dict):
                trigger = rule.get('trigger', {})
                trigger_desc = trigger.get('description', '')[:50] if isinstance(trigger, dict) else ''
                lines.append(f"- {rule.get('id', '')}: {rule.get('modality', '')} - {trigger_desc}...")
        context_parts.append("\n".join(lines))
    
    context_section = "\n".join(context_parts)
    
    last_ids_section = ""
    if last_ids and last_ids.get("rules"):
        last_ids_section = f"\n\n[Last Used ID]:\nContinue from: {last_ids['rules']}"
    
    user_prompt = f"""{context_section}{last_ids_section}

Text to analyze:
{text}

Extract ALL rules involving the available entities and relationships. Remember:
- Map modal verbs (shall, must, may) to correct modality
- trigger_entities and action_entities MUST be from [Available Entities Context]
- linked_relationships MUST be from [Available Relationships Context]
- Use sequential IDs (RULE001, RULE002...)
- Do NOT duplicate existing rules"""
    
    return SEQUENTIAL_RULE_SYSTEM_PROMPT, user_prompt


def build_procedure_prompt(
    text: str,
    available_entities: List[Dict[str, Any]],
    available_rules: List[Dict[str, Any]],
    available_events: List[Dict[str, Any]],
    previous_procedures: Optional[List[Dict[str, Any]]] = None,
    last_ids: Optional[Dict[str, str]] = None
) -> Tuple[str, str]:
    """Build prompt for Step 5: Procedure extraction."""
    entity_lines = ["[Available Entities Context]:",
                    "MUST use these IDs for steps[].required_entities[]:"]
    for ent in available_entities:
        if isinstance(ent, dict):
            entity_lines.append(f"- {ent.get('id', '')}: {ent.get('text', '')}")
    
    rule_lines = ["\n[Available Rules Context]:",
                  "MUST use these IDs for linked_rules[]:"]
    for rule in available_rules:
        if isinstance(rule, dict):
            trigger = rule.get('trigger', {})
            trigger_desc = trigger.get('description', '')[:40] if isinstance(trigger, dict) else ''
            rule_lines.append(f"- {rule.get('id', '')}: {rule.get('modality', '')} - {trigger_desc}...")
    
    event_lines = ["\n[Available Events Context]:",
                   "MUST use these IDs for steps[].required_events[]:"]
    for evt in available_events:
        if isinstance(evt, dict):
            event_lines.append(f"- {evt.get('id', '')}: {evt.get('event_type', '')}")
    
    context_parts = ["\n".join(entity_lines), "\n".join(rule_lines), "\n".join(event_lines)]
    
    if previous_procedures:
        lines = ["\n[Previously Extracted Procedures from Previous Pages]:",
                 "DO NOT extract these again:"]
        for proc in previous_procedures:
            if isinstance(proc, dict):
                lines.append(f"- {proc.get('id', '')}: {proc.get('name', '')}")
        context_parts.append("\n".join(lines))
    
    context_section = "\n".join(context_parts)
    
    last_ids_section = ""
    if last_ids and last_ids.get("procedures"):
        last_ids_section = f"\n\n[Last Used ID]:\nContinue from: {last_ids['procedures']}"
    
    user_prompt = f"""{context_section}{last_ids_section}

Text to analyze:
{text}

Extract ALL procedures involving the available entities, rules, and events. Remember:
- Steps must be numbered sequentially
- required_entities[] MUST be from [Available Entities Context]
- required_events[] MUST be from [Available Events Context]
- linked_rules[] MUST be from [Available Rules Context]
- Use sequential IDs (P001, P002...)
- Do NOT duplicate existing procedures"""
    
    return SEQUENTIAL_PROCEDURE_SYSTEM_PROMPT, user_prompt
