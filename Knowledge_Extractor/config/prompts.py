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
- If the term is just mentioned/used, leave formal_definition as null.
- The formal_definition should be the EXACT or NEAR-EXACT text from the document.

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
