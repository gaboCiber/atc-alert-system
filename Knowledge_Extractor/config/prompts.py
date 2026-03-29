"""
Prompts for LLM extraction.
"""
from typing import Optional, Tuple, List, Dict, Any

KEX_SYSTEM_PROMPT = """You are an expert Air Traffic Control (ATC) knowledge extractor.
Extract ALL instances of aeronautical entities and their operational relationships.

[Specify Constraints]:
1. STRICT ID CROSS-REFERENCING: Every item must have a unique ID (E001, R001, EV001, RULE001, P001, D001).
   Use these exact IDs when referencing entities inside relationships or rules.
2. ONTOLOGICAL SEPARATION:
   - "entities": Physical objects, roles, or abstract concepts.
   - "relationships": Static or structural connections.
   - "events": Dynamic actions happening in time.
   - "procedures": Step-by-step ordered workflows.
   - "definitions": Glossary terms.
   - "rules": Deontic constraints and obligations.
3. RULE EXTRACTION: Map modal verbs (shall, must, may) to correct modality.
   Separate trigger_condition from action_constraint.
4. FORMAL LOGIC: Generate pseudo-code for formal_if_then (e.g., "aircraft_visible == true").
5. DOMAIN FOCUS: Restrict to ATC operations, flight constraints, and safety procedures.
6. DYNAMIC DISCOVERY: Extract any domain-specific entity impacting flight operations.
"""

KEX_USER_PROMPT_TEMPLATE = """Extract entities, relationships, events, rules, procedures, and definitions from the following text.

{context_section}

{last_ids_section}

Text to analyze:
{text}

Return valid JSON matching the AeronauticalExtraction schema.
"""

SENTENCE_SEGMENTATION_PROMPT = """
You are given a list of sentences extracted from an aeronautical document. Each sentence has an index (starting from 0). Your task is to group these sentences into logical blocks that represent complete operational concepts (e.g., a complete rule, a procedure, a list, a warning, etc.).

**Rules:**
1. Each logical block is a contiguous range of sentence indices [start, end] where start <= end.
2. The blocks must be in increasing order.
3. **CRITICAL:** The blocks must cover ALL sentences without gaps or overlaps. Every sentence index from 0 to N-1 must appear in exactly one block.

**Output format:** A JSON object with a "chunks" array containing objects with "indices": [start, end].

**Important:**
- Always ensure start <= end
- Indices must be integers
- Do not create overlapping ranges

**Example:**
Input:
{
  0: "In all cases a conditional clearance shall be given in the following order and consist of:"},
  1: "1. Identification;"},
  2: "2. The condition"},
  3: "3. The clearance; and"},
  4: "4. Brief reiteration of the condition"},
  5: "Conditional phrases, such as “behind landing aircraft” or “after departing aircraft”, shall not be used for movements affecting the active runway(s), except when the aircraft or vehicles concerned are seen by the appropriate controller and pilot."}
}

Output:
[
  [0, 4],
  [5, 5]
]

Now process the following sentences. Return only valid JSON. Ensure all sentences are covered.

"""


def build_kex_prompt(
    text: str,
    context_entities: Optional[list] = None,
    last_ids: Optional[dict] = None
) -> Tuple[str, str]:
    """Build the complete prompt for KEX extraction."""
    
    # Build context section
    context_section = ""
    if context_entities:
        context_section = "[Previously Extracted Entities]:\n"
        for ent in context_entities:
            if isinstance(ent, dict):
                ent_id = ent.get("id", "")
                ent_text = ent.get("text", "")
                ent_label = ent.get("label", "Unknown")
                context_section += f"- ID: {ent_id}, Text: {ent_text}, Label: {ent_label}\n"
    
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
