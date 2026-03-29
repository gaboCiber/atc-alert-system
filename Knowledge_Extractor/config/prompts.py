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

**CRITICAL RULES (MUST FOLLOW):**
1. **FULL COVERAGE REQUIRED:** The first chunk MUST start at index 0. The last chunk MUST end at the last sentence index (N-1). NO EXCEPTIONS.
2. **NO GAPS:** Every single sentence from 0 to N-1 must be included in exactly one chunk. If you leave any sentence out, the output is INVALID.
3. **CONTIGUOUS RANGES:** Each chunk is [start, end] where start <= end. No negative indices.
4. **NO OVERLAPS:** Chunks cannot share indices. If chunk 1 is [0, 4], chunk 2 must start at 5.
5. **ORDERED:** Chunks must be in increasing order by start index.

**VALIDATION CHECK:** After generating chunks, verify that:
- First chunk starts at 0
- Last chunk ends at N-1 (where N = total number of sentences)
- No gaps between chunks

**Output format:** A JSON object with a "chunks" array containing objects with "indices": [start, end].

**Example:**
Input (6 sentences, indices 0-5):
{
  0: "In all cases a conditional clearance shall be given in the following order and consist of:"},
  1: "1. Identification;"},
  2: "2. The condition"},
  3: "3. The clearance; and"},
  4: "4. Brief reiteration of the condition"},
  5: "Conditional phrases shall not be used for movements affecting the active runway(s)."}
}

Output (covers 0-5 completely):
{
  "chunks": [
    {"indices": [0, 4]},
    {"indices": [5, 5]}
  ]
}

**REMINDER:** If there are N sentences (indices 0 to N-1), your output MUST cover exactly [0, N-1].

Now process the following sentences. Return only valid JSON covering ALL sentences from first to last.

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
