DEDUP_PROMPTS = {
    "entities": """You are evaluating if two entities from aeronautical documentation represent the SAME concept.
Determine if each CANDIDATE is semantically equivalent to the REFERENCE entity.

REFERENCE entity:
  text: {text}
  label: {label}
  subtype: {subtype}
  context: {context}

CANDIDATES:
{candidates}

For each candidate, determine:
- similarity_score (0.0-1.0): How similar is this entity to the reference?
- is_duplicate (true/false): Does it represent the SAME real-world concept?
- explanation: Brief justification

Consider two entities the SAME concept if they refer to the same real-world entity,
even if the wording differs (e.g. "ATC" and "Air Traffic Control" are the same).
Consider them DIFFERENT if they are distinct concepts even if text is similar
(e.g. "Controller" service vs "Controller" person, or "takeoff" vs "departure").""",

    "relationships": """You are evaluating if two relationships from aeronautical documentation represent the SAME relation.
Determine if each CANDIDATE is semantically equivalent to the REFERENCE relationship.

REFERENCE relationship:
  subject: {subject_text}
  predicate: {predicate}
  object: {object_text}
  type: {relation_type}

CANDIDATES:
{candidates}

For each candidate, determine:
- similarity_score (0.0-1.0): How similar is this relationship to the reference?
- is_duplicate (true/false): Does it represent the SAME real-world relationship?
- explanation: Brief justification

Consider two relationships the SAME if they describe the same connection between
the same entities. Consider them DIFFERENT if the subject, predicate, or object
differ in meaning.""",

    "events": """You are evaluating if two events from aeronautical documentation represent the SAME occurrence.
Determine if each CANDIDATE is semantically equivalent to the REFERENCE event.

REFERENCE event:
  type: {event_type}
  trigger: {trigger_text}
  phase: {phase}

CANDIDATES:
{candidates}

For each candidate, determine:
- similarity_score (0.0-1.0): How similar is this event to the reference?
- is_duplicate (true/false): Does it represent the SAME real-world event type?
- explanation: Brief justification

Consider two events the SAME if they describe the same type of occurrence
(e.g. "level bust" and "level busts"). Consider them DIFFERENT if they are
distinct event types even with similar wording.""",

    "rules": """You are evaluating if two rules from aeronautical documentation represent the SAME regulation.
Focus on the logical meaning: trigger, constraint, and formal condition.
Determine if each CANDIDATE is semantically equivalent to the REFERENCE rule.

REFERENCE rule:
  type: {rule_type}
  modality: {modality}
  trigger: {trigger_desc}
  constraint: {constraint_desc}
  if: {if_condition}
  then: {then_action}

CANDIDATES:
{candidates}

For each candidate, determine:
- similarity_score (0.0-1.0): How similar is this rule to the reference?
- is_duplicate (true/false): Does it represent the SAME regulation?
- explanation: Brief justification

Consider two rules the SAME if they share the same trigger, requirement/prohibition,
and logical structure. Consider them DIFFERENT if they govern distinct situations.""",

    "procedures": """You are evaluating if two procedures from aeronautical documentation represent the SAME procedure.
Focus on the purpose and overall workflow.
Determine if each CANDIDATE is semantically equivalent to the REFERENCE procedure.

REFERENCE procedure:
  name: {name}
  purpose: {purpose}
  mandatory_order: {mandatory_order}

CANDIDATES:
{candidates}

For each candidate, determine:
- similarity_score (0.0-1.0): How similar is this procedure to the reference?
- is_duplicate (true/false): Does it represent the SAME procedure?
- explanation: Brief justification

Consider two procedures the SAME if they serve the same purpose with the same
overall workflow. Consider them DIFFERENT if they are distinct procedures.""",
}

FIELD_EXTRACTORS = {
    "entities": ["text", "label", "subtype", "context"],
    "relationships": ["subject_text", "predicate", "object_text", "relation_type"],
    "events": ["event_type", "trigger_text", "phase"],
    "rules": ["rule_type", "modality", "trigger", "constraint", "formal_if_then"],
    "procedures": ["name", "purpose", "mandatory_order"],
}
