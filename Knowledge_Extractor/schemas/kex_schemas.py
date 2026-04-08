from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator, field_validator

# ==========================================
# ENUMS (Para restringir las alucinaciones)
# ==========================================

class RelationType(str, Enum):
    STRUCTURAL = "structural"
    SPATIAL = "spatial"
    PROCEDURAL = "procedural"
    COMMUNICATION = "communication"
    OPERATIONAL = "operational"
    TAXONOMIC = "taxonomic"

class FlightPhase(str, Enum):
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"

class RuleType(str, Enum):
    PROHIBITION = "prohibition"
    OBLIGATION = "obligation"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"
    SEQUENCE = "sequence"
    EXCEPTION = "exception"
    DEFINITION = "definition"
    SAFETY_CONSTRAINT = "safety_constraint"

class Modality(str, Enum):
    SHALL = "shall"
    SHALL_NOT = "shall_not"
    MUST = "must"
    MUST_NOT = "must_not"
    MAY = "may"
    SHOULD = "should"
    SHOULD_NOT = "should_not"
    CONDITIONAL = "conditional"

class DeonticStrength(str, Enum):
    MANDATORY = "mandatory"
    STRONG = "strong"
    ADVISORY = "advisory"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ==========================================
# SUB-MODELOS
# ==========================================

class EntityAttributes(BaseModel):
    status: Optional[str] = Field(None, description="Current state of the entity (e.g., active, vacated)")
    phase: Optional[str] = Field(None, description="Flight phase associated with this entity")
    role: Optional[str] = Field(None, description="Function of the entity (e.g., actor, target, location)")
    safety_critical: bool = Field(False, description="True if misidentification poses a safety risk")

class RuleTriggerCondition(BaseModel):
    description: str = Field(..., description="Under what exact circumstances does this rule apply?")
    trigger_entities: List[str] = Field(..., description="List of Entity IDs involved in triggering the rule")
    trigger_relation: Optional[str] = Field(None, description="How the trigger entities interact")

class RuleActionConstraint(BaseModel):
    description: str = Field(..., description="What is specifically allowed, required, or forbidden?")
    action_verb: str = Field(..., description="The core action verb (e.g., use, cross, maintain)")
    negation: bool = Field(False, description="True if this is a prohibition")
    action_entities: List[str] = Field(..., description="List of Entity IDs affected by the action")

class RuleException(BaseModel):
    description: str = Field(..., description="Natural language description of when the rule is bypassed")
    condition: str = Field(..., description="The specific overriding condition")

class FormalIfThen(BaseModel):
    if_condition: str = Field(..., alias="if", description="Logical string representation of the trigger")
    then_action: str = Field(..., alias="then", description="Logical string representation of the outcome")
    except_when: Optional[str] = Field(None, description="Logical string representation of exceptions")

class RuleApplicability(BaseModel):
    phase: List[FlightPhase] = Field(default_factory=list, description="Applicable flight phases")
    environment: List[str] = Field(default_factory=list, description="Applicable environments (e.g., aerodrome)")
    actors: List[str] = Field(default_factory=list, description="Actors bound by this rule")
    scope: Optional[str] = Field(None, description="Geographic or operational scope")

class ProcedureStep(BaseModel):
    step_no: int = Field(..., description="Integer representing step order")
    description: str = Field(..., description="What happens in this step")
    action: str = Field(..., description="Core action of the step")
    required_entities: List[str] = Field(default_factory=list, description="Entity IDs involved in this step")
    required_events: List[str] = Field(default_factory=list, description="Event IDs triggered by this step")

# ==========================================
# MODELOS PRINCIPALES
# ==========================================

class Entity(BaseModel):
    id: str = Field(..., description="UNIQUE_ID (e.g., E001)")
    text: str = Field(..., description="Exact text extracted from the document")
    label: str = Field(..., description="Category of the entity (e.g., ATC_Procedure, Runway)")
    subtype: Optional[str] = Field(None, description="More specific classification (e.g., active)")
    aliases: List[str] = Field(default_factory=list, description="List of synonyms or alternative phrasings")
    context: str = Field(..., description="Brief semantic description - MUST NOT be empty or N/A")
    formal_definition: Optional[str] = Field(
        None,
        description="Formal definition provided by the document for this term (only when explicitly defined, e.g., 'Term X means Y...')"
    )

    @field_validator('context')
    @classmethod
    def validate_context_not_empty(cls, v: str) -> str:
        """Validate that entity context is not empty, N/A, or None."""
        if not v or v.strip() == "" or v.strip().upper() == "N/A" or v.strip().lower() == "none":
            raise ValueError("Entity context cannot be empty, N/A, or None. Provide a meaningful semantic description.")
        return v

class Relationship(BaseModel):
    id: str = Field(..., description="UNIQUE_ID (e.g., R001)")
    subject_id: str = Field(..., description="ID of the subject entity (must exist in entities array)")
    subject_text: str = Field(..., description="Exact text of the subject")
    predicate: str = Field(..., description="The verb or action connecting subject and object")
    object_id: str = Field(..., description="ID of the object entity (must exist in entities array)")
    object_text: str = Field(..., description="Exact text of the object")
    relation_type: RelationType
    context: Optional[str] = Field(None, description="Brief explanation of this relationship")

class Event(BaseModel):
    id: str = Field(..., description="UNIQUE_ID (e.g., EV001)")
    event_type: str = Field(..., description="Classification of the event")
    trigger_text: str = Field(..., description="Exact phrase that indicates this event")
    actors: List[str] = Field(default_factory=list, description="List of Entity IDs performing the event")
    targets: List[str] = Field(default_factory=list, description="List of Entity IDs affected by the event")
    phase: FlightPhase
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Specific values (e.g., 'runway': '09')")
    temporal_context: Optional[str] = Field(None, description="When this occurs")

class Rule(BaseModel):
    id: str = Field(..., description="UNIQUE_ID (e.g., RULE001)")
    rule_type: RuleType
    modality: Modality
    deontic_strength: DeonticStrength
    trigger: RuleTriggerCondition
    constraint: RuleActionConstraint
    preconditions: List[str] = Field(default_factory=list, description="Conditions required before the rule applies")
    exceptions: List[RuleException] = Field(default_factory=list)
    formal_if_then: FormalIfThen
    applicability: RuleApplicability
    severity: Severity
    safety_critical: bool = Field(True, description="True if violation risks life or aircraft integrity")
    explainability: str = Field(..., description="Why this rule exists (safety rationale)")
    linked_entities: List[str] = Field(default_factory=list, description="ALL Entity IDs involved")
    linked_relationships: List[str] = Field(default_factory=list, description="Relationship IDs involved")

class Procedure(BaseModel):
    id: str = Field(..., description="UNIQUE_ID (e.g., P001)")
    name: str = Field(..., description="Name of the operational procedure")
    purpose: str = Field(..., description="Why this procedure is executed")
    context: str = Field(..., description="Situational context for the procedure")
    mandatory_order: bool = Field(True, description="True if steps must be followed exactly in sequence")
    steps: List[ProcedureStep] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list, description="Conditions met before starting")
    exceptions: List[str] = Field(default_factory=list, description="Conditions where procedure is aborted")
    linked_rules: List[str] = Field(default_factory=list, description="Rule IDs governing this procedure")

# ==========================================
# ROOT MODEL (Lo que retorna el LLM)
# ==========================================

class AeronauticalExtraction(BaseModel):
    """
    Root extraction model for Air Traffic Control documentation.
    """
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    rules: List[Rule] = Field(default_factory=list)
    procedures: List[Procedure] = Field(default_factory=list)
    # Note: definitions removed - use entity.formal_definition for explicit document definitions

    @model_validator(mode='after')
    def validate_unique_ids(self):
        """Validar que IDs no se repitan dentro de la extracción (Opción 1)"""
        all_ids = []
        
        for entity in self.entities:
            all_ids.append(("entity", entity.id))
        for rel in self.relationships:
            all_ids.append(("relationship", rel.id))
        for event in self.events:
            all_ids.append(("event", event.id))
        for rule in self.rules:
            all_ids.append(("rule", rule.id))
        for proc in self.procedures:
            all_ids.append(("procedure", proc.id))
        
        # Detectar duplicados
        seen = {}
        for type_name, id_val in all_ids:
            if id_val in seen:
                raise ValueError(f"Duplicate ID {id_val}: found in {type_name} and {seen[id_val]}")
            seen[id_val] = type_name
        
        return self

    @model_validator(mode='after')
    def validate_id_prefixes(self):
        """Validar que IDs tengan el prefijo correcto según su tipo (Opción 2)"""
        for entity in self.entities:
            if not entity.id.startswith("E"):
                raise ValueError(f"Entity ID must start with E (e.g., E001): got {entity.id}")
        
        for rel in self.relationships:
            if not rel.id.startswith("R"):
                raise ValueError(f"Relationship ID must start with R (e.g., R001): got {rel.id}")
        
        for event in self.events:
            if not event.id.startswith("EV"):
                raise ValueError(f"Event ID must start with EV (e.g., EV001): got {event.id}")
        
        for rule in self.rules:
            if not rule.id.startswith("RULE"):
                raise ValueError(f"Rule ID must start with RULE (e.g., RULE001): got {rule.id}")
        
        for proc in self.procedures:
            if not proc.id.startswith("P"):
                raise ValueError(f"Procedure ID must start with P (e.g., P001): got {proc.id}")
        
        return self


# ==========================================
# ESQUEMAS INDIVIDUALES PARA EXTRACCIÓN SECUENCIAL
# ==========================================

class EntityExtraction(BaseModel):
    """Schema for sequential entity extraction (Step 1)."""
    entities: List[Entity] = Field(default_factory=list, description="Extract all entities from the text")

class RelationshipExtraction(BaseModel):
    """Schema for sequential relationship extraction (Step 2).
    Requires entities from Step 1 for subject_id/object_id references."""
    relationships: List[Relationship] = Field(default_factory=list, description="Extract relationships referencing entities from context")

class EventExtraction(BaseModel):
    """Schema for sequential event extraction (Step 3).
    Requires entities from Step 1 for actors/targets references."""
    events: List[Event] = Field(default_factory=list, description="Extract events referencing entities from context")

class RuleExtraction(BaseModel):
    """Schema for sequential rule extraction (Step 4).
    Requires entities from Step 1 for trigger/action references.
    Requires relationships from Step 2 for linked_relationships."""
    rules: List[Rule] = Field(default_factory=list, description="Extract rules referencing entities and relationships from context")

class ProcedureExtraction(BaseModel):
    """Schema for sequential procedure extraction (Step 5).
    Requires entities from Step 1 for required_entities.
    Requires rules from Step 4 for linked_rules.
    Requires events from Step 3 for required_events in steps."""
    procedures: List[Procedure] = Field(default_factory=list, description="Extract procedures referencing entities, rules, and events from context")