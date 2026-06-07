import sys
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.llm_client_factory import create_instructor_client, ModelConfig

from .config import JudgeConfig
from .dedup_prompts import DEDUP_PROMPTS, FIELD_EXTRACTORS

JUDGE_PROMPTS = {
    "entities": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two entities and determine their semantic equivalence.

GROUND TRUTH ENTITY:
{text}
Label: {label}
Subtype: {subtype}
Context: {context}
Aliases: {aliases}

MODEL ENTITY:
{text_m}
Label: {label_m}
Subtype: {subtype_m}
Context: {context_m}
Aliases: {aliases_m}

Rate their semantic similarity from 0.0 to 1.0, where:
- 1.0: Exactly the same concept, possibly with different wording
- 0.8-0.9: Very similar, minor differences in scope or detail
- 0.6-0.7: Same general concept but with notable differences
- 0.4-0.5: Partially related but not equivalent
- 0.2-0.3: Barely related
- 0.0: Completely different""",

    "relationships": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two relationships and determine their semantic equivalence.

GROUND TRUTH RELATIONSHIP:
Subject: {subject_text}
Predicate: {predicate}
Object: {object_text}
Type: {relation_type}
Context: {context}

MODEL RELATIONSHIP:
Subject: {subject_text_m}
Predicate: {predicate_m}
Object: {object_text_m}
Type: {relation_type_m}
Context: {context_m}

Rate their semantic similarity from 0.0 to 1.0, where:
- 1.0: Same relationship between same entities with same type
- 0.8-0.9: Same entities and relationship, minor wording differences
- 0.6-0.7: Same general relationship but with different entities or type
- 0.4-0.5: Partially related relationships
- 0.2-0.3: Barely related
- 0.0: Completely different""",

    "events": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two events and determine their semantic equivalence.

GROUND TRUTH EVENT:
Type: {event_type}
Trigger: {trigger_text}
Actors: {actors}
Targets: {targets}
Phase: {phase}
Temporal Context: {temporal_context}

MODEL EVENT:
Type: {event_type_m}
Trigger: {trigger_text_m}
Actors: {actors_m}
Targets: {targets_m}
Phase: {phase_m}
Temporal Context: {temporal_context_m}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: Same event with same actors, targets, and phase
- 0.8-0.9: Same event with minor differences in actors or phase
- 0.6-0.7: Same general event type but with different specifics
- 0.4-0.5: Related but not equivalent events
- 0.2-0.3: Barely related
- 0.0: Completely different""",

    "rules": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two rules and determine their semantic equivalence.
Focus on the logical meaning: what triggers the rule, what it requires/prohibits, and why it exists.

GROUND TRUTH RULE:
Type: {rule_type}
Modality: {modality}
Strength: {deontic_strength}
Trigger: {trigger_desc}
Constraint: {constraint_desc}
If: {if_condition}
Then: {then_action}
Severity: {severity}
Explainability: {explainability}

MODEL RULE:
Type: {rule_type_m}
Modality: {modality_m}
Strength: {deontic_strength_m}
Trigger: {trigger_desc_m}
Constraint: {constraint_desc_m}
If: {if_condition_m}
Then: {then_action_m}
Severity: {severity_m}
Explainability: {explainability_m}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: Same rule with same trigger, constraint, and rationale
- 0.8-0.9: Same rule with minor differences in wording or scope
- 0.6-0.7: Same general rule concept but with different specifics
- 0.4-0.5: Related but not equivalent rules
- 0.2-0.3: Barely related
- 0.0: Completely different""",

    "procedures": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two procedures and determine their semantic equivalence.
Focus on the purpose, steps, and overall workflow.

GROUND TRUTH PROCEDURE:
Name: {name}
Purpose: {purpose}
Context: {context}
Mandatory Order: {mandatory_order}
Steps: {steps}
Preconditions: {preconditions}

MODEL PROCEDURE:
Name: {name_m}
Purpose: {purpose_m}
Context: {context_m}
Mandatory Order: {mandatory_order_m}
Steps: {steps_m}
Preconditions: {preconditions_m}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: Same procedure with same purpose, steps, and order
- 0.8-0.9: Same procedure with minor differences in step descriptions
- 0.6-0.7: Same general procedure but with different steps or order
- 0.4-0.5: Related but not equivalent procedures
- 0.2-0.3: Barely related
- 0.0: Completely different""",
}


class Judgment(BaseModel):
    similarity_score: float = Field(
        ...,
        description="Semantic similarity score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    explanation: str = Field(..., description="Brief explanation of the score")
    matched_fields: list[str] = Field(
        default_factory=list,
        description="Fields that match semantically",
    )
    unmatched_fields: list[str] = Field(
        default_factory=list,
        description="Fields that differ significantly",
    )


class CandidateJudgment(BaseModel):
    candidate_id: str = Field(..., description="ID of the candidate item")
    similarity_score: float = Field(
        ...,
        description="Semantic similarity score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    is_duplicate: bool = Field(
        ...,
        description="True if this candidate represents the SAME real-world concept as the reference",
    )
    explanation: str = Field(..., description="Brief justification for the judgment")


class BatchJudgment(BaseModel):
    reference_id: str = Field(..., description="ID of the reference item")
    judgments: List[CandidateJudgment] = Field(
        ...,
        description="List of judgments for each candidate",
    )


class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        if config.enabled:
            model_cfg = ModelConfig(
                name=config.model_name,
                provider=config.provider,
                base_url=config.base_url,
                api_key=config.api_key,
                max_retries=config.max_retries,
                timeout=config.timeout,
            )
            self.client, self.mode = create_instructor_client(model_cfg)
        else:
            self.client = None

    def judge(self, kex_type: str, gt_item: dict, model_item: dict) -> Optional[Judgment]:
        if not self.config.enabled or self.client is None:
            return None

        prompt_template = JUDGE_PROMPTS.get(kex_type)
        if not prompt_template:
            return None

        prompt = self._fill_prompt(prompt_template, gt_item, model_item)

        try:
            result = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=Judgment,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of knowledge extraction quality."},
                    {"role": "user", "content": prompt},
                ],
            )
            return result
        except Exception as e:
            if self.config.skip_on_error:
                return None
            raise

    def _fill_prompt(self, template: str, gt: dict, model: dict) -> str:
        gt_params = self._extract_params(gt)
        model_params = {f"{k}_m": v for k, v in self._extract_params(model).items()}
        params = {**gt_params, **model_params}
        return template.format(**params)

    def _extract_params(self, item: dict) -> dict:
        params = {}
        for k, v in item.items():
            if isinstance(v, dict):
                for dk, dv in v.items():
                    prefixed_key = f"{k}_{dk}"
                    if isinstance(dv, list):
                        params[prefixed_key] = ", ".join(str(x) for x in dv)
                    else:
                        params[prefixed_key] = str(dv) if dv is not None else "N/A"
            elif isinstance(v, list):
                params[k] = ", ".join(str(x) for x in v)
            else:
                params[k] = str(v) if v is not None else "N/A"

        if "trigger_description" in params:
            params["trigger_desc"] = params["trigger_description"]
        if "constraint_description" in params:
            params["constraint_desc"] = params["constraint_description"]
        if "formal_if_then_if_condition" in params:
            params["if_condition"] = params["formal_if_then_if_condition"]
        if "formal_if_then_if" in params:
            params["if_condition"] = params["formal_if_then_if"]
        if "formal_if_then_then_action" in params:
            params["then_action"] = params["formal_if_then_then_action"]
        if "formal_if_then_then" in params:
            params["then_action"] = params["formal_if_then_then"]
        if "formal_if_then_except_when" in params:
            params["except_when"] = params["formal_if_then_except_when"]

        return params

    def batch_judge(
        self,
        kex_type: str,
        reference: dict,
        candidates: List[dict],
    ) -> Optional[BatchJudgment]:
        if not self.config.enabled or self.client is None:
            return None

        prompt_template = DEDUP_PROMPTS.get(kex_type)
        if not prompt_template:
            return None

        fields = FIELD_EXTRACTORS.get(kex_type, [])

        ref_params = self._extract_fields(reference, fields)
        ref_str = "\n  ".join(f"{k}: {v}" for k, v in ref_params.items())

        cand_lines = []
        for i, cand in enumerate(candidates):
            cand_params = self._extract_fields(cand, fields)
            parts = [f"  {cand.get('global_id', cand.get('id', f'candidate_{i}'))}:"]
            for k, v in cand_params.items():
                parts.append(f"    {k}: {v}")
            cand_lines.append("\n".join(parts))

        candidates_str = "\n\n".join(
            f"Candidate {i+1}:\n{c}" for i, c in enumerate(cand_lines)
        )

        prompt = prompt_template.format(
            **ref_params,
            candidates=candidates_str,
        )

        try:
            result = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=BatchJudgment,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of aeronautical knowledge extraction quality."},
                    {"role": "user", "content": prompt},
                ],
            )
            result.reference_id = ref_params.get("id", reference.get("id", ""))
            return result
        except Exception as e:
            if self.config.skip_on_error:
                return None
            raise

    def holistic_judge(
        self,
        kex_type: str,
        gt_items: List[dict],
        model_items: List[dict],
    ) -> Optional[Judgment]:
        """Judge semantic equivalence of two sets of items (GT vs model) for a given type"""
        if not self.config.enabled or self.client is None:
            return None

        # Use holistic prompts for set comparison
        holistic_prompts = {
            "entities": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of entities and determine their semantic equivalence.

GROUND TRUTH ENTITIES:
{gt_entities}

MODEL ENTITIES:
{model_entities}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (cover the same real-world concepts, possibly with different granularity)
- 0.8-0.9: Very similar, minor differences in detail or granularity but same core concepts
- 0.6-0.7: Same general concepts but with notable differences in coverage or specificity
- 0.4-0.5: Partially related but with significant differences in what concepts are covered
- 0.2-0.3: Barely related, few overlapping concepts
- 0.0: Completely different, no shared semantic meaning

Focus on whether the model captured the same knowledge as the ground truth, allowing for:
- Different granularity (one GT entity may correspond to multiple model entities and vice versa)
- Different wording that preserves the same meaning
- Different organization as long as the semantic content is equivalent""",

            "relationships": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of relationships and determine their semantic equivalence.

GROUND TRUTH RELATIONSHIPS:
{gt_relationships}

MODEL RELATIONSHIPS:
{model_relationships}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (represent the same connections between concepts)
- 0.8-0.9: Very similar, minor differences in wording but same core relationships
- 0.6-0.7: Same general relationships but with notable differences in specificity or coverage
- 0.4-0.5: Partially related but with significant differences in what connections are represented
- 0.2-0.3: Barely related, few overlapping relationships
- 0.0: Completely different, no shared semantic meaning

Focus on whether the model captured the same knowledge as the ground truth, allowing for:
- Different granularity (one GT relationship may correspond to multiple model relationships and vice versa)
- Different wording that preserves the same meaning
- Different organization as long as the semantic content is equivalent""",

            "events": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of events and determine their semantic equivalence.

GROUND TRUTH EVENTS:
{gt_events}

MODEL EVENTS:
{model_events}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (represent the same occurrences or actions)
- 0.8-0.9: Very similar, minor differences in detail but same core events
- 0.6-0.7: Same general event types but with notable differences in specifics or coverage
- 0.4-0.5: Partially related but with significant differences in what events are represented
- 0.2-0.3: Barely related, few overlapping events
- 0.0: Completely different, no shared semantic meaning

Focus on whether the model captured the same knowledge as the ground truth, allowing for:
- Different granularity (one GT event may correspond to multiple model events and vice versa)
- Different wording that preserves the same meaning
- Different organization as long as the semantic content is equivalent""",

            "rules": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of rules and determine their semantic equivalence.

GROUND TRUTH RULES:
{gt_rules}

MODEL RULES:
{model_rules}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (represent the same obligations, prohibitions, and conditions)
- 0.8-0.9: Very similar, minor differences in wording but same core rules
- 0.6-0.7: Same general rule concepts but with notable differences in specificity or coverage
- 0.4-0.5: Partially related but with significant differences in what rules are represented
- 0.2-0.3: Barely related, few overlapping rules
- 0.0: Completely different, no shared semantic meaning

Focus on whether the model captured the same knowledge as the ground truth, allowing for:
- Different granularity (one GT rule may correspond to multiple model rules and vice versa)
- Different wording that preserves the same meaning (e.g., "must" vs "shall")
- Different organization as long as the semantic content is equivalent""",

            "procedures": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of procedures and determine their semantic equivalence.

GROUND TRUTH PROCEDURES:
{gt_procedures}

MODEL PROCEDURES:
{model_procedures}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (represent the same sequences of steps for accomplishing tasks)
- 0.8-0.9: Very similar, minor differences in wording but same core procedures
- 0.6-0.7: Same general procedures but with notable differences in specificity or coverage
- 0.4-0.5: Partially related but with significant differences in what procedures are represented
- 0.2-0.3: Barely related, few overlapping procedures
- 0.0: Completely different, no shared semantic meaning

Focus on whether the model captured the same knowledge as the ground truth, allowing for:
- Different granularity (one GT procedure may correspond to multiple model procedures and vice versa)
- Different wording that preserves the same meaning
- Different organization as long as the semantic content is equivalent""",
        }

        prompt_template = holistic_prompts.get(kex_type)
        if not prompt_template:
            return None

        # Extract semantic fields for each item (only the fields that matter for semantic equivalence)
        gt_formatted = self._format_items_for_holistic_judge(gt_items, kex_type)
        model_formatted = self._format_items_for_holistic_judge(model_items, kex_type)

        prompt = prompt_template.format(
            gt_entities=gt_formatted if kex_type == "entities" else "",
            gt_relationships=gt_formatted if kex_type == "relationships" else "",
            gt_events=gt_formatted if kex_type == "events" else "",
            gt_rules=gt_formatted if kex_type == "rules" else "",
            gt_procedures=gt_formatted if kex_type == "procedures" else "",
            model_entities=model_formatted if kex_type == "entities" else "",
            model_relationships=model_formatted if kex_type == "relationships" else "",
            model_events=model_formatted if kex_type == "events" else "",
            model_rules=model_formatted if kex_type == "rules" else "",
            model_procedures=model_formatted if kex_type == "procedures" else "",
        )

        try:
            result = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=Judgment,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of aeronautical knowledge extraction quality."},
                    {"role": "user", "content": prompt},
                ],
            )
            return result
        except Exception as e:
            if self.config.skip_on_error:
                return None
            raise

    def _format_items_for_holistic_judge(self, items: List[dict], kex_type: str) -> str:
        """Format a list of items for holistic judgment, showing only semantic fields"""
        if not items:
            return "  (none)"
        
        # Get the semantic fields for this type (from explain.md §11.5)
        semantic_fields = {
            "entities": ["text", "label", "subtype", "context", "aliases", "formal_definition"],
            "relationships": ["subject_text", "predicate", "object_text", "relation_type"],
            "events": ["event_type", "trigger_text", "actors", "targets", "phase"],
            "rules": ["rule_type", "modality", "deontic_strength", "trigger.description", "constraint.description", "formal_if_then"],
            "procedures": ["name", "purpose", "steps", "preconditions"],
        }
        
        fields = semantic_fields.get(kex_type, [])
        formatted_items = []
        
        for i, item in enumerate(items):
            item_lines = [f"  Item {i+1}:"]
            
            for field in fields:
                # Handle nested fields like trigger.description
                if "." in field:
                    parts = field.split(".")
                    value = item
                    try:
                        for part in parts:
                            if isinstance(value, dict):
                                value = value.get(part)
                            else:
                                value = "N/A"
                                break
                    except:
                        value = "N/A"
                else:
                    value = item.get(field, "N/A")
                
                # Format the value for display
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value) if value else "[]"
                elif value is None:
                    value_str = "N/A"
                else:
                    value_str = str(value)
                
                item_lines.append(f"    {field}: {value_str}")
            
            formatted_items.append("\n".join(item_lines))
        
        return "\n\n".join(formatted_items)

    def _extract_fields(self, item: dict, fields: List[str]) -> dict:
        result = {}
        for f in fields:
            if f == "trigger":
                trigger = item.get("trigger", {})
                if isinstance(trigger, dict):
                    result["trigger_desc"] = str(trigger.get("description", "N/A"))
                else:
                    result["trigger_desc"] = str(trigger)
            elif f == "constraint":
                constraint = item.get("constraint", {})
                if isinstance(constraint, dict):
                    result["constraint_desc"] = str(constraint.get("description", "N/A"))
                else:
                    result["constraint_desc"] = str(constraint)
            elif f == "formal_if_then":
                fit = item.get("formal_if_then", {})
                if isinstance(fit, dict):
                    result["if_condition"] = str(fit.get("if_condition", fit.get("if", "N/A")))
                    result["then_action"] = str(fit.get("then_action", fit.get("then", "N/A")))
                else:
                    result["if_condition"] = "N/A"
                    result["then_action"] = "N/A"
            else:
                v = item.get(f)
                if isinstance(v, list):
                    result[f] = ", ".join(str(x) for x in v)
                elif v is None:
                    result[f] = "N/A"
                else:
                    result[f] = str(v)
        result["id"] = item.get("global_id", item.get("id", "unknown"))
        return result
