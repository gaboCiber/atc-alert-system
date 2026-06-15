import sys
from pathlib import Path
import re
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

    # ------------------------------------------------------------------
    # Prompts for intrinsic validity (used when GT is empty but model has items)
    # ------------------------------------------------------------------
    INTRINSIC_VALIDITY_PROMPTS = {
        "entities": """The Ground Truth has NO entities for this text chunk, but the Model extracted the following entities. Evaluate their **intrinsic validity** as aeronautical knowledge.

MODEL ENTITIES:
{model_entities}

Rate the intrinsic validity of the Model's extraction from 0.0 to 1.0, where:
- 1.0: Excellent extraction. The entities represent real, useful aeronautical concepts that would be valuable in an ATC knowledge graph. They have clear labels, subtypes, and meaningful context.
- 0.8-0.9: Mostly valid. The entities are correct aeronautical concepts, but some have vague labels or incomplete context.
- 0.6-0.7: Partially valid. Some entities are genuine ATC concepts, but others are generic, ambiguous, or redundant.
- 0.4-0.5: Weak validity. The entities are loosely related to aviation but lack the precision needed for ATC operations.
- 0.2-0.3: Poor. Most entities are not genuine aeronautical concepts or are too vague to be useful.
- 0.0: Invalid. The entities are not valid aeronautical concepts and would add noise to a knowledge graph.

Consider:
- Are these real ATC/aeronautical concepts? (aircraft, runways, frequencies, clearances, phraseology, etc.)
- Do they have meaningful labels and subtypes?
- Is the context informative and accurate?
- Would they be useful in a knowledge graph supporting ATC operations?

IMPORTANT: The score represents VALIDITY, NOT similarity to GT (since GT is empty).""",

        "relationships": """The Ground Truth has NO relationships for this text chunk, but the Model extracted the following relationships. Evaluate their **intrinsic validity** as aeronautical knowledge.

MODEL RELATIONSHIPS:
{model_relationships}

Rate the intrinsic validity of the Model's extraction from 0.0 to 1.0, where:
- 1.0: Excellent. The relationships represent real, useful connections between aeronautical concepts that accurately reflect ATC operations.
- 0.8-0.9: Mostly valid. Correct connections but some are slightly vague or imprecise.
- 0.6-0.7: Partially valid. Some connections are genuine but others are inferred or speculative.
- 0.4-0.5: Weak. The relationships are tenuous or use incorrect predicates/relation types.
- 0.2-0.3: Poor. Most relationships don't reflect actual ATC operational connections.
- 0.0: Invalid. These are not real aeronautical relationships.

IMPORTANT: The score represents VALIDITY, NOT similarity to GT (since GT is empty).""",

        "events": """The Ground Truth has NO events for this text chunk, but the Model extracted the following events. Evaluate their **intrinsic validity** as aeronautical knowledge.

MODEL EVENTS:
{model_events}

Rate the intrinsic validity of the Model's extraction from 0.0 to 1.0, where:
- 1.0: Excellent. The events represent real, specific ATC occurrences (instructions, readbacks, clearances, state changes) with clear actors and actions.
- 0.8-0.9: Mostly valid. Actual ATC events but some have vague actors or imprecise types.
- 0.6-0.7: Partially valid. Some are genuine events, others are general statements misclassified as events.
- 0.4-0.5: Weak. The events are more like general rules/statements than specific occurrences.
- 0.2-0.3: Poor. Mostly misclassified general knowledge, not actual events.
- 0.0: Invalid. These are not real ATC events.

Consider:
- Are these specific occurrences/actions rather than general rules or facts?
- Do they have clear actors (who) and actions (what)?
- Would they be useful for tracking ATC operations?

IMPORTANT: The score represents VALIDITY, NOT similarity to GT (since GT is empty).""",

        "rules": """The Ground Truth has NO rules for this text chunk, but the Model extracted the following rules. Evaluate their **intrinsic validity** as compilable aeronautical knowledge.

MODEL RULES:
{model_rules}

Rate the intrinsic validity of the Model's extraction from 0.0 to 1.0, where:
- 1.0: Highly valid. These rules have clear triggers, precise constraints, and a logical structure that maps directly to TrafficState fields (altitude, heading, speed, runway, separation, flight phase). They could be compiled into executable Python code or evaluated at runtime by an LLM to check ATCO instructions in real time.
- 0.8-0.9: Valid. Well-defined rules with minor vagueness in triggers or constraints, but still potentially compilable or runtime-evaluable.
- 0.6-0.7: Partially valid. Some rules are genuine operational requirements but others are descriptive statements rather than prescriptive rules, or are too vague to compile/evaluate.
- 0.4-0.5: Weak validity. The rules are loosely related to ATC safety but lack the precision needed for automated evaluation (missing triggers, vague constraints, subjective conditions).
- 0.2-0.3: Poor. Most extractions are not actual rules (they are facts, definitions, or general statements misclassified as rules).
- 0.0: Invalid. These are not aeronautical rules and would be useless for a rule engine.

Consider:
- Does the rule have a clear TRIGGER (when does it apply)?
- Does it have a well-defined CONSTRAINT (what must/should/must not happen)?
- Does it map to TrafficState fields? (altitude, heading, speed, runway, separation, flight_phase, emergency_status, etc.)
- Is it compilable? (objective, checkable against data) or does it require subjective human judgment?
- Examples of HIGH validity: "Aircraft must not descend below MSA" (trigger: descent, constraint: ≥ MSA, maps to: altitude vs msa)
- Examples of LOW validity: "Use standard phraseology" (subjective, no clear trigger, not checkable against TrafficState)
- Would this rule be usable to evaluate an ATCO instruction in real time?

IMPORTANT: The score represents VALIDITY, NOT similarity to GT (since GT is empty).""",

        "procedures": """The Ground Truth has NO procedures for this text chunk, but the Model extracted the following procedures. Evaluate their **intrinsic validity** as aeronautical knowledge.

MODEL PROCEDURES:
{model_procedures}

Rate the intrinsic validity of the Model's extraction from 0.0 to 1.0, where:
- 1.0: Excellent. The procedures represent real ATC workflows with clear step sequences, preconditions, and purpose.
- 0.8-0.9: Mostly valid. Correct procedures but with slightly imprecise steps or missing preconditions.
- 0.6-0.7: Partially valid. Some are genuine procedures, others are sequences of facts rather than actionable workflows.
- 0.4-0.5: Weak. The steps are too vague or the procedure is not a standard ATC workflow.
- 0.2-0.3: Poor. Most procedures are not real ATC workflows.
- 0.0: Invalid. These are not real aeronautical procedures.

IMPORTANT: The score represents VALIDITY, NOT similarity to GT (since GT is empty).""",
    }

    # ------------------------------------------------------------------
    # Enhanced holistic comparison prompts (used when GT has items)
    # ------------------------------------------------------------------
    HOLISTIC_COMPARISON_PROMPTS = {
        "entities": """You are evaluating the quality of knowledge extraction from aeronautical documentation.
Compare the following two SETS of entities and determine their semantic equivalence.

GROUND TRUTH ENTITIES:
{gt_entities}

MODEL ENTITIES:
{model_entities}

Rate their semantic equivalence from 0.0 to 1.0, where:
- 1.0: The sets are semantically equivalent (cover the same real-world concepts)
- 0.8-0.9: Very similar, minor differences in detail or granularity but same core concepts
- 0.6-0.7: Same general concepts but with notable differences in coverage or specificity
- 0.4-0.5: Partially related but with significant differences in what concepts are covered
- 0.2-0.3: Barely related, few overlapping concepts
- 0.0: Completely different, no shared semantic meaning

CRITICAL GUIDELINES:
1. IGNORE IDs: Do not consider internal identifiers (E001, E069, R001, EV005, RULE001, P001, etc.). These are arbitrary cross-reference codes with no semantic meaning.
2. COVERAGE NOT MATCHING: Evaluate whether the TOTAL semantic content of the GT is present in the Model's set (accounting for merging/splitting). For example, if GT represents "contact Metro Tower 119.2" as three separate entities (ATC_Instruction + ATC_Unit + Radio_Frequency) and the Model represents it as one entity containing all that information, consider them EQUIVALENT.
3. CONTEXT MATTERS: Use the context and aliases to resolve cases where the same concept has different labels.
4. BE TOLERANT: Accept synonyms, different levels of detail, and alternative decompositions as long as the operational knowledge is preserved, even if the packaging differs.""",


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

CRITICAL GUIDELINES:
1. IGNORE IDs: Do not consider internal identifiers (E001, E069, R001, etc.). These are arbitrary cross-reference codes with no semantic meaning.
2. COVERAGE NOT MATCHING: Evaluate whether the TOTAL semantic content of the GT is present in the Model's set. If GT has many fine-grained relationships and the Model has fewer, more general ones that still capture the same operational knowledge, that is EQUIVALENT.
3. Focus on SUBJECT → PREDICATE → OBJECT meaning, not on specific IDs.
4. Accept different levels of granularity and different relation types as long as the operational meaning is preserved.""",

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

CRITICAL GUIDELINES:
1. IGNORE IDs: Do not consider internal identifiers (E001, EV005, etc.). These are arbitrary cross-reference codes with no semantic meaning.
2. COVERAGE NOT MATCHING: Evaluate the total set of occurrences. If the Model merges multiple GT events into one, that is EQUIVALENT if the combined event captures all the semantic content.
3. Evaluate based on TRIGGER TEXT, EVENT TYPE, PHASE, and the roles of actors/targets (not their specific IDs).
4. Different granularity in actor/target assignment is acceptable if the operational scenario is the same.""",

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

CRITICAL GUIDELINES:
1. IGNORE IDs: Do not consider internal identifiers (RULE001, RULE002, E001, etc.). These are arbitrary codes with no semantic meaning.
2. COVERAGE NOT MATCHING: Evaluate whether the TOTAL rule content is preserved. If the Model splits one GT rule into multiple or merges multiple GT rules into one, that is EQUIVALENT if the logic is preserved.
3. KEY FIELDS: Compare rule_type, modality (must/shall/may), deontic_strength, trigger, constraint, and formal_if_then logic.
4. "must" and "shall" are semantically equivalent in aeronautical contexts (both mean mandatory).
5. Different granularity in trigger/constraint formulation is acceptable as long as the operational requirement is the same.""",

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

CRITICAL GUIDELINES:
1. IGNORE IDs: Do not consider internal identifiers (P001, E001, EV005, etc.). These are arbitrary codes with no semantic meaning.
2. COVERAGE NOT MATCHING: Evaluate whether the total procedural knowledge is preserved. If the Model splits one GT procedure into multiple or merges multiple into one, that is EQUIVALENT if the same steps and purpose are captured.
3. Focus on purpose, step sequence, preconditions, and the overall operational workflow.
4. Different levels of detail in step descriptions are acceptable if the core workflow is the same.""",
    }

    def holistic_judge(
        self,
        kex_type: str,
        gt_items: List[dict],
        model_items: List[dict],
    ) -> Optional[Judgment]:
        """Judge semantic equivalence of two sets of items (GT vs model) for a given type.

        Three cases:
        1. Both empty → score = 1.0 (agreement)
        2. GT empty, model has items → evaluate intrinsic validity (not similarity)
        3. GT has items → semantic comparison with enhanced guidelines
        """
        if not self.config.enabled or self.client is None:
            return None

        gt_formatted = self._format_items_for_holistic_judge(gt_items, kex_type)
        model_formatted = self._format_items_for_holistic_judge(model_items, kex_type)

        # Case 1: Both empty → perfect agreement
        if not gt_items and not model_items:
            return Judgment(
                similarity_score=1.0,
                explanation="Both the Ground Truth and the Model output are empty sets. They are semantically equivalent.",
                matched_fields=["both_empty"],
                unmatched_fields=[],
            )

        # Case 2: GT empty, model has items → intrinsic validity
        if not gt_items and model_items:
            validity_prompts = self.INTRINSIC_VALIDITY_PROMPTS
            prompt_template = validity_prompts.get(kex_type)
            if not prompt_template:
                return None

            prompt = prompt_template.format(
                model_entities=model_formatted if kex_type == "entities" else "",
                model_relationships=model_formatted if kex_type == "relationships" else "",
                model_events=model_formatted if kex_type == "events" else "",
                model_rules=model_formatted if kex_type == "rules" else "",
                model_procedures=model_formatted if kex_type == "procedures" else "",
            )

            system_msg = "You are an expert evaluator of aeronautical knowledge extraction quality. Evaluate the intrinsic validity of the extracted knowledge."
        else:
            # Case 3: GT has items → semantic comparison
            comparison_prompts = self.HOLISTIC_COMPARISON_PROMPTS
            prompt_template = comparison_prompts.get(kex_type)
            if not prompt_template:
                return None

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

            system_msg = "You are an expert evaluator of aeronautical knowledge extraction quality."

        try:
            result = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=Judgment,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": system_msg},
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
                    except Exception:
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

                # Strip IDs from the value text to prevent ID-based scoring
                value_str = self._strip_ids(value_str)

                item_lines.append(f"    {field}: {value_str}")

            formatted_items.append("\n".join(item_lines))

        return "\n\n".join(formatted_items)

    def _strip_ids(self, text: str) -> str:
        """Remove internal ID patterns to prevent the LLM from using them in scoring"""
        # Patterns: E001, R001, EV001, RULE001, P001, E0001, etc.
        text = re.sub(r'\b[RE]V?\d{2,4}\b', '<ID>', text)
        text = re.sub(r'\bRULE\d{2,4}\b', '<ID>', text)
        text = re.sub(r'\bP\d{2,4}\b', '<ID>', text)
        # Clean up double spaces left by removals
        text = re.sub(r'  +', ' ', text)
        return text

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
