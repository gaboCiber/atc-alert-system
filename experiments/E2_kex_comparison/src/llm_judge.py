import sys
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.llm_client_factory import create_instructor_client, ModelConfig

from config import JudgeConfig
from dedup_prompts import DEDUP_PROMPTS, FIELD_EXTRACTORS

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
