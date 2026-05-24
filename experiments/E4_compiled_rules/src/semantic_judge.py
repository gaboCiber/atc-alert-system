import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from common.llm_client_factory import create_instructor_client, ModelConfig

from config import JudgeConfig


class CodeComparison(BaseModel):
    similarity_score: float = Field(..., description="Semantic equivalence score 0-1", ge=0.0, le=1.0)
    explanation: str = Field(..., description="Brief explanation of the comparison")
    structural_differences: list[str] = Field(default_factory=list)
    semantic_equivalence: str = Field(..., description="Are they semantically equivalent?")


SYSTEM_PROMPT = """You are evaluating two Python implementations of an ATC alert rule.
Compare them for SEMANTIC EQUIVALENCE -- do they produce the same results for the same inputs?

Ground Truth code and Generated code both implement the same rule logic.
Rate how semantically equivalent they are from 0.0 to 1.0, where:
- 1.0: Same logic, same behavior, possibly different style
- 0.7-0.9: Same behavior, minor implementation differences
- 0.4-0.6: Similar intent, different approach
- 0.1-0.3: Different logic, different results
- 0.0: Completely different

Focus on: evaluation logic, conditions checked, return values."""

USER_TEMPLATE = """GROUND TRUTH CODE:
```python
{gt_code}
```

GENERATED CODE:
```python
{gen_code}
```

Compare these two implementations and score their semantic equivalence."""


class SemanticJudge:
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

    def judge(self, rule_id: str, gt_code: str, gen_code: str) -> Optional[CodeComparison]:
        if not self.config.enabled or self.client is None:
            return None

        prompt = USER_TEMPLATE.format(gt_code=gt_code, gen_code=gen_code)

        try:
            result = self.client.chat.completions.create(
                model=self.config.model_name,
                response_model=CodeComparison,
                max_retries=self.config.max_retries,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return result
        except Exception as e:
            if self.config.skip_on_error:
                return None
            raise

    def judge_batch(
        self,
        rule_ids: list,
        gt_codes: dict,
        gen_codes: dict,
    ) -> dict:
        results = {}
        for rule_id in rule_ids:
            gt = gt_codes.get(rule_id, "")
            gen = gen_codes.get(rule_id, "")
            if gt and gen:
                result = self.judge(rule_id, gt, gen)
                results[rule_id] = result
        return results