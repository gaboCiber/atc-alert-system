import hashlib
import json
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


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


class SemanticJudge:
    def __init__(self, config: JudgeConfig, cache_dir: Optional[Path] = None):
        self.config = config
        self.cache_dir = cache_dir
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

    def _model_cache_key(self, model_name: str) -> str:
        safe = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
        return f"judge_cache_{safe}.json"

    def _load_cache(self, model_name: str) -> dict:
        if self.cache_dir is None:
            return {}
        path = self.cache_dir / self._model_cache_key(model_name)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, model_name: str, cache: dict):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / self._model_cache_key(model_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _make_cache_entry(self, gt_code: str, gen_code: str, result: CodeComparison) -> dict:
        return {
            "gt_hash": _code_hash(gt_code),
            "gen_hash": _code_hash(gen_code),
            "result": result.model_dump(),
        }

    def _check_cache(self, cache: dict, rule_id: str, gt_code: str, gen_code: str) -> Optional[CodeComparison]:
        entry = cache.get(rule_id)
        if entry is None:
            return None
        if entry.get("gt_hash") != _code_hash(gt_code):
            return None
        if entry.get("gen_hash") != _code_hash(gen_code):
            return None
        return CodeComparison(**entry["result"])

    def judge(
        self,
        rule_id: str,
        gt_code: str,
        gen_code: str,
        model_name: Optional[str] = None,
    ) -> Optional[CodeComparison]:
        if not self.config.enabled or self.client is None:
            return None

        cache = self._load_cache(model_name) if model_name else {}

        cached = self._check_cache(cache, rule_id, gt_code, gen_code)
        if cached is not None:
            return cached

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
        except Exception as e:
            if self.config.skip_on_error:
                return None
            raise

        if model_name:
            cache[rule_id] = self._make_cache_entry(gt_code, gen_code, result)
            self._save_cache(model_name, cache)

        return result

    def judge_batch(
        self,
        rule_ids: list,
        gt_codes: dict,
        gen_codes: dict,
        model_name: Optional[str] = None,
    ) -> dict:
        results = {}
        for rule_id in rule_ids:
            gt = gt_codes.get(rule_id, "")
            gen = gen_codes.get(rule_id, "")
            if gt and gen:
                result = self.judge(rule_id, gt, gen, model_name=model_name)
                results[rule_id] = result
        return results
