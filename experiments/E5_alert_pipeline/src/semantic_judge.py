from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel

from config import JudgeConfig
from loader import TestCase
from pipeline_runner import AlertResult


class JudgeVerdict(BaseModel):
    score: float
    explanation: str


@dataclass
class JudgeResult:
    score: float = 0.0
    explanation: str = ""
    error: Optional[str] = None


class SemanticJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.client = None
        self.mode = None
        if cfg.enabled:
            from common.llm_client_factory import ModelConfig, create_instructor_client

            mc = ModelConfig(
                name=cfg.model_name,
                provider=cfg.provider,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                max_retries=cfg.max_retries,
                timeout=cfg.timeout,
            )
            try:
                self.client, self.mode = create_instructor_client(mc)
            except Exception as e:
                print(f"  WARNING: could not create judge client: {e}")

    def judge(
        self,
        test_case: TestCase,
        strategy_name: str,
        alert_result: AlertResult,
        expected: Any,
    ) -> JudgeResult:
        if self.client is None:
            return JudgeResult()

        prompt = f"""You are a quality judge for an ATC alert system. Evaluate the quality of the alert generated.

TEST CASE: {test_case.id}
DESCRIPTION: {test_case.description}
INSTRUCTION: "{test_case.instruction}"

RULE: {alert_result.rule_id}
STRATEGY: {strategy_name}

PREDICTED ALERT:
  satisfied: {alert_result.satisfied}
  severity: {alert_result.severity}
  explanation: {alert_result.explanation}

EXPECTED:
  satisfied: {expected.satisfied}
  severity: {expected.severity or "N/A"}

Evaluate the alert quality (0.0 to 1.0):
- Consider correctness, severity accuracy, and explanation quality.
- 0.0 = completely wrong, 1.0 = perfect.
- If satisfied matches expected, score >= 0.5.
- If severity is also correct, score >= 0.7.
- If explanation is clear and relevant, score >= 0.85."""

        try:
            from common.llm_client_factory import ModelConfig

            opts = ModelConfig(
                name=self.cfg.model_name,
                provider=self.cfg.provider,
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                max_retries=self.cfg.max_retries,
                timeout=self.cfg.timeout,
            ).completion_kwargs()

            resp = self.client.chat.completions.create(
                response_model=JudgeVerdict,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                **opts,
            )
            return JudgeResult(
                score=float(resp.score),
                explanation=resp.explanation,
            )
        except Exception as e:
            return JudgeResult(score=0.5, explanation="", error=str(e))


def run_judge_evaluation(
    judge: SemanticJudge,
    test_cases: list,
    strategy_results: dict,
) -> Dict[str, Dict[str, float]]:
    if judge is None or not judge.cfg.enabled:
        return {}

    scores: Dict[str, Dict[str, float]] = {}
    for strategy_name, strategy_result in strategy_results.items():
        strat_scores: Dict[str, float] = {}
        for tc in test_cases:
            for rule_id in tc.expected_alerts.keys():
                alert = strategy_result.per_test_case.get(tc.id, {}).get(rule_id)
                if alert is None:
                    continue
                expected = tc.expected_alerts[rule_id]
                res = judge.judge(tc, strategy_name, alert, expected)
                key = f"{tc.id}_{rule_id}"
                strat_scores[key] = res.score
        scores[strategy_name] = strat_scores
    return scores
