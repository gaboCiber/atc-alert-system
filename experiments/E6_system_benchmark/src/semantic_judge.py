import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from config import LLMConfig


class JudgeVerdict(BaseModel):
    score: float
    explanation: str


@dataclass
class JudgeResult:
    score: float = 0.0
    explanation: str = ""
    error: Optional[str] = None


class SemanticJudge:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = None
        if cfg:
            from common.llm_client_factory import create_instructor_client, ModelConfig

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
                print(f"  WARNING: judge client: {e}")

    def judge_alert(
        self,
        tc_id: str,
        instruction: str,
        expected_alerts: Dict[str, Any],
        actual_alerts: Dict[str, Any],
    ) -> JudgeResult:
        if self.client is None:
            return JudgeResult()

        prompt = f"""You evaluate ATC alert system quality.

TEST: {tc_id}
INSTRUCTION: "{instruction}"

EXPECTED ALERTS:
{self._fmt_expected(expected_alerts)}

ACTUAL ALERTS GENERATED:
{self._fmt_actual(actual_alerts)}

Score 0.0-1.0 considering:
- Were all violations correctly detected? (recall)
- Were there any false alarms? (precision)
- Were severity levels correct?
- Response quality

Respond with JSON: {{"score": 0.0-1.0, "explanation": "brief reason"}}"""

        try:
            from common.llm_client_factory import ModelConfig as JudgeModelConfig
            mc = JudgeModelConfig(
                name=self.cfg.model_name,
                provider=self.cfg.provider,
                base_url=self.cfg.base_url,
                api_key=self.cfg.api_key,
                max_retries=self.cfg.max_retries,
                timeout=self.cfg.timeout,
            )
            opts = mc.completion_kwargs()

            resp = self.client.chat.completions.create(
                response_model=JudgeVerdict,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                **opts,
            )
            return JudgeResult(score=float(resp.score), explanation=resp.explanation)
        except Exception as e:
            return JudgeResult(score=0.5, explanation="", error=str(e))

    def _fmt_expected(self, alerts: Dict[str, Any]) -> str:
        if not alerts:
            return "  No alerts expected"
        lines = []
        for rid, exp in alerts.items():
            s = "VIOLATION" if not exp.satisfied else "SAFE"
            sev = f" ({exp.severity})" if exp.severity else ""
            lines.append(f"  {rid}: {s}{sev}")
        return "\n".join(lines)

    def _fmt_actual(self, alerts: Dict[str, "EvalResult"]) -> str:
        if not alerts:
            return "  No alerts generated"
        lines = []
        for rid, res in alerts.items():
            s = "VIOLATION" if not res.satisfied else "SAFE"
            sev = f" ({res.severity})" if res.severity else ""
            err = f" ERROR:{res.error}" if res.error else ""
            lines.append(f"  {rid}: {s}{sev}{err}")
        return "\n".join(lines)


def run_judge_all(
    judge: SemanticJudge,
    test_cases,
    all_benchmarks: List,
) -> Dict[str, float]:
    if judge is None or judge.client is None:
        return {}
    scores = {}
    for tc in test_cases:
        bench = next((b for b in all_benchmarks if b.test_case_id == tc.id), None)
        if bench is None:
            continue
        res = judge.judge_alert(
            tc.id, tc.instruction, tc.expected_alerts, bench.eval_results
        )
        scores[tc.id] = res.score
    return scores
