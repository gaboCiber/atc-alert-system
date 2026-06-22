"""Core del benchmark E7: pipeline integrado Alert_System."""

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from Alert_System.integration.asr_adapter import ASRAdapter
from Alert_System.integration.bert_atc_parser import BertATCParser
from Alert_System.core.state_manager import StateManager
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.pipeline.alert_pipeline import AlertPipeline, PipelineResult
from Alert_System.models.instruction import Speaker

from config import E7Config, to_model_config
from test_case_loader import TestCase, traffic_state_to_alert_system
from rule_loader import load_all_rules


@dataclass
class GeneratedAlertInfo:
    rule_id: str
    severity: str
    explanation: str
    condition_type: str


@dataclass
class TcPipelineResult:
    test_case_id: str
    total_ms: float = 0.0
    parse_ms: float = 0.0
    steps: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    generated_alerts: List[GeneratedAlertInfo] = field(default_factory=list)
    final_decision: str = "PENDING"
    has_errors: bool = False
    error: Optional[str] = None


class IntegratedPipelineBenchmark:
    """Ejecuta el pipeline completo del Alert_System por test case."""

    def __init__(self, cfg: E7Config):
        import time
        self.cfg = cfg
        t0 = time.perf_counter()
        self.model_config = to_model_config(cfg)

        if cfg.skip_bert:
            if cfg.verbose:
                print("⏭️  BERT omitido (--skip-bert), parseo con regex")
            self.bert_parser = None
        else:
            if cfg.verbose:
                print("🔄 Cargando BERT NER (primera vez puede tardar varios minutos)...")
            self.bert_parser = BertATCParser(
                model_name=cfg.bert.model_name,
                confidence_threshold=cfg.bert.confidence_threshold,
            )
            self.bert_parser._initialize()

        self.asr_adapter = ASRAdapter(bert_parser=self.bert_parser)
        if cfg.verbose:
            print(f"   ASRAdapter listo ({(time.perf_counter()-t0)*1000:.0f}ms)")

        t1 = time.perf_counter()
        self.rule_engine = RuleEngine()
        self.rule_filter = load_all_rules(
            rule_engine=self.rule_engine,
            llm_config=self.model_config,
            filter_config=cfg.filter,
            compiled_rules_dir=cfg.compiled_rules_dir,
            rules_json_path=cfg.rules_json_path,
            verbose=cfg.verbose,
            skip_generic=cfg.skip_generic,
        )
        if cfg.verbose:
            print(f"   RuleEngine cargado ({(time.perf_counter()-t1)*1000:.0f}ms)")
            print(f"   Init total: {(time.perf_counter()-t0)*1000:.0f}ms\n")

    def run_test_case(self, tc: TestCase) -> TcPipelineResult:
        import time
        tc_start = time.perf_counter()

        traffic = traffic_state_to_alert_system(tc)
        state_manager = StateManager(traffic)

        pipeline = AlertPipeline(
            state_manager=state_manager,
            rule_engine=self.rule_engine,
            llm_config=self.model_config,
            rule_filter=self.rule_filter,
            filter_timeout=self.cfg.filter.timeout_seconds,
            filter_top_k=self.cfg.filter.top_k,
            verbose=self.cfg.verbose,
        )

        if self.cfg.verbose:
            print(f"  [Setup] {(time.perf_counter() - tc_start)*1000:.0f}ms", flush=True)

        transcription = SimpleNamespace(
            text=tc.instruction,
            file_path="",
            model_name="text_input",
            confidence=1.0,
            timestamps=None,
            duration=None,
            metadata={},
        )

        parse_start = time.perf_counter()
        try:
            parsed = self.asr_adapter.adapt(transcription, speaker=Speaker.ATCO)
        except Exception as e:
            return TcPipelineResult(
                test_case_id=tc.id,
                error=f"ASRAdapter error: {e}",
                has_errors=True,
            )
        parse_ms = (time.perf_counter() - parse_start) * 1000

        if self.cfg.verbose:
            print(f"  [Parse] {parse_ms:.0f}ms", flush=True)

        try:
            result = pipeline.process_instruction(parsed.raw_text, pre_parsed=parsed)
        except Exception as e:
            return TcPipelineResult(
                test_case_id=tc.id,
                parse_ms=parse_ms,
                error=f"Pipeline error: {e}",
                has_errors=True,
            )

        if self.cfg.verbose:
            print(f"  [Pipeline] {(time.perf_counter() - parse_start)*1000:.0f}ms", flush=True)

        return self._build_tc_result(tc.id, result, parse_ms)

    @staticmethod
    def _build_tc_result(tc_id: str, result: PipelineResult, parse_ms: float) -> TcPipelineResult:
        steps: Dict[int, Dict[str, Any]] = {}
        for i, step in enumerate(result.steps):
            steps[i + 1] = {
                "name": step.step_name,
                "status": step.status,
                "execution_time_ms": step.execution_time_ms,
            }

        generated = IntegratedPipelineBenchmark._extract_alerts(result)

        return TcPipelineResult(
            test_case_id=tc_id,
            total_ms=result.total_execution_time_ms,
            parse_ms=parse_ms,
            steps=steps,
            generated_alerts=generated,
            final_decision=result.final_decision,
            has_errors=result.has_errors,
        )

    @staticmethod
    def _extract_alerts(result: PipelineResult) -> List[GeneratedAlertInfo]:
        alerts: List[GeneratedAlertInfo] = []
        seen: set[str] = set()

        for alert in result.alerts_generated:
            for violation in alert.violations:
                if violation.rule_id in seen:
                    continue
                seen.add(violation.rule_id)
                severity = violation.severity.value if hasattr(violation.severity, "value") else str(violation.severity)
                alerts.append(
                    GeneratedAlertInfo(
                        rule_id=violation.rule_id,
                        severity=severity,
                        explanation=violation.explanation or "",
                        condition_type=violation.condition_type,
                    )
                )

        for violation in result.violations_found:
            if violation.rule_id in seen:
                continue
            seen.add(violation.rule_id)
            severity = violation.severity.value if hasattr(violation.severity, "value") else str(violation.severity)
            alerts.append(
                GeneratedAlertInfo(
                    rule_id=violation.rule_id,
                    severity=severity,
                    explanation=violation.explanation or "",
                    condition_type=violation.condition_type,
                )
            )

        return alerts
