import importlib.util
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.llm_client_factory import ModelConfig

from test_case_loader import TestCase


@dataclass
class StepTiming:
    step_name: str
    times_ms: List[float] = field(default_factory=list)

    @property
    def avg_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def p50_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        return s[len(s) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]


@dataclass
class EvalResult:
    rule_id: str
    satisfied: bool
    severity: str
    latency_ms: float
    error: Optional[str] = None
    explanation: str = ""


@dataclass
class TcBenchmark:
    test_case_id: str
    bert_ms: float = 0.0
    native_rules: Dict[str, float] = field(default_factory=dict)
    compiled_rules: Dict[str, float] = field(default_factory=dict)
    generic_rules: Dict[str, float] = field(default_factory=dict)
    pipeline_e2e_ms: float = 0.0
    pipeline_steps: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    eval_results: Dict[str, EvalResult] = field(default_factory=dict)

    @property
    def total_rule_ms(self) -> float:
        n = sum(self.native_rules.values())
        c = sum(self.compiled_rules.values())
        g = sum(self.generic_rules.values())
        return n + c + g

    @property
    def total_ms(self) -> float:
        return self.bert_ms + self.total_rule_ms + self.pipeline_e2e_ms


def build_traffic_state(ts_data: dict) -> Any:
    from Alert_System.models.traffic_state import (
        TrafficState,
        AircraftState,
        Position,
        FlightPhase,
        RunwayState,
    )

    sector_id = ts_data.get("sector_id", "E6_BENCH")
    msa = ts_data.get("msa")
    aircrafts = {}
    for callsign, ac_data in ts_data.get("aircrafts", {}).items():
        pos_data = ac_data.get("position", {})
        phase_str = ac_data.get("flight_phase", "CRUISE")
        try:
            phase = FlightPhase(phase_str)
        except ValueError:
            phase = FlightPhase.CRUISE

        aircrafts[callsign] = AircraftState(
            callsign=callsign,
            position=Position(
                latitude=pos_data.get("latitude", 0.0),
                longitude=pos_data.get("longitude", 0.0),
                altitude=pos_data.get("altitude", 0),
                heading=pos_data.get("heading", 0),
                speed=pos_data.get("speed", 0),
                vertical_rate=pos_data.get("vertical_rate", 0),
            ),
            flight_phase=phase,
        )

    runways = {}
    for rw_id, rw_data in ts_data.get("runways", {}).items():
        runways[rw_id] = RunwayState(
            runway_id=rw_data.get("runway_id", rw_id),
            occupied=rw_data.get("occupied", False),
            occupied_by=rw_data.get("occupied_by"),
        )

    return TrafficState(
        sector_id=sector_id,
        msa=msa,
        aircrafts=aircrafts,
        runways=runways,
    )


class SystemBenchmark:
    def __init__(
        self,
        compiled_rules_dir: Path,
        llm_cfg: Any = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.llm_cfg = llm_cfg or ModelConfig(
            name="gemma4:31b-cloud",
            provider="openai",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        self.bert_parser = None
        self.compiled_rules: Dict[str, Any] = {}
        self._load_compiled_rules(compiled_rules_dir)

    def _load_compiled_rules(self, rules_dir: Path):
        if not rules_dir.exists():
            if self.verbose:
                print(f"  WARNING: compiled rules dir {rules_dir} not found")
            return
        for py_file in sorted(rules_dir.glob("RULE*.py")):
            rule_id = py_file.stem
            try:
                spec = importlib.util.spec_from_file_location(
                    f"e6_{rule_id}", py_file
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[f"e6_{rule_id}"] = mod
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "evaluate"):
                        self.compiled_rules[rule_id] = mod.evaluate
                        if self.verbose:
                            print(f"  Loaded compiled: {rule_id}")
            except Exception as e:
                if self.verbose:
                    print(f"  WARNING: {py_file.name}: {e}")

    def _ensure_bert_parser(self):
        if self.bert_parser is not None:
            return
        try:
            from Alert_System.integration.bert_atc_parser import BertATCParser

            self.bert_parser = BertATCParser()
        except Exception as e:
            if self.verbose:
                print(f"  WARNING: could not create BERT parser: {e}")
            self.bert_parser = None

    def _parse_instruction(self, raw_instruction: str):
        """Parse raw instruction string into a ParsedInstruction object."""
        if not raw_instruction:
            return None
        try:
            from Alert_System.models.instruction import (
                ParsedInstruction, InstructionType, Speaker,
            )
            text = raw_instruction.lower()
            callsign_match = re.search(r'\b([a-z]{3}\d+)\b', text)
            callsign = callsign_match.group(1).upper() if callsign_match else None

            instruction_type = InstructionType.UNKNOWN
            action_verb = "unknown"
            parameters = {}

            if "descend" in text or "descent" in text:
                instruction_type = InstructionType.DESCENT
                action_verb = "descend"
                fl_match = re.search(r'fl\s*(\d+)', text)
                if fl_match:
                    parameters["target_altitude"] = int(fl_match.group(1)) * 100
                    parameters["flight_level"] = int(fl_match.group(1))
                alt_match = re.search(r'(?:to|at)\s+(\d{4,6})\s*(?:feet|ft)?', text)
                if alt_match and "target_altitude" not in parameters:
                    parameters["target_altitude"] = int(alt_match.group(1))
            elif "climb" in text:
                instruction_type = InstructionType.CLIMB
                action_verb = "climb"
                fl_match = re.search(r'fl\s*(\d+)', text)
                if fl_match:
                    parameters["target_altitude"] = int(fl_match.group(1)) * 100
                alt_match = re.search(r'(?:to|at)\s+(\d{4,6})\s*(?:feet|ft)?', text)
                if alt_match and "target_altitude" not in parameters:
                    parameters["target_altitude"] = int(alt_match.group(1))
            elif "maintain" in text:
                instruction_type = InstructionType.MAINTAIN_ALTITUDE
                action_verb = "maintain"
                fl_match = re.search(r'fl\s*(\d+)', text)
                if fl_match:
                    parameters["target_altitude"] = int(fl_match.group(1)) * 100
                alt_match = re.search(r'maintain\s+(?:at\s+)?(?:fl\s*)?(\d{4,6})\s*(?:feet|ft)?', text)
                if alt_match and "target_altitude" not in parameters:
                    val = int(alt_match.group(1))
                    parameters["target_altitude"] = val * 100 if val < 1000 else val
            elif "heading" in text or "turn" in text:
                instruction_type = InstructionType.HEADING
                action_verb = "turn" if "turn" in text else "heading"
                heading_match = re.search(r'(?:heading|turn\s+(?:left|right)?)\s+(\d{3})', text)
                if heading_match:
                    parameters["heading"] = int(heading_match.group(1))
            elif "speed" in text:
                instruction_type = InstructionType.SPEED
                action_verb = "speed"
                spd_match = re.search(r'speed\s+(?:to\s+)?(\d{3})', text)
                if spd_match:
                    parameters["speed"] = int(spd_match.group(1))
            elif "cleared for takeoff" in text:
                instruction_type = InstructionType.TAKEOFF_CLEARANCE
                action_verb = "takeoff"
                rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
                if rw_match:
                    parameters["runway"] = rw_match.group(1)
            elif "cleared to land" in text:
                instruction_type = InstructionType.LANDING_CLEARANCE
                action_verb = "land"
                rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
                if rw_match:
                    parameters["runway"] = rw_match.group(1)
            elif "taxi" in text:
                instruction_type = InstructionType.TAXI
                action_verb = "taxi"
                wp_match = re.search(r'(?:to|via)\s+([\w\s]+)', text)
                if wp_match:
                    parameters["waypoint"] = wp_match.group(1).strip()
                rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
                if rw_match:
                    parameters["runway"] = rw_match.group(1)
            elif "hold short" in text or "hold position" in text:
                instruction_type = InstructionType.HOLD_SHORT
                action_verb = "hold"
                rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
                if rw_match:
                    parameters["runway"] = rw_match.group(1)
            elif "contact" in text:
                instruction_type = InstructionType.CONTACT
                action_verb = "contact"
            elif "squawk" in text:
                instruction_type = InstructionType.SQUAWK
                action_verb = "squawk"
                sq_match = re.search(r'squawk\s+(\d{4})', text)
                if sq_match:
                    parameters["squawk"] = sq_match.group(1)
            elif "line up" in text:
                instruction_type = InstructionType.LINE_UP
                action_verb = "line_up"
                rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
                if rw_match:
                    parameters["runway"] = rw_match.group(1)

            return ParsedInstruction(
                raw_text=raw_instruction,
                normalized_text=raw_instruction,
                speaker=Speaker.ATCO,
                callsign=callsign,
                instruction_type=instruction_type,
                action_verb=action_verb,
                parameters=parameters,
            )
        except Exception:
            return None

    # ── BERT Parse Timing ──

    def benchmark_bert(self, instruction: str) -> float:
        self._ensure_bert_parser()
        if self.bert_parser is None:
            return -1.0
        start = time.perf_counter()
        self.bert_parser.parse(instruction)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed

    # ── Native Rule Timing ──

    def benchmark_native(
        self, ts_data: dict, callsign: Optional[str]
    ) -> Dict[str, float]:
        timings = {}
        from Alert_System.rule_engine.engine import RuleEngine

        engine = RuleEngine()

        ts = build_traffic_state(ts_data)

        for cond_type in ["ALTITUDE", "SEPARATION", "RUNWAY"]:
            if not engine.has_evaluator(cond_type):
                continue
            start = time.perf_counter()
            engine.evaluate(cond_type, {}, ts, callsign)
            elapsed = (time.perf_counter() - start) * 1000
            timings[cond_type] = elapsed

        return timings

    # ── Compiled Rule Timing ──

    def benchmark_compiled(
        self, ts_data: dict, callsign: Optional[str], instruction=None
    ) -> Dict[str, float]:
        timings = {}
        ts = build_traffic_state(ts_data)
        parsed = self._parse_instruction(instruction) if isinstance(instruction, str) else instruction
        for rule_id, fn in self.compiled_rules.items():
            start = time.perf_counter()
            try:
                fn(ts, callsign=callsign, instruction=parsed)
            except Exception:
                pass
            elapsed = (time.perf_counter() - start) * 1000
            timings[rule_id] = elapsed
        return timings

    # ── Generic / LLM Rule Timing ──

    def benchmark_generic(
        self,
        tc: TestCase,
        rule_id: str,
        rule_desc: str,
    ) -> Tuple[float, EvalResult]:
        ts = build_traffic_state(tc.traffic_state)
        summary = self._build_traffic_summary(ts)

        prompt = f"""You are an air traffic control safety evaluator. Determine if the given rule is VIOLATED in this traffic state.

RULE ({rule_id}):
{rule_desc}

INSTRUCTION GIVEN: {tc.instruction}

CURRENT TRAFFIC SITUATION:
{summary}

IMPORTANT: The traffic state shown is the PROJECTED state AFTER the instruction was applied.
Carefully compare the rule requirements against the actual traffic data.

Respond with a JSON object:
{{"satisfied": true (if rule IS satisfied / safe), false (if rule IS violated / unsafe), "severity": "INFO"/"LOW"/"MEDIUM"/"HIGH"/"CRITICAL", "explanation": "concise reason citing specific values"}}"""

        from pydantic import BaseModel

        class GenericVerdict(BaseModel):
            satisfied: bool
            severity: str
            explanation: str

        from common.llm_client_factory import create_instructor_client

        mc = ModelConfig(
            name=self.llm_cfg.model_name,
            provider=self.llm_cfg.provider,
            base_url=self.llm_cfg.base_url,
            api_key=self.llm_cfg.api_key,
            max_retries=self.llm_cfg.max_retries,
            timeout=self.llm_cfg.timeout,
        )

        start = time.perf_counter()
        try:
            client, mode = create_instructor_client(mc)
            opts = mc.completion_kwargs()
            resp = client.chat.completions.create(
                response_model=GenericVerdict,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                **opts,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, EvalResult(
                rule_id=rule_id,
                satisfied=resp.satisfied,
                severity=resp.severity,
                latency_ms=elapsed,
                explanation=resp.explanation,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, EvalResult(
                rule_id=rule_id,
                satisfied=True,
                severity="INFO",
                latency_ms=elapsed,
                error=str(e),
            )

    # ── Full Pipeline Timing ──

    def benchmark_pipeline(
        self, tc: TestCase
    ) -> Tuple[float, Optional[Dict[int, Dict[str, Any]]], List[Any]]:
        from Alert_System.models.traffic_state import TrafficState
        from Alert_System.core.state_manager import StateManager
        from Alert_System.rule_engine.engine import RuleEngine
        from Alert_System.pipeline.alert_pipeline import AlertPipeline
        from common.llm_client_factory import ModelConfig as PipelineModelConfig

        llm_config_for_pipeline = PipelineModelConfig(
            name=self.llm_cfg.model_name,
            provider=self.llm_cfg.provider,
            base_url=self.llm_cfg.base_url,
            api_key=self.llm_cfg.api_key,
        )

        ts = build_traffic_state(tc.traffic_state)
        state_manager = StateManager(ts)
        rule_engine = RuleEngine()
        pipeline = AlertPipeline(
            state_manager,
            rule_engine,
            llm_config=llm_config_for_pipeline,
            verbose=False,
        )

        start = time.perf_counter()
        try:
            result = pipeline.process_instruction(tc.instruction)
            elapsed = (time.perf_counter() - start) * 1000

            steps_info = {}
            for i, step in enumerate(result.steps):
                steps_info[i + 1] = {
                    "name": step.step_name,
                    "status": step.status,
                    "execution_time_ms": step.execution_time_ms,
                }

            alerts = getattr(result, "alerts", []) or []
            return elapsed, steps_info, alerts
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            if self.verbose:
                print(f"  Pipeline error: {e}")
            return elapsed, None, []

    def _build_traffic_summary(self, ts: Any) -> str:
        lines = [f"MSA: {ts.msa} ft"]
        lines.append("Aircraft:")
        for cs, ac in ts.aircrafts.items():
            pos = ac.position
            lines.append(
                f"  {cs}: lat={pos.latitude}, lon={pos.longitude}, "
                f"alt={pos.altitude}ft, hdg={pos.heading}°, "
                f"spd={pos.speed}kts, phase={ac.flight_phase.value}"
            )
        if ts.runways:
            lines.append("Runways:")
            for rw_id, rw in ts.runways.items():
                occ = f"occupied by {rw.occupied_by}" if rw.occupied else "clear"
                lines.append(f"  {rw_id}: {occ}")
        return "\n".join(lines)
