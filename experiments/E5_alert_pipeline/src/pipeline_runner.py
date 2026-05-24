import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from loader import CompiledRule, TestCase
from config import GenericConfig


class GenericVerdict(BaseModel):
    satisfied: bool
    severity: str
    explanation: str


@dataclass
class AlertResult:
    rule_id: str
    satisfied: bool
    severity: Optional[str]
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class StrategyResult:
    strategy_name: str
    per_test_case: Dict[str, Dict[str, AlertResult]] = field(default_factory=dict)
    total_latency_ms: float = 0.0


def build_traffic_state(ts_data: dict) -> Any:
    from Alert_System.models.traffic_state import (
        TrafficState,
        AircraftState,
        Position,
        FlightPhase,
        RunwayState,
    )

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

    sector_id = ts_data.get("sector_id", "TEST")
    return TrafficState(
        sector_id=sector_id,
        msa=msa,
        aircrafts=aircrafts,
        runways=runways,
    )


def _build_traffic_summary(traffic_state: Any) -> str:
    lines = [f"MSA: {traffic_state.msa} ft"]
    lines.append("Aircraft:")
    for cs, ac in traffic_state.aircrafts.items():
        pos = ac.position
        lines.append(
            f"  {cs}: lat={pos.latitude}, lon={pos.longitude}, "
            f"alt={pos.altitude}ft, hdg={pos.heading}°, "
            f"spd={pos.speed}kts, phase={ac.flight_phase.value}"
        )
    if traffic_state.runways:
        lines.append("Runways:")
        for rw_id, rw in traffic_state.runways.items():
            occ = f"occupied by {rw.occupied_by}" if rw.occupied else "clear"
            lines.append(f"  {rw_id}: {occ}")
    return "\n".join(lines)


def evaluate_compiled(test_case: TestCase, rule: CompiledRule) -> AlertResult:
    start = time.perf_counter()
    try:
        ts = build_traffic_state(test_case.traffic_state)
        result = rule.evaluate_fn(ts, callsign=test_case.callsign)
        latency = (time.perf_counter() - start) * 1000
        return AlertResult(
            rule_id=rule.rule_id,
            satisfied=result.get("satisfied", True),
            severity=result.get("severity", "INFO"),
            explanation=result.get("explanation", ""),
            details=result.get("details", {}),
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return AlertResult(
            rule_id=rule.rule_id,
            satisfied=True,
            severity="INFO",
            explanation="",
            details={},
            latency_ms=latency,
            error=str(e),
        )


def evaluate_generic(
    test_case: TestCase,
    rule_id: str,
    rule_desc: str,
    generic_cfg: GenericConfig,
) -> AlertResult:
    start = time.perf_counter()
    try:
        ts = build_traffic_state(test_case.traffic_state)
        summary = _build_traffic_summary(ts)

        prompt = f"""You are an air traffic control safety evaluator. Determine if the given rule is VIOLATED in this traffic state.

RULE ({rule_id}):
{rule_desc}

INSTRUCTION GIVEN: {test_case.instruction}

CURRENT TRAFFIC SITUATION:
{summary}

IMPORTANT: The traffic state shown is the PROJECTED state AFTER the instruction was applied.
Carefully compare the rule requirements against the actual traffic data.

Respond with a JSON object:
{{"satisfied": true (if rule IS satisfied / safe), false (if rule IS violated / unsafe), "severity": "INFO"/"LOW"/"MEDIUM"/"HIGH"/"CRITICAL", "explanation": "concise reason citing specific values"}}"""

        from common.llm_client_factory import ModelConfig

        mc = ModelConfig(
            name=generic_cfg.model_name,
            provider=generic_cfg.provider,
            base_url=generic_cfg.base_url,
            api_key=generic_cfg.api_key,
            max_retries=generic_cfg.max_retries,
            timeout=generic_cfg.timeout,
        )
        from common.llm_client_factory import create_instructor_client
        import instructor

        client, mode = create_instructor_client(mc)
        opts = mc.completion_kwargs()

        resp = client.chat.completions.create(
            response_model=GenericVerdict,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            **opts,
        )

        latency = (time.perf_counter() - start) * 1000
        return AlertResult(
            rule_id=rule_id,
            satisfied=resp.satisfied,
            severity=resp.severity,
            explanation=resp.explanation,
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return AlertResult(
            rule_id=rule_id,
            satisfied=True,
            severity="INFO",
            explanation="",
            latency_ms=latency,
            error=str(e),
        )


def run_strategy(
    strategy: str,
    test_cases: List[TestCase],
    compiled_rules: Dict[str, CompiledRule],
    rule_descriptions: Dict[str, str],
    generic_cfg: GenericConfig,
) -> StrategyResult:
    result = StrategyResult(strategy_name=strategy)

    for tc in test_cases:
        tc_results = {}
        for rule_id in tc.expected_alerts.keys():
            if strategy == "compiled":
                rule = compiled_rules.get(rule_id)
                if rule is None:
                    tc_results[rule_id] = AlertResult(
                        rule_id=rule_id,
                        satisfied=True,
                        severity="INFO",
                        explanation="",
                        error=f"No compiled rule for {rule_id}",
                    )
                else:
                    tc_results[rule_id] = evaluate_compiled(tc, rule)
            elif strategy == "generic":
                desc = rule_descriptions.get(rule_id, "")
                tc_results[rule_id] = evaluate_generic(tc, rule_id, desc, generic_cfg)
        result.per_test_case[tc.id] = tc_results

    total_lat = 0.0
    count = 0
    for tc_id, tc_res in result.per_test_case.items():
        for rid, alert in tc_res.items():
            total_lat += alert.latency_ms
            count += 1
    result.total_latency_ms = total_lat / count if count > 0 else 0.0

    return result
