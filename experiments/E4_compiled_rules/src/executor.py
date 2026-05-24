import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from Alert_System.models.traffic_state import TrafficState, AircraftState, Position, FlightPhase, RunwayState

from config import E4Config
from loader import TestTrafficState, CompiledRule


@dataclass
class ExecutionResult:
    rule_id: str
    model_name: str
    test_name: str
    passed: bool
    output: Optional[dict] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    expected: Optional[dict] = None
    output_matches: Optional[bool] = None


@dataclass
class ExecutionMetrics:
    model_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    match_rate: float
    results: list = field(default_factory=list)


def build_traffic_state(state_dict: dict) -> TrafficState:
    state = TrafficState(
        sector_id=state_dict.get("sector_id", "TEST"),
        msa=state_dict.get("msa", 5000),
        qnh=state_dict.get("qnh"),
        wind=state_dict.get("wind"),
    )

    for callsign, ac_data in state_dict.get("aircrafts", {}).items():
        if isinstance(ac_data, dict):
            ac_data.pop("wake_turbulence", None)
            ac_data.pop("aircraft_type", None)
            ac_data.pop("is_emergency", None)
            ac_data.pop("emergency_type", None)
            ac_data.pop("last_contact", None)
            ac_data.pop("position_history", None)
            ac_data.pop("phase_history", None)
            ac_data.pop("previous_phase", None)
            ac_data.pop("phase_transition_timestamp", None)
            ac_data.pop("squawk_history", None)
            ac_data.pop("squawk_assigned_timestamp", None)
            ac_data.pop("clearances", None)
            ac_data.pop("restrictions", None)

            pos_data = ac_data.get("position", {})

            phase_str = ac_data.get("flight_phase", "CRUISE")
            try:
                from Alert_System.models.traffic_state import FlightPhase
                phase = FlightPhase(phase_str)
            except ValueError:
                from Alert_System.models.traffic_state import FlightPhase
                phase = FlightPhase.CRUISE

            ac = AircraftState(
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
            state.add_aircraft(ac)

    for rw_data in state_dict.get("runways", {}).values():
        if isinstance(rw_data, dict):
            rw = RunwayState(
                runway_id=rw_data.get("runway_id", "00"),
                occupied=rw_data.get("occupied", False),
                occupied_by=rw_data.get("occupied_by"),
            )
            state.add_runway(rw)

    return state


def execute_code(
    code: str,
    traffic_state: TrafficState,
    callsign: Optional[str] = None,
) -> tuple:
    import time
    start = time.time()

    namespace = {"__builtins__": {}}
    try:
        exec(code, namespace)
    except Exception as e:
        return None, f"exec_failed:{e}", time.time() - start

    if "evaluate" not in namespace:
        return None, "no_evaluate_function", time.time() - start

    try:
        result = namespace["evaluate"](traffic_state, callsign)
        elapsed = (time.time() - start) * 1000
        return result, None, elapsed
    except Exception as e:
        return None, f"evaluation_failed:{e}", (time.time() - start) * 1000


def compare_outputs(output: dict, expected: Optional[dict]) -> bool:
    if expected is None:
        return None
    if not output or not expected:
        return False

    sat_out = output.get("satisfied")
    sat_exp = expected.get("satisfied")
    if sat_out is not None and sat_exp is not None:
        if sat_out != sat_exp:
            return False

    sev_out = output.get("severity")
    sev_exp = expected.get("severity")
    if sev_out is not None and sev_exp is not None:
        if sev_out != sev_exp:
            return False

    return True


def run_execution_test(
    rule: CompiledRule,
    test_state: TestTrafficState,
) -> ExecutionResult:
    ts = build_traffic_state(test_state.state)

    output, error, elapsed = execute_code(rule.compiled_code, ts, callsign=None)

    expected = test_state.expected_outcome
    match = compare_outputs(output, expected) if output else False

    passed = error is None and (match is True or match is None)

    return ExecutionResult(
        rule_id=rule.rule_id,
        model_name=rule.model_name,
        test_name=test_state.name,
        passed=passed,
        output=output,
        error=error,
        execution_time_ms=elapsed,
        expected=expected,
        output_matches=match,
    )


def evaluate_compiled_rules(
    compiled_rules: dict,
    test_traffic_states: dict,
    model_name: str,
) -> ExecutionMetrics:
    results = []
    successful = 0
    failed = 0

    for rule_id, rule in compiled_rules.items():
        for test_name, test_state in test_traffic_states.items():
            result = run_execution_test(rule, test_state)
            results.append(result)
            if result.error is None:
                successful += 1
            else:
                failed += 1

    total = len(results)
    match_count = sum(1 for r in results if r.output_matches is True)
    match_rate = match_count / total if total > 0 else 0.0

    return ExecutionMetrics(
        model_name=model_name,
        total_executions=total,
        successful_executions=successful,
        failed_executions=failed,
        match_rate=match_rate,
        results=results,
    )