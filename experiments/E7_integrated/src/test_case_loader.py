"""Cargador de test cases para E7_integrated."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExpectedAlert:
    """Alerta esperada en ground truth."""
    rule_id: str
    satisfied: bool
    severity: str


@dataclass
class TrafficStateData:
    """Estado de tráfico para test case."""
    sector_id: str
    msa: int
    aircrafts: Dict[str, Dict[str, Any]]
    runways: Dict[str, Dict[str, Any]]


@dataclass
class TestCase:
    """Test case completo para benchmark."""
    id: str
    instruction: str
    callsign: str
    description: str
    traffic_state: TrafficStateData
    expected_alerts: Dict[str, ExpectedAlert]
    eval_type: str = "native+compiled+generic"


def load_test_cases(ground_truth_dir: Path) -> List[TestCase]:
    """
    Carga test cases desde ground_truth/test_cases.json.
    
    Valida que cada TC tenga expected_alerts (no vacío a menos que sea clean state).
    """
    test_cases_path = ground_truth_dir / "test_cases.json"
    
    if not test_cases_path.exists():
        raise FileNotFoundError(f"No se encontró {test_cases_path}")
    
    with open(test_cases_path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)
    
    test_cases = []
    
    for raw in raw_cases:
        # Parse expected_alerts
        expected = {}
        for rule_id, alert_data in raw.get("expected_alerts", {}).items():
            expected[rule_id] = ExpectedAlert(
                rule_id=rule_id,
                satisfied=alert_data.get("satisfied", False),
                severity=alert_data.get("severity", "INFO")
            )
        
        # Parse traffic_state
        ts = raw["traffic_state"]
        traffic_state = TrafficStateData(
            sector_id=ts.get("sector_id", "DEFAULT"),
            msa=ts.get("msa", 3000),
            aircrafts=ts.get("aircrafts", {}),
            runways=ts.get("runways", {})
        )
        
        # Validar: TCs sin expected_alerts deben ser clean states explícitos
        if not expected:
            desc_lower = raw.get("description", "").lower()
            clean_markers = (
                "clean", "no violation", "no violations", "compliant",
                "safely", "adequately separated", "single aircraft",
                "runway 27l is clear", "runway 27l clear", "runway clear",
                "stopped at c1", "holding short", "without explicit",
                "information only", "standard professional", "clear and concise",
                "just above", "well above", "no altitude violation",
                "no separation to check", "condition met", "correct format",
                " ok", "above 3 nm", "above minimum",
            )
            if not any(m in desc_lower for m in clean_markers):
                print(f"⚠️  ADVERTENCIA: {raw['id']} no tiene expected_alerts y no parece clean state")
                print(f"   Descripción: {raw['description']}")
        
        tc = TestCase(
            id=raw["id"],
            instruction=raw["instruction"],
            callsign=raw["callsign"],
            description=raw["description"],
            traffic_state=traffic_state,
            expected_alerts=expected,
            eval_type=raw.get("eval_type", "native+compiled+generic")
        )
        
        test_cases.append(tc)
    
    print(f"✅ Cargados {len(test_cases)} test cases desde {test_cases_path}")
    print(f"   Con expected_alerts: {sum(1 for tc in test_cases if tc.expected_alerts)}")
    print(f"   Clean states (sin expected): {sum(1 for tc in test_cases if not tc.expected_alerts)}")
    
    return test_cases


def traffic_state_to_alert_system(tc: TestCase):
    """
    Convierte TrafficStateData al formato del Alert_System.
    
    Retorna TrafficState listo para StateManager.
    """
    from Alert_System.models.traffic_state import (
        TrafficState, AircraftState, Position, FlightPhase, RunwayState,
    )

    aircrafts = {}
    for callsign, ac_data in tc.traffic_state.aircrafts.items():
        pos_data = ac_data.get("position", {})
        phase_str = ac_data.get("flight_phase", "CRUISE")
        try:
            phase = FlightPhase(phase_str.lower())
        except ValueError:
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
    for rwy_id, rwy_data in tc.traffic_state.runways.items():
        runways[rwy_id] = RunwayState(
            runway_id=rwy_data.get("runway_id", rwy_id),
            occupied=rwy_data.get("occupied", False),
            occupied_by=rwy_data.get("occupied_by"),
        )

    return TrafficState(
        sector_id=tc.traffic_state.sector_id,
        msa=tc.traffic_state.msa,
        aircrafts=aircrafts,
        runways=runways,
    )