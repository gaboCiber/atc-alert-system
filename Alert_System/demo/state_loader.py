"""Carga un TrafficState desde un archivo JSON para el demo CLI."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from Alert_System.models.traffic_state import (
    AircraftState,
    Clearances,
    FlightPhase,
    OccupantType,
    PhaseTransition,
    Position,
    RunwayOperationMode,
    RunwayState,
    SquawkChange,
    TrafficState,
    WakeTurbulenceCategory,
)


class TrafficStateLoader:
    """Carga un TrafficState desde un archivo JSON."""

    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path

    @classmethod
    def from_file(cls, path: str) -> TrafficState:
        """Carga un TrafficState desde un archivo JSON."""
        loader = cls(path)
        return loader.load()

    def load(self) -> TrafficState:
        """Carga y retorna el TrafficState desde el archivo JSON."""
        if not self.json_path:
            raise ValueError("No se proporcionó path al archivo JSON")

        path = Path(self.json_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.json_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._build_traffic_state(data)

    def _build_traffic_state(self, data: Dict[str, Any]) -> TrafficState:
        """Construye un TrafficState a partir del dict cargado del JSON."""
        # Sector
        sector_id = data.get("sector_id", "DEFAULT")
        msa = data.get("msa")
        qnh = data.get("qnh")
        wind = data.get("wind")

        state = TrafficState(
            sector_id=sector_id,
            msa=msa,
            qnh=qnh,
            wind=wind,
        )

        # Aeronaves
        for callsign, ac_data in data.get("aircrafts", {}).items():
            aircraft = self._build_aircraft(ac_data)
            state.add_aircraft(aircraft)

        # Pistas
        for rw_id, rw_data in data.get("runways", {}).items():
            runway = self._build_runway(rw_data)
            state.add_runway(runway)

        return state

    def _build_aircraft(self, data: Dict[str, Any]) -> AircraftState:
        """Construye un AircraftState desde dict."""
        pos_data = data.get("position", {})
        position = Position(
            latitude=pos_data.get("latitude", 0.0),
            longitude=pos_data.get("longitude", 0.0),
            altitude=pos_data.get("altitude", 0),
            heading=pos_data.get("heading", 0),
            speed=pos_data.get("speed", 0),
            vertical_rate=pos_data.get("vertical_rate"),
        )

        # Clearances
        clearances_data = data.get("clearances", {})
        clearances = Clearances(
            altitude_assigned=clearances_data.get("altitude_assigned"),
            heading_assigned=clearances_data.get("heading_assigned"),
            runway_assigned=clearances_data.get("runway_assigned"),
            route=clearances_data.get("route"),
            squawk=clearances_data.get("squawk"),
            speed_assigned=clearances_data.get("speed_assigned"),
        )

        # Fase de vuelo
        phase_str = data.get("flight_phase", "cruise")
        try:
            flight_phase = FlightPhase(phase_str)
        except ValueError:
            flight_phase = FlightPhase.CRUISE

        # Previous phase (Fase 1)
        prev_phase_str = data.get("previous_phase")
        previous_phase = None
        if prev_phase_str:
            try:
                previous_phase = FlightPhase(prev_phase_str)
            except ValueError:
                pass

        # Phase transition timestamp (Fase 1)
        phase_transition_ts = None
        if data.get("phase_transition_timestamp"):
            try:
                phase_transition_ts = datetime.fromisoformat(data["phase_transition_timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Phase history (Fase 1)
        phase_history = []
        for transition_data in data.get("phase_history", []):
            try:
                from_phase = FlightPhase(transition_data.get("from_phase", "cruise"))
                to_phase = FlightPhase(transition_data.get("to_phase", "cruise"))
                timestamp = datetime.utcnow()
                if transition_data.get("timestamp"):
                    try:
                        timestamp = datetime.fromisoformat(transition_data["timestamp"].replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass
                phase_history.append(PhaseTransition(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    timestamp=timestamp,
                    reason=transition_data.get("reason")
                ))
            except (ValueError, KeyError):
                continue

        # Squawk history (Fase 1)
        squawk_history = []
        for squawk_data in data.get("squawk_history", []):
            try:
                timestamp = datetime.utcnow()
                if squawk_data.get("timestamp"):
                    try:
                        timestamp = datetime.fromisoformat(squawk_data["timestamp"].replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass
                squawk_history.append(SquawkChange(
                    from_squawk=squawk_data.get("from_squawk"),
                    to_squawk=squawk_data.get("to_squawk", "0000"),
                    timestamp=timestamp,
                    changed_by=squawk_data.get("changed_by")
                ))
            except (KeyError, ValueError):
                continue

        # Squawk assigned timestamp (Fase 1)
        squawk_assigned_ts = None
        if data.get("squawk_assigned_timestamp"):
            try:
                squawk_assigned_ts = datetime.fromisoformat(data["squawk_assigned_timestamp"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Wake turbulence
        wake_str = data.get("wake_turbulence", "M")
        try:
            wake_turbulence = WakeTurbulenceCategory(wake_str)
        except ValueError:
            wake_turbulence = WakeTurbulenceCategory.MEDIUM

        return AircraftState(
            callsign=data.get("callsign", "UNKNOWN"),
            position=position,
            flight_phase=flight_phase,
            clearances=clearances,
            restrictions=data.get("restrictions", []),
            wake_turbulence=wake_turbulence,
            aircraft_type=data.get("aircraft_type"),
            is_emergency=data.get("is_emergency", False),
            emergency_type=data.get("emergency_type"),
            # Nuevos campos Fase 1
            phase_history=phase_history,
            previous_phase=previous_phase,
            phase_transition_timestamp=phase_transition_ts,
            squawk_history=squawk_history,
            squawk_assigned_timestamp=squawk_assigned_ts,
        )

    def _build_runway(self, data: Dict[str, Any]) -> RunwayState:
        """Construye un RunwayState desde dict."""
        mode_str = data.get("operation_mode", "mixed")
        try:
            operation_mode = RunwayOperationMode(mode_str)
        except ValueError:
            operation_mode = RunwayOperationMode.MIXED

        # Occupant type (Fase 1)
        occupant_type = None
        if data.get("occupant_type"):
            try:
                occupant_type = OccupantType(data.get("occupant_type"))
            except ValueError:
                pass

        return RunwayState(
            runway_id=data.get("runway_id", "UNKNOWN"),
            occupied=data.get("occupied", False),
            occupied_by=data.get("occupied_by"),
            operation_mode=operation_mode,
            holding_short=data.get("holding_short", []),
            landing_queue=data.get("landing_queue", []),
            closed_until=data.get("closed_until"),
            closure_reason=data.get("closure_reason"),
            # Nuevo campo Fase 1
            occupant_type=occupant_type,
        )

    def save(self, state: TrafficState, path: str) -> None:
        """Guarda un TrafficState a un archivo JSON."""
        # Usar el serializador de Pydantic
        data = state.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
