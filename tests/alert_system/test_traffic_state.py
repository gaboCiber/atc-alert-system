"""Tests para modelos de estado de tráfico aéreo."""

from datetime import datetime

import pytest

from Alert_System.models.traffic_state import (
    AircraftState,
    Clearances,
    FlightPhase,
    Position,
    RunwayOperationMode,
    RunwayState,
    TrafficState,
    WakeTurbulenceCategory,
)


class TestPosition:
    """Tests para el modelo Position."""

    def test_position_creation(self):
        """Crear una posición válida."""
        pos = Position(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=30000,
            heading=90,
            speed=450,
            vertical_rate=-500,
        )
        assert pos.latitude == 40.7128
        assert pos.altitude == 30000
        assert pos.heading == 90

    def test_position_heading_validation(self):
        """Validación de heading entre 0 y 360."""
        with pytest.raises(ValueError):
            Position(
                latitude=0,
                longitude=0,
                altitude=0,
                heading=400,  # Inválido
                speed=0,
            )

    def test_position_speed_validation(self):
        """Validación de speed >= 0."""
        with pytest.raises(ValueError):
            Position(
                latitude=0,
                longitude=0,
                altitude=0,
                heading=0,
                speed=-10,  # Inválido
            )


class TestClearances:
    """Tests para el modelo Clearances."""

    def test_clearances_default(self):
        """Clearances por defecto vacíos."""
        clearances = Clearances()
        assert clearances.altitude_assigned is None
        assert clearances.runway_assigned is None

    def test_clearances_with_values(self):
        """Clearances con valores asignados."""
        clearances = Clearances(
            altitude_assigned=24000,
            heading_assigned=270,
            runway_assigned="27R",
            squawk="1234",
        )
        assert clearances.altitude_assigned == 24000
        assert clearances.heading_assigned == 270
        assert clearances.runway_assigned == "27R"
        #assert clearances.squawk == "1234"


class TestAircraftState:
    """Tests para el modelo AircraftState."""

    def test_aircraft_state_creation(self):
        """Crear estado de aeronave válido."""
        position = Position(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=30000,
            heading=90,
            speed=450,
        )
        
        aircraft = AircraftState(
            callsign="AAL123",
            position=position,
            flight_phase=FlightPhase.CRUISE,
            aircraft_type="B738",
        )
        
        assert aircraft.callsign == "AAL123"
        assert aircraft.flight_phase == FlightPhase.CRUISE
        assert aircraft.wake_turbulence == WakeTurbulenceCategory.MEDIUM
        assert not aircraft.is_emergency

    def test_aircraft_state_with_clearances(self):
        """Estado de aeronave con clearances."""
        position = Position(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=30000,
            heading=90,
            speed=450,
        )
        
        clearances = Clearances(
            altitude_assigned=24000,
            heading_assigned=270,
        )
        
        aircraft = AircraftState(
            callsign="UAL456",
            position=position,
            flight_phase=FlightPhase.CRUISE,
            clearances=clearances,
        )
        
        assert aircraft.clearances.altitude_assigned == 24000

    def test_aircraft_state_serialization(self):
        """Serialización a JSON."""
        position = Position(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=30000,
            heading=90,
            speed=450,
        )
        
        aircraft = AircraftState(
            callsign="DAL789",
            position=position,
            flight_phase=FlightPhase.DESCENT,
        )
        
        json_str = aircraft.model_dump_json()
        assert "DAL789" in json_str
        assert "descent" in json_str


class TestRunwayState:
    """Tests para el modelo RunwayState."""

    def test_runway_empty(self):
        """Pista vacía por defecto."""
        runway = RunwayState(runway_id="09L")
        assert not runway.occupied
        assert runway.occupied_by is None
        assert runway.operation_mode == RunwayOperationMode.MIXED

    def test_runway_occupied(self):
        """Pista ocupada."""
        runway = RunwayState(
            runway_id="27R",
            occupied=True,
            occupied_by="AAL123",
            operation_mode=RunwayOperationMode.LANDING,
        )
        assert runway.occupied
        assert runway.occupied_by == "AAL123"

    def test_runway_holding_queue(self):
        """Cola de holding short."""
        runway = RunwayState(runway_id="09R")
        runway.holding_short.append("UAL456")
        runway.holding_short.append("DAL789")
        
        assert len(runway.holding_short) == 2
        assert runway.holding_short[0] == "UAL456"


class TestTrafficState:
    """Tests para el modelo TrafficState."""

    def test_traffic_state_empty(self):
        """Estado vacío."""
        state = TrafficState(sector_id="JFK_APP")
        assert state.sector_id == "JFK_APP"
        assert len(state.aircrafts) == 0
        assert len(state.runways) == 0

    def test_traffic_state_add_aircraft(self):
        """Añadir aeronaves."""
        state = TrafficState(sector_id="LAX_TWR")
        
        position = Position(
            latitude=34.0522,
            longitude=-118.2437,
            altitude=5000,
            heading=180,
            speed=150,
        )
        
        aircraft = AircraftState(
            callsign="AAL123",
            position=position,
            flight_phase=FlightPhase.APPROACH,
        )
        
        state.add_aircraft(aircraft)
        
        assert len(state.aircrafts) == 1
        assert state.get_aircraft("AAL123") is not None
        assert state.get_aircraft("aal123") is not None  # Case insensitive

    def test_traffic_state_remove_aircraft(self):
        """Remover aeronaves."""
        state = TrafficState(sector_id="MIA_APP")
        
        position = Position(
            latitude=25.7617,
            longitude=-80.1918,
            altitude=10000,
            heading=90,
            speed=250,
        )
        
        aircraft = AircraftState(
            callsign="UAL456",
            position=position,
            flight_phase=FlightPhase.CLIMB,
        )
        
        state.add_aircraft(aircraft)
        assert len(state.aircrafts) == 1
        
        state.remove_aircraft("UAL456")
        assert len(state.aircrafts) == 0
        assert state.get_aircraft("UAL456") is None

    def test_traffic_state_add_runway(self):
        """Añadir pistas usando add_runway."""
        state = TrafficState(sector_id="ORD_TWR")
        
        runway = RunwayState(
            runway_id="14R",
            occupied=True,
            occupied_by="DAL123",
        )
        
        state.add_runway(runway)
        
        assert len(state.runways) == 1
        assert state.get_runway("14R").occupied_by == "DAL123"
        assert state.get_runway("14r") is not None  # Case insensitive

    def test_traffic_state_get_runway(self):
        """Obtener pistas usando get_runway."""
        state = TrafficState(sector_id="ORD_TWR")
        
        runway = RunwayState(runway_id="27L")
        state.add_runway(runway)
        
        # Case insensitive
        assert state.get_runway("27L") is not None
        assert state.get_runway("27l") is not None
        assert state.get_runway("09R") is None  # No existe

    def test_traffic_state_remove_runway(self):
        """Remover pistas usando remove_runway."""
        state = TrafficState(sector_id="MIA_TWR")
        
        runway = RunwayState(runway_id="09L")
        state.add_runway(runway)
        assert len(state.runways) == 1
        
        state.remove_runway("09L")
        assert len(state.runways) == 0
        assert state.get_runway("09L") is None

    def test_traffic_state_get_nearby(self):
        """Obtener aeronaves cercanas."""
        state = TrafficState(sector_id="TEST")
        
        # Añadir aeronave 1
        ac1 = AircraftState(
            callsign="AC1",
            position=Position(
                latitude=40.0,
                longitude=-74.0,
                altitude=30000,
                heading=90,
                speed=450,
            ),
            flight_phase=FlightPhase.CRUISE,
        )
        state.add_aircraft(ac1)
        
        # Añadir aeronave 2 cercana (aprox 10 NM al norte)
        ac2 = AircraftState(
            callsign="AC2",
            position=Position(
                latitude=40.17,  # ~10 NM north
                longitude=-74.0,
                altitude=30000,
                heading=90,
                speed=450,
            ),
            flight_phase=FlightPhase.CRUISE,
        )
        state.add_aircraft(ac2)
        
        # Añadir aeronave 3 lejana (aprox 100 NM al norte)
        ac3 = AircraftState(
            callsign="AC3",
            position=Position(
                latitude=41.67,  # ~100 NM north
                longitude=-74.0,
                altitude=30000,
                heading=90,
                speed=450,
            ),
            flight_phase=FlightPhase.CRUISE,
        )
        state.add_aircraft(ac3)
        
        nearby = state.get_nearby_aircraft("AC1", max_distance_nm=20)
        assert len(nearby) == 1
        assert nearby[0].callsign == "AC2"


class TestFlightPhase:
    """Tests para el enum FlightPhase."""

    def test_flight_phase_values(self):
        """Valores del enum."""
        assert FlightPhase.GROUND == "ground"
        assert FlightPhase.TAKEOFF == "takeoff"
        assert FlightPhase.CLIMB == "climb"
        assert FlightPhase.CRUISE == "cruise"
        assert FlightPhase.DESCENT == "descent"
        assert FlightPhase.APPROACH == "approach"
        assert FlightPhase.LANDING == "landing"


class TestWakeTurbulence:
    """Tests para el enum WakeTurbulenceCategory."""

    def test_wake_turbulence_values(self):
        """Valores del enum."""
        assert WakeTurbulenceCategory.LIGHT == "L"
        assert WakeTurbulenceCategory.MEDIUM == "M"
        assert WakeTurbulenceCategory.HEAVY == "H"
        assert WakeTurbulenceCategory.SUPER == "S"
