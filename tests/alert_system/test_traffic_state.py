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


class TestNewModelsPhase1:
    """Tests para los nuevos modelos agregados en Fase 1."""

    def test_occupant_type_values(self):
        """Valores del enum OccupantType."""
        from Alert_System.models.traffic_state import OccupantType
        assert OccupantType.AIRCRAFT == "aircraft"
        assert OccupantType.VEHICLE == "vehicle"
        assert OccupantType.UNKNOWN == "unknown"

    def test_phase_transition_creation(self):
        """Crear PhaseTransition válido."""
        from Alert_System.models.traffic_state import PhaseTransition
        transition = PhaseTransition(
            from_phase=FlightPhase.CLIMB,
            to_phase=FlightPhase.CRUISE,
            reason="Reached cruise altitude"
        )
        assert transition.from_phase == FlightPhase.CLIMB
        assert transition.to_phase == FlightPhase.CRUISE
        assert transition.reason == "Reached cruise altitude"
        assert transition.timestamp is not None

    def test_squawk_change_creation(self):
        """Crear SquawkChange válido."""
        from Alert_System.models.traffic_state import SquawkChange
        change = SquawkChange(
            from_squawk="1234",
            to_squawk="5678",
            changed_by="ATC"
        )
        assert change.from_squawk == "1234"
        assert change.to_squawk == "5678"
        assert change.changed_by == "ATC"
        assert change.timestamp is not None

    def test_squawk_change_without_from(self):
        """SquawkChange sin from_squawk (nuevo squawk)."""
        from Alert_System.models.traffic_state import SquawkChange
        change = SquawkChange(
            to_squawk="0000",
            changed_by="PILOT"
        )
        assert change.from_squawk is None
        assert change.to_squawk == "0000"

    def test_aircraft_state_with_phase_history(self):
        """AircraftState con campos de historial de fase."""
        from Alert_System.models.traffic_state import PhaseTransition
        pos = Position(
            latitude=40.0,
            longitude=-74.0,
            altitude=10000,
            heading=180,
            speed=250
        )
        transition = PhaseTransition(
            from_phase=FlightPhase.CLIMB,
            to_phase=FlightPhase.CRUISE,
            reason="Reached cruise"
        )
        aircraft = AircraftState(
            callsign="TEST123",
            position=pos,
            flight_phase=FlightPhase.CRUISE,
            phase_history=[transition],
            previous_phase=FlightPhase.CLIMB,
            phase_transition_timestamp=datetime.utcnow()
        )
        assert len(aircraft.phase_history) == 1
        assert aircraft.previous_phase == FlightPhase.CLIMB
        assert aircraft.phase_transition_timestamp is not None

    def test_aircraft_state_with_squawk_history(self):
        """AircraftState con historial de squawk."""
        from Alert_System.models.traffic_state import SquawkChange
        pos = Position(
            latitude=40.0,
            longitude=-74.0,
            altitude=5000,
            heading=270,
            speed=180
        )
        squawk_change = SquawkChange(
            from_squawk="2100",
            to_squawk="3456",
            changed_by="ATC"
        )
        aircraft = AircraftState(
            callsign="TEST456",
            position=pos,
            flight_phase=FlightPhase.APPROACH,
            squawk_history=[squawk_change],
            squawk_assigned_timestamp=datetime.utcnow()
        )
        assert len(aircraft.squawk_history) == 1
        assert aircraft.squawk_history[0].to_squawk == "3456"
        assert aircraft.squawk_assigned_timestamp is not None

    def test_aircraft_state_defaults_for_new_fields(self):
        """AircraftState tiene defaults para los nuevos campos."""
        pos = Position(
            latitude=40.0,
            longitude=-74.0,
            altitude=10000,
            heading=90,
            speed=300
        )
        aircraft = AircraftState(
            callsign="TEST789",
            position=pos,
            flight_phase=FlightPhase.CRUISE
        )
        assert aircraft.phase_history == []
        assert aircraft.previous_phase is None
        assert aircraft.phase_transition_timestamp is None
        assert aircraft.squawk_history == []
        assert aircraft.squawk_assigned_timestamp is None

    def test_runway_state_with_occupant_type(self):
        """RunwayState con occupant_type."""
        from Alert_System.models.traffic_state import OccupantType
        runway = RunwayState(
            runway_id="09L",
            occupied=True,
            occupied_by="AAL123",
            occupant_type=OccupantType.AIRCRAFT
        )
        assert runway.occupied is True
        assert runway.occupant_type == OccupantType.AIRCRAFT

    def test_runway_state_vehicle_occupant(self):
        """RunwayState con vehículo como ocupante."""
        from Alert_System.models.traffic_state import OccupantType
        runway = RunwayState(
            runway_id="27R",
            occupied=True,
            occupied_by="FIRE_TRUCK_1",
            occupant_type=OccupantType.VEHICLE
        )
        assert runway.occupant_type == OccupantType.VEHICLE

    def test_runway_state_defaults_for_occupant_type(self):
        """RunwayState tiene default None para occupant_type."""
        runway = RunwayState(
            runway_id="18C",
            occupied=False
        )
        assert runway.occupant_type is None

    def test_traffic_state_with_new_fields(self):
        """TrafficState con nuevos campos en AircraftState y RunwayState."""
        from Alert_System.models.traffic_state import PhaseTransition, SquawkChange, OccupantType

        pos = Position(
            latitude=40.0,
            longitude=-74.0,
            altitude=8000,
            heading=180,
            speed=220
        )
        transition = PhaseTransition(
            from_phase=FlightPhase.DESCENT,
            to_phase=FlightPhase.APPROACH
        )
        aircraft = AircraftState(
            callsign="AAL123",
            position=pos,
            flight_phase=FlightPhase.APPROACH,
            phase_history=[transition],
            previous_phase=FlightPhase.DESCENT
        )

        runway = RunwayState(
            runway_id="09L",
            occupied=True,
            occupied_by="UAL456",
            occupant_type=OccupantType.AIRCRAFT
        )

        traffic = TrafficState(
            sector_id="K JFK",
            aircrafts={"AAL123": aircraft},
            runways={"09L": runway},
            msa=3000
        )

        assert traffic.get_aircraft("AAL123") is not None
        assert traffic.get_aircraft("AAL123").phase_history[0].to_phase == FlightPhase.APPROACH
        assert traffic.get_runway("09L").occupant_type == OccupantType.AIRCRAFT
