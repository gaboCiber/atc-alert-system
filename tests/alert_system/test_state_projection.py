"""Tests para State Projection y State Manager."""

import pytest
from datetime import datetime

from Alert_System.core.state_manager import StateManager, StateTransaction, Transaction
from Alert_System.core.state_projection import (
    ProjectedSeparation,
    ProjectedState,
    ProjectedTrajectory,
    ProjectedWaypoint,
    StateProjector,
)
from Alert_System.models.instruction import InstructionType, ParsedInstruction, Speaker
from Alert_System.models.traffic_state import (
    AircraftState,
    FlightPhase,
    Position,
    RunwayState,
    TrafficState,
)


class TestProjectedWaypoint:
    """Tests para ProjectedWaypoint."""

    def test_waypoint_creation(self):
        """Crear waypoint proyectado."""
        waypoint = ProjectedWaypoint(
            latitude=40.7128,
            longitude=-74.0060,
            altitude=30000,
            estimated_time=60,
            speed=450,
            vertical_rate=-1000,
        )
        assert waypoint.altitude == 30000
        assert waypoint.estimated_time == 60


class TestProjectedTrajectory:
    """Tests para ProjectedTrajectory."""

    def test_trajectory_creation(self):
        """Crear trayectoria proyectada."""
        waypoints = [
            ProjectedWaypoint(40.0, -74.0, 30000, 60, 450, -1000),
            ProjectedWaypoint(40.1, -74.1, 29000, 120, 450, -1000),
        ]
        trajectory = ProjectedTrajectory(
            callsign="AAL123",
            waypoints=waypoints,
            estimated_duration_sec=120,
            final_phase=FlightPhase.DESCENT,
        )
        assert trajectory.callsign == "AAL123"
        assert len(trajectory.waypoints) == 2
        assert trajectory.final_phase == FlightPhase.DESCENT


class TestProjectedSeparation:
    """Tests para ProjectedSeparation."""

    def test_separation_creation(self):
        """Crear separación proyectada."""
        separation = ProjectedSeparation(
            aircraft_1="AAL123",
            aircraft_2="UAL456",
            vertical_separation_ft=1500,
            horizontal_separation_nm=8.5,
            time_to_conflict=None,
            conflict_predicted=False,
        )
        assert not separation.conflict_predicted
        assert separation.vertical_separation_ft == 1500

    def test_conflict_detection(self):
        """Separación con conflicto."""
        separation = ProjectedSeparation(
            aircraft_1="AAL123",
            aircraft_2="UAL456",
            vertical_separation_ft=800,  # Menor a 1000ft
            horizontal_separation_nm=3.2,  # Menor a 5NM
            time_to_conflict=120,
            conflict_predicted=True,
        )
        assert separation.conflict_predicted
        assert separation.time_to_conflict == 120


class TestProjectedState:
    """Tests para ProjectedState."""

    @pytest.fixture
    def basic_projected_state(self):
        """Fixture con estado proyectado básico."""
        traffic = TrafficState(sector_id="TEST")
        instruction = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 24000},
        )
        return ProjectedState(
            traffic_state=traffic,
            source_instruction=instruction,
        )

    def test_projected_state_creation(self, basic_projected_state):
        """Crear estado proyectado."""
        assert basic_projected_state.source_instruction.callsign == "AAL123"
        assert basic_projected_state.is_valid_projection

    def test_has_conflicts_false(self, basic_projected_state):
        """Sin conflictos."""
        assert not basic_projected_state.has_conflicts()

    def test_has_conflicts_true(self, basic_projected_state):
        """Con conflictos."""
        basic_projected_state.projected_separations["AAL123"] = [
            ProjectedSeparation(
                aircraft_1="AAL123",
                aircraft_2="UAL456",
                vertical_separation_ft=800,
                horizontal_separation_nm=3.0,
                conflict_predicted=True,
            ),
        ]
        assert basic_projected_state.has_conflicts()

    def test_get_conflicts(self, basic_projected_state):
        """Obtener conflictos."""
        basic_projected_state.projected_separations["AAL123"] = [
            ProjectedSeparation(
                aircraft_1="AAL123",
                aircraft_2="UAL456",
                vertical_separation_ft=800,
                horizontal_separation_nm=3.0,
                conflict_predicted=True,
            ),
            ProjectedSeparation(
                aircraft_1="AAL123",
                aircraft_2="DAL789",
                vertical_separation_ft=2000,
                horizontal_separation_nm=10.0,
                conflict_predicted=False,
            ),
        ]
        conflicts = basic_projected_state.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].aircraft_2 == "UAL456"


class TestStateProjector:
    """Tests para StateProjector."""

    @pytest.fixture
    def traffic_state_with_aircraft(self):
        """Fixture con estado de tráfico."""
        state = TrafficState(sector_id="JFK_APP", msa=5000)
        
        aircraft = AircraftState(
            callsign="AAL123",
            position=Position(
                latitude=40.7128,
                longitude=-74.0060,
                altitude=30000,
                heading=90,
                speed=450,
            ),
            flight_phase=FlightPhase.CRUISE,
        )
        state.add_aircraft(aircraft)
        
        return state

    def test_projector_creation(self):
        """Crear proyector."""
        projector = StateProjector()
        assert projector is not None

    def test_create_projection_descent(self, traffic_state_with_aircraft):
        """Proyección de descenso."""
        projector = StateProjector()
        
        instruction = ParsedInstruction(
            raw_text="AAL123 descend to flight level two four zero",
            normalized_text="AAL123 descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 24000, "flight_level": 240},
        )
        
        projected = projector.create_projection(
            traffic_state_with_aircraft,
            instruction,
            projection_minutes=5,
        )
        
        assert projected.is_valid_projection
        assert projected.source_instruction.callsign == "AAL123"
        
        # Verificar que la aeronave tiene la nueva altitud
        aircraft = projected.get_aircraft("AAL123")
        assert aircraft.position.altitude == 24000
        assert aircraft.clearances.altitude_assigned == 24000
        
        # Verificar que se generó trayectoria
        trajectory = projected.get_trajectory("AAL123")
        assert trajectory is not None
        assert len(trajectory.waypoints) == 5  # 5 minutos = 5 waypoints

    def test_create_projection_no_callsign(self, traffic_state_with_aircraft):
        """Error si no hay callsign."""
        projector = StateProjector()
        
        instruction = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        projected = projector.create_projection(
            traffic_state_with_aircraft,
            instruction,
        )
        
        assert not projected.is_valid_projection
        assert "No callsign" in projected.projection_errors[0]

    def test_create_projection_aircraft_not_found(self, traffic_state_with_aircraft):
        """Error si aeronave no existe."""
        projector = StateProjector()
        
        instruction = ParsedInstruction(
            raw_text="UNKNOWN descend to FL240",
            normalized_text="UNKNOWN descend to FL240",
            speaker=Speaker.ATCO,
            callsign="UNKNOWN",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        projected = projector.create_projection(
            traffic_state_with_aircraft,
            instruction,
        )
        
        assert not projected.is_valid_projection
        assert "not found" in projected.projection_errors[0]

    def test_projection_climb(self, traffic_state_with_aircraft):
        """Proyección de ascenso."""
        projector = StateProjector()
        
        # Cambiar aeronave a altitud baja
        ac = traffic_state_with_aircraft.get_aircraft("AAL123")
        ac.position.altitude = 10000
        ac.flight_phase = FlightPhase.CLIMB
        
        instruction = ParsedInstruction(
            raw_text="AAL123 climb to FL300",
            normalized_text="AAL123 climb to FL300",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 30000},
        )
        
        projected = projector.create_projection(
            traffic_state_with_aircraft,
            instruction,
        )
        
        aircraft = projected.get_aircraft("AAL123")
        assert aircraft.position.altitude == 30000

    def test_projection_heading_change(self, traffic_state_with_aircraft):
        """Proyección de cambio de rumbo."""
        projector = StateProjector()
        
        instruction = ParsedInstruction(
            raw_text="AAL123 turn left heading 270",
            normalized_text="AAL123 turn left heading 270",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
            parameters={"heading": 270},
        )
        
        projected = projector.create_projection(
            traffic_state_with_aircraft,
            instruction,
        )
        
        aircraft = projected.get_aircraft("AAL123")
        assert aircraft.position.heading == 270
        assert aircraft.clearances.heading_assigned == 270

    def test_estimate_time_to_altitude(self):
        """Estimar tiempo a altitud."""
        projector = StateProjector()
        
        time = projector.estimate_time_to_altitude(
            current_altitude=30000,
            target_altitude=24000,
            instruction_type=InstructionType.DESCENT,
        )
        
        # 6000ft a 1000ft/min = 6 minutos
        assert time == 6.0

    def test_estimate_time_same_altitude(self):
        """Tiempo cero si misma altitud."""
        projector = StateProjector()
        
        time = projector.estimate_time_to_altitude(
            current_altitude=30000,
            target_altitude=30000,
            instruction_type=InstructionType.DESCENT,
        )
        
        assert time == 0.0


class TestStateManager:
    """Tests para StateManager."""

    @pytest.fixture
    def state_manager(self):
        """Fixture con StateManager inicializado."""
        initial_state = TrafficState(sector_id="TEST", msa=5000)
        return StateManager(initial_state)

    @pytest.fixture
    def projected_state(self):
        """Fixture con proyección válida."""
        traffic = TrafficState(sector_id="TEST", msa=5000)
        instruction = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        return ProjectedState(
            traffic_state=traffic,
            source_instruction=instruction,
            is_valid_projection=True,
        )

    def test_state_manager_creation(self, state_manager):
        """Crear StateManager."""
        assert state_manager.sector_id == "TEST"
        assert state_manager.current_state is not None

    def test_propose_change(self, state_manager, projected_state):
        """Proponer cambio."""
        transaction = state_manager.propose_change(projected_state)
        
        assert transaction is not None
        assert transaction.status == "PENDING"
        assert state_manager.has_pending_transaction()

    def test_commit(self, state_manager, projected_state):
        """Commit de cambio."""
        transaction = state_manager.propose_change(projected_state)
        
        result = state_manager.commit(transaction.transaction_id)
        
        assert result is True
        assert transaction.status == "COMMITTED"
        assert not state_manager.has_pending_transaction()

    def test_commit_force(self, state_manager, projected_state):
        """Commit forzado con alertas."""
        projected_state.has_alerts = True
        transaction = state_manager.propose_change(projected_state)
        transaction.has_alerts = True
        
        # Sin force, no debe permitir
        result = state_manager.commit(transaction.transaction_id)
        assert not result
        
        # Con force, debe permitir
        result = state_manager.commit(
            transaction.transaction_id,
            force=True,
            reason="ATCO override"
        )
        assert result
        assert transaction.force_committed
        assert transaction.atco_reason == "ATCO override"

    def test_rollback(self, state_manager, projected_state):
        """Rollback de cambio."""
        transaction = state_manager.propose_change(projected_state)
        
        result = state_manager.rollback(
            transaction.transaction_id,
            reason="Unsafe"
        )
        
        assert result is True
        assert transaction.status == "ROLLBACK"
        assert transaction.atco_reason == "Unsafe"
        assert not state_manager.has_pending_transaction()

    def test_undo_last_commit(self, state_manager, projected_state):
        """Deshacer último commit."""
        # Guardar estado inicial
        initial_sector = state_manager.sector_id
        
        # Hacer commit
        transaction = state_manager.propose_change(projected_state)
        state_manager.commit(transaction.transaction_id)
        
        # Deshacer
        result = state_manager.undo_last_commit()
        
        assert result is True

    def test_get_pending_transaction(self, state_manager, projected_state):
        """Obtener transacción pendiente."""
        transaction = state_manager.propose_change(projected_state)
        
        pending = state_manager.get_pending_transaction()
        
        assert pending is not None
        assert pending.transaction_id == transaction.transaction_id

    def test_get_transaction_history(self, state_manager, projected_state):
        """Obtener historial de transacciones."""
        # Crear varias transacciones
        txn1 = state_manager.propose_change(projected_state)
        state_manager.commit(txn1.transaction_id)
        
        txn2 = state_manager.propose_change(projected_state)
        state_manager.rollback(txn2.transaction_id)
        
        history = state_manager.get_transaction_history()
        
        assert len(history) == 2


class TestTransactionContextManager:
    """Tests para Transaction context manager."""

    @pytest.fixture
    def state_manager(self):
        """Fixture con StateManager."""
        return StateManager(TrafficState(sector_id="TEST"))

    @pytest.fixture
    def projected_state(self):
        """Fixture con proyección."""
        traffic = TrafficState(sector_id="TEST")
        instruction = ParsedInstruction(
            raw_text="test",
            normalized_text="test",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
        )
        return ProjectedState(
            traffic_state=traffic,
            source_instruction=instruction,
            is_valid_projection=True,
        )

    def test_transaction_context_success(self, state_manager, projected_state):
        """Context manager exitoso."""
        with Transaction(state_manager, projected_state, auto_commit=True) as txn:
            assert txn.status == "PENDING"
            # No hay excepción, debe hacer commit
        
        assert txn.status == "COMMITTED"

    def test_transaction_context_exception(self, state_manager, projected_state):
        """Context manager con excepción."""
        try:
            with Transaction(state_manager, projected_state) as txn:
                assert txn.status == "PENDING"
                raise ValueError("Test error")
        except ValueError:
            pass  # Esperamos que la excepción se propague
        
        # Debe haber hecho rollback
        assert txn.status == "ROLLBACK"

    def test_transaction_context_no_auto_commit(self, state_manager, projected_state):
        """Context manager sin auto-commit."""
        with Transaction(state_manager, projected_state, auto_commit=False) as txn:
            pass
        
        # Sin auto_commit, debe quedar pendiente
        assert txn.status == "PENDING"
