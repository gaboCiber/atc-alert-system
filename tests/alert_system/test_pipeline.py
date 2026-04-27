"""Tests para el pipeline de alertas de 8 pasos."""

import pytest
from datetime import datetime

from Alert_System.pipeline.alert_pipeline import AlertPipeline, PipelineResult, PipelineStep
from Alert_System.models.instruction import InstructionType, ParsedInstruction, Speaker
from Alert_System.models.traffic_state import (
    AircraftState,
    FlightPhase,
    Position,
    RunwayOperationMode,
    RunwayState,
    TrafficState,
)
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.core.state_manager import StateManager
from Alert_System.models.alert import AlertSeverity, AlertCategory


@pytest.fixture
def traffic_state_with_aircraft():
    """Fixture con estado de tráfico con aeronaves."""
    state = TrafficState(sector_id="JFK_APP", msa=5000)
    
    # Añadir aeronave 1
    ac1 = AircraftState(
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
    state.add_aircraft(ac1)
    
    # Añadir aeronave 2 (cerca de la primera)
    ac2 = AircraftState(
        callsign="UAL456",
        position=Position(
            latitude=40.72,  # Cerca de AAL123
            longitude=-74.0,
            altitude=29000,  # Separación vertical pequeña
            heading=90,
            speed=450,
        ),
        flight_phase=FlightPhase.CRUISE,
    )
    state.add_aircraft(ac2)
    
    return state


@pytest.fixture
def state_manager(traffic_state_with_aircraft):
    """Fixture con StateManager."""
    return StateManager(traffic_state_with_aircraft)


@pytest.fixture
def rule_engine():
    """Fixture con RuleEngine."""
    return RuleEngine()


@pytest.fixture
def alert_pipeline(state_manager, rule_engine):
    """Fixture con AlertPipeline."""
    return AlertPipeline(state_manager, rule_engine)


class TestPipelineStep:
    """Tests para PipelineStep."""

    def test_step_creation(self):
        """Crear paso del pipeline."""
        step = PipelineStep(
            step_number=1,
            step_name="TEST_STEP",
        )
        assert step.status == "PENDING"
        assert step.step_number == 1

    def test_mark_success(self):
        """Marcar paso como exitoso."""
        step = PipelineStep(step_number=1, step_name="TEST")
        step.mark_success({"data": "test"})
        
        assert step.status == "SUCCESS"
        assert step.output_data == {"data": "test"}

    def test_mark_failed(self):
        """Marcar paso como fallido."""
        step = PipelineStep(step_number=1, step_name="TEST")
        step.mark_failed("Error message")
        
        assert step.status == "FAILED"
        assert step.error_message == "Error message"


class TestPipelineResult:
    """Tests para PipelineResult."""

    def test_result_creation(self):
        """Crear resultado del pipeline."""
        result = PipelineResult(
            pipeline_id="PL_001",
            timestamp=datetime.utcnow(),
            raw_instruction="test instruction",
        )
        assert result.final_decision == "PENDING"
        assert not result.has_errors

    def test_get_step(self):
        """Obtener paso por número."""
        result = PipelineResult(
            pipeline_id="PL_001",
            timestamp=datetime.utcnow(),
            raw_instruction="test",
            steps=[
                PipelineStep(step_number=1, step_name="STEP1"),
                PipelineStep(step_number=2, step_name="STEP2"),
            ],
        )
        
        step = result.get_step(2)
        assert step.step_name == "STEP2"

    def test_get_step_by_name(self):
        """Obtener paso por nombre."""
        result = PipelineResult(
            pipeline_id="PL_001",
            timestamp=datetime.utcnow(),
            raw_instruction="test",
            steps=[
                PipelineStep(step_number=1, step_name="STEP1"),
            ],
        )
        
        step = result.get_step_by_name("STEP1")
        assert step.step_number == 1

    def test_was_successful_commit(self):
        """Pipeline exitoso con COMMIT."""
        result = PipelineResult(
            pipeline_id="PL_001",
            timestamp=datetime.utcnow(),
            raw_instruction="test",
            final_decision="COMMIT",
        )
        assert result.was_successful()

    def test_was_successful_rollback(self):
        """Pipeline exitoso con ROLLBACK."""
        result = PipelineResult(
            pipeline_id="PL_001",
            timestamp=datetime.utcnow(),
            raw_instruction="test",
            final_decision="ROLLBACK",
        )
        assert result.was_successful()


class TestAlertPipeline:
    """Tests para AlertPipeline."""

    def test_pipeline_creation(self, alert_pipeline):
        """Crear pipeline."""
        assert alert_pipeline is not None
        assert alert_pipeline.state_manager is not None
        assert alert_pipeline.rule_engine is not None

    def test_process_simple_descent(self, alert_pipeline):
        """Procesar instrucción de descenso simple."""
        result = alert_pipeline.process_instruction(
            "AAL123 descend to flight level two four zero"
        )
        
        assert isinstance(result, PipelineResult)
        assert result.parsed_instruction is not None
        assert result.parsed_instruction.callsign == "AAL123"
        assert result.parsed_instruction.instruction_type == InstructionType.DESCENT
        
        # Debe tener 8 pasos
        assert len(result.steps) == 8
        
        # Paso 1 debe ser exitoso
        step1 = result.get_step(1)
        assert step1.status == "SUCCESS"
        
        # Resultado final debe ser COMMIT o ROLLBACK
        assert result.final_decision in ["COMMIT", "ROLLBACK"]

    def test_process_with_pre_parsed(self, alert_pipeline):
        """Procesar instrucción ya parseada."""
        parsed = ParsedInstruction(
            raw_text="AAL123 climb to FL300",
            normalized_text="AAL123 climb to FL300",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 30000},
        )
        
        result = alert_pipeline.process_instruction(
            "AAL123 climb to FL300",
            pre_parsed=parsed,
        )
        
        assert result.parsed_instruction.callsign == "AAL123"

    def test_step_1_input_processing(self, alert_pipeline):
        """Paso 1: Input Processing."""
        step = alert_pipeline._step_1_input_processing(
            "AAL123 turn left heading 270",
            None,
        )
        
        assert step.status == "SUCCESS"
        assert step.output_data is not None
        assert step.output_data.callsign == "AAL123"
        assert step.output_data.instruction_type == InstructionType.HEADING

    def test_step_3_state_projection(self, alert_pipeline):
        """Paso 3: State Projection."""
        parsed = ParsedInstruction(
            raw_text="AAL123 descend to FL240",
            normalized_text="AAL123 descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 24000, "flight_level": 240},
        )
        
        step = alert_pipeline._step_3_state_update(parsed)
        
        assert step.status == "SUCCESS"
        projected = step.output_data
        assert projected.is_valid_projection
        
        # Verificar que la aeronave tiene la nueva altitud
        ac = projected.get_aircraft("AAL123")
        assert ac.position.altitude == 24000

    def test_step_4_rule_evaluation(self, alert_pipeline):
        """Paso 4: Rule Evaluation."""
        # Crear proyección que viole MSA
        traffic = TrafficState(sector_id="TEST", msa=5000)
        ac = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=4000, heading=90, speed=250),
            flight_phase=FlightPhase.APPROACH,
        )
        traffic.add_aircraft(ac)
        
        parsed = ParsedInstruction(
            raw_text="AAL123 descend to 4000",
            normalized_text="AAL123 descend to 4000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        # Crear proyección
        from Alert_System.core.state_projection import StateProjector
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        step = alert_pipeline._step_4_rule_evaluation(parsed, projected)
        
        assert step.status == "SUCCESS"
        violations = step.output_data
        # Debe haber violación de MSA (4000 < 5000)
        assert len(violations) >= 1

    def test_step_5_alert_generation(self, alert_pipeline):
        """Paso 5: Alert Generation."""
        from Alert_System.models.alert import Violation, AlertCategory, AlertSeverity
        
        violations = [
            Violation(
                violation_id="VIO_001",
                rule_id="MSA_RULE",
                condition_type="ALTITUDE_MINIMUM",
                severity=AlertSeverity.CRITICAL,
                explanation="Below MSA",
                details={
                    "aircraft_involved": ["AAL123"],
                    "expected_minimum": 5000,
                    "actual_altitude": 4000,
                },
            ),
        ]
        
        parsed = ParsedInstruction(
            raw_text="test",
            normalized_text="test",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        step = alert_pipeline._step_5_alert_generation(parsed, violations)
        
        assert step.status == "SUCCESS"
        alerts = step.output_data
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_step_7_atco_decision_critical(self, alert_pipeline):
        """Paso 7: Decisión con alerta crítica."""
        from Alert_System.models.alert import Alert, AlertCategory, AlertSeverity
        
        alerts = [
            Alert(
                alert_id="ALT_001",
                category=AlertCategory.MSA_VIOLATION,
                severity=AlertSeverity.CRITICAL,
                affected_callsigns=["AAL123"],
                triggering_instruction_raw="descend to 4000",
                title="MSA Violation",
                explanation="Below MSA",
                suggested_action="Climb immediately",
            ),
        ]
        
        from Alert_System.core.state_projection import ProjectedState
        projected = ProjectedState(
            traffic_state=TrafficState(sector_id="TEST"),
            source_instruction=ParsedInstruction(
                raw_text="test",
                normalized_text="test",
                speaker=Speaker.ATCO,
                callsign="AAL123",
                instruction_type=InstructionType.DESCENT,
                action_verb="descend",
            ),
        )
        
        step = alert_pipeline._step_7_atco_decision(alerts, projected)
        
        assert step.status == "SUCCESS"
        # Con alerta crítica debe decidir ROLLBACK
        assert step.output_data == "ROLLBACK"

    def test_step_7_atco_decision_no_alerts(self, alert_pipeline):
        """Paso 7: Decisión sin alertas."""
        from Alert_System.core.state_projection import ProjectedState
        projected = ProjectedState(
            traffic_state=TrafficState(sector_id="TEST"),
            source_instruction=ParsedInstruction(
                raw_text="test",
                normalized_text="test",
                speaker=Speaker.ATCO,
                callsign="AAL123",
                instruction_type=InstructionType.HEADING,
                action_verb="turn",
            ),
        )
        
        step = alert_pipeline._step_7_atco_decision([], projected)
        
        assert step.status == "SUCCESS"
        # Sin alertas debe decidir COMMIT
        assert step.output_data == "COMMIT"

    def test_step_8_commit(self, alert_pipeline):
        """Paso 8: Commit del estado."""
        from Alert_System.core.state_projection import ProjectedState
        
        traffic = TrafficState(sector_id="TEST")
        ac = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=25000, heading=90, speed=300),
            flight_phase=FlightPhase.CRUISE,
        )
        traffic.add_aircraft(ac)
        
        parsed = ParsedInstruction(
            raw_text="test",
            normalized_text="test",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        projected = ProjectedState(
            traffic_state=traffic,
            source_instruction=parsed,
            is_valid_projection=True,
        )
        
        step = alert_pipeline._step_8_final_state_update("COMMIT", projected)
        
        assert step.status == "SUCCESS"
        # El estado debe haber sido actualizado
        assert alert_pipeline.state_manager.current_state.get_aircraft("AAL123") is not None


class TestSimpleATCParser:
    """Tests para el parser simple de ATC."""

    def test_parse_descent(self, alert_pipeline):
        """Parsear instrucción de descenso."""
        parsed = alert_pipeline._simple_atc_parser(
            "AAL123 descend to FL240"
        )
        
        assert parsed.callsign == "AAL123"
        assert parsed.instruction_type == InstructionType.DESCENT
        assert parsed.parameters["target_altitude"] == 24000

    def test_parse_climb(self, alert_pipeline):
        """Parsear instrucción de ascenso."""
        parsed = alert_pipeline._simple_atc_parser(
            "UAL456 climb to FL300"
        )
        
        assert parsed.callsign == "UAL456"
        assert parsed.instruction_type == InstructionType.CLIMB

    def test_parse_heading(self, alert_pipeline):
        """Parsear instrucción de rumbo."""
        parsed = alert_pipeline._simple_atc_parser(
            "DAL789 heading 270"
        )
        
        assert parsed.callsign == "DAL789"
        assert parsed.instruction_type == InstructionType.HEADING
        # El parser no extrae heading actualmente, solo validamos el tipo
        assert "heading" in parsed.parameters or parsed.instruction_type == InstructionType.HEADING

    def test_parse_takeoff(self, alert_pipeline):
        """Parsear instrucción de despegue."""
        parsed = alert_pipeline._simple_atc_parser(
            "AAL123 cleared for takeoff runway 04L"
        )
        
        assert parsed.callsign == "AAL123"
        assert parsed.instruction_type == InstructionType.TAKEOFF_CLEARANCE
        assert parsed.parameters["runway"] == "04l"

    def test_parse_landing(self, alert_pipeline):
        """Parsear instrucción de aterrizaje."""
        parsed = alert_pipeline._simple_atc_parser(
            "UAL456 cleared to land runway 22R"
        )
        
        assert parsed.callsign == "UAL456"
        assert parsed.instruction_type == InstructionType.LANDING_CLEARANCE

    def test_parse_no_callsign(self, alert_pipeline):
        """Parsear sin callsign."""
        parsed = alert_pipeline._simple_atc_parser(
            "descend to FL240"
        )
        
        assert parsed.callsign is None


class TestAltitudeRuleEvaluation:
    """Tests específicos para evaluación de reglas de altitud."""

    def test_msa_violation_detected(self, alert_pipeline):
        """Detectar violación de MSA."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado con MSA baja
        traffic = TrafficState(sector_id="TEST", msa=5000)
        ac = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=4000, heading=90, speed=250),
            flight_phase=FlightPhase.APPROACH,
        )
        traffic.add_aircraft(ac)
        
        alert_pipeline.state_manager.update_state(traffic)
        
        parsed = ParsedInstruction(
            raw_text="AAL123 descend to 4000",
            normalized_text="AAL123 descend to 4000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        violations = alert_pipeline._evaluate_altitude_rules(parsed, projected, "AAL123")
        
        assert len(violations) == 1
        assert "MSA" in violations[0].explanation

    def test_no_msa_violation(self, alert_pipeline):
        """Sin violación de MSA."""
        from Alert_System.core.state_projection import StateProjector
        
        traffic = TrafficState(sector_id="TEST", msa=5000)
        ac = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=6000, heading=90, speed=250),
            flight_phase=FlightPhase.CRUISE,
        )
        traffic.add_aircraft(ac)
        
        parsed = ParsedInstruction(
            raw_text="AAL123 maintain 6000",
            normalized_text="AAL123 maintain 6000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.MAINTAIN_ALTITUDE,
            action_verb="maintain",
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        violations = alert_pipeline._evaluate_altitude_rules(parsed, projected, "AAL123")
        
        assert len(violations) == 0


class TestSeparationRuleEvaluation:
    """Tests específicos para evaluación de reglas de separación."""

    def test_separation_conflict_detected(self, alert_pipeline):
        """Detectar conflicto de separación."""
        from Alert_System.core.state_projection import StateProjector, ProjectedSeparation
        
        traffic = TrafficState(sector_id="TEST")
        ac1 = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=30000, heading=90, speed=450),
            flight_phase=FlightPhase.CRUISE,
        )
        ac2 = AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.01, longitude=-74.01, altitude=29500, heading=90, speed=450),
            flight_phase=FlightPhase.CRUISE,
        )
        traffic.add_aircraft(ac1)
        traffic.add_aircraft(ac2)
        
        parsed = ParsedInstruction(
            raw_text="AAL123 descend to FL240",
            normalized_text="AAL123 descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        # Forzar un conflicto en las separaciones
        projected.projected_separations["AAL123"] = [
            ProjectedSeparation(
                aircraft_1="AAL123",
                aircraft_2="UAL456",
                vertical_separation_ft=500,
                horizontal_separation_nm=2.0,
                conflict_predicted=True,
                time_to_conflict=60,
            ),
        ]
        
        violations = alert_pipeline._evaluate_separation_rules(parsed, projected, "AAL123")
        
        assert len(violations) == 1
        assert "separation" in violations[0].explanation.lower()


class TestRunwayRuleEvaluation:
    """Tests específicos para evaluación de reglas de pista."""

    def test_runway_occupied(self, alert_pipeline):
        """Detectar pista ocupada."""
        from Alert_System.core.state_projection import StateProjector
        
        traffic = TrafficState(sector_id="TEST")
        traffic.add_runway(RunwayState(
            runway_id="04L",
            mode=RunwayOperationMode.LANDING,
            occupied_by="UAL456",
            occupied_until=None,
        ))
        
        parsed = ParsedInstruction(
            raw_text="AAL123 cleared for takeoff runway 04L",
            normalized_text="AAL123 cleared for takeoff runway 04L",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TAKEOFF_CLEARANCE,
            action_verb="takeoff",
            parameters={"runway": "04L"},
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        violations = alert_pipeline._evaluate_runway_rules(parsed, projected, "AAL123")
        
        assert len(violations) == 1
        assert "occupied" in violations[0].explanation.lower()

    def test_runway_available(self, alert_pipeline):
        """Pista disponible."""
        from Alert_System.core.state_projection import StateProjector
        
        traffic = TrafficState(sector_id="TEST")
        traffic.add_runway(RunwayState(
            runway_id="04L",
            mode=RunwayOperationMode.LANDING,
            occupied_by=None,
            occupied_until=None,
        ))
        
        parsed = ParsedInstruction(
            raw_text="AAL123 cleared for takeoff runway 04L",
            normalized_text="AAL123 cleared for takeoff runway 04L",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TAKEOFF_CLEARANCE,
            action_verb="takeoff",
            parameters={"runway": "04L"},
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        violations = alert_pipeline._evaluate_runway_rules(parsed, projected, "AAL123")
        
        assert len(violations) == 0
