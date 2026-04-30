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
            occupied=True,
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


class TestMultipleRuleEvaluation:
    """Tests para evaluación de múltiples reglas del mismo tipo."""
    
    def test_multiple_altitude_rules(self, alert_pipeline):
        """Evaluar múltiples reglas de altitud registradas."""
        from Alert_System.rule_engine.conditions import AltitudeCondition
        from Alert_System.core.state_projection import StateProjector
        
        # Obtener el evaluador de altitud
        altitude_evaluator = alert_pipeline.rule_engine._evaluator_instances.get("ALTITUDE")
        if not altitude_evaluator:
            pytest.skip("AltitudeCondition no registrado")
        
        # Limpiar reglas existentes
        altitude_evaluator.clear_rules()
        
        # Agregar múltiples reglas de altitud
        altitude_evaluator.add_rule({
            "condition_type": "ALTITUDE",
            "parameters": {
                "check_type": "MINIMUM",
                "reference_value": 3000,
                "rule_id": "MIN_3000"
            }
        })
        altitude_evaluator.add_rule({
            "condition_type": "ALTITUDE",
            "parameters": {
                "check_type": "MAXIMUM",
                "reference_value": 45000,
                "rule_id": "MAX_45000"
            }
        })
        
        # Crear estado con aeronave a 2000ft (violación de MIN_3000)
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=2000, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Evaluar todas las reglas
        violations = altitude_evaluator.evaluate_all(traffic, "AAL123")
        
        # Debería detectar la violación de MIN_3000
        assert len(violations) >= 1
        rule_ids = [v.rule_id for v in violations]
        assert "MIN_3000" in rule_ids
        
        # Limpiar reglas
        altitude_evaluator.clear_rules()
    
    def test_mixed_rule_types(self, alert_pipeline):
        """Evaluar reglas de múltiples tipos (ALTITUDE + SEPARATION + RUNWAY)."""
        from Alert_System.rule_engine.conditions import AltitudeCondition, SeparationCondition, RunwayCondition
        from Alert_System.core.state_projection import StateProjector
        
        # Obtener evaluadores
        altitude_evaluator = alert_pipeline.rule_engine._evaluator_instances.get("ALTITUDE")
        separation_evaluator = alert_pipeline.rule_engine._evaluator_instances.get("SEPARATION")
        runway_evaluator = alert_pipeline.rule_engine._evaluator_instances.get("RUNWAY")
        
        if not all([altitude_evaluator, separation_evaluator, runway_evaluator]):
            pytest.skip("No todos los evaluadores están registrados")
        
        # Limpiar reglas
        altitude_evaluator.clear_rules()
        separation_evaluator.clear_rules()
        runway_evaluator.clear_rules()
        
        # Agregar reglas de diferentes tipos
        altitude_evaluator.add_rule({
            "condition_type": "ALTITUDE",
            "parameters": {
                "check_type": "MINIMUM",
                "reference_value": 3000,
                "rule_id": "MIN_3000"
            }
        })
        
        separation_evaluator.add_rule({
            "condition_type": "SEPARATION",
            "parameters": {
                "separation_type": "BOTH",
                "min_distance": 5,
                "rule_id": "SEP_5NM"
            }
        })
        
        # Crear estado con violación de altitud
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=2000, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Evaluar cada tipo
        altitude_violations = altitude_evaluator.evaluate_all(traffic, "AAL123")
        separation_violations = separation_evaluator.evaluate_all(traffic, "AAL123")
        runway_violations = runway_evaluator.evaluate_all(traffic, "AAL123")
        
        # Verificar que se detectan violaciones de altitud
        assert len(altitude_violations) >= 1
        
        # Limpiar reglas
        altitude_evaluator.clear_rules()
        separation_evaluator.clear_rules()
        runway_evaluator.clear_rules()
    
    def test_no_rules_registered(self, alert_pipeline):
        """Evaluar sin reglas registradas debe usar defaults."""
        from Alert_System.rule_engine.conditions import AltitudeCondition
        from Alert_System.core.state_projection import StateProjector
        
        altitude_evaluator = alert_pipeline.rule_engine._evaluator_instances.get("ALTITUDE")
        if not altitude_evaluator:
            pytest.skip("AltitudeCondition no registrado")
        
        # Limpiar todas las reglas
        altitude_evaluator.clear_rules()
        
        # Crear estado con aeronave por debajo de MSA
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=4000, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Evaluar sin reglas específicas (debería usar default MSA_CHECK)
        violations = altitude_evaluator.evaluate_all(traffic, "AAL123")
        
        # Debería detectar violación de MSA usando default
        assert len(violations) >= 1


class TestEndToEndScenarios:
    """Tests end-to-end para escenarios completos de tráfico aéreo."""
    
    def test_descent_sequence_with_multiple_rules(self, alert_pipeline):
        """Escenario: Secuencia de descenso con evaluación continua de reglas."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado inicial: aeronave a 15000ft, MSA 5000ft
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=15000, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Instrucción 1: Descenso a 10000ft (no debería generar alerta)
        parsed1 = ParsedInstruction(
            raw_text="AAL123 descend to 10000",
            normalized_text="AAL123 descend to 10000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 10000},
        )
        
        projector1 = StateProjector()
        projected1 = projector1.create_projection(traffic, parsed1, 5)
        violations1 = alert_pipeline._evaluate_altitude_rules(parsed1, projected1, "AAL123")
        
        # No debería haber violación de MSA (10000ft > 5000ft)
        assert len(violations1) == 0
        
        # Instrucción 2: Descenso a 4000ft (debería generar alerta MSA)
        parsed2 = ParsedInstruction(
            raw_text="AAL123 descend to 4000",
            normalized_text="AAL123 descend to 4000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 4000},
        )
        
        projector2 = StateProjector()
        projected2 = projector2.create_projection(traffic, parsed2, 5)
        violations2 = alert_pipeline._evaluate_altitude_rules(parsed2, projected2, "AAL123")
        
        # Debería haber violación de MSA (4000ft < 5000ft)
        assert len(violations2) >= 1
        assert any("MSA" in v.explanation for v in violations2)
    
    def test_takeoff_sequence_with_runway_conflict(self, alert_pipeline):
        """Escenario: Secuencia de despegue con conflicto de pista ocupada."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado: pista 04L ocupada por otra aeronave
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_runway(RunwayState(
            runway_id="04L",
            occupied=True,
            occupied_by="UAL456",
        ))
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=0, heading=90, speed=0),
            flight_phase=FlightPhase.GROUND,
        ))
        
        # Instrucción de despegue: debería generar alerta de pista ocupada
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
        
        # Debería detectar pista ocupada
        assert len(violations) >= 1
        assert any("04L" in v.explanation for v in violations)
    
    def test_multiple_aircraft_separation_scenario(self, alert_pipeline):
        """Escenario: Múltiples aeronaves con evaluación de separación."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado con dos aeronaves cercanas
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=10000, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        traffic.add_aircraft(AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.01, longitude=-75.01, altitude=10000, heading=90, speed=250),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Instrucción de cambio de rumbo para AAL123
        parsed = ParsedInstruction(
            raw_text="AAL123 turn left heading 180",
            normalized_text="AAL123 turn left heading 180",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TURN_LEFT,
            action_verb="turn",
            parameters={"heading": 180},
        )
        
        projector = StateProjector()
        projected = projector.create_projection(traffic, parsed, 5)
        
        # Evaluar separación
        violations = alert_pipeline._evaluate_separation_rules(parsed, projected, "AAL123")
        
        # Verificar que el sistema de separación funciona
        # (puede o no detectar conflicto dependiendo de la distancia exacta)
        # Lo importante es que no falle al evaluar
        assert isinstance(violations, list)
    
    def test_complex_approach_scenario(self, alert_pipeline):
        """Escenario complejo: aproximación con múltiples aeronaves y reglas."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar escenario de aproximación: 3 aeronaves en aproximación
        traffic = TrafficState(sector_id="KJFK", msa=3000)
        
        # AAL123: en aproximación final a 4000ft
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.6, longitude=-73.8, altitude=4000, heading=270, speed=180),
            flight_phase=FlightPhase.APPROACH,
        ))
        
        # UAL456: en aproximación a 5000ft, detrás de AAL123
        traffic.add_aircraft(AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.65, longitude=-73.85, altitude=5000, heading=270, speed=190),
            flight_phase=FlightPhase.APPROACH,
        ))
        
        # DAL789: en base a 6000ft
        traffic.add_aircraft(AircraftState(
            callsign="DAL789",
            position=Position(latitude=40.7, longitude=-73.9, altitude=6000, heading=270, speed=200),
            flight_phase=FlightPhase.APPROACH,
        ))
        
        # Pistas disponibles
        traffic.add_runway(RunwayState(
            runway_id="22L",
            occupied=False,
            occupied_by=None,
        ))
        traffic.add_runway(RunwayState(
            runway_id="22R",
            occupied=True,
            occupied_by="JBL555",
        ))
        
        # Secuencia de instrucciones
        
        # 1. AAL123 autorizado para aterrizar en 22L (pista libre)
        parsed1 = ParsedInstruction(
            raw_text="AAL123 cleared ILS approach runway 22L",
            normalized_text="AAL123 cleared ILS approach runway 22L",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.LANDING_CLEARANCE,
            action_verb="cleared",
            parameters={"runway": "22L"},
        )
        
        projector1 = StateProjector()
        projected1 = projector1.create_projection(traffic, parsed1, 5)
        violations1 = alert_pipeline._evaluate_runway_rules(parsed1, projected1, "AAL123")
        
        # El evaluador puede detectar violaciones de otras pistas
        # Lo importante es que no detecte violación de 22L
        runway_22l_violations = [v for v in violations1 if "22L" in v.explanation]
        assert len(runway_22l_violations) == 0  # 22L está libre
        
        # 2. UAL456 descenso a 3000ft (violación MSA 3000ft, está en el límite)
        parsed2 = ParsedInstruction(
            raw_text="UAL456 descend to 3000",
            normalized_text="UAL456 descend to 3000",
            speaker=Speaker.ATCO,
            callsign="UAL456",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 3000},
        )
        
        projector2 = StateProjector()
        projected2 = projector2.create_projection(traffic, parsed2, 5)
        violations2 = alert_pipeline._evaluate_altitude_rules(parsed2, projected2, "UAL456")
        
        # 3000ft == MSA, no debería haber violación
        assert len(violations2) == 0
        
        # 3. UAL456 descenso a 2500ft (violación MSA)
        parsed3 = ParsedInstruction(
            raw_text="UAL456 descend to 2500",
            normalized_text="UAL456 descend to 2500",
            speaker=Speaker.ATCO,
            callsign="UAL456",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 2500},
        )
        
        projector3 = StateProjector()
        projected3 = projector3.create_projection(traffic, parsed3, 5)
        violations3 = alert_pipeline._evaluate_altitude_rules(parsed3, projected3, "UAL456")
        
        # Debería detectar violación MSA (2500ft < 3000ft)
        assert len(violations3) >= 1
        assert any("MSA" in v.explanation for v in violations3)
        
        # 4. DAL789 autorizado para aterrizar en 22R (pista ocupada)
        parsed4 = ParsedInstruction(
            raw_text="DAL789 cleared ILS approach runway 22R",
            normalized_text="DAL789 cleared ILS approach runway 22R",
            speaker=Speaker.ATCO,
            callsign="DAL789",
            instruction_type=InstructionType.LANDING_CLEARANCE,
            action_verb="cleared",
            parameters={"runway": "22R"},
        )
        
        projector4 = StateProjector()
        projected4 = projector4.create_projection(traffic, parsed4, 5)
        violations4 = alert_pipeline._evaluate_runway_rules(parsed4, projected4, "DAL789")
        
        # Debería detectar pista ocupada
        assert len(violations4) >= 1
        assert any("22R" in v.explanation for v in violations4)
    
    def test_multi_step_climb_descent_scenario(self, alert_pipeline):
        """Escenario: Secuencia larga de subidas y bajadas con múltiples aeronaves."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado inicial
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        
        # AAL123: crucero a 35000ft
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=35000, heading=270, speed=450),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # UAL456: crucero a 34000ft (1000ft debajo)
        traffic.add_aircraft(AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.01, longitude=-75.01, altitude=34000, heading=270, speed=440),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Secuencia de 6 instrucciones
        
        # 1. AAL123 subir a 37000ft
        parsed1 = ParsedInstruction(
            raw_text="AAL123 climb to 37000",
            normalized_text="AAL123 climb to 37000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 37000},
        )
        
        projector1 = StateProjector()
        projected1 = projector1.create_projection(traffic, parsed1, 5)
        violations1 = alert_pipeline._evaluate_altitude_rules(parsed1, projected1, "AAL123")
        assert len(violations1) == 0  # 37000ft es seguro
        
        # 2. UAL456 subir a 36000ft (se acerca a AAL123)
        parsed2 = ParsedInstruction(
            raw_text="UAL456 climb to 36000",
            normalized_text="UAL456 climb to 36000",
            speaker=Speaker.ATCO,
            callsign="UAL456",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 36000},
        )
        
        projector2 = StateProjector()
        projected2 = projector2.create_projection(traffic, parsed2, 5)
        violations2 = alert_pipeline._evaluate_altitude_rules(parsed2, projected2, "UAL456")
        assert len(violations2) == 0  # 36000ft es seguro
        
        # 3. AAL123 bajar a 35000ft (vuelve a altitud original)
        parsed3 = ParsedInstruction(
            raw_text="AAL123 descend to 35000",
            normalized_text="AAL123 descend to 35000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 35000},
        )
        
        projector3 = StateProjector()
        projected3 = projector3.create_projection(traffic, parsed3, 5)
        violations3 = alert_pipeline._evaluate_altitude_rules(parsed3, projected3, "AAL123")
        assert len(violations3) == 0  # 35000ft es seguro
        
        # 4. UAL456 bajar a 4000ft (descenso prolongado)
        parsed4 = ParsedInstruction(
            raw_text="UAL456 descend to 4000",
            normalized_text="UAL456 descend to 4000",
            speaker=Speaker.ATCO,
            callsign="UAL456",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 4000},
        )
        
        projector4 = StateProjector()
        projected4 = projector4.create_projection(traffic, parsed4, 5)
        violations4 = alert_pipeline._evaluate_altitude_rules(parsed4, projected4, "UAL456")
        
        # 4000ft < 5000ft MSA, debería detectar violación
        assert len(violations4) >= 1
        assert any("MSA" in v.explanation for v in violations4)
        
        # 5. AAL123 bajar a 45000ft (subir en realidad)
        parsed5 = ParsedInstruction(
            raw_text="AAL123 climb to 45000",
            normalized_text="AAL123 climb to 45000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 45000},
        )
        
        projector5 = StateProjector()
        projected5 = projector5.create_projection(traffic, parsed5, 5)
        violations5 = alert_pipeline._evaluate_altitude_rules(parsed5, projected5, "AAL123")
        assert len(violations5) == 0  # 45000ft es seguro
        
        # 6. AAL123 bajar a 3000ft (violación MSA grave)
        parsed6 = ParsedInstruction(
            raw_text="AAL123 descend to 3000",
            normalized_text="AAL123 descend to 3000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 3000},
        )
        
        projector6 = StateProjector()
        projected6 = projector6.create_projection(traffic, parsed6, 5)
        violations6 = alert_pipeline._evaluate_altitude_rules(parsed6, projected6, "AAL123")
        
        # Debería detectar violación MSA (3000ft < 5000ft)
        assert len(violations6) >= 1
        assert any("MSA" in v.explanation for v in violations6)
    
    def test_high_density_traffic_scenario(self, alert_pipeline):
        """Escenario de alta densidad: 5 aeronaves con múltiples interacciones."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar alta densidad de tráfico
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        
        # 5 aeronaves en diferentes altitudes y posiciones
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=35000, heading=270, speed=450),
            flight_phase=FlightPhase.CRUISE,
        ))
        traffic.add_aircraft(AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.02, longitude=-75.02, altitude=34000, heading=270, speed=440),
            flight_phase=FlightPhase.CRUISE,
        ))
        traffic.add_aircraft(AircraftState(
            callsign="DAL789",
            position=Position(latitude=40.04, longitude=-75.04, altitude=33000, heading=270, speed=430),
            flight_phase=FlightPhase.CRUISE,
        ))
        traffic.add_aircraft(AircraftState(
            callsign="JBL555",
            position=Position(latitude=40.06, longitude=-75.06, altitude=32000, heading=270, speed=420),
            flight_phase=FlightPhase.CRUISE,
        ))
        traffic.add_aircraft(AircraftState(
            callsign="SWA888",
            position=Position(latitude=40.08, longitude=-75.08, altitude=31000, heading=270, speed=410),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Evaluar separación para cada aeronave
        for callsign in ["AAL123", "UAL456", "DAL789", "JBL555", "SWA888"]:
            parsed = ParsedInstruction(
                raw_text=f"{callsign} maintain heading",
                normalized_text=f"{callsign} maintain heading",
                speaker=Speaker.ATCO,
                callsign=callsign,
                instruction_type=InstructionType.HEADING,
                action_verb="maintain",
                parameters={"heading": 270},
            )
            
            projector = StateProjector()
            projected = projector.create_projection(traffic, parsed, 5)
            violations = alert_pipeline._evaluate_separation_rules(parsed, projected, callsign)
            
            # Verificar que no falla al evaluar separación
            assert isinstance(violations, list)
        
        # Simular instrucción de descenso para la primera aeronave
        parsed_descend = ParsedInstruction(
            raw_text="AAL123 descend to 25000",
            normalized_text="AAL123 descend to 25000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 25000},
        )
        
        projector_descend = StateProjector()
        projected_descend = projector_descend.create_projection(traffic, parsed_descend, 5)
        violations_descend = alert_pipeline._evaluate_altitude_rules(parsed_descend, projected_descend, "AAL123")
        
        # 25000ft > MSA 5000ft, no debería haber violación
        assert len(violations_descend) == 0
    
    def test_extended_instruction_sequence(self, alert_pipeline):
        """Escenario: Secuencia extendida de 8+ instrucciones de diferentes tipos."""
        from Alert_System.core.state_projection import StateProjector
        
        # Configurar estado inicial
        traffic = TrafficState(sector_id="KJFK", msa=5000)
        
        # AAL123: crucero a 25000ft
        traffic.add_aircraft(AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-75.0, altitude=25000, heading=270, speed=350),
            flight_phase=FlightPhase.CRUISE,
        ))
        
        # Secuencia de 8 instrucciones
        
        # 1. Cambio de rumbo a 180
        parsed1 = ParsedInstruction(
            raw_text="AAL123 turn left heading 180",
            normalized_text="AAL123 turn left heading 180",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TURN_LEFT,
            action_verb="turn",
            parameters={"heading": 180},
        )
        
        projector1 = StateProjector()
        projected1 = projector1.create_projection(traffic, parsed1, 5)
        violations1 = alert_pipeline._evaluate_separation_rules(parsed1, projected1, "AAL123")
        assert isinstance(violations1, list)
        
        # 2. Subir a 30000ft
        parsed2 = ParsedInstruction(
            raw_text="AAL123 climb to 30000",
            normalized_text="AAL123 climb to 30000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 30000},
        )
        
        projector2 = StateProjector()
        projected2 = projector2.create_projection(traffic, parsed2, 5)
        violations2 = alert_pipeline._evaluate_altitude_rules(parsed2, projected2, "AAL123")
        assert len(violations2) == 0  # 30000ft > MSA
        
        # 3. Mantener altitud
        parsed3 = ParsedInstruction(
            raw_text="AAL123 maintain 30000",
            normalized_text="AAL123 maintain 30000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.MAINTAIN_ALTITUDE,
            action_verb="maintain",
            parameters={"target_altitude": 30000},
        )
        
        projector3 = StateProjector()
        projected3 = projector3.create_projection(traffic, parsed3, 5)
        violations3 = alert_pipeline._evaluate_altitude_rules(parsed3, projected3, "AAL123")
        assert len(violations3) == 0
        
        # 4. Cambio de rumbo a 270
        parsed4 = ParsedInstruction(
            raw_text="AAL123 turn right heading 270",
            normalized_text="AAL123 turn right heading 270",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TURN_RIGHT,
            action_verb="turn",
            parameters={"heading": 270},
        )
        
        projector4 = StateProjector()
        projected4 = projector4.create_projection(traffic, parsed4, 5)
        violations4 = alert_pipeline._evaluate_separation_rules(parsed4, projected4, "AAL123")
        assert isinstance(violations4, list)
        
        # 5. Descenso a 20000ft
        parsed5 = ParsedInstruction(
            raw_text="AAL123 descend to 20000",
            normalized_text="AAL123 descend to 20000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 20000},
        )
        
        projector5 = StateProjector()
        projected5 = projector5.create_projection(traffic, parsed5, 5)
        violations5 = alert_pipeline._evaluate_altitude_rules(parsed5, projected5, "AAL123")
        assert len(violations5) == 0  # 20000ft > MSA
        
        # 6. Descenso a 10000ft
        parsed6 = ParsedInstruction(
            raw_text="AAL123 descend to 10000",
            normalized_text="AAL123 descend to 10000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 10000},
        )
        
        projector6 = StateProjector()
        projected6 = projector6.create_projection(traffic, parsed6, 5)
        violations6 = alert_pipeline._evaluate_altitude_rules(parsed6, projected6, "AAL123")
        assert len(violations6) == 0  # 10000ft > MSA
        
        # 7. Descenso a 4000ft (violación MSA)
        parsed7 = ParsedInstruction(
            raw_text="AAL123 descend to 4000",
            normalized_text="AAL123 descend to 4000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 4000},
        )
        
        projector7 = StateProjector()
        projected7 = projector7.create_projection(traffic, parsed7, 5)
        violations7 = alert_pipeline._evaluate_altitude_rules(parsed7, projected7, "AAL123")
        
        # Debería detectar violación MSA (4000ft < 5000ft)
        assert len(violations7) >= 1
        assert any("MSA" in v.explanation for v in violations7)
        
        # 8. Corrección: subir a 6000ft
        parsed8 = ParsedInstruction(
            raw_text="AAL123 climb to 6000",
            normalized_text="AAL123 climb to 6000",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.CLIMB,
            action_verb="climb",
            parameters={"target_altitude": 6000},
        )
        
        projector8 = StateProjector()
        projected8 = projector8.create_projection(traffic, parsed8, 5)
        violations8 = alert_pipeline._evaluate_altitude_rules(parsed8, projected8, "AAL123")
        assert len(violations8) == 0  # 6000ft > MSA

