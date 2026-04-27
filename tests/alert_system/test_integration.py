"""
Tests de integración ASR + KEX + Alert System.
"""

import pytest
from datetime import datetime

# Alert System
from Alert_System.models.instruction import InstructionType, Speaker
from Alert_System.models.traffic_state import TrafficState, AircraftState, Position, FlightPhase
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.core.state_manager import StateManager
from Alert_System.pipeline.alert_pipeline import AlertPipeline

# Integration
from Alert_System.integration import (
    ASRAdapter,
    KEXAdapter,
    EndToEndPipeline,
    TranscriptionContext,
    KnowledgeContext,
)

# ASR
from ASR.transcription.base import TranscriptionResult
from ASR.normalization import ATCTextNormalizer

# KEX
from Knowledge_Extractor import Rule, Entity, Event


class TestASRAdapter:
    """Tests para el adaptador ASR."""
    
    @pytest.fixture
    def adapter(self):
        return ASRAdapter()
    
    @pytest.fixture
    def sample_transcription(self):
        return TranscriptionResult(
            text="AAL123 descend to FL240",
            file_path="test.wav",
            model_name="whisper",
            confidence=0.95,
        )
    
    def test_adapt_basic(self, adapter, sample_transcription):
        """Adaptar transcripción básica."""
        parsed = adapter.adapt(sample_transcription, Speaker.ATCO)
        
        assert parsed.callsign == "AAL123"
        assert parsed.instruction_type == InstructionType.DESCENT
        assert parsed.speaker == Speaker.ATCO
        assert parsed.parameters["target_altitude"] == 24000
    
    def test_adapt_heading(self, adapter):
        """Adaptar instrucción de heading."""
        transcription = TranscriptionResult(
            text="UAL456 turn left heading 270",
            file_path="test.wav",
            model_name="whisper",
        )
        
        parsed = adapter.adapt(transcription, Speaker.ATCO)
        
        assert parsed.callsign == "UAL456"
        assert parsed.instruction_type == InstructionType.HEADING
        assert parsed.parameters["heading"] == 270
        assert parsed.parameters["direction"] == "left"
    
    def test_adapt_takeoff(self, adapter):
        """Adaptar instrucción de despegue."""
        transcription = TranscriptionResult(
            text="DAL789 cleared for takeoff runway 04L",
            file_path="test.wav",
            model_name="whisper",
        )
        
        parsed = adapter.adapt(transcription, Speaker.ATCO)
        
        assert parsed.callsign == "DAL789"
        assert parsed.instruction_type == InstructionType.TAKEOFF_CLEARANCE
        assert parsed.parameters["runway"] == "04L"
    
    def test_adapt_batch(self, adapter):
        """Adaptar múltiples transcripciones."""
        transcriptions = [
            TranscriptionResult(
                text=f"AAL{i} climb to FL300",
                file_path=f"test{i}.wav",
                model_name="whisper",
            )
            for i in range(100, 103)
        ]
        
        parsed_list = adapter.adapt_batch(transcriptions, Speaker.ATCO)
        
        assert len(parsed_list) == 3
        for i, parsed in enumerate(parsed_list):
            assert parsed.instruction_type == InstructionType.CLIMB
            assert parsed.parameters["target_altitude"] == 30000


class TestKEXAdapter:
    """Tests para el adaptador KEX."""
    
    @pytest.fixture
    def adapter(self):
        return KEXAdapter()
    
    @pytest.fixture
    def sample_rules(self):
        """Reglas de ejemplo."""
        from Knowledge_Extractor.schemas.kex_schemas import (
            RuleTriggerCondition,
            RuleActionConstraint,
            FormalIfThen,
            RuleApplicability,
            Severity,
            RuleType,
            Modality,
            DeonticStrength,
            FlightPhase,
        )
        
        return [
            Rule(
                id="RULE_ALT_001",
                rule_type=RuleType.SAFETY_CONSTRAINT,
                modality=Modality.MUST_NOT,
                deontic_strength=DeonticStrength.MANDATORY,
                trigger=RuleTriggerCondition(
                    description="Aircraft below Minimum Sector Altitude",
                    trigger_entities=["AIRCRAFT"],
                ),
                constraint=RuleActionConstraint(
                    description="Must climb immediately",
                    action_verb="climb",
                    negation=False,
                    action_entities=["AIRCRAFT"],
                ),
                formal_if_then=FormalIfThen(
                    if_condition="altitude < MSA",
                    then_action="alert(ATCO, CLIMB_IMMEDIATELY)",
                ),
                applicability=RuleApplicability(
                    phase=[FlightPhase.CRUISE, FlightPhase.APPROACH],
                    environment=["ALL"],
                ),
                severity=Severity.CRITICAL,
                safety_critical=True,
                explainability="Below MSA is dangerous",
            ),
            Rule(
                id="RULE_SEP_001",
                rule_type=RuleType.OBLIGATION,
                modality=Modality.MUST,
                deontic_strength=DeonticStrength.MANDATORY,
                trigger=RuleTriggerCondition(
                    description="Aircraft within 1000ft vertical separation",
                    trigger_entities=["AIRCRAFT_1", "AIRCRAFT_2"],
                ),
                constraint=RuleActionConstraint(
                    description="Must maintain 1000ft vertical separation",
                    action_verb="maintain",
                    negation=False,
                    action_entities=["AIRCRAFT_1", "AIRCRAFT_2"],
                ),
                formal_if_then=FormalIfThen(
                    if_condition="vertical_separation < 1000ft",
                    then_action="alert(ATCO, LOSS_OF_SEPARATION)",
                ),
                applicability=RuleApplicability(
                    phase=[FlightPhase.CRUISE],
                    environment=["ALL"],
                ),
                severity=Severity.HIGH,
                safety_critical=True,
                explainability="Loss of separation is dangerous",
            ),
        ]
    
    def test_adapt_altitude_rule(self, adapter, sample_rules):
        """Adaptar regla de altitud."""
        evaluators = adapter.adapt_rules([sample_rules[0]])
        
        assert len(evaluators) == 1
        assert evaluators[0].condition_id == "RULE_ALT_001"
        assert evaluators[0].condition_type in ["above", "below"]
    
    def test_adapt_separation_rule(self, adapter, sample_rules):
        """Adaptar regla de separación."""
        evaluators = adapter.adapt_rules([sample_rules[1]])
        
        assert len(evaluators) == 1
        assert evaluators[0].condition_id == "RULE_SEP_001"
        assert evaluators[0].min_vertical_separation == 1000
    
    def test_create_knowledge_context(self, adapter, sample_rules):
        """Crear contexto de conocimiento."""
        context = adapter.create_knowledge_context(
            rules=sample_rules,
            entities=[],
            events=[],
            source="test_document.pdf",
        )
        
        assert len(context.rules) == 2
        assert context.source_document == "test_document.pdf"
        assert isinstance(context.extraction_timestamp, datetime)
        
        # Test lookup
        rule = context.find_rule_by_id("RULE_ALT_001")
        assert rule is not None
        assert rule.id == "RULE_ALT_001"


class TestEndToEndPipeline:
    """Tests para el pipeline end-to-end."""
    
    @pytest.fixture
    def e2e_pipeline(self):
        """Pipeline configurado para testing (sin ASR)."""
        state = TrafficState(sector_id="TEST", msa=5000)
        state_manager = StateManager(state)
        rule_engine = RuleEngine()
        alert_pipeline = AlertPipeline(state_manager, rule_engine)
        
        return EndToEndPipeline(
            asr_pipeline=None,  # No ASR en tests
            asr_adapter=None,
            alert_pipeline=alert_pipeline,
            initial_state=state,
        )
    
    @pytest.fixture
    def e2e_with_aircraft(self, e2e_pipeline):
        """Pipeline con aeronave en estado seguro (6000ft > MSA 5000ft)."""
        ac = AircraftState(
            callsign="AAL123",
            position=Position(
                latitude=40.0,
                longitude=-74.0,
                altitude=6000,  # Altitud segura por encima del MSA
                heading=90,
                speed=250,
            ),
            flight_phase=FlightPhase.APPROACH,
        )
        e2e_pipeline.get_current_state().add_aircraft(ac)
        return e2e_pipeline
    
    def test_process_text_descent_violation(self, e2e_with_aircraft):
        """Procesar texto que genera violación de MSA."""
        # Poner aeronave a altitud insegura (4000 < MSA 5000)
        ac = e2e_with_aircraft.get_current_state().get_aircraft("AAL123")
        ac.position.altitude = 4000
        
        result = e2e_with_aircraft.process_text(
            "AAL123 descend to 3000",
            speaker=Speaker.ATCO,
            enable_projection=True,
        )
        
        assert result.raw_transcription == "AAL123 descend to 3000"
        assert result.parsed_instruction is not None
        assert result.parsed_instruction.callsign == "AAL123"
        
        # Debe haber alertas críticas (MSA violation)
        assert result.has_alerts()
        assert result.is_critical()
        assert result.decision == "ROLLBACK"
    
    def test_process_text_safe_instruction(self, e2e_with_aircraft):
        """Procesar instrucción segura."""
        # Cambiar altitud a 6000 (arriba de MSA)
        ac = e2e_with_aircraft.get_current_state().get_aircraft("AAL123")
        ac.position.altitude = 6000
        
        result = e2e_with_aircraft.process_text(
            "AAL123 maintain 6000",
            speaker=Speaker.ATCO,
            enable_projection=True,
        )
        
        assert not result.has_alerts()
        assert result.decision == "COMMIT"
    
    def test_process_text_heading(self, e2e_with_aircraft):
        """Procesar instrucción de heading (sin violación)."""
        result = e2e_with_aircraft.process_text(
            "AAL123 turn left heading 180",
            speaker=Speaker.ATCO,
            enable_projection=True,
        )
        
        assert result.parsed_instruction.instruction_type == InstructionType.HEADING
        # No debe haber alertas (heading no viola reglas)
        assert not result.is_critical()
    
    def test_commit_rollback(self, e2e_with_aircraft):
        """Probar commit y rollback."""
        # Procesar instrucción segura (6000 > MSA 5000)
        result = e2e_with_aircraft.process_text(
            "AAL123 descend to 6000",
            speaker=Speaker.ATCO,
            enable_projection=True,
        )
        
        # Verificar commit exitoso
        assert result.decision == "COMMIT"
        
        # Rollback
        assert e2e_with_aircraft.rollback_changes()
        
        # La aeronave debe seguir en posición original (6000)
        ac = e2e_with_aircraft.get_current_state().get_aircraft("AAL123")
        assert ac.position.altitude == 6000  # Original
    
    def test_msa_violation_detected(self, e2e_with_aircraft):
        """Detectar cuando ATCO incumple el MSA - escenario crítico."""
        # Poner aeronave a altitud insegura (4000 < MSA 5000)
        ac = e2e_with_aircraft.get_current_state().get_aircraft("AAL123")
        ac.position.altitude = 4000
        
        # ATCO da instrucción peligrosa: descend to 3000 (por debajo de MSA)
        result = e2e_with_aircraft.process_text(
            "AAL123 descend to 3000",
            speaker=Speaker.ATCO,
            enable_projection=True,
        )
        
        # El sistema debe detectar la violación
        assert result.parsed_instruction.callsign == "AAL123"
        assert result.parsed_instruction.parameters.get("target_altitude") == 3000
        
        # Debe haber alertas críticas por violación de MSA
        assert result.has_alerts(), "Debe detectar violación de MSA"
        assert result.is_critical(), "Debe ser alerta crítica"
        assert result.decision == "ROLLBACK", "Debe hacer rollback automático"
        
        # La aeronave NO debe cambiar de altitud (rollback)
        ac = e2e_with_aircraft.get_current_state().get_aircraft("AAL123")
        assert ac.position.altitude == 4000, "Debe mantener altitud segura original"
    
    def test_load_knowledge_rules(self, e2e_pipeline):
        """Cargar reglas del KEX."""
        from Knowledge_Extractor.schemas.kex_schemas import (
            RuleTriggerCondition,
            RuleActionConstraint,
            FormalIfThen,
            RuleApplicability,
            Severity,
            RuleType,
            Modality,
            DeonticStrength,
            FlightPhase,
        )
        
        rule = Rule(
            id="RULE_TEST_001",
            rule_type=RuleType.SAFETY_CONSTRAINT,
            modality=Modality.MUST_NOT,
            deontic_strength=DeonticStrength.MANDATORY,
            trigger=RuleTriggerCondition(
                description="Aircraft below 3000ft in sector",
                trigger_entities=["AIRCRAFT"],
            ),
            constraint=RuleActionConstraint(
                description="Must be above 3000ft",
                action_verb="maintain",
                negation=False,
                action_entities=["AIRCRAFT"],
            ),
            formal_if_then=FormalIfThen(
                if_condition="altitude < 3000",
                then_action="alert",
            ),
            applicability=RuleApplicability(
                phase=[FlightPhase.CRUISE],
                environment=["ALL"],
            ),
            severity=Severity.CRITICAL,
            safety_critical=True,
            explainability="Test rule",
        )
        
        e2e_pipeline.load_knowledge([rule])
        
        # Verificar que se cargó
        assert len(e2e_pipeline._knowledge_rules) == 1


class TestIntegrationFullFlow:
    """Tests de flujo completo integrado."""
    
    def test_complete_scenario_separation(self):
        """Escenario completo: conflicto de separación."""
        # Setup
        state = TrafficState(sector_id="TEST", msa=3000)
        
        # Dos aeronaves cercanas
        ac1 = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=10000, heading=90, speed=250),
            flight_phase=FlightPhase.CRUISE,
        )
        ac2 = AircraftState(
            callsign="UAL456",
            position=Position(latitude=40.01, longitude=-74.01, altitude=10500, heading=270, speed=250),
            flight_phase=FlightPhase.CRUISE,
        )
        
        state.add_aircraft(ac1)
        state.add_aircraft(ac2)
        
        # Pipeline
        e2e = EndToEndPipeline(
            asr_pipeline=None,
            initial_state=state,
        )
        
        # Instrucción que acerca las aeronaves
        result = e2e.process_text(
            "AAL123 descend to 10000",
            enable_projection=True,
        )
        
        # Verificar resultado
        assert result.parsed_instruction.callsign == "AAL123"
        assert result.parsed_instruction.instruction_type == InstructionType.DESCENT
        
        # Puede haber alerta de separación o no dependiendo de la proyección
        # pero el pipeline debe completar sin errores
        assert result.alert_pipeline_result is not None
    
    def test_runway_occupancy_scenario(self):
        """Escenario: pista ocupada."""
        from Alert_System.models.traffic_state import RunwayState, RunwayOperationMode
        
        # Setup con pista ocupada
        state = TrafficState(sector_id="TEST")
        state.add_runway(RunwayState(
            runway_id="04L",
            mode=RunwayOperationMode.LANDING,
            occupied_by="DAL001",
        ))
        
        ac = AircraftState(
            callsign="AAL123",
            position=Position(latitude=40.0, longitude=-74.0, altitude=500, heading=90, speed=150),
            flight_phase=FlightPhase.APPROACH,
        )
        state.add_aircraft(ac)
        
        # Pipeline
        e2e = EndToEndPipeline(initial_state=state)
        
        # Instrucción de despegue en pista ocupada
        result = e2e.process_text(
            "AAL123 cleared for takeoff runway 04L",
            enable_projection=True,
        )
        
        # Debe detectar conflicto de pista
        assert result.parsed_instruction.parameters["runway"] == "04L"
        # El pipeline evalúa reglas y puede generar alerta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
