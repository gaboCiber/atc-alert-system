"""Tests para modelos de instrucciones y alertas."""

import pytest

from Alert_System.models.alert import (
    Alert,
    AlertCategory,
    AlertResult,
    AlertSeverity,
    Violation,
)
from Alert_System.models.instruction import (
    InstructionType,
    ParsedInstruction,
    Speaker,
)


class TestParsedInstruction:
    """Tests para ParsedInstruction."""

    def test_instruction_creation(self):
        """Crear instrucción válida."""
        instruction = ParsedInstruction(
            raw_text="American one two three descend to flight level two four zero",
            normalized_text="AAL123 descend to FL240",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 24000, "flight_level": 240},
        )
        
        assert instruction.callsign == "AAL123"
        assert instruction.instruction_type == InstructionType.DESCENT
        assert instruction.speaker == Speaker.ATCO
        assert instruction.is_valid

    def test_instruction_with_entities(self):
        """Instrucción con entidades referenciadas."""
        instruction = ParsedInstruction(
            raw_text="United four five six turn left heading zero nine zero",
            normalized_text="UAL456 turn left heading 090",
            speaker=Speaker.ATCO,
            callsign="UAL456",
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
            parameters={"heading": 90, "direction": "left"},
            entities=["WAYPOINT_KORLI", "RUNWAY_09L"],
        )
        
        assert len(instruction.entities) == 2
        assert "WAYPOINT_KORLI" in instruction.entities

    def test_instruction_clearance_detection(self):
        """Detectar si es clearance."""
        takeoff_clearance = ParsedInstruction(
            raw_text="American one two three cleared for takeoff",
            normalized_text="AAL123 cleared for takeoff",
            speaker=Speaker.ATCO,
            callsign="AAL123",
            instruction_type=InstructionType.TAKEOFF_CLEARANCE,
            action_verb="cleared",
        )
        assert takeoff_clearance.is_clearance()
        
        heading_instruction = ParsedInstruction(
            raw_text="turn left heading 090",
            normalized_text="turn left heading 090",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
        )
        assert not heading_instruction.is_clearance()

    def test_instruction_altitude_change_detection(self):
        """Detectar cambio de altitud."""
        descent = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
        )
        assert descent.is_altitude_change()
        
        heading = ParsedInstruction(
            raw_text="turn right 090",
            normalized_text="turn right 090",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
        )
        assert not heading.is_altitude_change()

    def test_get_target_altitude(self):
        """Obtener altitud objetivo."""
        instruction = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"target_altitude": 24000},
        )
        assert instruction.get_target_altitude() == 24000

    def test_get_target_altitude_from_flight_level(self):
        """Obtener altitud desde flight_level."""
        instruction = ParsedInstruction(
            raw_text="descend to FL240",
            normalized_text="descend to FL240",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            parameters={"flight_level": 240},  # Sin target_altitude explícito
        )
        assert instruction.get_target_altitude() == 24000

    def test_get_target_heading(self):
        """Obtener rumbo objetivo."""
        instruction = ParsedInstruction(
            raw_text="turn left heading 090",
            normalized_text="turn left heading 090",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.HEADING,
            action_verb="turn",
            parameters={"heading": 90},
        )
        assert instruction.get_target_heading() == 90

    def test_requires_immediate_action(self):
        """Detectar urgencia."""
        urgent = ParsedInstruction(
            raw_text="immediately descend",
            normalized_text="immediately descend",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            temporal_marker="immediately",
        )
        assert urgent.requires_immediate_action()
        
        normal = ParsedInstruction(
            raw_text="descend when ready",
            normalized_text="descend when ready",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            temporal_marker="when_ready",
        )
        assert not normal.requires_immediate_action()

    def test_instruction_validation_errors(self):
        """Instrucción con errores de validación."""
        instruction = ParsedInstruction(
            raw_text="descend",
            normalized_text="descend",
            speaker=Speaker.ATCO,
            instruction_type=InstructionType.DESCENT,
            action_verb="descend",
            is_valid=False,
            validation_errors=["Missing callsign", "Missing target altitude"],
        )
        assert not instruction.is_valid
        assert len(instruction.validation_errors) == 2


class TestAlertSeverity:
    """Tests para AlertSeverity."""

    def test_severity_values(self):
        """Valores del enum."""
        assert AlertSeverity.INFO == "info"
        assert AlertSeverity.LOW == "low"
        assert AlertSeverity.MEDIUM == "medium"
        assert AlertSeverity.HIGH == "high"
        assert AlertSeverity.CRITICAL == "critical"


class TestAlertCategory:
    """Tests para AlertCategory."""

    def test_category_values(self):
        """Valores del enum."""
        assert AlertCategory.ALTITUDE_VIOLATION == "altitude_violation"
        assert AlertCategory.SEPARATION_LOSS == "separation_loss"
        assert AlertCategory.RUNWAY_CONFLICT == "runway_conflict"
        assert AlertCategory.EMERGENCY_DETECTED == "emergency_detected"


class TestViolation:
    """Tests para Violation."""

    def test_violation_creation(self):
        """Crear violación válida."""
        violation = Violation(
            rule_id="RULE001",
            condition_type="ALTITUDE_MINIMUM",
            severity=AlertSeverity.HIGH,
            details={
                "expected_minimum": 5000,
                "actual": 4200,
                "msa": 5000,
            },
            explanation="Aircraft descended below Minimum Sector Altitude",
            suggestion="Climb immediately to 5000ft or above",
        )
        
        assert violation.rule_id == "RULE001"
        assert violation.condition_type == "ALTITUDE_MINIMUM"
        assert violation.severity == AlertSeverity.HIGH
        assert violation.get_detail("actual") == 4200


class TestAlert:
    """Tests para Alert."""

    def test_alert_creation(self):
        """Crear alerta válida."""
        violation = Violation(
            rule_id="RULE001",
            condition_type="ALTITUDE_MINIMUM",
            severity=AlertSeverity.HIGH,
            details={"expected_minimum": 5000, "actual": 4200},
            explanation="Below MSA",
        )
        
        alert = Alert(
            severity=AlertSeverity.HIGH,
            category=AlertCategory.MSA_VIOLATION,
            affected_callsigns=["AAL123"],
            primary_callsign="AAL123",
            triggering_instruction_raw="AAL123 descend to 4000",
            violations=[violation],
            title="MSA Violation",
            explanation="Aircraft AAL123 would descend below Minimum Sector Altitude",
            suggested_action="Maintain altitude at or above 5000ft",
        )
        
        assert alert.severity == AlertSeverity.HIGH
        assert alert.category == AlertCategory.MSA_VIOLATION
        assert not alert.acknowledged
        assert alert.commit_decision == "PENDING"

    def test_alert_with_multiple_violations(self):
        """Alerta con múltiples violaciones."""
        violations = [
            Violation(
                rule_id="RULE001",
                condition_type="SEPARATION_VERTICAL",
                severity=AlertSeverity.HIGH,
                details={"separation_ft": 800, "required_ft": 1000},
                explanation="Vertical separation below 1000ft",
            ),
            Violation(
                rule_id="RULE002",
                condition_type="SEPARATION_HORIZONTAL",
                severity=AlertSeverity.MEDIUM,
                details={"separation_nm": 4.5, "required_nm": 5},
                explanation="Horizontal separation below 5 NM",
            ),
        ]
        
        alert = Alert(
            severity=AlertSeverity.HIGH,
            category=AlertCategory.SEPARATION_LOSS,
            affected_callsigns=["AAL123", "UAL456"],
            primary_callsign="AAL123",
            triggering_instruction_raw="AAL123 descend to FL240",
            violations=violations,
            title="Separation Loss Risk",
            explanation="AAL123 and UAL456 would lose separation",
            suggested_action="Assign different altitudes immediately",
        )
        
        assert len(alert.violations) == 2

    def test_alert_get_primary_violation(self):
        """Obtener violación más severa."""
        violations = [
            Violation(
                rule_id="RULE002",
                condition_type="SEPARATION_HORIZONTAL",
                severity=AlertSeverity.MEDIUM,
                details={},
                explanation="Medium severity",
            ),
            Violation(
                rule_id="RULE001",
                condition_type="SEPARATION_VERTICAL",
                severity=AlertSeverity.CRITICAL,
                details={},
                explanation="Critical severity",
            ),
            Violation(
                rule_id="RULE003",
                condition_type="SPEED",
                severity=AlertSeverity.LOW,
                details={},
                explanation="Low severity",
            ),
        ]
        
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SEPARATION_LOSS,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="test",
            violations=violations,
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        primary = alert.get_primary_violation()
        assert primary is not None
        assert primary.severity == AlertSeverity.CRITICAL

    def test_alert_acknowledge(self):
        """Reconocer alerta."""
        alert = Alert(
            severity=AlertSeverity.HIGH,
            category=AlertCategory.ALTITUDE_VIOLATION,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="test",
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        assert not alert.acknowledged
        alert.acknowledge("ATCO_John")
        assert alert.acknowledged
        assert alert.acknowledged_by == "ATCO_John"
        assert alert.acknowledged_at is not None

    def test_alert_set_commit_decision(self):
        """Establecer decisión de commit."""
        alert = Alert(
            severity=AlertSeverity.HIGH,
            category=AlertCategory.ALTITUDE_VIOLATION,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="test",
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        alert.set_commit_decision("COMMIT", "ATCO override: safe to proceed")
        assert alert.commit_decision == "COMMIT"
        assert alert.force_committed
        assert alert.force_commit_reason == "ATCO override: safe to proceed"

    def test_alert_is_critical(self):
        """Detectar alerta crítica."""
        critical = Alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.EMERGENCY_DETECTED,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="MAYDAY MAYDAY",
            title="Emergency",
            explanation="Aircraft declared emergency",
            suggested_action="Provide priority handling",
        )
        assert critical.is_critical()
        
        medium = Alert(
            severity=AlertSeverity.MEDIUM,
            category=AlertCategory.SPEED_VIOLATION,
            affected_callsigns=["UAL456"],
            triggering_instruction_raw="speed 300",
            title="Speed",
            explanation="Speed above 250 below 10000ft",
            suggested_action="Reduce speed to 250",
        )
        assert not medium.is_critical()

    def test_alert_add_violation(self):
        """Añadir violación a alerta existente."""
        alert = Alert(
            severity=AlertSeverity.LOW,
            category=AlertCategory.PROCEDURAL_ERROR,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="test",
            violations=[],
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        assert len(alert.violations) == 0
        
        violation = Violation(
            rule_id="RULE001",
            condition_type="TEST",
            severity=AlertSeverity.HIGH,
            details={},
            explanation="Test violation",
        )
        
        alert.add_violation(violation)
        
        assert len(alert.violations) == 1
        assert alert.severity == AlertSeverity.HIGH  # Severidad actualizada


class TestAlertResult:
    """Tests para AlertResult."""

    def test_alert_result_ok(self):
        """Resultado sin alerta."""
        result = AlertResult(
            instruction={"type": "descent"},
            status="OK",
            processing_time_ms=45.5,
        )
        
        assert result.status == "OK"
        assert not result.has_alert()
        assert result.is_safe()

    def test_alert_result_with_alert(self):
        """Resultado con alerta."""
        alert = Alert(
            severity=AlertSeverity.HIGH,
            category=AlertCategory.ALTITUDE_VIOLATION,
            affected_callsigns=["AAL123"],
            triggering_instruction_raw="test",
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        result = AlertResult(
            instruction={"type": "descent"},
            status="ALERT",
            alert=alert,
            violations_count=1,
            processing_time_ms=52.3,
        )
        
        assert result.has_alert()
        assert result.violations_count == 1

    def test_alert_result_critical_not_safe(self):
        """Alerta crítica no es segura."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SEPARATION_LOSS,
            affected_callsigns=["AAL123", "UAL456"],
            triggering_instruction_raw="test",
            title="Test",
            explanation="Test",
            suggested_action="Test",
        )
        
        result = AlertResult(
            instruction={"type": "descent"},
            status="ALERT",
            alert=alert,
            violations_count=2,
            processing_time_ms=48.0,
        )
        
        assert not result.is_safe()
