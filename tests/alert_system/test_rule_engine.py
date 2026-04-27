"""Tests para el motor de reglas."""

import pytest

from Alert_System.models.alert import AlertSeverity
from Alert_System.models.traffic_state import (
    AircraftState,
    FlightPhase,
    Position,
    RunwayState,
    TrafficState,
)
from Alert_System.rule_engine.conditions import (
    AltitudeCondition,
    ConditionResult,
    RunwayCondition,
    SeparationCondition,
)
from Alert_System.rule_engine.engine import RuleEngine


class TestConditionResult:
    """Tests para ConditionResult."""

    def test_satisfied_result(self):
        """Resultado satisfactorio."""
        result = ConditionResult(satisfied=True, details={"check": "passed"})
        assert result.satisfied
        assert result.violation is None
        assert result.details["check"] == "passed"

    def test_violation_result(self):
        """Resultado con violación."""
        from Alert_System.models.alert import Violation
        
        violation = Violation(
            rule_id="TEST_RULE",
            condition_type="ALTITUDE_MINIMUM",
            severity=AlertSeverity.HIGH,
            details={},
            explanation="Test violation",
        )
        
        result = ConditionResult(satisfied=False, violation=violation)
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.severity == AlertSeverity.HIGH


class TestAltitudeCondition:
    """Tests para AltitudeCondition."""

    @pytest.fixture
    def traffic_state(self):
        """Fixture con estado de tráfico básico."""
        state = TrafficState(sector_id="TEST", msa=5000)
        
        aircraft = AircraftState(
            callsign="AAL123",
            position=Position(
                latitude=40.0,
                longitude=-74.0,
                altitude=6000,
                heading=90,
                speed=250,
            ),
            flight_phase=FlightPhase.APPROACH,
        )
        state.add_aircraft(aircraft)
        
        return state

    def test_altitude_minimum_pass(self, traffic_state):
        """Altitud por encima del mínimo - pasa."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "MINIMUM",
            "reference_value": 5000,
            "rule_id": "ALT_MIN_5000",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert result.satisfied
        assert result.violation is None
        assert result.details["check"] == "minimum_passed"

    def test_altitude_minimum_fail(self, traffic_state):
        """Altitud por debajo del mínimo - falla."""
        condition = AltitudeCondition()
        
        # Cambiar altitud a 4000 (debajo del mínimo de 5000)
        ac = traffic_state.get_aircraft("AAL123")
        ac.position.altitude = 4000
        
        params = {
            "check_type": "MINIMUM",
            "reference_value": 5000,
            "rule_id": "ALT_MIN_5000",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "ALTITUDE_MINIMUM"
        assert result.violation.details["current_altitude"] == 4000
        assert result.violation.details["required_minimum"] == 5000

    def test_msa_check_pass(self, traffic_state):
        """Verificación MSA pasa."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "MSA_CHECK",
            "rule_id": "MSA_CHECK",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert result.satisfied

    def test_msa_check_fail(self, traffic_state):
        """Verificación MSA falla - CRITICAL."""
        condition = AltitudeCondition()
        
        # Cambiar altitud a 4000 (debajo de MSA 5000)
        ac = traffic_state.get_aircraft("AAL123")
        ac.position.altitude = 4000
        
        params = {
            "check_type": "MSA_CHECK",
            "rule_id": "MSA_CHECK",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.severity == AlertSeverity.CRITICAL
        assert "CRITICAL" in result.violation.explanation

    def test_altitude_maximum_pass(self, traffic_state):
        """Altitud por debajo del máximo - pasa."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "MAXIMUM",
            "reference_value": 10000,
            "rule_id": "ALT_MAX_10000",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert result.satisfied

    def test_altitude_maximum_fail(self, traffic_state):
        """Altitud por encima del máximo - falla."""
        condition = AltitudeCondition()
        
        # Cambiar altitud a 11000 (encima del máximo de 10000)
        ac = traffic_state.get_aircraft("AAL123")
        ac.position.altitude = 11000
        
        params = {
            "check_type": "MAXIMUM",
            "reference_value": 10000,
            "rule_id": "ALT_MAX_10000",
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "ALTITUDE_MAXIMUM"

    def test_no_aircraft_callsign(self, traffic_state):
        """Error si no se proporciona callsign."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "MINIMUM",
            "reference_value": 5000,
        }
        
        result = condition.evaluate(traffic_state, params, None)
        
        assert not result.satisfied
        assert "error" in result.details

    def test_aircraft_not_found(self, traffic_state):
        """Error si aeronave no existe."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "MINIMUM",
            "reference_value": 5000,
        }
        
        result = condition.evaluate(traffic_state, params, "UNKNOWN")
        
        assert not result.satisfied
        assert "error" in result.details

    def test_unknown_check_type(self, traffic_state):
        """Error con tipo de chequeo desconocido."""
        condition = AltitudeCondition()
        
        params = {
            "check_type": "UNKNOWN",
            "reference_value": 5000,
        }
        
        result = condition.evaluate(traffic_state, params, "AAL123")
        
        assert not result.satisfied
        assert "error" in result.details

    def test_required_parameters(self):
        """Parámetros base requeridos (check_type)."""
        condition = AltitudeCondition()
        required = condition.get_required_parameters()
        
        assert "check_type" in required
        # reference_value es opcional según el tipo de chequeo

    def test_validate_parameters(self):
        """Validación de parámetros."""
        condition = AltitudeCondition()
        
        # Parámetros válidos
        valid, errors = condition.validate_parameters({
            "check_type": "MINIMUM",
            "reference_value": 5000,
        })
        assert valid
        assert len(errors) == 0
        
        # Parámetros inválidos
        valid, errors = condition.validate_parameters({
            "check_type": "MINIMUM",
        })
        assert not valid
        assert len(errors) > 0


class TestSeparationCondition:
    """Tests para SeparationCondition."""

    @pytest.fixture
    def traffic_state_with_aircrafts(self):
        """Fixture con dos aeronaves cercanas."""
        state = TrafficState(sector_id="TEST")
        
        # Aeronave 1
        ac1 = AircraftState(
            callsign="AAL123",
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
        
        # Aeronave 2 - cercana horizontalmente, misma altitud
        ac2 = AircraftState(
            callsign="UAL456",
            position=Position(
                latitude=40.08,  # ~5 NM north
                longitude=-74.0,
                altitude=30000,  # Misma altitud - separación vertical 0
                heading=90,
                speed=450,
            ),
            flight_phase=FlightPhase.CRUISE,
        )
        state.add_aircraft(ac2)
        
        return state

    def test_vertical_separation_fail(self, traffic_state_with_aircrafts):
        """Separación vertical insuficiente - falla."""
        condition = SeparationCondition()
        
        params = {
            "separation_type": "VERTICAL",
            "min_distance": 1000,
            "rule_id": "VERT_SEP_1000",
        }
        
        result = condition.evaluate(
            traffic_state_with_aircrafts,
            params,
            "AAL123"
        )
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "SEPARATION_VERTICAL"
        assert result.violation.details["aircraft_2"] == "UAL456"

    def test_vertical_separation_pass(self, traffic_state_with_aircrafts):
        """Separación vertical suficiente - pasa."""
        condition = SeparationCondition()
        
        # Cambiar altitud de UAL456 para tener separación vertical
        ac2 = traffic_state_with_aircrafts.get_aircraft("UAL456")
        ac2.position.altitude = 31000  # 1000ft arriba
        
        params = {
            "separation_type": "VERTICAL",
            "min_distance": 1000,
            "rule_id": "VERT_SEP_1000",
        }
        
        result = condition.evaluate(
            traffic_state_with_aircrafts,
            params,
            "AAL123"
        )
        
        assert result.satisfied

    def test_horizontal_separation_fail(self, traffic_state_with_aircrafts):
        """Separación horizontal insuficiente - falla."""
        condition = SeparationCondition()
        
        params = {
            "separation_type": "HORIZONTAL",
            "min_distance": 5,
            "rule_id": "HOR_SEP_5NM",
        }
        
        result = condition.evaluate(
            traffic_state_with_aircrafts,
            params,
            "AAL123"
        )
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "SEPARATION_HORIZONTAL"

    def test_both_separation_pass(self, traffic_state_with_aircrafts):
        """Ambas separaciones OK - pasa."""
        condition = SeparationCondition()
        
        # Separar aeronaves verticalmente y horizontalmente
        ac2 = traffic_state_with_aircrafts.get_aircraft("UAL456")
        ac2.position.altitude = 32000  # 2000ft separación vertical
        ac2.position.latitude = 40.2  # ~12 NM separación horizontal
        
        params = {
            "separation_type": "BOTH",
            "rule_id": "BOTH_SEP",
        }
        
        result = condition.evaluate(
            traffic_state_with_aircrafts,
            params,
            "AAL123"
        )
        
        assert result.satisfied

    def test_no_nearby_aircrafts(self, traffic_state_with_aircrafts):
        """Sin aeronaves cercanas - siempre pasa."""
        condition = SeparationCondition()
        
        # Remover UAL456
        traffic_state_with_aircrafts.remove_aircraft("UAL456")
        
        params = {
            "separation_type": "VERTICAL",
            "min_distance": 1000,
        }
        
        result = condition.evaluate(
            traffic_state_with_aircrafts,
            params,
            "AAL123"
        )
        
        assert result.satisfied


class TestRunwayCondition:
    """Tests para RunwayCondition."""

    @pytest.fixture
    def traffic_state_with_runway(self):
        """Fixture con pista ocupada."""
        state = TrafficState(sector_id="TEST")
        
        runway = RunwayState(
            runway_id="27L",
            occupied=True,
            occupied_by="AAL123",
        )
        state.add_runway(runway)
        
        return state

    def test_runway_occupied_fail(self, traffic_state_with_runway):
        """Pista ocupada - falla para clearance."""
        condition = RunwayCondition()
        
        params = {
            "check_type": "OCCUPIED",
            "runway_id": "27L",
            "rule_id": "RUNWAY_CHECK",
        }
        
        result = condition.evaluate(
            traffic_state_with_runway,
            params,
            "UAL456"
        )
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "RUNWAY_OCCUPIED"
        assert result.violation.severity == AlertSeverity.CRITICAL

    def test_runway_available_pass(self, traffic_state_with_runway):
        """Pista disponible - pasa."""
        condition = RunwayCondition()
        
        # Liberar pista
        runway = traffic_state_with_runway.get_runway("27L")
        runway.occupied = False
        runway.occupied_by = None
        
        params = {
            "check_type": "OCCUPIED",
            "runway_id": "27L",
        }
        
        result = condition.evaluate(
            traffic_state_with_runway,
            params,
            "UAL456"
        )
        
        assert result.satisfied

    def test_runway_not_found(self, traffic_state_with_runway):
        """Pista no existe - falla."""
        condition = RunwayCondition()
        
        params = {
            "check_type": "EXISTS",
            "runway_id": "99X",
            "rule_id": "RUNWAY_EXISTS",
        }
        
        result = condition.evaluate(
            traffic_state_with_runway,
            params,
            "AAL123"
        )
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "RUNWAY_NOT_FOUND"

    def test_holding_short_full(self, traffic_state_with_runway):
        """Cola de holding short llena - falla."""
        condition = RunwayCondition()
        
        # Llenar holding short
        runway = traffic_state_with_runway.get_runway("27L")
        runway.holding_short = ["UAL1", "UAL2", "UAL3", "UAL4"]
        
        params = {
            "check_type": "HOLDING_SHORT_FULL",
            "runway_id": "27L",
            "max_holding": 3,
            "rule_id": "HOLDING_CHECK",
        }
        
        result = condition.evaluate(
            traffic_state_with_runway,
            params,
            "UAL456"
        )
        
        assert not result.satisfied
        assert result.violation is not None
        assert result.violation.condition_type == "HOLDING_SHORT_CONGESTION"

    def test_holding_short_ok(self, traffic_state_with_runway):
        """Cola de holding short normal - pasa."""
        condition = RunwayCondition()
        
        runway = traffic_state_with_runway.get_runway("27L")
        runway.holding_short = ["UAL1"]
        
        params = {
            "check_type": "HOLDING_SHORT_FULL",
            "runway_id": "27L",
            "max_holding": 3,
        }
        
        result = condition.evaluate(
            traffic_state_with_runway,
            params,
            "UAL456"
        )
        
        assert result.satisfied


class TestRuleEngine:
    """Tests para RuleEngine."""

    @pytest.fixture
    def engine(self):
        """Fixture con motor inicializado."""
        return RuleEngine()

    @pytest.fixture
    def traffic_state(self):
        """Fixture con estado de tráfico."""
        state = TrafficState(sector_id="TEST", msa=5000)
        
        aircraft = AircraftState(
            callsign="AAL123",
            position=Position(
                latitude=40.0,
                longitude=-74.0,
                altitude=4000,  # Debajo de MSA
                heading=90,
                speed=250,
            ),
            flight_phase=FlightPhase.APPROACH,
        )
        state.add_aircraft(aircraft)
        
        return state

    def test_engine_initialization(self, engine):
        """Motor inicializado con evaluadores por defecto."""
        assert engine.has_evaluator("ALTITUDE")
        assert engine.has_evaluator("SEPARATION")
        assert engine.has_evaluator("RUNWAY")

    def test_register_evaluator(self, engine):
        """Registrar nuevo evaluador."""
        from Alert_System.rule_engine.conditions import ConditionEvaluator
        
        class TestCondition(ConditionEvaluator):
            condition_type = "TEST"
            def evaluate(self, traffic_state, parameters, aircraft_callsign=None):
                return ConditionResult(satisfied=True)
        
        engine.register_evaluator("TEST", TestCondition)
        
        assert engine.has_evaluator("TEST")

    def test_evaluate_altitude(self, engine, traffic_state):
        """Evaluar condición de altitud."""
        params = {
            "check_type": "MSA_CHECK",
        }
        
        result = engine.evaluate("ALTITUDE", params, traffic_state, "AAL123")
        
        assert not result.satisfied  # Porque está a 4000ft, MSA es 5000ft
        assert result.violation is not None

    def test_evaluate_unknown_type(self, engine, traffic_state):
        """Error al evaluar tipo desconocido."""
        result = engine.evaluate("UNKNOWN", {}, traffic_state, "AAL123")
        
        assert not result.satisfied
        assert "error" in result.details

    def test_batch_evaluate(self, engine, traffic_state):
        """Evaluar múltiples condiciones."""
        conditions = [
            {
                "type": "ALTITUDE",
                "parameters": {"check_type": "MSA_CHECK"},
            },
            {
                "type": "ALTITUDE",
                "parameters": {"check_type": "MINIMUM", "reference_value": 3000},
            },
        ]
        
        results = engine.batch_evaluate(conditions, traffic_state, "AAL123")
        
        assert len(results) == 2
        # Primera falla (MSA 5000 vs 4000)
        assert not results[0].satisfied
        # Segunda pasa (mínimo 3000, actual 4000)
        assert results[1].satisfied

    def test_evaluate_all_violations(self, engine, traffic_state):
        """Obtener solo violaciones."""
        conditions = [
            {
                "type": "ALTITUDE",
                "parameters": {"check_type": "MSA_CHECK"},
            },
            {
                "type": "ALTITUDE",
                "parameters": {"check_type": "MINIMUM", "reference_value": 3000},
            },
        ]
        
        violations = engine.evaluate_all_violations(conditions, traffic_state, "AAL123")
        
        # Solo la primera condición genera violación
        assert len(violations) == 1
        assert violations[0].condition_type == "MSA_VIOLATION"

    def test_check_rule(self, engine, traffic_state):
        """Verificar regla completa."""
        rule = {
            "id": "RULE_001",
            "conditions": [
                {
                    "type": "ALTITUDE",
                    "parameters": {"check_type": "MSA_CHECK"},
                },
            ],
            "logic": "ALL",
        }
        
        result = engine.check_rule(rule, traffic_state, "AAL123")
        
        assert result["rule_id"] == "RULE_001"
        assert not result["passed"]
        assert len(result["violations"]) == 1
        assert result["severity"] == AlertSeverity.CRITICAL

    def test_check_rule_any_logic(self, engine, traffic_state):
        """Verificar regla con lógica ANY."""
        rule = {
            "id": "RULE_002",
            "conditions": [
                {
                    "type": "ALTITUDE",
                    "parameters": {"check_type": "MSA_CHECK"},  # Falla
                },
                {
                    "type": "ALTITUDE",
                    "parameters": {"check_type": "MINIMUM", "reference_value": 3000},  # Pasa
                },
            ],
            "logic": "ANY",
        }
        
        result = engine.check_rule(rule, traffic_state, "AAL123")
        
        # Con lógica ANY, si al menos una pasa, la regla pasa
        assert result["passed"]

    def test_get_registered_evaluators(self, engine):
        """Obtener lista de evaluadores registrados."""
        evaluators = engine.get_registered_evaluators()
        
        assert "ALTITUDE" in evaluators
        assert "SEPARATION" in evaluators
        assert "RUNWAY" in evaluators

    def test_get_evaluator_info(self, engine):
        """Obtener información de evaluador."""
        info = engine.get_evaluator_info("ALTITUDE")
        
        assert info is not None
        assert info["type"] == "ALTITUDE"
        assert "AltitudeCondition" in info["class"]
        assert "check_type" in info["required_parameters"]
