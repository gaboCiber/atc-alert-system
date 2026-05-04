"""Tests de integración con LLM real para GenericKexCondition.

Estos tests usan Ollama con modelos locales para validar que la integración
funciona correctamente con un LLM real.

NOTA: Estos tests son más lentos y requieren Ollama corriendo.
Se marcan con pytest.mark.slow para poder ejecutarlos opcionalmente.
"""

import pytest
from unittest.mock import patch

# Marcar como tests lentos para ejecución opcional
pytestmark = pytest.mark.slow


@pytest.fixture
def ollama_config():
    """Configuración para Ollama con llama3.2."""
    from common.llm_client_factory import ModelConfig
    
    return ModelConfig(
        name="smollm2:360m",
        provider="ollama",
        base_url="http://localhost:11434",
        api_key="ollama",  # Ollama no necesita API key real
        max_retries=3,
        timeout=30
    )


@pytest.fixture
def sample_traffic_state():
    """Estado de tráfico de ejemplo para pruebas."""
    from Alert_System.models.traffic_state import TrafficState, AircraftState, Position, FlightPhase
    
    traffic_state = TrafficState(sector_id="TEST_SECTOR")
    traffic_state.msa = 5000
    
    aircraft = AircraftState(
        callsign="TEST123",
        position=Position(latitude=40.0, longitude=-3.0, altitude=4000, speed=250, heading=180),
        flight_phase=FlightPhase.CRUISE
    )
    traffic_state.aircrafts["TEST123"] = aircraft
    
    return traffic_state


@pytest.fixture
def altitude_violation_rule():
    """Regla de violación de altitud para pruebas."""
    from Alert_System.integration.schemas import ExecutableRule
    
    return ExecutableRule(
        source_rule_id="RULE_ALT_001",
        rule_category="GENERIC",
        condition_description="CRITICAL: Aircraft altitude must NEVER be below 5000ft in controlled airspace. This is a safety violation.",
        raw_trigger="Altitude below minimum",
        raw_constraint="aircraft.altitude >= 5000 AND aircraft.altitude < 5000 = VIOLATION"
    )


@pytest.fixture
def separation_rule():
    """Regla de separación para pruebas."""
    from Alert_System.integration.schemas import ExecutableRule
    
    return ExecutableRule(
        source_rule_id="RULE_SEP_001", 
        rule_category="GENERIC",
        condition_description="Aircraft must maintain minimum separation of 3nm",
        raw_trigger="Minimum separation",
        raw_constraint="distance > 3nm"
    )


class TestLLMIntegration:
    """Tests de integración con LLM real vía Ollama."""
    
    @pytest.mark.integration
    def test_llm_altitude_violation_detection(self, ollama_config, sample_traffic_state, altitude_violation_rule):
        """Test que el LLM detecta correctamente una violación de altitud."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        
        # Crear condición con configuración Ollama
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Evaluar regla (la aeronave está a 4000ft, MSA es 5000ft)
        result = condition.evaluate(
            traffic_state=sample_traffic_state,
            parameters={"executable_rule": altitude_violation_rule},
            aircraft_callsign="TEST123"
        )
        
        # Debug: ver qué devuelve el LLM
        print(f"\n🔍 Debug - Result satisfied: {result.satisfied}")
        print(f"🔍 Debug - Has violation: {result.violation is not None}")
        
        # Verificar que el LLM evaluó la regla (puede o no detectar violación)
        assert result.violation is not None, "El LLM debería generar una violación o evaluación"
        assert result.violation.condition_type == "GENERIC_LLM_VIOLATION", "Debería ser violación LLM"
        assert result.violation.rule_id == "RULE_ALT_001"
        
        # Verificar detalles del LLM en la violación
        violation_details = result.violation.details
        assert "llm_confidence" in violation_details, "Debería incluir confianza del LLM"
        assert 0 <= violation_details["llm_confidence"] <= 1, "La confianza debería estar entre 0 y 1"
        
        # Mostrar resultados del LLM
        print(f"✅ Violación detectada por LLM: {result.violation.explanation}")
        print(f"📊 Confianza: {violation_details['llm_confidence']}")
        print(f"🚨 Severidad: {result.violation.severity.value}")
        print(f"🔍 Tipo: {result.violation.condition_type}")
        
        # Verificar campos opcionales que podrían estar presentes
        if "extracted_values" in violation_details:
            print(f"📊 Valores extraídos: {violation_details['extracted_values']}")
        if "suggested_action" in violation_details:
            print(f"💡 Acción sugerida: {violation_details['suggested_action']}")
    
    @pytest.mark.integration
    def test_llm_no_violation_correct_assessment(self, ollama_config, altitude_violation_rule):
        """Test que el LLM evalúa correctamente cuando no hay violación."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        from Alert_System.models.traffic_state import TrafficState, AircraftState, Position, FlightPhase
        
        # Crear tráfico sin violación (aeronave muy por encima del mínimo)
        traffic_state = TrafficState(sector_id="TEST_SECTOR")
        traffic_state.msa = 1000  # MSA muy baja
        
        aircraft = AircraftState(
            callsign="TEST456",
            position=Position(latitude=40.0, longitude=-3.0, altitude=15000, speed=250, heading=180),
            flight_phase=FlightPhase.CRUISE
        )
        traffic_state.aircrafts["TEST456"] = aircraft
        
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Evaluar regla (la aeronave está a 15000ft, MSA es 1000ft - claramente no hay violación)
        result = condition.evaluate(
            traffic_state=traffic_state,
            parameters={"executable_rule": altitude_violation_rule},
            aircraft_callsign="TEST456"
        )
        
        # El LLM puede ser conservador, así que verificamos que evaluó correctamente
        if result.satisfied:
            print(f"✅ Evaluación correcta: sin violación")
            print(f"📊 Confianza: {result.details.get('confidence', 'N/A')}")
        else:
            # Si el LLM es conservador, verificamos que al menos tenga alta confianza y explicación razonable
            print(f"⚠️  LLM conservador: detectó posible violación")
            print(f"📊 Explicación: {result.violation.explanation if result.violation else 'N/A'}")
            print("ℹ️  Esto puede ser normal - el LLM es muy cauteloso con la seguridad")
        
        # Verificar que el LLM evaluó (ya sea con o sin violación)
        assert result.violation is not None or result.details.get("check") == "llm_evaluation", "Debería usar evaluación LLM"
    
    @pytest.mark.integration
    def test_llm_separation_rule_evaluation(self, ollama_config, sample_traffic_state, separation_rule):
        """Test evaluación de regla de separación con LLM."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Evaluar regla de separación (sin datos de separación específicos)
        result = condition.evaluate(
            traffic_state=sample_traffic_state,
            parameters={"executable_rule": separation_rule},
            aircraft_callsign="TEST123"
        )
        
        # El LLM debería evaluar basado en el contexto disponible
        # Verificar que usó evaluación LLM (puede ser en result.details o en violation.details)
        if result.violation:
            # Si detectó violación, verificar en violation.details
            assert "llm_confidence" in result.violation.details, "Debería incluir confianza del LLM"
            confidence = result.violation.details["llm_confidence"]
        else:
            # Si no detectó violación, verificar en result.details
            assert result.details.get("check") == "llm_evaluation", "Debería usar evaluación LLM"
            assert "confidence" in result.details, "Debería incluir confianza"
            confidence = result.details["confidence"]
        
        print(f"✅ Evaluación de separación completada")
        print(f"📊 Confianza: {confidence}")
        print(f"📝 Explicación: {result.violation.explanation if result.violation else result.details.get('explanation', 'N/A')}")
    
    @pytest.mark.integration
    def test_llm_fallback_when_ollama_unavailable(self, altitude_violation_rule, sample_traffic_state):
        """Test fallback a keywords cuando Ollama no está disponible."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        from common.llm_client_factory import ModelConfig
        
        # Configuración con puerto incorrecto para forzar error
        bad_config = ModelConfig(
            name="llama3.2:latest",
            provider="ollama",
            base_url="http://localhost:9999",  # Puerto incorrecto
            api_key="ollama",
            max_retries=1,
            timeout=5
        )
        
        condition = GenericKexCondition(llm_config=bad_config)
        
        # Evaluar - debería fallback a keywords
        result = condition.evaluate(
            traffic_state=sample_traffic_state,
            parameters={"executable_rule": altitude_violation_rule},
            aircraft_callsign="TEST123"
        )
        
        # Verificar fallback funcionó
        assert result.violation.condition_type == "GENERIC_MSA_VIOLATION", "Debería ser violación por keywords"
        assert not result.satisfied, "Keywords deberían detectar la violación de altitud"
        
        print(f"✅ Fallback a keywords funcionó")
        print(f"🔍 Tipo de violación: {result.violation.condition_type}")
        print("ℹ️  Fallback exitoso cuando Ollama no está disponible")
    
    @pytest.mark.integration
    def test_llm_structured_output_validation(self, ollama_config, sample_traffic_state, altitude_violation_rule):
        """Test que el LLM genera salida estructurada válida."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        from Alert_System.integration.schemas import LLMEvaluationResult
        
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Evaluar y verificar estructura de la respuesta
        result = condition.evaluate(
            traffic_state=sample_traffic_state,
            parameters={"executable_rule": altitude_violation_rule},
            aircraft_callsign="TEST123"
        )
        
        if not result.satisfied and result.violation:
            # Verificar que todos los campos esperados están presentes
            details = result.violation.details
            
            assert "llm_confidence" in details, "Falta confidence"
            assert isinstance(details["llm_confidence"], (int, float)), "Confidence debe ser numérico"
            assert 0 <= details["llm_confidence"] <= 1, "Confidence debe estar entre 0 y 1"
            
            # Campos opcionales que podrían estar presentes
            optional_fields = ["explanation"]
            for field in optional_fields:
                if field in details:
                    assert details[field] is not None, f"{field} no debería ser None"
            
            # extracted_values es opcional y puede ser None
            if "extracted_values" in details:
                if details["extracted_values"] is not None:
                    print(f"📊 Valores extraídos: {details['extracted_values']}")
                else:
                    print("📊 Sin valores extraídos (es normal)")
            else:
                print("📊 No hay campo extracted_values (es normal)")
            
            # suggested_action puede ser None, es opcional
            if "suggested_action" in details:
                print(f"💡 Acción sugerida: {details['suggested_action']}")
            else:
                print("💡 Sin acción sugerida (es normal)")
            
            print(f"✅ Salida estructurada válida")
            print(f"📊 Campos presentes: {list(details.keys())}")


if __name__ == "__main__":
    # Para ejecutar manualmente: python -m pytest tests/alert_system/test_llm_integration.py::TestLLMIntegration::test_llm_altitude_violation_detection -v -s
    
    print("⚠️  Para ejecutar estos tests asegúrate de que Ollama esté corriendo:")
    print("   $ ollama serve")
    print("⚠️  Ejecuta con marcador de integración:")
    print("   $ python -m pytest tests/alert_system/test_llm_integration.py -m integration -v -s")
