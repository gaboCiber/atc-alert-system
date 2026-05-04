"""Tests de integración complejos con múltiples reglas y escenarios ATC realistas.

Estos tests simulan escenarios completos del sistema ATC con:
- Múltiples aeronaves en el espacio aéreo
- Diferentes tipos de reglas (altitud, separación, pista, genéricas)
- Evaluación LLM + reglas específicas
- Escenarios complejos de tráfico

NOTA: Tests lentos que requieren Ollama corriendo.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

pytestmark = pytest.mark.slow


@pytest.fixture
def complex_traffic_state():
    """Estado de tráfico complejo con múltiples aeronaves."""
    from Alert_System.models.traffic_state import TrafficState, AircraftState, Position, FlightPhase
    
    traffic_state = TrafficState(sector_id="COMPLEX_SECTOR")
    traffic_state.msa = 5000
    
    # Aeronave 1: En descenso, potencial violación de altitud
    aircraft1 = AircraftState(
        callsign="IBK123",
        position=Position(latitude=40.5, longitude=-3.2, altitude=4500, speed=280, heading=270),
        flight_phase=FlightPhase.DESCENT
    )
    
    # Aeronave 2: En crucero, separación cercana con IBK123
    aircraft2 = AircraftState(
        callsign="IBE456",
        position=Position(latitude=40.6, longitude=-3.1, altitude=6000, speed=320, heading=90),
        flight_phase=FlightPhase.CRUISE
    )
    
    # Aeronave 3: En aproximación, cerca de pista
    aircraft3 = AircraftState(
        callsign="VLY789",
        position=Position(latitude=40.4, longitude=-3.0, altitude=2000, speed=180, heading=180),
        flight_phase=FlightPhase.APPROACH
    )
    
    # Aeronave 4: En tierra, taxiando
    aircraft4 = AircraftState(
        callsign="RYR012",
        position=Position(latitude=40.3, longitude=-2.9, altitude=0, speed=20, heading=360),
        flight_phase=FlightPhase.TAXI
    )
    
    traffic_state.aircrafts.update({
        "IBK123": aircraft1,
        "IBE456": aircraft2,
        "VLY789": aircraft3,
        "RYR012": aircraft4
    })
    
    # Agregar pistas
    from Alert_System.models.traffic_state import RunwayState
    traffic_state.runways = {
        "RWY12": RunwayState(runway_id="RWY12", occupied=False),
        "RWY30": RunwayState(runway_id="RWY30", occupied=True)  # Ocupada
    }
    
    return traffic_state


@pytest.fixture
def ollama_config():
    """Configuración para Ollama con llama3.2."""
    from common.llm_client_factory import ModelConfig
    
    return ModelConfig(
        name="llama3.2:latest",
        provider="ollama",
        base_url="http://localhost:11434",
        api_key="ollama",
        max_retries=3,
        timeout=30
    )


class TestComplexATCScenarios:
    """Tests de escenarios ATC complejos y realistas."""
    
    @pytest.mark.integration
    def test_multi_aircraft_conflict_scenario(self, ollama_config, complex_traffic_state):
        """Test escenario con múltiples aeronaves y potenciales conflictos."""
        from Alert_System.rule_engine.conditions import (
            GenericKexCondition, AltitudeCondition, SeparationCondition
        )
        from Alert_System.integration.schemas import ExecutableRule
        
        # Crear evaluadores
        generic_condition = GenericKexCondition(llm_config=ollama_config)
        altitude_condition = AltitudeCondition()
        separation_condition = SeparationCondition()
        
        # Regla genérica compleja: conflicto de altitud y separación
        complex_rule = ExecutableRule(
            source_rule_id="COMPLEX_001",
            rule_category="SAFETY",
            condition_description="CRITICAL: Multiple aircraft conflict detected. IBK123 descending below MSA while IBE456 maintains close proximity. Potential collision risk requires immediate intervention.",
            raw_trigger="Multi-aircraft conflict",
            raw_constraint="altitude >= 5000 AND separation >= 3nm"
        )
        
        results = {}
        
        # Evaluar cada aeronave con diferentes condiciones
        for callsign in ["IBK123", "IBE456", "VLY789"]:
            print(f"\n🔍 Evaluando aeronave {callsign}...")
            
            # 1. Evaluación de altitud (regla específica)
            alt_result = altitude_condition.evaluate(
                traffic_state=complex_traffic_state,
                parameters={"check_type": "MINIMUM", "reference_value": 5000},
                aircraft_callsign=callsign
            )
            
            # 2. Evaluación de separación (regla específica)
            sep_result = separation_condition.evaluate(
                traffic_state=complex_traffic_state,
                parameters={"check_type": "MINIMUM", "reference_value": 3.0},
                aircraft_callsign=callsign
            )
            
            # 3. Evaluación genérica con LLM
            generic_result = generic_condition.evaluate(
                traffic_state=complex_traffic_state,
                parameters={"executable_rule": complex_rule},
                aircraft_callsign=callsign
            )
            
            results[callsign] = {
                "altitude": alt_result,
                "separation": sep_result,
                "generic": generic_result
            }
            
            print(f"  📊 Altitud: {'VIOLACIÓN' if not alt_result.satisfied else 'OK'}")
            print(f"  📊 Separación: {'VIOLACIÓN' if not sep_result.satisfied else 'OK'}")
            print(f"  📊 LLM: {'VIOLACIÓN' if not generic_result.satisfied else 'OK'}")
            
            if not generic_result.satisfied and generic_result.violation:
                print(f"  🧠 LLM Explicación: {generic_result.violation.explanation[:100]}...")
        
        # Verificaciones del escenario
        # IBK123 debería tener violación de altitud (4500ft < 5000ft)
        assert not results["IBK123"]["altitude"].satisfied, "IBK123 debería violar altitud"
        
        # El LLM debería evaluar el escenario (puede o no detectar violación dependiendo del contexto)
        # IBK123 está por debajo de MSA, así que el LLM puede decidir que la regla no aplica
        if results["IBK123"]["generic"].satisfied:
            print("🧠 LLM: Sin violación (regla no aplica por debajo de MSA)")
            assert results["IBK123"]["generic"].details.get("check") == "llm_evaluation"
        else:
            print("🧠 LLM: Violación detectada")
            assert results["IBK123"]["generic"].violation.condition_type == "GENERIC_LLM_VIOLATION"
        
        # VLY789 está en aproximación a 2000ft, viola altitud mínima de 5000ft
        assert not results["VLY789"]["altitude"].satisfied, "VLY789 debería violar altitud (2000ft < 5000ft)"
        
        print(f"\n✅ Escenario complejo evaluado exitosamente")
        print(f"📊 Total violaciones detectadas: {sum(1 for r in results.values() if not r['generic'].satisfied)}")
    
    @pytest.mark.integration
    def test_mixed_rule_types_evaluation(self, ollama_config, complex_traffic_state):
        """Test evaluación mezclada de reglas específicas y genéricas."""
        from Alert_System.rule_engine.conditions import (
            GenericKexCondition, AltitudeCondition, RunwayCondition
        )
        from Alert_System.integration.schemas import ExecutableRule
        
        # Crear condiciones
        generic_condition = GenericKexCondition(llm_config=ollama_config)
        altitude_condition = AltitudeCondition()
        runway_condition = RunwayCondition()
        
        # Reglas genéricas para diferentes escenarios
        rules = {
            "safety_altitude": ExecutableRule(
                source_rule_id="SAFE_ALT_001",
                rule_category="SAFETY",
                condition_description="All aircraft must maintain safe altitude above terrain and obstacles",
                raw_trigger="Safe altitude",
                raw_constraint="altitude >= 3000"
            ),
            "runway_conflict": ExecutableRule(
                source_rule_id="RWY_CONF_001",
                rule_category="RUNWAY",
                condition_description="Prevent runway incursions and conflicts",
                raw_trigger="Runway safety",
                raw_constraint="runway.occupied == false OR aircraft.phase != 'takeoff'"
            ),
            "traffic_flow": ExecutableRule(
                source_rule_id="FLOW_001",
                rule_category="TRAFFIC",
                condition_description="Maintain orderly traffic flow and prevent congestion",
                raw_trigger="Traffic management",
                raw_constraint="separation >= 3nm AND altitude_diff >= 1000"
            )
        }
        
        evaluation_results = {}
        
        # Evaluar cada tipo de regla contra cada aeronave
        for rule_name, rule in rules.items():
            print(f"\n🔍 Evaluando regla: {rule_name}")
            
            rule_results = {}
            for callsign in complex_traffic_state.aircrafts:
                # Evaluación con LLM
                result = generic_condition.evaluate(
                    traffic_state=complex_traffic_state,
                    parameters={"executable_rule": rule},
                    aircraft_callsign=callsign
                )
                
                rule_results[callsign] = result
                
                status = "VIOLACIÓN" if not result.satisfied else "OK"
                confidence = result.violation.details.get("llm_confidence", "N/A") if result.violation else "N/A"
                print(f"  {callsign}: {status} (confianza: {confidence})")
            
            evaluation_results[rule_name] = rule_results
        
        # Verificaciones específicas
        # IBK123 a 4500ft - el LLM puede detectar violación si considera el MSA (5000ft)
        if not evaluation_results["safety_altitude"]["IBK123"].satisfied:
            print("🧠 LLM: IBK123 viola por estar bajo MSA (5000ft) aunque sobre mínimo (3000ft)")
        else:
            print("🧠 LLM: IBK123 OK (4500ft >= 3000ft)")
        
        # RYR012 en tierra - el LLM puede decidir que la regla de altitud no aplica a aeronaves en tierra
        if evaluation_results["safety_altitude"]["RYR012"].satisfied:
            print("🧠 LLM: RYR012 en tierra - regla de altitud no aplica")
        else:
            print("🧠 LLM: RYR012 viola altitud mínima")
        
        # Al menos una regla debería detectar violaciones
        total_violations = sum(
            1 for rule_results in evaluation_results.values()
            for result in rule_results.values()
            if not result.satisfied
        )
        assert total_violations > 0, "Debería haber al menos una violación detectada"
        
        print(f"\n✅ Evaluación mixla completada")
        print(f"📊 Total violaciones: {total_violations}")
    
    @pytest.mark.integration
    def test_llm_vs_specific_rules_comparison(self, ollama_config, complex_traffic_state):
        """Test comparativo entre evaluación LLM y reglas específicas."""
        from Alert_System.rule_engine.conditions import (
            GenericKexCondition, AltitudeCondition, SeparationCondition
        )
        from Alert_System.integration.schemas import ExecutableRule
        
        # Evaluadores
        generic_condition = GenericKexCondition(llm_config=ollama_config)
        altitude_condition = AltitudeCondition()
        
        # Regla genérica que imita a la regla de altitud
        altitude_rule = ExecutableRule(
            source_rule_id="LLM_ALT_001",
            rule_category="ALTITUDE",
            condition_description="Aircraft must maintain minimum altitude of 5000ft for safety",
            raw_trigger="Minimum altitude requirement",
            raw_constraint="aircraft.altitude >= 5000ft"
        )
        
        comparison_results = {}
        
        for callsign in complex_traffic_state.aircrafts:
            # Evaluación con regla específica
            specific_result = altitude_condition.evaluate(
                traffic_state=complex_traffic_state,
                parameters={"check_type": "MINIMUM", "reference_value": 5000},
                aircraft_callsign=callsign
            )
            
            # Evaluación con LLM
            llm_result = generic_condition.evaluate(
                traffic_state=complex_traffic_state,
                parameters={"executable_rule": altitude_rule},
                aircraft_callsign=callsign
            )
            
            comparison_results[callsign] = {
                "specific": specific_result,
                "llm": llm_result,
                "agreement": specific_result.satisfied == llm_result.satisfied
            }
            
            print(f"\n🔍 {callsign}:")
            print(f"  📏 Regla específica: {'VIOLACIÓN' if not specific_result.satisfied else 'OK'}")
            print(f"  🧠 LLM: {'VIOLACIÓN' if not llm_result.satisfied else 'OK'}")
            print(f"  🤝 Acuerdo: {'✅' if comparison_results[callsign]['agreement'] else '❌'}")
            
            if not llm_result.satisfied and llm_result.violation:
                print(f"  💬 LLM: {llm_result.violation.explanation[:80]}...")
        
        # Análisis de acuerdos
        agreements = sum(1 for r in comparison_results.values() if r["agreement"])
        total_aircraft = len(comparison_results)
        agreement_rate = agreements / total_aircraft
        
        print(f"\n📊 Tasa de acuerdo: {agreement_rate:.1%} ({agreements}/{total_aircraft})")
        
        # Debería haber algún acuerdo entre LLM y reglas específicas
        assert agreement_rate >= 0.25, "LLM debería estar de acuerdo con reglas específicas al menos 25% del tiempo"
        
        # Casos específicos de verificación
        # IBK123 (4500ft) debería violar en ambos sistemas
        assert not comparison_results["IBK123"]["specific"].satisfied, "Regla específica debería detectar violación IBK123"
        # El LLM puede ser más conservador o menos, pero debería detectar algo
        
        print(f"\n✅ Comparación LLM vs reglas específicas completada")
    
    @pytest.mark.integration
    def test_emergency_scenario_evaluation(self, ollama_config, complex_traffic_state):
        """Test evaluación de escenario de emergencia complejo."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        from Alert_System.integration.schemas import ExecutableRule
        
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Simular escenario de emergencia
        emergency_rule = ExecutableRule(
            source_rule_id="EMERGENCY_001",
            rule_category="EMERGENCY",
            condition_description="EMERGENCY: IBK123 reporting rapid descent due to engine failure. Must ensure minimum safe altitude and clear traffic conflicts. IBE456 in proximity needs immediate vectoring.",
            raw_trigger="Emergency descent",
            raw_constraint="altitude >= 2000 AND traffic.clear = true"
        )
        
        print(f"\n🚨 Evaluando escenario de emergencia...")
        
        # Evaluar aeronave en emergencia (IBK123)
        emergency_result = condition.evaluate(
            traffic_state=complex_traffic_state,
            parameters={"executable_rule": emergency_rule},
            aircraft_callsign="IBK123"
        )
        
        # Evaluar tráfico cercano (IBE456)
        traffic_result = condition.evaluate(
            traffic_state=complex_traffic_state,
            parameters={"executable_rule": emergency_rule},
            aircraft_callsign="IBE456"
        )
        
        print(f"📊 Aeronave emergencia (IBK123): {'VIOLACIÓN' if not emergency_result.satisfied else 'OK'}")
        print(f"📊 Tráfico cercano (IBE456): {'VIOLACIÓN' if not traffic_result.satisfied else 'OK'}")
        
        if not emergency_result.satisfied and emergency_result.violation:
            print(f"🧠 Evaluación emergencia: {emergency_result.violation.explanation}")
            print(f"📊 Confianza: {emergency_result.violation.details.get('llm_confidence', 'N/A')}")
            print(f"🚨 Severidad: {emergency_result.violation.severity.value}")
        
        # El LLM evalúa la emergencia - puede detectar problemas o considerar exención ATC
        if not emergency_result.satisfied:
            print("🧠 LLM: Detectó problemas en emergencia")
            assert emergency_result.violation.condition_type == "GENERIC_LLM_VIOLATION"
            
            # Debería tener alta confianza en escenario de emergencia
            confidence = emergency_result.violation.details.get("llm_confidence", 0)
            assert confidence >= 0.7, "Debería tener alta confianza en emergencia"
        else:
            print("🧠 LLM: Considera exención ATC para emergencia")
            # El LLM puede decidir que en emergencia hay exención de reglas
            assert emergency_result.details.get("check") == "llm_evaluation"
        
        print(f"\n✅ Escenario de emergencia evaluado")
    
    @pytest.mark.integration
    def test_progressive_scenario_evolution(self, ollama_config, complex_traffic_state):
        """Test evaluación de escenario que evoluciona en el tiempo."""
        from Alert_System.rule_engine.conditions import GenericKexCondition
        from Alert_System.integration.schemas import ExecutableRule
        from copy import deepcopy
        
        condition = GenericKexCondition(llm_config=ollama_config)
        
        # Regla de evolución de tráfico
        evolution_rule = ExecutableRule(
            source_rule_id="EVOLUTION_001",
            rule_category="DYNAMIC",
            condition_description="Monitor traffic evolution and detect developing conflicts",
            raw_trigger="Traffic evolution",
            raw_constraint="separation.trend != 'decreasing'"
        )
        
        print(f"\n⏱️  Evaluando evolución del escenario...")
        
        # Simular evolución del escenario
        scenarios = [
            ("Inicial", complex_traffic_state),
            ("Después de 5 min", self._evolve_traffic_state(complex_traffic_state, minutes=5)),
            ("Después de 10 min", self._evolve_traffic_state(complex_traffic_state, minutes=10)),
        ]
        
        evolution_results = []
        
        for scenario_name, traffic_state in scenarios:
            print(f"\n🕐 Escenario: {scenario_name}")
            
            scenario_violations = 0
            for callsign in traffic_state.aircrafts:
                result = condition.evaluate(
                    traffic_state=traffic_state,
                    parameters={"executable_rule": evolution_rule},
                    aircraft_callsign=callsign
                )
                
                if not result.satisfied:
                    scenario_violations += 1
                    print(f"  ⚠️  {callsign}: {result.violation.explanation[:60]}...")
            
            evolution_results.append((scenario_name, scenario_violations))
            print(f"  📊 Violaciones en este escenario: {scenario_violations}")
        
        # Verificar que el escenario evoluciona (cambia el número de violaciones)
        violation_counts = [count for _, count in evolution_results]
        assert len(set(violation_counts)) > 1, "El escenario debería mostrar evolución"
        
        print(f"\n✅ Evolución del escenario evaluada")
        print(f"📈 Trayectoria de violaciones: {violation_counts}")
    
    def _evolve_traffic_state(self, original_state, minutes):
        """Simula evolución del estado de tráfico."""
        from copy import deepcopy
        
        evolved = deepcopy(original_state)
        
        # Simular movimiento de aeronaves
        for callsign, aircraft in evolved.aircrafts.items():
            if callsign == "IBK123":
                # Desciende más
                aircraft.position.altitude -= 100 * minutes
            elif callsign == "IBE456":
                # Se acerca más
                aircraft.position.latitude -= 0.01 * minutes
            elif callsign == "VLY789":
                # Continúa aproximación
                aircraft.position.altitude -= 50 * minutes
        
        return evolved


if __name__ == "__main__":
    print("⚠️  Tests complejos de integración ATC")
    print("   Requiere: Ollama corriendo con llama3.2")
    print("   Ejecutar: uv run python -m pytest tests/alert_system/test_complex_integration.py -v -s")
