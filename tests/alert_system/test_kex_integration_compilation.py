"""Tests de integración completa: KEX → KEXAdapter → Compilación → RuleEngine."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from Alert_System.integration.kex_adapter import KEXAdapter
from Alert_System.compilation.compiler import RuleCompiler
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.models.traffic_state import (
    TrafficState, AircraftState, Position, FlightPhase, RunwayState
)
from common.llm_client_factory import ModelConfig


class TestKexIntegrationCompilation:
    """Tests del flujo completo desde datos KEX reales hasta compilación."""
    
    @pytest.fixture
    def kex_data_dir(self):
        """Directorio con datos reales del KEX."""
        base_dir = Path(__file__).parent.parent.parent / "Knowledge_Extractor" / "output" / "10_kex_output"
        kex_dir = base_dir / "ICAO Standard Phraseology(gpt-oss:20b)"
        
        if not kex_dir.exists():
            pytest.skip(f"KEX data directory not found: {kex_dir}")
        
        return kex_dir
    
    @pytest.fixture
    def sample_kex_rules(self, kex_data_dir):
        """Carga reglas reales desde archivos KEX."""
        rules = []
        
        # Cargar reglas de los primeros 3 archivos JSON
        json_files = sorted(kex_data_dir.glob("pagina_*.json"))[:3]
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraer reglas de cada chunk
                for chunk in data.get('sentence_results', []):
                    chunk_rules = chunk.get('ner', {}).get('rules', [])
                    rules.extend(chunk_rules)
                    
            except Exception as e:
                print(f"⚠️ Error loading {json_file}: {e}")
        
        return rules
    
    @pytest.fixture
    def mock_llm_config(self):
        """Configuración LLM mock para tests."""
        config = Mock()
        config.name = "test-model"
        config.provider = "test"
        config.base_url = "http://test"
        return config
    
    def test_load_kex_rules(self, sample_kex_rules):
        """Verificar que podemos cargar reglas KEX reales."""
        assert len(sample_kex_rules) > 0, "No se cargaron reglas KEX"
        
        print(f"\n📋 Reglas KEX cargadas: {len(sample_kex_rules)}")
        
        # Mostrar algunas reglas de ejemplo
        for i, rule in enumerate(sample_kex_rules[:3]):
            print(f"\n📄 Regla {i+1}: {rule.get('id', 'UNKNOWN')}")
            print(f"   Tipo: {rule.get('rule_type', 'unknown')}")
            print(f"   Modalidad: {rule.get('modality', 'unknown')}")
            
            trigger = rule.get('trigger', {})
            if isinstance(trigger, dict):
                trigger_text = trigger.get('description', 'No description')
            else:
                trigger_text = str(trigger)
            print(f"   Trigger: {trigger_text[:100]}...")
    
    def test_kex_adapter_with_real_rules(self, sample_kex_rules):
        """Probar KEXAdapter con reglas reales del KEX."""
        print(f"\n🔄 Adaptando {len(sample_kex_rules)} reglas KEX...")
        
        adapter = KEXAdapter()
        
        # Convertir reglas KEX a formato interno
        from Knowledge_Extractor import Rule
        
        kex_rule_objects = []
        for rule_data in sample_kex_rules:
            try:
                # Crear objeto Rule del KEX
                rule = Rule(
                    id=rule_data.get('id', 'UNKNOWN'),
                    rule_type=rule_data.get('rule_type', 'unknown'),
                    modality=rule_data.get('modality', 'unknown'),
                    trigger=rule_data.get('trigger', {}),
                    formal_if_then=rule_data.get('formal_if_then', {}),
                    severity=rule_data.get('severity', 'MEDIUM'),
                    safety_critical=rule_data.get('safety_critical', False)
                )
                kex_rule_objects.append(rule)
            except Exception as e:
                print(f"⚠️ Error creando Rule object: {e}")
        
        print(f"✅ {len(kex_rule_objects)} objetos Rule creados")
        
        # Adaptar reglas a evaluadores
        evaluators = adapter.adapt_rules(kex_rule_objects)
        
        print(f"📊 {len(evaluators)} evaluadores creados")
        
        # Mostrar tipos de evaluadores creados
        evaluator_types = {}
        for evaluator in evaluators:
            evaluator_type = type(evaluator).__name__
            evaluator_types[evaluator_type] = evaluator_types.get(evaluator_type, 0) + 1
        
        print("📋 Tipos de evaluadores:")
        for eval_type, count in evaluator_types.items():
            print(f"  - {eval_type}: {count}")
        
        assert len(evaluators) > 0, "No se crearon evaluadores"
    
    def test_compile_real_kex_rules(self, sample_kex_rules, mock_llm_config):
        """Intentar compilar reglas KEX reales."""
        print(f"\n🔨 Compilando {len(sample_kex_rules)} reglas KEX reales...")
        
        # Convertir a ExecutableRule
        adapter = KEXAdapter(llm_config=mock_llm_config)
        executables = []
        
        for rule_data in sample_kex_rules[:5]:  # Limitar a 5 para el test
            try:
                # Crear objeto Rule del KEX
                from Knowledge_Extractor import Rule
                rule = Rule(
                    id=rule_data.get('id', 'UNKNOWN'),
                    rule_type=rule_data.get('rule_type', 'unknown'),
                    modality=rule_data.get('modality', 'unknown'),
                    trigger=rule_data.get('trigger', {}),
                    formal_if_then=rule_data.get('formal_if_then', {}),
                    severity=rule_data.get('severity', 'MEDIUM'),
                    safety_critical=rule_data.get('safety_critical', False)
                )
                
                # Convertir a ExecutableRule
                executable = adapter.compile_to_executable(rule)
                executables.append(executable)
                
            except Exception as e:
                print(f"⚠️ Error procesando regla {rule_data.get('id')}: {e}")
        
        print(f"📋 {len(executables)} reglas ejecutables creadas")
        
        # Mock del compilador para evitar llamadas LLM reales
        compiler = RuleCompiler(llm_config=mock_llm_config)
        compiler._instructor_client = Mock()
        compiler._raw_client = Mock()
        
        # Mock de respuesta LLM
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
def evaluate(traffic_state, callsign=None):
    aircraft = traffic_state.get_aircraft(callsign) if callsign else None
    if callsign and not aircraft:
        return {"satisfied": True, "details": {}, "explanation": "Aircraft not found", "severity": "INFO"}
    
    msa = traffic_state.msa or 5000
    if aircraft and aircraft.position.altitude < msa:
        return {
            "satisfied": False,
            "details": {"altitude": aircraft.position.altitude, "msa": msa},
            "explanation": f"Aircraft below MSA",
            "severity": "HIGH"
        }
    
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
"""
        compiler._raw_client.chat.completions.create.return_value = mock_response
        
        # Compilar reglas
        compiled_rules = []
        for executable in executables:
            try:
                compiled = compiler.compile_executable_rule(executable)
                compiled_rules.append(compiled)
                print(f"✅ {executable.source_rule_id}: compiled")
            except Exception as e:
                print(f"❌ {executable.source_rule_id}: failed - {e}")
        
        print(f"\n📊 Resultados compilación:")
        print(f"  Compiladas: {len([r for r in compiled_rules if r.compilation_status.value == 'compiled'])}")
        print(f"  Fallidas: {len([r for r in compiled_rules if r.compilation_status.value == 'failed'])}")
        
        assert len(compiled_rules) > 0, "No se compiló ninguna regla"
    
    def test_full_pipeline_simulation(self, sample_kex_rules, mock_llm_config):
        """Simulación completa del pipeline con datos reales."""
        print(f"\n🚀 Pipeline completo con {len(sample_kex_rules)} reglas KEX")
        
        # Paso 1: KEX → KEXAdapter → ExecutableRules
        adapter = KEXAdapter(llm_config=mock_llm_config)
        executables = []
        
        for rule_data in sample_kex_rules[:3]:  # Limitar a 3 para el test
            try:
                from Knowledge_Extractor import Rule
                rule = Rule(
                    id=rule_data.get('id', 'UNKNOWN'),
                    rule_type=rule_data.get('rule_type', 'unknown'),
                    modality=rule_data.get('modality', 'unknown'),
                    trigger=rule_data.get('trigger', {}),
                    formal_if_then=rule_data.get('formal_if_then', {}),
                    severity=rule_data.get('severity', 'MEDIUM'),
                    safety_critical=rule_data.get('safety_critical', False)
                )
                
                executable = adapter.compile_to_executable(rule)
                executables.append(executable)
                print(f"📋 {executable.source_rule_id}: {executable.rule_category}")
                
            except Exception as e:
                print(f"⚠️ Error en regla {rule_data.get('id')}: {e}")
        
        # Paso 2: Mock compilación con guardado incremental
        compiler = RuleCompiler(llm_config=mock_llm_config)
        compiler._instructor_client = Mock()
        compiler._raw_client = Mock()
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "Test rule", "severity": "INFO"}
"""
        compiler._raw_client.chat.completions.create.return_value = mock_response
        
        # Paso 3: Compilar con guardado incremental en directorio temporal
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = compiler.compile_batch(
                executables, 
                save_incrementally=True, 
                output_dir=temp_dir
            )
            
            # Las reglas ya se guardaron incrementalmente durante la compilación
            loader = CompiledRuleLoader(compiled_rules_dir=temp_dir, llm_config=mock_llm_config)
            
            # Verificar cuántas se guardaron
            saved_count = 0
            for rule_id, rule in manifest.rules.items():
                if rule.compilation_status.value == 'compiled':
                    rule_file = Path(temp_dir) / f"{rule_id}.py"
                    if rule_file.exists():
                        saved_count += 1
            
            print(f"\n💾 {saved_count} reglas guardadas en {temp_dir}")
            
            # Paso 4: Cargar en RuleEngine
            loader.load_all_compiled_rules()
            rule_engine = RuleEngine()
            loaded_count = loader.register_in_engine(rule_engine)
            
            print(f"🔧 {loaded_count} reglas registradas en RuleEngine")
            
            # Paso 5: Evaluar con datos de prueba
            traffic_state = TrafficState(
                sector_id="TEST",
                msa=5000,
                aircrafts={
                    "TEST123": AircraftState(
                        callsign="TEST123",
                        position=Position(40.0, -3.0, 6000, 90, 250),
                        flight_phase=FlightP.hase.CRUISE,
                    )
                }
            )
            
            # Evaluar reglas compiladas
            violations = 0
            for condition_type in rule_engine.get_registered_evaluators():
                if condition_type.startswith("COMPILED_"):
                    result = rule_engine.evaluate(
                        condition_type=condition_type,
                        parameters={},
                        traffic_state=traffic_state,
                        aircraft_callsign="TEST123",
                    )
                    if not result.satisfied:
                        violations += 1
            
            print(f"📊 Evaluación: {violations} violaciones detectadas")
            
            assert loaded_count > 0, "No se registraron reglas en RuleEngine"


def test_demo_with_real_kex_data():
    """Demo que usa datos reales del KEX (marcado como slow test)."""
    pytest.skip("Slow test - requires real KEX data and LLM")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
