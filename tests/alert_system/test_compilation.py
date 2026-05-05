"""Tests para el módulo de compilación de reglas."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from Alert_System.compilation.schemas import CompiledRule, CompilationManifest, CompilationStatus
from Alert_System.compilation.validator import validate_code, validate_return_structure, CodeValidationError
from Alert_System.compilation.compiler import RuleCompiler
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.conditions import CompiledCondition
from Alert_System.integration.schemas import ExecutableRule


class TestCodeValidator:
    """Tests para validación de código generado por LLM."""
    
    def test_validate_safe_code(self):
        """Código seguro debe pasar validación."""
        safe_code = """
def evaluate(traffic_state, callsign=None):
    aircraft = traffic_state.get_aircraft(callsign) if callsign else None
    if callsign and not aircraft:
        return {"satisfied": True, "details": {}, "explanation": "Aircraft not found", "severity": "INFO"}
    
    msa = traffic_state.msa or 5000
    aircrafts_to_check = [aircraft] if aircraft else list(traffic_state.aircrafts.values())
    
    worst_result = {"satisfied": True, "details": {}, "explanation": "All aircraft above MSA", "severity": "INFO"}
    for ac in aircrafts_to_check:
        alt = ac.position.altitude
        if alt < msa:
            result = {
                "satisfied": False,
                "details": {"callsign": ac.callsign, "altitude": alt, "msa": msa, "difference_ft": msa - alt},
                "explanation": f"{ac.callsign} at {alt}ft is below MSA {msa}ft",
                "severity": "HIGH"
            }
            if worst_result["satisfied"]:
                worst_result = result
    return worst_result
"""
        is_valid, issues = validate_code(safe_code)
        assert is_valid, f"Safe code should pass validation: {issues}"
    
    def test_validate_forbidden_imports(self):
        """Imports prohibidos deben fallar validación."""
        forbidden_code = """
import os
def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
"""
        is_valid, issues = validate_code(forbidden_code)
        assert not is_valid
        assert any("Forbidden import" in issue for issue in issues)
    
    def test_validate_forbidden_names(self):
        """Nombres prohibidos deben fallar validación."""
        forbidden_code = """
def evaluate(traffic_state, callsign=None):
    exec('print("malicious")')
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
"""
        is_valid, issues = validate_code(forbidden_code)
        assert not is_valid
        assert any("Forbidden name access" in issue for issue in issues)
    
    def test_validate_syntax_error(self):
        """Errores de sintaxis deben fallar validación."""
        invalid_code = """
def evaluate(traffic_state, callsign=None):
    if aircraft
        return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
"""
        is_valid, issues = validate_code(invalid_code)
        assert not is_valid
        assert any("Syntax error" in issue for issue in issues)
    
    def test_validate_missing_function(self):
        """Sin función evaluate debe fallar validación."""
        no_function_code = """
def some_other_function():
    return {}
"""
        is_valid, issues = validate_code(no_function_code)
        assert not is_valid
        assert any("No 'evaluate' function found" in issue for issue in issues)
    
    def test_validate_return_structure(self):
        """Validar estructura de retorno correcta."""
        good_return_code = """
def evaluate(traffic_state, callsign=None):
    return {
        "satisfied": True,
        "details": {"altitude": 5000},
        "explanation": "All good",
        "severity": "INFO"
    }
"""
        is_valid, issues = validate_return_structure(good_return_code)
        assert is_valid, f"Good return structure should pass: {issues}"
    
    def test_validate_missing_return_keys(self):
        """Keys faltantes deben fallar validación."""
        missing_keys_code = """
def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}}
"""
        is_valid, issues = validate_return_structure(missing_keys_code)
        assert not is_valid
        assert any("missing required keys" in issue for issue in issues)


class TestRuleCompiler:
    """Tests para el compilador de reglas."""
    
    @pytest.fixture
    def mock_llm_config(self):
        """Configuración LLM mock."""
        config = Mock()
        config.name = "test-model"
        config.provider = "test"
        config.base_url = "http://test"
        return config
    
    @pytest.fixture
    def mock_compiler(self, mock_llm_config):
        """Compilador con LLM mock."""
        compiler = RuleCompiler(llm_config=mock_llm_config)
        # Inicializar clientes mock
        compiler._instructor_client = Mock()
        compiler._raw_client = Mock()
        return compiler
    
    def test_compile_rule_success(self, mock_compiler):
        """Compilación exitosa de una regla."""
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
        
        with patch.object(mock_compiler, '_raw_client') as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            
            compiled = mock_compiler.compile_rule(
                rule_id="TEST_RULE",
                category="GENERIC",
                description="Check altitude above MSA",
                trigger="Aircraft below MSA",
                constraint="Maintain altitude above MSA",
                severity="HIGH",
                safety_critical=True,
            )
        
        assert compiled.compilation_status == CompilationStatus.COMPILED
        assert compiled.source_rule_id == "TEST_RULE"
        assert "def evaluate" in compiled.compiled_code
        assert compiled.compilation_metadata["attempts"] == 1
    
    def test_compile_rule_validation_failure(self, mock_compiler):
        """Fallo de validación debe marcar regla como FAILED."""
        # Mock de respuesta con código inválido
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
import os
def evaluate(traffic_state, callsign=None):
    os.system("rm -rf /")
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
"""
        
        with patch.object(mock_compiler, '_raw_client') as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            
            compiled = mock_compiler.compile_rule(
                rule_id="BAD_RULE",
                category="GENERIC",
                description="Malicious rule",
                max_retries=1,
            )
        
        assert compiled.compilation_status == CompilationStatus.FAILED
        assert "Static validation failed" in compiled.failure_reason
    
    def test_compile_executable_rule(self, mock_compiler):
        """Compilación desde ExecutableRule."""
        executable = ExecutableRule(
            source_rule_id="EXEC_TEST",
            rule_category="GENERIC",
            condition_description="Test rule",
            raw_trigger="Trigger",
            raw_constraint="Constraint",
            severity="MEDIUM",
        )
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "Test", "severity": "INFO"}
"""
        
        with patch.object(mock_compiler, '_raw_client') as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            
            compiled = mock_compiler.compile_executable_rule(executable)
        
        assert compiled.source_rule_id == "EXEC_TEST"
        assert compiled.rule_category == "GENERIC"
        assert compiled.compilation_status == CompilationStatus.COMPILED


class TestCompiledRuleLoader:
    """Tests para el cargador de reglas compiladas."""
    
    @pytest.fixture
    def temp_rules_dir(self, tmp_path):
        """Directorio temporal para reglas compiladas."""
        rules_dir = tmp_path / "compiled_rules"
        rules_dir.mkdir()
        return rules_dir
    
    @pytest.fixture
    def sample_manifest(self):
        """Manifest de ejemplo."""
        return CompilationManifest(
            model_used="test-model",
            rules={
                "RULE_001": CompiledRule(
                    source_rule_id="RULE_001",
                    rule_category="GENERIC",
                    condition_description="Test rule 1",
                    compiled_code="def evaluate(): return {'satisfied': True, 'details': {}, 'explanation': 'OK', 'severity': 'INFO'}",
                    compilation_status=CompilationStatus.COMPILED,
                ),
                "RULE_002": CompiledRule(
                    source_rule_id="RULE_002",
                    rule_category="GENERIC",
                    condition_description="Test rule 2",
                    compiled_code="def evaluate(): return {'satisfied': False, 'details': {}, 'explanation': 'FAIL', 'severity': 'HIGH'}",
                    compilation_status=CompilationStatus.FAILED,
                    failure_reason="Test failure",
                ),
            },
            total_compiled=1,
            total_failed=1,
            total_fallback=1,
        )
    
    def test_load_manifest(self, temp_rules_dir, sample_manifest):
        """Cargar manifest desde archivo."""
        # Guardar manifest
        manifest_path = temp_rules_dir / "manifest.json"
        manifest_path.write_text(sample_manifest.model_dump_json(indent=2))
        
        loader = CompiledRuleLoader(compiled_rules_dir=str(temp_rules_dir))
        loaded = loader.load_manifest()
        
        assert loaded is not None
        assert loaded.model_used == "test-model"
        assert len(loaded.rules) == 2
        assert "RULE_001" in loaded.rules
        assert "RULE_002" in loaded.rules
    
    def test_load_compiled_rule_from_file(self, temp_rules_dir):
        """Cargar regla compilada desde archivo .py."""
        rule_file = temp_rules_dir / "RULE_001.py"
        rule_file.write_text("""def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
""")
        
        loader = CompiledRuleLoader(compiled_rules_dir=str(temp_rules_dir))
        rule = loader.load_compiled_rule("RULE_001")
        
        assert rule is not None
        assert rule.source_rule_id == "RULE_001"
        assert rule.compilation_status == CompilationStatus.COMPILED
        assert "def evaluate" in rule.compiled_code
    
    def test_load_all_compiled_rules(self, temp_rules_dir, sample_manifest):
        """Cargar todas las reglas compiladas."""
        # Guardar manifest
        manifest_path = temp_rules_dir / "manifest.json"
        manifest_path.write_text(sample_manifest.model_dump_json(indent=2))
        
        # Crear archivo .py para RULE_001
        rule_file = temp_rules_dir / "RULE_001.py"
        rule_file.write_text("""def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
""")
        
        loader = CompiledRuleLoader(compiled_rules_dir=str(temp_rules_dir))
        all_rules = loader.load_all_compiled_rules()
        
        assert len(all_rules) == 1  # Solo RULE_001 (compilada)
        assert "RULE_001" in all_rules
        assert all_rules["RULE_001"].compilation_status == CompilationStatus.COMPILED
    
    def test_create_compiled_conditions(self, temp_rules_dir):
        """Crear CompiledCondition instances."""
        rule_file = temp_rules_dir / "RULE_001.py"
        rule_file.write_text("""def evaluate(traffic_state, callsign=None):
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
""")
        
        loader = CompiledRuleLoader(compiled_rules_dir=str(temp_rules_dir))
        conditions = loader.create_compiled_conditions()
        
        assert len(conditions) == 1
        assert isinstance(conditions[0], CompiledCondition)
        assert conditions[0].condition_id == "RULE_001"


class TestCompiledCondition:
    """Tests para CompiledCondition wrapper."""
    
    @pytest.fixture
    def sample_compiled_rule(self):
        """Regla compilada de ejemplo."""
        return CompiledRule(
            source_rule_id="TEST_RULE",
            rule_category="GENERIC",
            condition_description="Test altitude check",
            compiled_code="""
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
""",
            compilation_status=CompilationStatus.COMPILED,
        )
    
    @pytest.fixture
    def sample_traffic_state(self):
        """TrafficState de ejemplo."""
        from Alert_System.models.traffic_state import (
            TrafficState, AircraftState, Position, FlightPhase
        )
        
        return TrafficState(
            sector_id="TEST_SECTOR",
            msa=5000,
            aircrafts={
                "TEST123": AircraftState(
                    callsign="TEST123",
                    position=Position(
                        latitude=40.0, longitude=-3.0,
                        altitude=6000, heading=90, speed=250
                    ),
                    flight_phase=FlightPhase.CRUISE,
                ),
                "TEST456": AircraftState(
                    callsign="TEST456",
                    position=Position(
                        latitude=40.01, longitude=-3.01,
                        altitude=4500, heading=270, speed=200
                    ),
                    flight_phase=FlightPhase.DESCENT,
                ),
            },
        )
    
    def test_load_function_success(self, sample_compiled_rule):
        """Cargar función exitosamente."""
        condition = CompiledCondition(compiled_rule=sample_compiled_rule)
        
        evaluate_fn = condition._load_function()
        assert evaluate_fn is not None
        assert callable(evaluate_fn)
    
    def test_load_function_no_code(self):
        """Fallo cuando no hay código compilado."""
        empty_rule = CompiledRule(
            source_rule_id="EMPTY",
            rule_category="GENERIC",
            condition_description="Empty rule",
            compiled_code="",
            compilation_status=CompilationStatus.COMPILED,
        )
        
        condition = CompiledCondition(compiled_rule=empty_rule)
        evaluate_fn = condition._load_function()
        
        assert evaluate_fn is None
        assert condition._load_error == "No compiled code available"
    
    def test_evaluate_success(self, sample_compiled_rule, sample_traffic_state):
        """Evaluación exitosa."""
        condition = CompiledCondition(compiled_rule=sample_compiled_rule)
        
        # Aircraft above MSA
        result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign="TEST123")
        
        assert result.satisfied is True
        assert result.violation is None
        assert result.details["compiled"] is True
        assert result.details["rule_id"] == "TEST_RULE"
        
        # Aircraft below MSA
        result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign="TEST456")
        
        assert result.satisfied is False
        assert result.violation is not None
        assert result.violation.severity.value.lower() == "high"
        assert "below MSA" in result.violation.explanation
    
    def test_evaluate_no_callsign(self, sample_compiled_rule, sample_traffic_state):
        """Evaluación sin callsign específico (todas las aeronaves)."""
        condition = CompiledCondition(compiled_rule=sample_compiled_rule)
        
        result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign=None)
        
        # El código compilado actual retorna OK cuando callsign=None (no evalúa todas)
        # En una implementación completa, debería evaluar todas las aeronaves
        assert result.satisfied is True
        assert result.violation is None
        assert result.details["compiled"] is True
    
    def test_evaluate_aircraft_not_found(self, sample_compiled_rule, sample_traffic_state):
        """Evaluación con callsign que no existe."""
        condition = CompiledCondition(compiled_rule=sample_compiled_rule)
        
        result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign="NONEXISTENT")
        
        assert result.satisfied is True  # Aircraft not found = no violation
        assert result.violation is None
    
    def test_evaluate_execution_error(self, sample_traffic_state):
        """Manejo de errores de ejecución."""
        # Regla con código que causa error
        bad_rule = CompiledRule(
            source_rule_id="BAD_RULE",
            rule_category="GENERIC",
            condition_description="Bad rule",
            compiled_code="""
def evaluate(traffic_state, callsign=None):
    # Esto causará un error
    x = undefined_variable
    return {"satisfied": True, "details": {}, "explanation": "OK", "severity": "INFO"}
""",
            compilation_status=CompilationStatus.COMPILED,
        )
        
        condition = CompiledCondition(compiled_rule=bad_rule)
        
        # Sin fallback, debe retornar error
        result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign="TEST123")
        
        assert result.satisfied is False
        assert result.violation is None
        assert "error" in result.details
        assert "Compiled execution failed" in result.details["error"]
    
    def test_evaluate_with_fallback(self, sample_traffic_state):
        """Fallback a GenericKexCondition cuando ejecución falla."""
        # Mock de GenericKexCondition
        mock_fallback = Mock()
        mock_result = Mock()
        mock_result.satisfied = True
        mock_result.violation = None
        mock_fallback.evaluate.return_value = mock_result
        
        # Regla con código que causa error
        bad_rule = CompiledRule(
            source_rule_id="BAD_RULE",
            rule_category="GENERIC",
            condition_description="Bad rule",
            compiled_code="""
def evaluate(traffic_state, callsign=None):
    raise Exception("Test error")
""",
            compilation_status=CompilationStatus.COMPILED,
        )
        
        mock_llm_config = Mock()
        condition = CompiledCondition(compiled_rule=bad_rule, llm_config=mock_llm_config)
        
        # Mock del fallback
        with patch.object(condition, '_get_fallback_condition', return_value=mock_fallback):
            result = condition.evaluate(sample_traffic_state, {}, aircraft_callsign="TEST123")
        
        assert result.satisfied is True
        mock_fallback.evaluate.assert_called_once()


class TestIntegration:
    """Tests de integración del sistema completo."""
    
    def test_end_to_end_compilation_flow(self):
        """Flujo completo de compilación."""
        # Este test requiere un LLM real, así que lo marcamos como slow
        pytest.skip("Requires real LLM - marked as slow test")
    
    def test_kex_adapter_compiled_priority(self):
        """KEXAdapter debe priorizar reglas compiladas."""
        # Mock de loader que tiene regla compilada
        mock_loader = Mock()
        mock_compiled_rule = CompiledRule(
            source_rule_id="TEST_RULE",
            rule_category="GENERIC",
            condition_description="Test",
            compiled_code="def evaluate(): return {'satisfied': True, 'details': {}, 'explanation': 'OK', 'severity': 'INFO'}",
            compilation_status=CompilationStatus.COMPILED,
        )
        mock_loader.has_compiled_rule.return_value = True
        mock_loader.get_compiled_rule.return_value = mock_compiled_rule
        
        from Alert_System.integration.kex_adapter import KEXAdapter
        
        adapter = KEXAdapter()
        adapter._compiled_loader = mock_loader
        
        executable = ExecutableRule(
            source_rule_id="TEST_RULE",
            rule_category="GENERIC",
            condition_description="Test rule",
        )
        
        evaluator = adapter._adapt_executable_rule(executable)
        
        assert evaluator is not None
        assert isinstance(evaluator, CompiledCondition)
        assert evaluator.condition_id == "TEST_RULE"
        mock_loader.has_compiled_rule.assert_called_once_with("TEST_RULE")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
