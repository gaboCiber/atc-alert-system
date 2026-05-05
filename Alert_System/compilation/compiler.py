"""Compilador de reglas KEX a código Python usando LLM."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .schemas import CompiledRule, CompilationManifest, CompilationStatus
from .validator import validate_code, validate_return_structure, CodeValidationError
from .prompts import (
    COMPILATION_SYSTEM_PROMPT,
    COMPILATION_USER_PROMPT_TEMPLATE,
    TRAFFIC_STATE_SCHEMA,
)


class RuleCompiler:
    """
    Compila reglas KEX genéricas a funciones Python evaluadoras usando LLM.
    
    Flujo:
    1. Genera código Python a partir de la descripción de la regla
    2. Valida estáticamente el código (seguridad, estructura)
    3. Prueba el código con un TrafficState de prueba
    4. Guarda la regla compilada si pasa todas las validaciones
    """
    
    def __init__(self, llm_config: Any = None):
        """
        Inicializa el compilador.
        
        Args:
            llm_config: Configuración del modelo LLM (ModelConfig)
        """
        self.llm_config = llm_config
        self._instructor_client = None
        self._raw_client = None
    
    def _initialize_clients(self):
        """Inicializa clientes LLM de forma lazy."""
        if self.llm_config and not self._instructor_client:
            try:
                from common.llm_client_factory import create_instructor_client, create_raw_client
                self._instructor_client, _ = create_instructor_client(self.llm_config)
                self._raw_client = create_raw_client(self.llm_config)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM clients: {e}")
    
    def compile_rule(
        self,
        rule_id: str,
        category: str,
        description: str,
        trigger: str = "",
        constraint: str = "",
        severity: str = "MEDIUM",
        safety_critical: bool = False,
        required_state_fields: List[str] = None,
        max_retries: int = 2,
    ) -> CompiledRule:
        """
        Compila una regla a código Python.
        
        Args:
            rule_id: ID de la regla KEX
            category: Categoría (GENERIC, ALTITUDE, etc.)
            description: Descripción de la condición
            trigger: Texto del trigger original
            constraint: Texto de la constraint original
            severity: Severidad de la regla
            safety_critical: Si es crítica para seguridad
            required_state_fields: Campos del TrafficState necesarios
            max_retries: Intentos máximos si la compilación falla
            
        Returns:
            CompiledRule con el código generado o estado FAILED
        """
        compiled_rule = CompiledRule(
            source_rule_id=rule_id,
            rule_category=category,
            condition_description=description,
            compiled_code="",
            required_state_fields=required_state_fields or [],
            compilation_metadata={
                "model": self.llm_config.name if self.llm_config else "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "attempts": 0,
            },
            compilation_status=CompilationStatus.PENDING,
            raw_trigger=trigger,
            raw_constraint=constraint,
            severity=severity,
            safety_critical=safety_critical,
        )
        
        for attempt in range(1, max_retries + 1):
            compiled_rule.compilation_metadata["attempts"] = attempt
            
            try:
                # Paso 1: Generar código con LLM
                code = self._generate_code(
                    rule_id=rule_id,
                    category=category,
                    description=description,
                    trigger=trigger,
                    constraint=constraint,
                    severity=severity,
                    safety_critical=safety_critical,
                )
                
                # Paso 2: Validar estáticamente
                is_valid, issues = validate_code(code)
                if not is_valid:
                    # Intentar corregir si hay issues menores
                    if attempt < max_retries:
                        print(f"⚠️ Attempt {attempt}: Validation issues: {issues}. Retrying...")
                        continue
                    compiled_rule.compilation_status = CompilationStatus.FAILED
                    compiled_rule.failure_reason = f"Static validation failed: {issues}"
                    compiled_rule.compiled_code = code  # Guardar para debug
                    return compiled_rule
                
                # Paso 2b: Validar estructura de retorno
                return_valid, return_issues = validate_return_structure(code)
                if not return_valid:
                    if attempt < max_retries:
                        print(f"⚠️ Attempt {attempt}: Return structure issues: {return_issues}. Retrying...")
                        continue
                    # No es fatal, pero registrar
                    compiled_rule.compilation_metadata["return_warnings"] = return_issues
                
                # Paso 3: Probar el código
                test_ok, test_error = self._test_code(code)
                if not test_ok:
                    if attempt < max_retries:
                        print(f"⚠️ Attempt {attempt}: Test execution failed: {test_error}. Retrying...")
                        continue
                    compiled_rule.compilation_status = CompilationStatus.FAILED
                    compiled_rule.failure_reason = f"Test execution failed: {test_error}"
                    compiled_rule.compiled_code = code
                    return compiled_rule
                
                # Éxito
                compiled_rule.compiled_code = code
                compiled_rule.compilation_status = CompilationStatus.COMPILED
                compiled_rule.compilation_metadata["compiled_at"] = datetime.utcnow().isoformat()
                print(f"✅ Rule {rule_id} compiled successfully (attempt {attempt})")
                return compiled_rule
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️ Attempt {attempt}: Compilation error: {e}. Retrying...")
                    continue
                compiled_rule.compilation_status = CompilationStatus.FAILED
                compiled_rule.failure_reason = f"Compilation error: {str(e)}"
                return compiled_rule
        
        return compiled_rule
    
    def compile_executable_rule(self, executable_rule: Any) -> CompiledRule:
        """
        Compila un ExecutableRule a código Python.
        
        Args:
            executable_rule: Instancia de ExecutableRule
            
        Returns:
            CompiledRule con el código generado
        """
        return self.compile_rule(
            rule_id=executable_rule.source_rule_id,
            category=executable_rule.rule_category,
            description=executable_rule.condition_description or "",
            trigger=executable_rule.raw_trigger or "",
            constraint=executable_rule.raw_constraint or "",
            severity=executable_rule.severity or "MEDIUM",
            safety_critical=executable_rule.safety_critical,
            required_state_fields=executable_rule.required_state_fields,
        )
    
    def compile_batch(
        self,
        executable_rules: List[Any],
        save_incrementally: bool = True,
        output_dir: Optional[str] = None,
    ) -> CompilationManifest:
        """
        Compila un lote de reglas ExecutableRule.
        
        Args:
            executable_rules: Lista de ExecutableRule
            save_incrementally: Si True, guarda cada regla exitosa inmediatamente
            output_dir: Directorio de salida (usa default si save_incrementally=True)
            
        Returns:
            CompilationManifest con resultados de todas las compilaciones
        """
        model_name = self.llm_config.name if self.llm_config else "unknown"
        manifest = CompilationManifest(model_used=model_name)
        
        # Configurar loader para guardado incremental
        loader = None
        if save_incrementally:
            from .loader import CompiledRuleLoader
            if output_dir:
                loader = CompiledRuleLoader(compiled_rules_dir=output_dir, llm_config=self.llm_config)
            else:
                loader = CompiledRuleLoader(llm_config=self.llm_config)
            
            # Asegurar que el directorio exista
            loader.compiled_rules_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rule in enumerate(executable_rules):
            print(f"\n🔨 Compiling rule {i+1}/{len(executable_rules)}: {rule.source_rule_id}")
            
            compiled = self.compile_executable_rule(rule)
            manifest.add_rule(compiled)
            
            status = "✅" if compiled.compilation_status == CompilationStatus.COMPILED else "❌"
            print(f"  {status} {compiled.source_rule_id}: {compiled.compilation_status.value}")
            if compiled.failure_reason:
                print(f"  Reason: {compiled.failure_reason}")
            
            # Guardar inmediatamente si la compilación fue exitosa
            if save_incrementally and loader and compiled.compilation_status == CompilationStatus.COMPILED:
                try:
                    # Guardar archivo .py individual
                    loader.save_compiled_rule(compiled)
                    print(f"  💾 Saved to {loader.compiled_rules_dir / f'{compiled.source_rule_id}.py'}")
                    
                    # Actualizar manifest incrementalmente
                    loader.save_manifest(manifest)
                    
                except Exception as e:
                    print(f"  ⚠️ Error saving rule: {e}")
        
        print(f"\n📊 Compilation summary:")
        print(f"  Compiled: {manifest.total_compiled}")
        print(f"  Failed (fallback): {manifest.total_failed}")
        print(f"  Total: {len(manifest.rules)}")
        
        if save_incrementally and loader:
            print(f"  📁 Rules saved in: {loader.compiled_rules_dir}")
        
        return manifest
    
    def _generate_code(
        self,
        rule_id: str,
        category: str,
        description: str,
        trigger: str,
        constraint: str,
        severity: str,
        safety_critical: bool,
    ) -> str:
        """
        Genera código Python usando el LLM.
        
        Returns:
            Código Python de la función evaluate
        """
        self._initialize_clients()
        
        if not self._raw_client:
            raise RuntimeError("No LLM client available for compilation")
        
        user_prompt = COMPILATION_USER_PROMPT_TEMPLATE.format(
            rule_id=rule_id,
            category=category,
            description=description,
            trigger=trigger,
            constraint=constraint,
            severity=severity,
            safety_critical=safety_critical,
            traffic_state_schema=TRAFFIC_STATE_SCHEMA,
        )
        
        response = self._raw_client.chat.completions.create(
            model=self.llm_config.name,
            messages=[
                {"role": "system", "content": COMPILATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Baja temperatura para código determinista
            max_tokens=2000,
        )
        
        raw_code = response.choices[0].message.content
        
        # Extraer código de markdown code blocks si está presente
        code = self._extract_code(raw_code)
        
        return code
    
    def _extract_code(self, raw_response: str) -> str:
        """
        Extrae código Python de la respuesta del LLM.
        
        Maneja respuestas que vienen envueltas en ```python ... ```
        """
        # Intentar extraer de code blocks
        patterns = [
            r"```python\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"```python\s+(.*?)\n```",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Si no hay code blocks, asumir que toda la respuesta es código
        # (eliminar líneas que no sean código)
        lines = raw_response.strip().split("\n")
        code_lines = []
        in_function = False
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines at start
            if not in_function and not stripped:
                continue
            # Detect start of function
            if stripped.startswith("def "):
                in_function = True
            if in_function:
                code_lines.append(line)
        
        if code_lines:
            return "\n".join(code_lines)
        
        # Last resort: return entire response
        return raw_response.strip()
    
    def _test_code(self, code: str) -> Tuple[bool, str]:
        """
        Prueba el código generado ejecutándolo con un TrafficState de prueba.
        
        Returns:
            Tuple de (success, error_message)
        """
        from Alert_System.models.traffic_state import (
            TrafficState, AircraftState, Position, FlightPhase
        )
        
        # Crear TrafficState de prueba
        test_state = TrafficState(
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
        
        # Namespace restringido para ejecución
        import math
        namespace = {
            "math": math,
            "TrafficState": TrafficState,
            "AircraftState": AircraftState,
            "Position": Position,
            "FlightPhase": FlightPhase,
        }
        
        try:
            # Compilar y ejecutar la función
            exec(code, namespace)
            
            # Verificar que 'evaluate' existe
            if "evaluate" not in namespace:
                return False, "Function 'evaluate' not defined in generated code"
            
            evaluate_fn = namespace["evaluate"]
            
            # Ejecutar con callsign específico
            result = evaluate_fn(test_state, callsign="TEST123")
            
            # Verificar estructura del resultado
            if not isinstance(result, dict):
                return False, f"evaluate() returned {type(result).__name__}, expected dict"
            
            required_keys = {"satisfied", "details", "explanation", "severity"}
            missing_keys = required_keys - set(result.keys())
            if missing_keys:
                return False, f"Result dict missing keys: {missing_keys}"
            
            # Verificar tipos de los valores
            if not isinstance(result["satisfied"], bool):
                return False, f"'satisfied' should be bool, got {type(result['satisfied']).__name__}"
            
            if not isinstance(result["details"], dict):
                return False, f"'details' should be dict, got {type(result['details']).__name__}"
            
            if not isinstance(result["explanation"], str):
                return False, f"'explanation' should be str, got {type(result['explanation']).__name__}"
            
            if result["severity"] not in {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                return False, f"Invalid severity: {result['severity']}"
            
            # Ejecutar sin callsign (para todos los aircraft)
            result_all = evaluate_fn(test_state, callsign=None)
            if not isinstance(result_all, dict):
                return False, f"evaluate(callsign=None) returned {type(result_all).__name__}, expected dict"
            
            return True, ""
            
        except Exception as e:
            return False, f"Execution error: {type(e).__name__}: {str(e)}"
