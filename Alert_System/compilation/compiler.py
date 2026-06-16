"""Compilador de reglas KEX a código Python usando LLM."""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    CompiledRule, CompilationManifest, CompilationStatus, RuleVerdict,
    ClassificationResponse, GeneratedCodeResponse, ValidatedCodeResponse
)
# Validación ahora manejada por Pydantic en ValidatedCodeResponse
from .prompts import (
    COMPILATION_SYSTEM_PROMPT,
    COMPILATION_USER_PROMPT_TEMPLATE,
    TRAFFIC_STATE_SCHEMA,
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
)


class RuleCompiler:
    """
    Compila reglas KEX genéricas a funciones Python evaluadoras usando LLM.
    
    Flujo:
    0. Clasifica si la regla es compilable con TrafficState (o es subjetiva)
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
        self.mode = None  # Para referencia del modo de Instructor
    
    def _initialize_clients(self):
        """Inicializa cliente LLM de forma lazy usando solo Instructor."""
        if self.llm_config and not self._instructor_client:
            try:
                from common.llm_client_factory import create_instructor_client
                self._instructor_client, self.mode = create_instructor_client(self.llm_config)
                
                # Agregar callback para logging de errores de validación (como en Knowledge_Extractor)
                self._instructor_client.on("parse:error", self._log_validation_error)
                
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM client: {e}")
    
    def _log_validation_error(self, error: Exception):
        """Callback que se ejecuta cuando Instructor falla una validación."""
        print(f"⚠️ Instructor validation error: {error}. Retrying automatically...")
    
    def classify_rule(
        self,
        rule_id: str,
        rule_type: str = "",
        modality: str = "",
        trigger: str = "",
        constraint: str = "",
        formal_if_then: str = "",
        applicability: str = "",
        severity: str = "",
        explainability: str = "",
    ) -> RuleVerdict:
        """
        Clasifica si una regla es compilable con TrafficState o es subjetiva.
        
        Args:
            rule_id: ID de la regla
            rule_type: Tipo de regla (prohibition, obligation, etc.)
            modality: Modalidad (shall, may, etc.)
            trigger: Descripción del trigger
            constraint: Descripción de la constraint
            formal_if_then: Representación formal if-then
            applicability: Ámbito de aplicación
            severity: Severidad
            explainability: Razón de la regla
            
        Returns:
            RuleVerdict con is_compilable, reason, required_fields, confidence
        """
        self._initialize_clients()
        
        if not self._instructor_client:
            # Sin LLM, asumir compilable por defecto
            return RuleVerdict(
                is_compilable=True,
                reason="No LLM available for classification, assuming compilable",
                required_fields=[],
                confidence=0.5,
            )
        
        user_prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
            rule_id=rule_id,
            rule_type=rule_type,
            modality=modality,
            trigger=trigger,
            constraint=constraint,
            formal_if_then=formal_if_then,
            applicability=applicability,
            severity=severity,
            explainability=explainability,
        )
        
        try:
            
            # Usar Instructor con response_model estructurado
            response = self._instructor_client.chat.completions.create(
                model=self.llm_config.name,
                response_model=ClassificationResponse,
                max_retries=self.llm_config.max_retries,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            # Convertir a RuleVerdict manteniendo compatibilidad
            return RuleVerdict(
                is_compilable=response.is_compilable,
                reason=response.reason,
                required_fields=response.required_fields,
                confidence=response.confidence,
            )
            
        except Exception as e:
            # En caso de error, asumir compilable para no bloquear el pipeline
            return RuleVerdict(
                is_compilable=True,
                reason=f"Classification error: {e}",
                required_fields=[],
                confidence=0.3,
            )
    
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
        rule_type: str = "",
        modality: str = "",
        formal_if_then: str = "",
        applicability: str = "",
        explainability: str = "",
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
        
        # Paso 0: Clasificar si la regla es compilable con TrafficState
        verdict = self.classify_rule(
            rule_id=rule_id,
            rule_type=rule_type,
            modality=modality,
            trigger=trigger,
            constraint=constraint,
            formal_if_then=formal_if_then,
            applicability=applicability,
            severity=severity,
            explainability=explainability,
        )
        
        compiled_rule.compilation_metadata["classification"] = verdict.model_dump()
        
        if not verdict.is_compilable:
            compiled_rule.compilation_status = CompilationStatus.NOT_COMPILABLE
            compiled_rule.failure_reason = f"Not compilable: {verdict.reason}"
            print(f"🚫 Rule {rule_id} is not compilable with TrafficState: {verdict.reason}")
            return compiled_rule
        
        print(f"✓ Rule {rule_id} classified as compilable (confidence: {verdict.confidence:.1f}): {verdict.reason}")
        if verdict.required_fields:
            compiled_rule.required_state_fields = verdict.required_fields
        
        try:
            # Paso 1: Generar código con LLM (Instructor maneja validación y reintentos)
            code = self._generate_code(
                rule_id=rule_id,
                category=category,
                description=description,
                trigger=trigger,
                constraint=constraint,
                severity=severity,
                safety_critical=safety_critical,
            )
            
            # Paso 2: Probar el código (última verificación de ejecución)
            test_ok, test_error = self._test_code(code)
            if not test_ok:
                compiled_rule.compilation_status = CompilationStatus.FAILED
                compiled_rule.failure_reason = f"Test execution failed: {test_error}"
                compiled_rule.compiled_code = code
                return compiled_rule
            
            # Éxito - el código ya está validado por Instructor/Pydantic
            compiled_rule.compiled_code = code
            compiled_rule.compilation_status = CompilationStatus.COMPILED
            compiled_rule.compilation_metadata["compiled_at"] = datetime.utcnow().isoformat()
            compiled_rule.compilation_metadata["attempts"] = 1  # Instructor maneja reintentos internamente
            print(f"✅ Rule {rule_id} compiled successfully")
            return compiled_rule
            
        except Exception as e:
            compiled_rule.compilation_status = CompilationStatus.FAILED
            compiled_rule.failure_reason = f"Compilation error: {str(e)}"
            compiled_rule.compilation_metadata["attempts"] = 1
            return compiled_rule
    
    def compile_executable_rule(self, executable_rule: Any) -> CompiledRule:
        """
        Compila un ExecutableRule a código Python.
        
        Args:
            executable_rule: Instancia de ExecutableRule
            
        Returns:
            CompiledRule con el código generado
        """
        # Extraer campos adicionales para clasificación
        rule_type = getattr(executable_rule, 'rule_type', '') or ''
        modality = getattr(executable_rule, 'modality', '') or ''
        formal_if_then = ''
        if hasattr(executable_rule, 'raw_formal_if_then') and executable_rule.raw_formal_if_then:
            fit = executable_rule.raw_formal_if_then
            if isinstance(fit, dict):
                formal_if_then = f"IF {fit.get('if', fit.get('if_condition', ''))} THEN {fit.get('then', fit.get('then_action', ''))}"
            else:
                formal_if_then = str(fit)
        applicability = ''
        if hasattr(executable_rule, 'raw_applicability') and executable_rule.raw_applicability:
            app = executable_rule.raw_applicability
            if isinstance(app, dict):
                applicability = app.get('scope', '') or ', '.join(app.get('actors', []))
            else:
                applicability = str(app)
        explainability = getattr(executable_rule, 'explainability', '') or ''
        
        return self.compile_rule(
            rule_id=executable_rule.source_rule_id,
            category=executable_rule.rule_category,
            description=executable_rule.condition_description or "",
            trigger=executable_rule.raw_trigger or "",
            constraint=executable_rule.raw_constraint or "",
            severity=executable_rule.severity or "MEDIUM",
            safety_critical=executable_rule.safety_critical,
            required_state_fields=executable_rule.required_state_fields,
            rule_type=rule_type,
            modality=modality,
            formal_if_then=formal_if_then,
            applicability=applicability,
            explainability=explainability,
        )
    
    def compile_batch(
        self,
        executable_rules: List[Any],
        save_incrementally: bool = True,
        output_dir: Optional[str] = None,
        start_rule_index = 1
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
        
        # Configurar loader para guardado incremental
        loader = None
        manifest = None
        if save_incrementally:
            from .loader import CompiledRuleLoader
            if output_dir:
                loader = CompiledRuleLoader(compiled_rules_dir=output_dir, llm_config=self.llm_config)
            else:
                loader = CompiledRuleLoader(llm_config=self.llm_config)
            
            # Asegurar que el directorio exista
            loader.compiled_rules_dir.mkdir(parents=True, exist_ok=True)
            manifest = loader.load_manifest()
            
        if not manifest:
            manifest = CompilationManifest(model_used=model_name)

        for i, rule in enumerate(executable_rules, 1):
            if i < start_rule_index:
                print(f"\n🔨 Skipping rule {i}")
                continue

            print(f"\n🔨 Compiling rule {i}/{len(executable_rules)}: {rule.source_rule_id}")
            
            compiled = self.compile_executable_rule(rule)
            manifest.add_rule(compiled)
            
            status = "✅" if compiled.compilation_status == CompilationStatus.COMPILED else ("🚫" if compiled.compilation_status == CompilationStatus.NOT_COMPILABLE else "❌")
            print(f"  {status} {compiled.source_rule_id}: {compiled.compilation_status.value}")
            if compiled.failure_reason:
                print(f"  Reason: {compiled.failure_reason}")
            
            # Guardar inmediatamente si la compilación fue exitosa
            if save_incrementally and loader:
                try:
                    if compiled.compilation_status == CompilationStatus.COMPILED:
                        # Guardar archivo .py individual
                        loader.save_compiled_rule(compiled)
                        print(f"  💾 Saved to {loader.compiled_rules_dir / f'{compiled.source_rule_id}.py'}")
                    
                    # Actualizar manifest incrementalmente
                    loader.save_manifest(manifest)
                    
                except Exception as e:
                    print(f"  ⚠️ Error saving rule: {e}")
            
        
        print(f"\n📊 Compilation summary:")
        print(f"  Compiled: {manifest.total_compiled}")
        print(f"  Not compilable (subjective): {manifest.total_not_compilable}")
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
        Genera código Python usando el LLM con Instructor.
        
        Returns:
            Código Python de la función evaluate
        """
        self._initialize_clients()
        
        if not self._instructor_client:
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
        
        # Usar Instructor con response_model validado
        response = self._instructor_client.chat.completions.create(
            model=self.llm_config.name,
            response_model=ValidatedCodeResponse,
            max_retries=self.llm_config.max_retries,
            messages=[
                {"role": "system", "content": COMPILATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Baja temperatura para código determinista
        )
        
        # El código ya viene validado por Instructor y Pydantic
        # Incluye validación de sintaxis, imports prohibidos, estructura de retorno, etc.
        return response.code
    
        
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
