"""Pipeline de 8 pasos para el sistema de alertas ATC."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from Alert_System.models.instruction import ParsedInstruction, InstructionType, Speaker
from Alert_System.models.traffic_state import AircraftState, TrafficState
from Alert_System.models.alert import Alert, AlertResult, AlertSeverity, Violation
from Alert_System.rule_engine.conditions import ConditionResult
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.core.state_manager import StateManager
from Alert_System.core.state_projection import ProjectedState, StateProjector


@dataclass
class PipelineStep:
    """Representa un paso del pipeline con su resultado."""
    step_number: int
    step_name: str
    status: str = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED, SKIPPED
    input_data: Any = None
    output_data: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def mark_success(self, output: Any) -> None:
        """Marca el paso como exitoso."""
        self.status = "SUCCESS"
        self.output_data = output
    
    def mark_failed(self, error: str) -> None:
        """Marca el paso como fallido."""
        self.status = "FAILED"
        self.error_message = error
    
    def mark_skipped(self, reason: str = "") -> None:
        """Marca el paso como omitido."""
        self.status = "SKIPPED"
        if reason:
            self.error_message = reason


@dataclass
class PipelineResult:
    """Resultado completo de la ejecución del pipeline."""
    
    # Identificación
    pipeline_id: str
    timestamp: datetime
    
    # Instrucción procesada
    raw_instruction: str
    parsed_instruction: Optional[ParsedInstruction] = None
    
    # Pasos del pipeline
    steps: List[PipelineStep] = field(default_factory=list)
    
    # Resultado final
    final_decision: str = "PENDING"  # COMMIT, ROLLBACK, PENDING
    atco_override: bool = False
    atco_reason: Optional[str] = None
    
    # Alertas generadas
    alerts_generated: List[Alert] = field(default_factory=list)
    violations_found: List[Violation] = field(default_factory=list)
    
    # Estado final
    committed_state: Optional[TrafficState] = None
    projected_state: Optional[ProjectedState] = None
    
    # Metadata
    total_execution_time_ms: float = 0.0
    has_errors: bool = False
    
    def get_step(self, step_number: int) -> Optional[PipelineStep]:
        """Obtiene un paso específico del pipeline."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None
    
    def get_step_by_name(self, name: str) -> Optional[PipelineStep]:
        """Obtiene un paso por nombre."""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None
    
    def was_successful(self) -> bool:
        """¿El pipeline completó exitosamente?"""
        return self.final_decision in ["COMMIT", "ROLLBACK"] and not self.has_errors


class AlertPipeline:
    """
    Pipeline de 8 pasos para procesar instrucciones ATC y generar alertas.
    
    Pasos:
    1. Input Processing: Parseo de instrucción
    2. Normalization: Normalización a formato interno
    3. State Update: Aplicación tentativa al estado proyectado
    4. Rule Evaluation: Evaluación de reglas contra el estado proyectado
    5. Alert Generation: Generación de alertas si hay violaciones
    6. Alert Presentation: Presentación al ATCO
    7. ATCO Decision: Decisión del ATCO (COMMIT/ROLLBACK/OVERRIDE)
    8. Final State Update: Commit o rollback del estado real
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        rule_engine: RuleEngine,
    ):
        """
        Inicializa el pipeline.
        
        Args:
            state_manager: Gestor de estado con soporte commit/rollback
            rule_engine: Motor de reglas para evaluación
        """
        self.state_manager = state_manager
        self.rule_engine = rule_engine
        self.state_projector = StateProjector()
    
    def process_instruction(
        self,
        raw_instruction: str,
        pre_parsed: Optional[ParsedInstruction] = None,
    ) -> PipelineResult:
        """
        Procesa una instrucción ATC a través del pipeline de 8 pasos.
        
        Args:
            raw_instruction: Texto crudo de la instrucción (ASR output)
            pre_parsed: Si se proporciona, usa este parseo en lugar del paso 1
            
        Returns:
            PipelineResult con todos los resultados
        """
        import time
        start_time = time.time()
        
        # Crear resultado inicial
        result = PipelineResult(
            pipeline_id=f"PL_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            timestamp=datetime.utcnow(),
            raw_instruction=raw_instruction,
            steps=[],
        )
        
        try:
            # Paso 1: Input Processing
            step1 = self._step_1_input_processing(raw_instruction, pre_parsed)
            result.steps.append(step1)
            
            if step1.status == "FAILED":
                result.has_errors = True
                result.final_decision = "ROLLBACK"
                return result
            
            parsed = step1.output_data
            result.parsed_instruction = parsed
            
            # Paso 2: Normalization (ya normalizado en paso 1 si usamos ParsedInstruction)
            step2 = self._step_2_normalization(parsed)
            result.steps.append(step2)
            
            # Paso 3: State Update (proyección)
            step3 = self._step_3_state_update(parsed)
            result.steps.append(step3)
            
            if step3.status == "FAILED":
                result.has_errors = True
                result.final_decision = "ROLLBACK"
                return result
            
            projected = step3.output_data
            result.projected_state = projected
            
            # Paso 4: Rule Evaluation
            step4 = self._step_4_rule_evaluation(parsed, projected)
            result.steps.append(step4)
            
            violations = step4.output_data or []
            result.violations_found = violations
            
            # Paso 5: Alert Generation
            step5 = self._step_5_alert_generation(parsed, violations)
            result.steps.append(step5)
            
            alerts = step5.output_data or []
            result.alerts_generated = alerts
            
            # Paso 6: Alert Presentation
            step6 = self._step_6_alert_presentation(alerts, violations)
            result.steps.append(step6)
            
            # Paso 7: ATCO Decision
            step7 = self._step_7_atco_decision(alerts, projected)
            result.steps.append(step7)
            
            decision = step7.output_data
            result.final_decision = decision
            
            # Paso 8: Final State Update
            step8 = self._step_8_final_state_update(decision, projected)
            result.steps.append(step8)
            
            if step8.output_data:
                result.committed_state = step8.output_data
            
        except Exception as e:
            result.has_errors = True
            # Añadir paso fallido
            result.steps.append(PipelineStep(
                step_number=0,
                step_name="PIPELINE_ERROR",
                status="FAILED",
                error_message=str(e),
            ))
        
        finally:
            result.total_execution_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _step_1_input_processing(
        self,
        raw_instruction: str,
        pre_parsed: Optional[ParsedInstruction],
    ) -> PipelineStep:
        """
        Paso 1: Input Processing
        Parsea la instrucción en formato estructurado.
        """
        step = PipelineStep(
            step_number=1,
            step_name="INPUT_PROCESSING",
            input_data=raw_instruction,
        )
        
        if pre_parsed:
            # Usar instrucción ya parseada
            step.mark_success(pre_parsed)
            return step
        
        # Aquí se integraría con un parser de ATC (KEX o LLM)
        # Por ahora, simulamos un parseo básico
        try:
            parsed = self._simple_atc_parser(raw_instruction)
            step.mark_success(parsed)
        except Exception as e:
            step.mark_failed(f"Parse error: {str(e)}")
        
        return step
    
    def _step_2_normalization(
        self,
        parsed: ParsedInstruction,
    ) -> PipelineStep:
        """
        Paso 2: Normalization
        Normaliza a formato interno estandarizado.
        """
        step = PipelineStep(
            step_number=2,
            step_name="NORMALIZATION",
            input_data=parsed,
        )
        
        # Ya está normalizado en el ParsedInstruction
        # Aquí se podría añadir validaciones adicionales
        step.mark_success(parsed)
        return step
    
    def _step_3_state_update(
        self,
        parsed: ParsedInstruction,
    ) -> PipelineStep:
        """
        Paso 3: State Update (proyección)
        Crea estado proyectado aplicando la instrucción.
        """
        step = PipelineStep(
            step_number=3,
            step_name="STATE_PROJECTION",
            input_data=parsed,
        )
        
        try:
            current_state = self.state_manager.current_state
            
            projected = self.state_projector.create_projection(
                current_state,
                parsed,
                projection_minutes=10,
            )
            
            if not projected.is_valid_projection:
                step.mark_failed(
                    f"Invalid projection: {projected.projection_errors}"
                )
            else:
                step.mark_success(projected)
        
        except Exception as e:
            step.mark_failed(f"Projection error: {str(e)}")
        
        return step
    
    def _step_4_rule_evaluation(
        self,
        parsed: ParsedInstruction,
        projected: ProjectedState,
    ) -> PipelineStep:
        """
        Paso 4: Rule Evaluation
        Evalúa reglas contra el estado proyectado.
        """
        step = PipelineStep(
            step_number=4,
            step_name="RULE_EVALUATION",
            input_data=(parsed, projected),
        )
        
        try:
            violations = []
            callsign = parsed.callsign
            
            if not callsign:
                step.mark_success([])
                return step
            
            # Evaluar condiciones según el tipo de instrucción
            violations.extend(
                self._evaluate_altitude_rules(parsed, projected, callsign)
            )
            violations.extend(
                self._evaluate_separation_rules(parsed, projected, callsign)
            )
            violations.extend(
                self._evaluate_runway_rules(parsed, projected, callsign)
            )
            
            step.mark_success(violations)
        
        except Exception as e:
            step.mark_failed(f"Rule evaluation error: {str(e)}")
        
        return step
    
    def _step_5_alert_generation(
        self,
        parsed: ParsedInstruction,
        violations: List[Violation],
    ) -> PipelineStep:
        """
        Paso 5: Alert Generation
        Genera alertas estructuradas a partir de violaciones.
        """
        step = PipelineStep(
            step_number=5,
            step_name="ALERT_GENERATION",
            input_data=violations,
        )
        
        try:
            alerts = []
            
            for violation in violations:
                # Extraer callsigns de los detalles de la violación
                aircraft_involved = violation.details.get("aircraft_involved", [])
                # Inferir categoría del condition_type
                category = self._infer_alert_category(violation.condition_type)
                
                alert = Alert(
                    alert_id=f"ALT_{uuid.uuid4().hex[:8]}",
                    category=category,
                    severity=violation.severity,
                    affected_callsigns=aircraft_involved,
                    primary_callsign=aircraft_involved[0] if aircraft_involved else None,
                    triggering_instruction_raw=parsed.raw_text if parsed else "",
                    violations=[violation],
                    title=f"Alert: {violation.condition_type}",
                    explanation=violation.explanation,
                    suggested_action="Review instruction",
                )
                alerts.append(alert)
            
            step.mark_success(alerts)
        
        except Exception as e:
            step.mark_failed(f"Alert generation error: {str(e)}")
        
        return step
    
    def _step_6_alert_presentation(
        self,
        alerts: List[Alert],
        violations: List[Violation],
    ) -> PipelineStep:
        """
        Paso 6: Alert Presentation
        Prepara alertas para presentación al ATCO.
        """
        step = PipelineStep(
            step_number=6,
            step_name="ALERT_PRESENTATION",
            input_data=(alerts, violations),
        )
        
        # Aquí se formatearía para la UI
        presentation = {
            "alert_count": len(alerts),
            "violation_count": len(violations),
            "has_critical": any(a.severity == AlertSeverity.CRITICAL for a in alerts),
            "alerts": alerts,
        }
        
        step.mark_success(presentation)
        return step
    
    def _step_7_atco_decision(
        self,
        alerts: List[Alert],
        projected: ProjectedState,
    ) -> PipelineStep:
        """
        Paso 7: ATCO Decision
        Obtiene decisión del ATCO.
        
        En implementación real, esto sería interactivo.
        Por ahora, usa lógica automática.
        """
        step = PipelineStep(
            step_number=7,
            step_name="ATCO_DECISION",
            input_data=alerts,
        )
        
        # Lógica automática:
        # - Si hay alertas CRITICAL → ROLLBACK automático
        # - Si hay alertas WARNING → COMMIT con alerta
        # - Si no hay alertas → COMMIT normal
        
        has_critical = any(a.severity == AlertSeverity.CRITICAL for a in alerts)
        
        if has_critical and not projected.has_conflicts():
            # Paranoia: double check
            pass
        
        if has_critical:
            decision = "ROLLBACK"
        else:
            decision = "COMMIT"
        
        step.mark_success(decision)
        return step
    
    def _step_8_final_state_update(
        self,
        decision: str,
        projected: ProjectedState,
    ) -> PipelineStep:
        """
        Paso 8: Final State Update
        Commit o rollback del estado real.
        """
        step = PipelineStep(
            step_number=8,
            step_name="FINAL_STATE_UPDATE",
            input_data=decision,
        )
        
        try:
            if decision == "COMMIT":
                # Crear transacción y hacer commit
                txn = self.state_manager.propose_change(projected)
                success = self.state_manager.commit(txn.transaction_id)
                
                if success:
                    step.mark_success(self.state_manager.current_state)
                else:
                    step.mark_failed("Commit failed")
            
            elif decision == "ROLLBACK":
                # Solo marcar como rollback, no cambiar estado
                step.mark_success(None)
            
            else:
                step.mark_failed(f"Unknown decision: {decision}")
        
        except Exception as e:
            step.mark_failed(f"State update error: {str(e)}")
        
        return step
    
    def _infer_alert_category(self, condition_type: str) -> Any:
        """Infiere la categoría de alerta del tipo de condición."""
        from Alert_System.models.alert import AlertCategory
        
        condition_lower = condition_type.lower()
        
        if "altitude" in condition_lower or "msa" in condition_lower:
            return AlertCategory.ALTITUDE_VIOLATION
        elif "separation" in condition_lower:
            return AlertCategory.SEPARATION_LOSS
        elif "runway" in condition_lower:
            return AlertCategory.RUNWAY_CONFLICT
        elif "speed" in condition_lower:
            return AlertCategory.SPEED_VIOLATION
        else:
            return AlertCategory.PROCEDURAL_ERROR
    
    # =========================================================================
    # Métodos auxiliares de evaluación de reglas
    # =========================================================================
    
    def _evaluate_altitude_rules(
        self,
        parsed: ParsedInstruction,
        projected: ProjectedState,
        callsign: str,
    ) -> List[Violation]:
        """Evalúa reglas de altitud."""
        violations = []
        
        aircraft = projected.get_aircraft(callsign)
        if not aircraft:
            return violations
        
        altitude = aircraft.position.altitude
        msa = projected.traffic_state.msa
        
        # Verificar MSA
        if altitude < msa:
            from Alert_System.models.alert import AlertCategory, AlertSeverity
            violations.append(Violation(
                violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                rule_id="MSA_RULE",
                condition_type="ALTITUDE_MINIMUM",
                severity=AlertSeverity.CRITICAL,
                explanation=f"Altitude {altitude}ft below MSA {msa}ft",
                details={
                    "aircraft_involved": [callsign],
                    "expected_minimum": msa,
                    "actual_altitude": altitude,
                },
            ))
        
        return violations
    
    def _evaluate_separation_rules(
        self,
        parsed: ParsedInstruction,
        projected: ProjectedState,
        callsign: str,
    ) -> List[Violation]:
        """Evalúa reglas de separación."""
        violations = []
        
        # Usar separaciones pre-calculadas en la proyección
        separations = projected.projected_separations.get(callsign, [])
        
        for sep in separations:
            if sep.conflict_predicted:
                from Alert_System.models.alert import AlertCategory, AlertSeverity
                violations.append(Violation(
                    violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                    rule_id="SEPARATION_RULE",
                    condition_type="SEPARATION_VERTICAL",
                    severity=AlertSeverity.CRITICAL,
                    explanation=(
                        f"Predicted loss of separation with {sep.aircraft_2} "
                        f"in {sep.time_to_conflict}s"
                    ),
                    details={
                        "aircraft_involved": [callsign, sep.aircraft_2],
                        "expected_vertical": 1000,
                        "expected_horizontal_nm": 5,
                        "actual_vertical": sep.vertical_separation_ft,
                        "actual_horizontal_nm": sep.horizontal_separation_nm,
                    },
                ))
        
        return violations
    
    def _evaluate_runway_rules(
        self,
        parsed: ParsedInstruction,
        projected: ProjectedState,
        callsign: str,
    ) -> List[Violation]:
        """Evalúa reglas de pista."""
        violations = []
        
        # Verificar si es instrucción de pista
        if parsed.instruction_type not in [
            InstructionType.TAKEOFF_CLEARANCE,
            InstructionType.LANDING_CLEARANCE,
        ]:
            return violations
        
        runway = parsed.parameters.get("runway")
        if not runway:
            return violations
        
        # Verificar si pista está ocupada
        runway_state = projected.traffic_state.runways.get(runway)
        if runway_state and runway_state.occupied_by:
            from Alert_System.models.alert import AlertCategory, AlertSeverity
            violations.append(Violation(
                violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                rule_id="RUNWAY_RULE",
                condition_type="RUNWAY_AVAILABLE",
                severity=AlertSeverity.CRITICAL,
                explanation=f"Runway {runway} occupied by {runway_state.occupied_by}",
                details={
                    "aircraft_involved": [callsign, runway_state.occupied_by],
                    "runway": runway,
                    "occupied_by": runway_state.occupied_by,
                },
            ))
        
        return violations
    
    def _simple_atc_parser(self, raw_instruction: str) -> ParsedInstruction:
        """
        Parser simple para instrucciones ATC.
        
        Esto sería reemplazado por el parser real del KEX/ASR.
        """
        text = raw_instruction.lower()
        
        # Detectar callsign (patrón simple: 3 letras + números)
        import re
        callsign_match = re.search(r'\b([a-z]{3}\d+)\b', text)
        callsign = callsign_match.group(1).upper() if callsign_match else None
        
        # Detectar tipo de instrucción
        instruction_type = InstructionType.UNKNOWN
        action_verb = "unknown"
        parameters = {}
        
        if "descend" in text or "descent" in text:
            instruction_type = InstructionType.DESCENT
            action_verb = "descend"
            # Buscar FL o altitud
            fl_match = re.search(r'fl\s*(\d+)', text)
            if fl_match:
                parameters["target_altitude"] = int(fl_match.group(1)) * 100
                parameters["flight_level"] = int(fl_match.group(1))
        
        elif "climb" in text:
            instruction_type = InstructionType.CLIMB
            action_verb = "climb"
            fl_match = re.search(r'fl\s*(\d+)', text)
            if fl_match:
                parameters["target_altitude"] = int(fl_match.group(1)) * 100
        
        elif "heading" in text or "turn" in text:
            instruction_type = InstructionType.HEADING
            action_verb = "turn" if "turn" in text else "heading"
            heading_match = re.search(r'(\d{3})', text)
            if heading_match:
                parameters["heading"] = int(heading_match.group(1))
        
        elif "cleared for takeoff" in text:
            instruction_type = InstructionType.TAKEOFF_CLEARANCE
            action_verb = "takeoff"
            rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
            if rw_match:
                parameters["runway"] = rw_match.group(1)
        
        elif "cleared to land" in text:
            instruction_type = InstructionType.LANDING_CLEARANCE
            action_verb = "land"
            rw_match = re.search(r'runway\s+(\d+[lr]?)', text)
            if rw_match:
                parameters["runway"] = rw_match.group(1)
        
        return ParsedInstruction(
            raw_text=raw_instruction,
            normalized_text=raw_instruction,
            speaker=Speaker.ATCO,
            callsign=callsign,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=parameters,
        )
