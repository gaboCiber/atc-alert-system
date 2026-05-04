"""Implementaciones de condiciones evaluables para el motor de reglas."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Tuple
from uuid import uuid4
from pydantic import BaseModel
from ..models import TrafficState, AlertSeverity, AircraftState, RunwayState, Violation


@dataclass
class ConditionResult:
    """Resultado de evaluar una condición."""
    
    satisfied: bool  # True si la condición se cumple (no hay violación)
    violation: Optional[Violation] = None  # Violación si la hay
    details: Dict[str, Any] = None  # Detalles adicionales
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ConditionEvaluator(ABC):
    """Clase base para evaluadores de condiciones."""
    
    condition_type: str = ""
    
    def __init__(self):
        """Inicializa el evaluador con registro vacío de reglas."""
        self._rules: List[Dict[str, Any]] = []
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """
        Agrega una regla específica a este evaluador.
        
        Args:
            rule: Diccionario con condition_type y parameters
        """
        self._rules.append(rule)
    
    def clear_rules(self) -> None:
        """Limpia todas las reglas registradas."""
        self._rules.clear()
    
    @abstractmethod
    def evaluate(
        self,
        traffic_state: TrafficState,
        parameters: Dict[str, Any],
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """
        Evalúa la condición contra el estado del tráfico.
        
        Args:
            traffic_state: Estado actual o proyectado del tráfico
            parameters: Parámetros de la condición desde la regla
            aircraft_callsign: Callsign de la aeronave objetivo (si aplica)
            
        Returns:
            ConditionResult con el resultado de la evaluación
        """
        pass
    
    @abstractmethod
    def evaluate_all(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa TODAS las reglas registradas contra el estado.
        
        Args:
            traffic_state: Estado actual o proyectado del tráfico
            aircraft_callsign: Callsign de la aeronave objetivo (si aplica)
            
        Returns:
            Lista de violaciones encontradas
        """
        pass
    
    def get_required_parameters(self) -> List[str]:
        """Retorna los parámetros requeridos por esta condición."""
        return []
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida que los parámetros contengan lo necesario."""
        required = self.get_required_parameters()
        missing = [p for p in required if p not in parameters]
        if missing:
            return False, [f"Missing required parameter: {p}" for p in missing]
        return True, []


class AltitudeCondition(ConditionEvaluator):
    """
    Evalúa condiciones de altitud.
    
    Tipos de condición:
    - ALTITUDE_MINIMUM: Verifica que altitud >= mínimo
    - ALTITUDE_MAXIMUM: Verifica que altitud <= máximo
    - ALTITUDE_RANGE: Verifica que altitud esté en rango
    """
    
    condition_type = "ALTITUDE"
    
    def get_required_parameters(self) -> List[str]:
        # reference_value solo requerido para MINIMUM/MAXIMUM, no para MSA_CHECK
        return ["check_type"]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida parámetros específicos según el tipo de chequeo."""
        check_type = parameters.get("check_type")
        
        if not check_type:
            return False, ["Missing required parameter: check_type"]
        
        # MINIMUM y MAXIMUM requieren reference_value
        if check_type in ["MINIMUM", "MAXIMUM"] and "reference_value" not in parameters:
            return False, ["Missing required parameter: reference_value for MINIMUM/MAXIMUM check"]
        
        return True, []
    
    def evaluate(
        self,
        traffic_state: TrafficState,
        parameters: Dict[str, Any],
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """Evalúa condición de altitud."""
        if not aircraft_callsign:
            return ConditionResult(
                satisfied=False,
                violation=None,
                details={"error": "No aircraft callsign provided"}
            )
        
        aircraft = traffic_state.get_aircraft(aircraft_callsign)
        if not aircraft:
            return ConditionResult(
                satisfied=False,
                violation=None,
                details={"error": f"Aircraft {aircraft_callsign} not found"}
            )
        
        check_type = parameters.get("check_type")
        reference_value = parameters.get("reference_value")
        current_altitude = aircraft.position.altitude
        
        if check_type == "MINIMUM":
            # Verificar que altitud >= mínimo
            if current_altitude < reference_value:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "ALTITUDE_RULE"),
                    condition_type="ALTITUDE_MINIMUM",
                    severity=AlertSeverity.HIGH,
                    details={
                        "current_altitude": current_altitude,
                        "required_minimum": reference_value,
                        "difference_ft": reference_value - current_altitude,
                        "callsign": aircraft_callsign,
                    },
                    explanation=(
                        f"Aircraft {aircraft_callsign} at {current_altitude}ft "
                        f"is below minimum altitude of {reference_value}ft"
                    ),
                    suggestion=f"Climb immediately to {reference_value}ft or above",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "minimum_passed"})
        
        elif check_type == "MAXIMUM":
            # Verificar que altitud <= máximo
            if current_altitude > reference_value:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "ALTITUDE_RULE"),
                    condition_type="ALTITUDE_MAXIMUM",
                    severity=AlertSeverity.MEDIUM,
                    details={
                        "current_altitude": current_altitude,
                        "required_maximum": reference_value,
                        "difference_ft": current_altitude - reference_value,
                        "callsign": aircraft_callsign,
                    },
                    explanation=(
                        f"Aircraft {aircraft_callsign} at {current_altitude}ft "
                        f"exceeds maximum altitude of {reference_value}ft"
                    ),
                    suggestion=f"Descend to {reference_value}ft or below",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "maximum_passed"})
        
        elif check_type == "MSA_CHECK":
            # Verificar contra Minimum Sector Altitude
            msa = traffic_state.msa
            if msa and current_altitude < msa:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "MSA_RULE"),
                    condition_type="MSA_VIOLATION",
                    severity=AlertSeverity.CRITICAL,
                    details={
                        "current_altitude": current_altitude,
                        "msa": msa,
                        "sector_id": traffic_state.sector_id,
                        "callsign": aircraft_callsign,
                    },
                    explanation=(
                        f"CRITICAL: Aircraft {aircraft_callsign} at {current_altitude}ft "
                        f"is below MSA ({msa}ft)"
                    ),
                    suggestion=f"CLIMB IMMEDIATELY to {msa}ft or above",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "msa_passed"})
        
        return ConditionResult(
            satisfied=False,
            details={"error": f"Unknown check_type: {check_type}"}
        )
    
    def evaluate_all(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa TODAS las reglas de altitud registradas.
        
        Si no hay reglas registradas, usa reglas por defecto (MSA_CHECK).
        """
        violations = []
        
        # Si no hay reglas registradas, usar reglas por defecto
        if not self._rules:
            # Regla por defecto: MSA_CHECK
            default_rules = [
                {"condition_type": "ALTITUDE", "parameters": {"check_type": "MSA_CHECK", "rule_id": "MSA_RULE"}}
            ]
            rules_to_evaluate = default_rules
        else:
            rules_to_evaluate = self._rules
        
        for rule in rules_to_evaluate:
            parameters = rule.get("parameters", {})
            result = self.evaluate(traffic_state, parameters, aircraft_callsign)
            if not result.satisfied and result.violation:
                violations.append(result.violation)
        
        return violations


class SeparationCondition(ConditionEvaluator):
    """
    Evalúa condiciones de separación entre aeronaves.
    
    Tipos de condición:
    - VERTICAL_MINIMUM: Verifica separación vertical mínima
    - HORIZONTAL_MINIMUM: Verifica separación horizontal mínima
    """
    
    condition_type = "SEPARATION"
    
    # Estándares ICAO
    VERTICAL_SEPARATION_STD = 1000  # pies
    HORIZONTAL_SEPARATION_STD = 5   # NM
    
    def get_required_parameters(self) -> List[str]:
        return ["separation_type", "min_distance"]
    
    def evaluate(
        self,
        traffic_state: TrafficState,
        parameters: Dict[str, Any],
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """Evalúa condición de separación."""
        if not aircraft_callsign:
            return ConditionResult(
                satisfied=False,
                details={"error": "No aircraft callsign provided"}
            )
        
        aircraft = traffic_state.get_aircraft(aircraft_callsign)
        if not aircraft:
            return ConditionResult(
                satisfied=False,
                details={"error": f"Aircraft {aircraft_callsign} not found"}
            )
        
        separation_type = parameters.get("separation_type")
        min_distance = parameters.get("min_distance")
        
        # Obtener aeronaves cercanas
        nearby_distance = parameters.get("search_radius_nm", 20)
        nearby_aircraft = traffic_state.get_nearby_aircraft(aircraft_callsign, nearby_distance)
        
        if separation_type == "VERTICAL":
            return self._check_vertical_separation(
                aircraft, nearby_aircraft, min_distance, parameters
            )
        
        elif separation_type == "HORIZONTAL":
            return self._check_horizontal_separation(
                aircraft, nearby_aircraft, min_distance, parameters
            )
        
        elif separation_type == "BOTH":
            # Verificar ambas separaciones
            vertical_result = self._check_vertical_separation(
                aircraft, nearby_aircraft, self.VERTICAL_SEPARATION_STD, parameters
            )
            horizontal_result = self._check_horizontal_separation(
                aircraft, nearby_aircraft, self.HORIZONTAL_SEPARATION_STD, parameters
            )
            
            # Si alguna falla, retornar la violación
            if not vertical_result.satisfied:
                return vertical_result
            if not horizontal_result.satisfied:
                return horizontal_result
            
            return ConditionResult(
                satisfied=True,
                details={"check": "separation_passed", "nearby_count": len(nearby_aircraft)}
            )
        
        return ConditionResult(
            satisfied=False,
            details={"error": f"Unknown separation_type: {separation_type}"}
        )
    
    def _check_vertical_separation(
        self,
        aircraft: AircraftState,
        nearby: List[AircraftState],
        min_separation: int,
        parameters: Dict[str, Any],
    ) -> ConditionResult:
        """Verifica separación vertical."""
        own_altitude = aircraft.position.altitude
        
        for other in nearby:
            other_altitude = other.position.altitude
            vertical_sep = abs(own_altitude - other_altitude)
            
            if vertical_sep < min_separation:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "SEPARATION_RULE"),
                    condition_type="SEPARATION_VERTICAL",
                    severity=AlertSeverity.HIGH,
                    details={
                        "aircraft_1": aircraft.callsign,
                        "aircraft_2": other.callsign,
                        "vertical_separation_ft": vertical_sep,
                        "required_separation_ft": min_separation,
                        "altitude_1": own_altitude,
                        "altitude_2": other_altitude,
                    },
                    explanation=(
                        f"Vertical separation between {aircraft.callsign} and "
                        f"{other.callsign} is {vertical_sep}ft, "
                        f"below minimum {min_separation}ft"
                    ),
                    suggestion=f"Assign different altitudes immediately",
                )
                return ConditionResult(satisfied=False, violation=violation)
        
        return ConditionResult(satisfied=True, details={"check": "vertical_separation_passed"})
    
    def _check_horizontal_separation(
        self,
        aircraft: AircraftState,
        nearby: List[AircraftState],
        min_separation: float,
        parameters: Dict[str, Any],
    ) -> ConditionResult:
        """Verifica separación horizontal."""
        from Alert_System.models.traffic_state import TrafficState
        
        for other in nearby:
            distance = TrafficState.calculate_distance(aircraft.position, other.position)
            
            if distance < min_separation:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "SEPARATION_RULE"),
                    condition_type="SEPARATION_HORIZONTAL",
                    severity=AlertSeverity.HIGH,
                    details={
                        "aircraft_1": aircraft.callsign,
                        "aircraft_2": other.callsign,
                        "horizontal_separation_nm": distance,
                        "required_separation_nm": min_separation,
                    },
                    explanation=(
                        f"Horizontal separation between {aircraft.callsign} and "
                        f"{other.callsign} is {distance:.1f}NM, "
                        f"below minimum {min_separation}NM"
                    ),
                    suggestion="Issue heading instructions to increase separation",
                )
                return ConditionResult(satisfied=False, violation=violation)
        
        return ConditionResult(satisfied=True, details={"check": "horizontal_separation_passed"})
    
    def evaluate_all(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa TODAS las reglas de separación registradas.
        
        Si no hay reglas registradas, usa reglas por defecto (separación BOTH).
        """
        violations = []
        
        # Si no hay reglas registradas, usar reglas por defecto
        if not self._rules:
            # Regla por defecto: verificar separación BOTH
            default_rules = [
                {
                    "condition_type": "SEPARATION",
                    "parameters": {
                        "separation_type": "BOTH",
                        "min_distance": 5,
                        "rule_id": "SEPARATION_RULE"
                    }
                }
            ]
            rules_to_evaluate = default_rules
        else:
            rules_to_evaluate = self._rules
        
        for rule in rules_to_evaluate:
            parameters = rule.get("parameters", {})
            result = self.evaluate(traffic_state, parameters, aircraft_callsign)
            if not result.satisfied and result.violation:
                violations.append(result.violation)
        
        return violations


class RunwayCondition(ConditionEvaluator):
    """
    Evalúa condiciones de pista.
    
    Tipos de condición:
    - RUNWAY_OCCUPIED: Verifica si la pista está ocupada
    - RUNWAY_AVAILABLE: Verifica si la pista está disponible
    - HOLDING_SHORT: Verifica cola de espera
    """
    
    condition_type = "RUNWAY"
    
    def get_required_parameters(self) -> List[str]:
        return ["check_type", "runway_id"]
    
    def evaluate(
        self,
        traffic_state: TrafficState,
        parameters: Dict[str, Any],
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """Evalúa condición de pista."""
        check_type = parameters.get("check_type")
        runway_id = parameters.get("runway_id")
        
        runway = traffic_state.runways.get(runway_id)
        
        if check_type == "OCCUPIED":
            # Verificar si la pista está ocupada (para takeoff/landing clearance)
            if runway and runway.occupied:
                occupied_by = runway.occupied_by or "unknown"
                violation = Violation(
                    rule_id=parameters.get("rule_id", "RUNWAY_RULE"),
                    condition_type="RUNWAY_OCCUPIED",
                    severity=AlertSeverity.CRITICAL,
                    details={
                        "runway_id": runway_id,
                        "occupied_by": occupied_by,
                        "requesting_aircraft": aircraft_callsign,
                    },
                    explanation=(
                        f"Runway {runway_id} is occupied by {occupied_by}. "
                        f"Cannot issue clearance to {aircraft_callsign}"
                    ),
                    suggestion=f"Wait for {occupied_by} to clear runway",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "runway_available"})
        
        elif check_type == "HOLDING_SHORT_FULL":
            # Verificar si la cola de holding short está llena
            max_holding = parameters.get("max_holding", 3)
            if runway and len(runway.holding_short) >= max_holding:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "RUNWAY_RULE"),
                    condition_type="HOLDING_SHORT_CONGESTION",
                    severity=AlertSeverity.MEDIUM,
                    details={
                        "runway_id": runway_id,
                        "holding_count": len(runway.holding_short),
                        "max_allowed": max_holding,
                        "holding_aircrafts": runway.holding_short,
                    },
                    explanation=(
                        f"Holding short of {runway_id} is congested "
                        f"({len(runway.holding_short)} aircrafts)"
                    ),
                    suggestion="Consider alternative runway or sequence optimization",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "holding_ok"})
        
        elif check_type == "EXISTS":
            # Verificar que la pista existe
            if runway_id not in traffic_state.runways:
                violation = Violation(
                    rule_id=parameters.get("rule_id", "RUNWAY_RULE"),
                    condition_type="RUNWAY_NOT_FOUND",
                    severity=AlertSeverity.HIGH,
                    details={
                        "runway_id": runway_id,
                        "available_runways": list(traffic_state.runways.keys()),
                    },
                    explanation=f"Runway {runway_id} does not exist in this sector",
                    suggestion=f"Use available runway: {list(traffic_state.runways.keys())}",
                )
                return ConditionResult(satisfied=False, violation=violation)
            
            return ConditionResult(satisfied=True, details={"check": "runway_exists"})
        
        return ConditionResult(
            satisfied=False,
            details={"error": f"Unknown check_type: {check_type}"}
        )

    def evaluate_all(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa TODAS las reglas de pista registradas.
        
        Si no hay reglas registradas, verifica todas las pistas disponibles
        para detectar conflictos de ocupación.
        """
        violations = []
        
        # Si hay reglas registradas, evaluarlas
        if self._rules:
            for rule in self._rules:
                parameters = rule.get("parameters", {})
                result = self.evaluate(traffic_state, parameters, aircraft_callsign)
                if not result.satisfied and result.violation:
                    violations.append(result.violation)
        else:
            # Sin reglas registradas: verificar todas las pistas para conflicto
            # de ocupación (para aeronaves que solicitan clearance)
            for runway_id, runway_state in traffic_state.runways.items():
                if runway_state.occupied and runway_state.occupied_by:
                    # Solo reportar si la aeronave involucrada no es la ocupante
                    if aircraft_callsign and runway_state.occupied_by != aircraft_callsign:
                        from Alert_System.models.alert import AlertSeverity
                        violation = Violation(
                            violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                            rule_id="RUNWAY_RULE",
                            condition_type="RUNWAY_AVAILABLE",
                            severity=AlertSeverity.CRITICAL,
                            explanation=f"Runway {runway_id} occupied by {runway_state.occupied_by}",
                            details={
                                "aircraft_involved": [aircraft_callsign, runway_state.occupied_by],
                                "runway": runway_id,
                                "occupied_by": runway_state.occupied_by,
                            },
                        )
                        violations.append(violation)
        
        return violations


class GenericKexCondition(ConditionEvaluator):
    """
    Evaluador genérico para reglas KEX que no encajan en categorías predefinidas.
    
    Este evaluador almacena reglas del KEX en formato ExecutableRule y
    las evalúa mediante interpretación dinámica contra el estado del tráfico.
    
    Es el "catch-all" para reglas arbitrarias extraídas de documentos aeronáuticos.
    """
    
    condition_type = "GENERIC"
    
    def __init__(self, llm_config: Optional[Any] = None):
        """Inicializa el evaluador genérico.
        
        Args:
            llm_config: Configuración para evaluación LLM. Si None, solo usa keywords.
        """
        super().__init__()
        self._executable_rule = None  # Almacena ExecutableRule para evaluación
        self.condition_id = ""
        self.llm_config = llm_config
        self._instructor_client = None  # Lazy initialization
        self._raw_client = None
    
    def get_required_parameters(self) -> List[str]:
        """Parámetros requeridos: la regla ejecutable completa."""
        return ["executable_rule"]
    
    def _initialize_clients(self):
        """Initialize LLM clients lazily when needed."""
        if self.llm_config and not self._instructor_client:
            try:
                # Lazy imports to avoid circular dependency
                from common.llm_client_factory import create_instructor_client, create_raw_client
                self._instructor_client, _ = create_instructor_client(self.llm_config)
                self._raw_client = create_raw_client(self.llm_config)
            except Exception as e:
                # Log error but continue with keyword fallback
                print(f"Warning: Failed to initialize LLM clients: {e}")
                self.llm_config = None  # Disable LLM for this instance
    
    def evaluate(
        self,
        traffic_state: Any,  # TrafficState or ProjectedState
        parameters: Dict[str, Any],
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """
        Evalúa una regla genérica contra el estado del tráfico.
        
        Intenta evaluación LLM primero si está disponible, con fallback a keywords.
        
        Args:
            traffic_state: Estado actual del tráfico (TrafficState o ProjectedState)
            parameters: Parámetros de la regla (incluyendo executable_rule)
            aircraft_callsign: Callsign de la aeronave a evaluar
            
        Returns:
            ConditionResult indicando si se cumple la condición
        """
        # Obtener la regla ejecutable
        executable = parameters.get("executable_rule") or self._executable_rule
        
        if not executable:
            return ConditionResult(
                satisfied=False,
                violation=None,
                details={"error": "No executable rule provided"}
            )
        
        # Intentar evaluación LLM primero si está disponible
        if self.llm_config:
            self._initialize_clients()
            if self._instructor_client:
                try:
                    return self._evaluate_with_llm(executable, traffic_state, aircraft_callsign)
                except Exception as e:
                    # Log error and fallback to keywords
                    print(f"Warning: LLM evaluation failed: {e}, falling back to keywords")
        
        # Fallback: evaluación por keywords
        return self._evaluate_with_keywords(executable, traffic_state, aircraft_callsign)
    
    def _evaluate_with_llm(
        self,
        executable: Any,  # ExecutableRule from integration.schemas
        traffic_state: Any,  # TrafficState or ProjectedState
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """Evaluate rule using LLM with structured output."""
        # Lazy imports to avoid circular dependency
        from ..integration.schemas import LLMEvaluationResult
        from ..config.evaluation_prompts import build_evaluation_prompt
        
        # Handle both TrafficState and ProjectedState
        if hasattr(traffic_state, 'traffic_state'):
            # It's a ProjectedState
            actual_state = traffic_state.traffic_state
        else:
            # It's a TrafficState
            actual_state = traffic_state
        
        # Build traffic state summary
        traffic_summary = self._build_traffic_state_summary(actual_state, aircraft_callsign)
        
        # Build prompts
        system_prompt, user_prompt = build_evaluation_prompt(
            rule_id=executable.source_rule_id,
            rule_category=executable.rule_category,
            rule_description=executable.condition_description or "",
            raw_rule_text=f"{executable.raw_trigger or ''} {executable.raw_constraint or ''}",
            traffic_state_summary=traffic_summary["traffic"],
            aircraft_summary=traffic_summary["aircraft"],
            msa_value=str(actual_state.msa or "N/A"),
            runway_status=traffic_summary["runways"],
            separation_summary=traffic_summary["separations"],
        )
        
        # Call LLM with structured output
        response = self._instructor_client.chat.completions.create(
            model=self.llm_config.name,
            response_model=LLMEvaluationResult,
            max_retries=self.llm_config.max_retries,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # Convert LLM result to ConditionResult
        if response.is_violated and response.confidence > 0.5:
            # Determine severity
            severity = AlertSeverity.MEDIUM
            if response.severity_override:
                severity_map = {
                    "LOW": AlertSeverity.LOW,
                    "MEDIUM": AlertSeverity.MEDIUM,
                    "HIGH": AlertSeverity.HIGH,
                    "CRITICAL": AlertSeverity.CRITICAL,
                }
                severity = severity_map.get(response.severity_override, AlertSeverity.MEDIUM)
            
            violation = Violation(
                violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                rule_id=executable.source_rule_id,
                condition_type="GENERIC_LLM_VIOLATION",
                severity=severity,
                explanation=response.explanation,
                details={
                    "rule_category": executable.rule_category,
                    "llm_confidence": response.confidence,
                    "suggested_action": response.suggested_action,
                    "extracted_values": response.extracted_values,
                    "aircraft": aircraft_callsign,
                },
            )
            return ConditionResult(satisfied=False, violation=violation)
        
        # No violation detected
        return ConditionResult(
            satisfied=True,
            details={
                "check": "llm_evaluation",
                "rule_id": executable.source_rule_id,
                "confidence": response.confidence,
                "explanation": response.explanation,
            }
        )
    
    def _evaluate_with_keywords(
        self,
        executable: Any,  # ExecutableRule from integration.schemas
        traffic_state: Any,  # TrafficState or ProjectedState
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """Fallback evaluation using keyword matching."""
        condition_desc = executable.condition_description or ""
        condition_lower = condition_desc.lower()
        
        # Handle both TrafficState and ProjectedState
        if hasattr(traffic_state, 'traffic_state'):
            # It's a ProjectedState
            actual_state = traffic_state.traffic_state
        else:
            # It's a TrafficState
            actual_state = traffic_state
        
        # Verificar si la regla menciona conceptos que podemos evaluar
        if "altitude" in condition_lower or "below" in condition_lower:
            # Delegar a evaluación de altitud si hay aeronave
            if aircraft_callsign:
                aircraft = actual_state.get_aircraft(aircraft_callsign)
                if aircraft and actual_state.msa:
                    if aircraft.position.altitude < actual_state.msa:
                        violation = Violation(
                            violation_id=f"VIO_{uuid.uuid4().hex[:8]}",
                            rule_id=executable.source_rule_id,
                            condition_type="GENERIC_MSA_VIOLATION",
                            severity=AlertSeverity.HIGH,
                            explanation=f"Generic rule violation: {condition_desc}",
                            details={
                                "rule_category": executable.rule_category,
                                "aircraft": aircraft_callsign,
                                "altitude": aircraft.position.altitude,
                                "msa": actual_state.msa,
                            },
                        )
                        return ConditionResult(satisfied=False, violation=violation)
        
        # Por defecto: condición satisfecha (no se detectó violación)
        return ConditionResult(
            satisfied=True,
            details={"check": "keyword_evaluation", "rule_id": executable.source_rule_id}
        )
    
    def _build_traffic_state_summary(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build human-readable summary of traffic state for LLM."""
        summary = {
            "traffic": "",
            "aircraft": "",
            "runways": "",
            "separations": "",
        }
        
        # Traffic state overview
        summary["traffic"] = f"MSA: {traffic_state.msa or 'N/A'}, Total aircraft: {len(traffic_state.aircrafts)}"
        
        # Aircraft of interest
        if aircraft_callsign and aircraft_callsign in traffic_state.aircrafts:
            aircraft = traffic_state.aircrafts[aircraft_callsign]
            summary["aircraft"] = (
                f"{aircraft_callsign}: "
                f"Alt={aircraft.position.altitude}ft, "
                f"Speed={aircraft.position.speed}kts, "
                f"Heading={aircraft.position.heading}°"
            )
        else:
            summary["aircraft"] = "No specific aircraft provided"
        
        # Runway status
        if traffic_state.runways:
            runway_list = []
            for runway_id, runway in traffic_state.runways.items():
                status = "occupied" if runway.occupied else "free"
                runway_list.append(f"{runway_id}: {status}")
            summary["runways"] = ", ".join(runway_list)
        else:
            summary["runways"] = "No runway data"
        
        # Separation concerns
        if hasattr(traffic_state, 'projected_separations') and traffic_state.projected_separations:
            sep_list = []
            for sep in traffic_state.projected_separations[:3]:  # Limit to first 3
                sep_list.append(f"{sep.aircraft1}-{sep.aircraft2}: {sep.distance:.1f}nm")
            summary["separations"] = ", ".join(sep_list)
        else:
            summary["separations"] = "No separation data"
        
        return summary
    
    def evaluate_all(
        self,
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa todas las reglas genéricas registradas.
        
        Para GenericKexCondition, típicamente hay una sola regla
        almacenada en _executable_rule.
        """
        violations = []
        
        if self._executable_rule:
            result = self.evaluate(
                traffic_state=traffic_state,
                parameters={},
                aircraft_callsign=aircraft_callsign
            )
            if not result.satisfied and result.violation:
                violations.append(result.violation)
        
        # También evaluar reglas registradas vía add_rule
        for rule in self._rules:
            parameters = rule.get("parameters", {})
            result = self.evaluate(traffic_state, parameters, aircraft_callsign)
            if not result.satisfied and result.violation:
                violations.append(result.violation)
        
        return violations
