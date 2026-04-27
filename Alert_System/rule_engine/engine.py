"""Motor de reglas para evaluación de condiciones ATC."""

from typing import Any, Dict, List, Optional, Type

from Alert_System.models.alert import AlertSeverity, Violation
from Alert_System.models.traffic_state import TrafficState

from .conditions import ConditionEvaluator, ConditionResult


class RuleEngine:
    """
    Motor de reglas genérico y extensible para evaluación ATC.
    
    Permite registrar evaluadores de condiciones y evaluar reglas
    contra el estado del tráfico aéreo.
    """
    
    def __init__(self):
        """Inicializa el motor de reglas."""
        self._evaluators: Dict[str, Type[ConditionEvaluator]] = {}
        self._evaluator_instances: Dict[str, ConditionEvaluator] = {}
        
        # Registrar evaluadores por defecto
        self._register_default_evaluators()
    
    def _register_default_evaluators(self) -> None:
        """Registra los evaluadores de condiciones por defecto."""
        from .conditions import AltitudeCondition, RunwayCondition, SeparationCondition
        
        self.register_evaluator("ALTITUDE", AltitudeCondition)
        self.register_evaluator("SEPARATION", SeparationCondition)
        self.register_evaluator("RUNWAY", RunwayCondition)
    
    def register_evaluator(
        self,
        condition_type: str,
        evaluator_class: Type[ConditionEvaluator],
    ) -> None:
        """
        Registra un evaluador de condiciones.
        
        Args:
            condition_type: Tipo de condición (ej: "ALTITUDE", "SEPARATION")
            evaluator_class: Clase evaluadora que hereda de ConditionEvaluator
        """
        self._evaluators[condition_type] = evaluator_class
        self._evaluator_instances[condition_type] = evaluator_class()
    
    def has_evaluator(self, condition_type: str) -> bool:
        """Verifica si existe un evaluador para el tipo de condición."""
        return condition_type in self._evaluators
    
    def evaluate(
        self,
        condition_type: str,
        parameters: Dict[str, Any],
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> ConditionResult:
        """
        Evalúa una condición específica.
        
        Args:
            condition_type: Tipo de condición a evaluar
            parameters: Parámetros de la condición
            traffic_state: Estado del tráfico (actual o proyectado)
            aircraft_callsign: Callsign de la aeronave objetivo
            
        Returns:
            ConditionResult con el resultado de la evaluación
        """
        if not self.has_evaluator(condition_type):
            return ConditionResult(
                satisfied=False,
                violation=None,
                details={"error": f"No evaluator registered for type: {condition_type}"}
            )
        
        evaluator = self._evaluator_instances[condition_type]
        
        # Validar parámetros
        valid, errors = evaluator.validate_parameters(parameters)
        if not valid:
            return ConditionResult(
                satisfied=False,
                violation=None,
                details={"error": "Parameter validation failed", "errors": errors}
            )
        
        # Evaluar la condición
        return evaluator.evaluate(traffic_state, parameters, aircraft_callsign)
    
    def batch_evaluate(
        self,
        conditions: List[Dict[str, Any]],
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[ConditionResult]:
        """
        Evalúa múltiples condiciones.
        
        Args:
            conditions: Lista de condiciones, cada una con 'type' y 'parameters'
            traffic_state: Estado del tráfico
            aircraft_callsign: Callsign de la aeronave objetivo
            
        Returns:
            Lista de ConditionResult
        """
        results = []
        
        for condition in conditions:
            cond_type = condition.get("type")
            parameters = condition.get("parameters", {})
            
            result = self.evaluate(cond_type, parameters, traffic_state, aircraft_callsign)
            results.append(result)
        
        return results
    
    def evaluate_all_violations(
        self,
        conditions: List[Dict[str, Any]],
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> List[Violation]:
        """
        Evalúa todas las condiciones y retorna solo las violaciones.
        
        Args:
            conditions: Lista de condiciones a evaluar
            traffic_state: Estado del tráfico
            aircraft_callsign: Callsign de la aeronave objetivo
            
        Returns:
            Lista de Violaciones (solo las que no se satisfacen)
        """
        results = self.batch_evaluate(conditions, traffic_state, aircraft_callsign)
        
        violations = []
        for result in results:
            if not result.satisfied and result.violation:
                violations.append(result.violation)
        
        return violations
    
    def check_rule(
        self,
        rule: Dict[str, Any],
        traffic_state: TrafficState,
        aircraft_callsign: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verifica una regla completa contra el estado del tráfico.
        
        Args:
            rule: Definición de la regla con 'conditions' y 'logic'
            traffic_state: Estado del tráfico
            aircraft_callsign: Callsign de la aeronave objetivo
            
        Returns:
            Diccionario con resultado de la evaluación
        """
        conditions = rule.get("conditions", [])
        logic = rule.get("logic", "ALL")  # ALL = todas deben pasar, ANY = alguna puede fallar
        
        results = self.batch_evaluate(conditions, traffic_state, aircraft_callsign)
        
        violations = [r.violation for r in results if not r.satisfied and r.violation]
        
        if logic == "ALL":
            # Todas las condiciones deben pasar
            passed = all(r.satisfied for r in results)
        else:  # logic == "ANY"
            # Al menos una condición debe pasar
            passed = any(r.satisfied for r in results)
        
        # Determinar severidad global
        severity = AlertSeverity.INFO
        for v in violations:
            if v.severity == AlertSeverity.CRITICAL:
                severity = AlertSeverity.CRITICAL
                break
            elif v.severity == AlertSeverity.HIGH and severity != AlertSeverity.CRITICAL:
                severity = AlertSeverity.HIGH
            elif v.severity == AlertSeverity.MEDIUM and severity == AlertSeverity.INFO:
                severity = AlertSeverity.MEDIUM
        
        return {
            "rule_id": rule.get("id", "UNKNOWN"),
            "passed": passed,
            "violations": violations,
            "severity": severity,
            "condition_results": [
                {
                    "satisfied": r.satisfied,
                    "details": r.details,
                }
                for r in results
            ],
        }
    
    def get_registered_evaluators(self) -> List[str]:
        """Retorna lista de tipos de condiciones registradas."""
        return list(self._evaluators.keys())
    
    def get_evaluator_info(self, condition_type: str) -> Optional[Dict[str, Any]]:
        """Retorna información sobre un evaluador específico."""
        if not self.has_evaluator(condition_type):
            return None
        
        evaluator = self._evaluator_instances[condition_type]
        
        return {
            "type": condition_type,
            "class": self._evaluators[condition_type].__name__,
            "required_parameters": evaluator.get_required_parameters(),
        }
