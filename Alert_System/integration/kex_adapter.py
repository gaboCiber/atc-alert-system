"""
Adaptador KEX para el Alert System.

Convierte reglas extraídas por KEX a condiciones ejecutables
del Rule Engine.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

# KEX imports
from Knowledge_Extractor import Rule, Entity, Event

# Alert System imports
from Alert_System.rule_engine.conditions import (
    ConditionEvaluator,
    AltitudeCondition,
    SeparationCondition,
    RunwayCondition,
)
from Alert_System.models.alert import AlertCategory, AlertSeverity
from Alert_System.integration.schemas import ExecutableRule


@dataclass
class KnowledgeContext:
    """Contexto de conocimiento extraído para alertas."""
    rules: List[Rule]
    entities: List[Entity]
    events: List[Event]
    extraction_timestamp: datetime
    source_document: str
    
    def find_rule_by_id(self, rule_id: str) -> Optional[Rule]:
        """Busca una regla por ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def find_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Busca una entidad por ID."""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None


class KEXAdapter:
    """
    Adaptador que convierte reglas KEX a ConditionEvaluators.
    
    Responsabilidades:
    1. Recibir reglas del KEX
    2. Mapear reglas a condiciones ejecutables
    3. Configurar el Rule Engine con reglas dinámicas
    """
    
    def __init__(self):
        """Inicializa el adaptador KEX."""
        import json
        import os
        
        # Cargar configuracion de patrones desde JSON
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "rule_patterns.json"
        )
        self._rule_patterns = {}
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    for pattern in config.get("patterns", []):
                        self._rule_patterns[pattern["name"]] = pattern
            except (json.JSONDecodeError, IOError):
                pass  # Fallback: usar categorizacion por defecto
        
        # Mapeo de severidad
        self._severity_mapping = {
            "CRITICAL": AlertSeverity.CRITICAL,
            "HIGH": AlertSeverity.HIGH,
            "MEDIUM": AlertSeverity.MEDIUM,
            "LOW": AlertSeverity.LOW,
            "INFO": AlertSeverity.INFO,
        }
    
    def adapt_rules(self, rules: List[Rule]) -> List[ConditionEvaluator]:
        """
        Convierte reglas KEX a evaluadores de condiciones.
        
        Nuevo flujo:
        1. Primero compila cada Rule a ExecutableRule (formato intermedio)
        2. Luego adapta ExecutableRule a ConditionEvaluator
        
        Args:
            rules: Lista de reglas del KEX
            
        Returns:
            Lista de ConditionEvaluators para el Rule Engine
        """
        evaluators = []
        
        for rule in rules:
            try:
                # Paso 1: Compilar a formato ejecutable intermedio
                executable = self.compile_to_executable(rule)
                
                # Paso 2: Adaptar a evaluador
                evaluator = self._adapt_executable_rule(executable)
                if evaluator:
                    evaluators.append(evaluator)
            except Exception as e:
                # Log error pero no detener proceso
                print(f"⚠️ Error adaptando regla {rule.id}: {e}")
        
        return evaluators
    
    def compile_to_executable(self, rule: Rule) -> ExecutableRule:
        """
        Compila una regla KEX a formato ejecutable intermedio.
        
        Este paso analiza la regla y decide:
        - Si es una regla conocida (ALTITUDE, SEPARATION, RUNWAY)
        - Si es una regla genérica que requiere interpretación
        - Si es una regla no evaluable automáticamente
        
        Args:
            rule: Regla del KEX
            
        Returns:
            ExecutableRule con categoría y parámetros determinados
        """
        # Determinar categoría
        category = self._categorize_rule(rule)
        
        # Extraer texto relevante
        trigger_text = rule.trigger.description if rule.trigger else ""
        formal_if = rule.formal_if_then.if_condition if rule.formal_if_then else ""
        formal_then = rule.formal_if_then.then_action if rule.formal_if_then else ""
        
        # Crear ExecutableRule base
        executable = ExecutableRule(
            source_rule_id=rule.id,
            rule_category=category,
            raw_trigger=trigger_text,
            raw_constraint=formal_then,
            severity=rule.severity if hasattr(rule, 'severity') else None,
            safety_critical=getattr(rule, 'safety_critical', False),
        )
        
        # Si es categoría conocida, extraer parámetros estructurados
        if category in ["ALTITUDE", "SEPARATION", "RUNWAY"]:
            executable.parameters = self._extract_parameters(rule)
            executable.condition_description = f"{formal_if} -> {formal_then}"
        
        # Si es genérica, almacenar descripción para interpretación LLM
        elif category == "GENERIC":
            executable.condition_description = (
                f"Trigger: {trigger_text}. "
                f"Condition: {formal_if}. "
                f"Action required: {formal_then}"
            )
            executable.required_state_fields = self._infer_required_fields(rule)
        
        # Si no es evaluable, almacenar razón
        elif category == "UNEVALUABLE":
            executable.reason_unexecutable = (
                "Regla requiere juicio humano o información no disponible en TrafficState"
            )
        
        return executable
    
    def _categorize_rule(self, rule: Rule) -> str:
        """
        Categoriza una regla KEX según su contenido usando patrones del JSON.
        
        Returns:
            "ALTITUDE", "SEPARATION", "RUNWAY", "GENERIC", o "UNEVALUABLE"
        """
        trigger_text = rule.trigger.description.lower() if rule.trigger else ""
        formal_if = rule.formal_if_then.if_condition.lower() if rule.formal_if_then else ""
        formal_then = rule.formal_if_then.then_action.lower() if rule.formal_if_then else ""
        full_text = trigger_text + " " + formal_if + " " + formal_then
        
        # Usar patrones del JSON si están disponibles
        if self._rule_patterns:
            for pattern_name, pattern in self._rule_patterns.items():
                keywords = pattern.get("keywords", [])
                if any(keyword.lower() in full_text for keyword in keywords):
                    return pattern.get("category", "GENERIC")
        
        # Fallback: categorización por defecto si JSON no está disponible
        # Altitude-related
        if any(word in full_text for word in [
            "altitude", "msa", "minimum sector altitude", "flight level", "fl",
            "climb", "descend", "descent"
        ]):
            return "ALTITUDE"
        
        # Separation
        if any(word in full_text for word in [
            "separation", "distance", "conflict", "loss of separation",
            "vertical separation", "horizontal separation"
        ]):
            return "SEPARATION"
        
        # Runway
        if any(word in full_text for word in [
            "runway", "rwy", "occupied", "closed", "landing", "takeoff",
            "taxi", "holding point"
        ]):
            return "RUNWAY"
        
        # Verificar si es evaluable (requiere información del TrafficState)
        if any(word in full_text for word in [
            "pilot", "crew", "fatigue", "weather", "visibility",
            "judgment", "discretion", "decide"
        ]):
            return "UNEVALUABLE"
        
        # Por defecto: regla genérica que puede intentar evaluarse
        return "GENERIC"
    
    def _infer_required_fields(self, rule: Rule) -> List[str]:
        """Infiere qué campos del TrafficState se necesitan para evaluar esta regla."""
        trigger_text = rule.trigger.description.lower() if rule.trigger else ""
        formal_if = rule.formal_if_then.if_condition.lower() if rule.formal_if_then else ""
        full_text = trigger_text + " " + formal_if
        
        # Usar patrones del JSON si están disponibles
        if self._rule_patterns:
            for pattern_name, pattern in self._rule_patterns.items():
                keywords = pattern.get("keywords", [])
                if any(keyword.lower() in full_text for keyword in keywords):
                    return pattern.get("required_state_fields", [])
        
        # Fallback: inferencia por defecto
        fields = []
        
        if any(word in full_text for word in ["altitude", "flight level", "fl", "climb", "descend"]):
            fields.append("aircraft.position.altitude")
        
        if any(word in full_text for word in ["runway", "rwy", "landing", "takeoff"]):
            fields.append("runways")
        
        if any(word in full_text for word in ["separation", "distance", "conflict"]):
            fields.append("aircraft")
            fields.append("projected_separations")
        
        return fields
    
    def _adapt_executable_rule(self, executable: ExecutableRule) -> Optional[ConditionEvaluator]:
        """
        Adapta un ExecutableRule a un ConditionEvaluator.
        
        Args:
            executable: Regla en formato ejecutable
            
        Returns:
            ConditionEvaluator o None si no se puede adaptar
        """
        from Alert_System.rule_engine.conditions import GenericKexCondition
        
        category = executable.rule_category
        
        # Reglas no evaluables: no crear evaluador
        if category == "UNEVALUABLE":
            return None
        
        # Reglas genéricas: usar GenericKexCondition
        if category == "GENERIC":
            condition = GenericKexCondition()
            condition.condition_id = executable.source_rule_id
            condition._executable_rule = executable
            return condition
        
        # Reglas conocidas: mapear a evaluadores específicos
        if category == "ALTITUDE":
            condition = AltitudeCondition()
            condition.condition_id = executable.source_rule_id
            # Agregar reglas al evaluador
            if executable.parameters:
                condition.add_rule({
                    "condition_type": "ALTITUDE",
                    "parameters": executable.parameters,
                })
            return condition
        
        if category == "SEPARATION":
            condition = SeparationCondition()
            condition.condition_id = executable.source_rule_id
            if executable.parameters:
                condition.add_rule({
                    "condition_type": "SEPARATION",
                    "parameters": executable.parameters,
                })
            return condition
        
        if category == "RUNWAY":
            condition = RunwayCondition()
            condition.condition_id = executable.source_rule_id
            if executable.parameters:
                condition.add_rule({
                    "condition_type": "RUNWAY",
                    "parameters": executable.parameters,
                })
            return condition
        
        return None
    
    def _adapt_single_rule(self, rule: Rule) -> Optional[ConditionEvaluator]:
        """
        Adapta una regla individual (método legacy, ahora usa ExecutableRule).
        
        Args:
            rule: Regla del KEX
            
        Returns:
            ConditionEvaluator o None si no se puede adaptar
        """
        # Compilar a ExecutableRule primero
        executable = self.compile_to_executable(rule)
        return self._adapt_executable_rule(executable)
    
    def _infer_condition_type(self, rule: Rule) -> Optional[str]:
        """
        Infiere el tipo de condición de una regla KEX.
        
        Analiza el trigger y formal_if_then para determinar
        qué tipo de evaluador necesitamos.
        """
        trigger_text = rule.trigger.description.lower() if rule.trigger else ""
        formal_if = rule.formal_if_then.if_condition.lower() if rule.formal_if_then else ""
        
        # Altitude-related
        if any(word in trigger_text + formal_if for word in [
            "altitude", "msa", "minimum sector altitude", "flight level", "fl"
        ]):
            if any(word in trigger_text + formal_if for word in ["below", "less than", "<"]):
                return "ALTITUDE_MINIMUM"
            if any(word in trigger_text + formal_if for word in ["above", "greater than", ">"]):
                return "ALTITUDE_MAXIMUM"
        
        # Separation
        if any(word in trigger_text + formal_if for word in [
            "separation", "distance", "conflict", "loss"
        ]):
            return "SEPARATION_VERTICAL"
        
        # Runway
        if any(word in trigger_text + formal_if for word in [
            "runway", "rwy", "occupied", "closed"
        ]):
            if "occupied" in trigger_text + formal_if or "closed" in trigger_text + formal_if:
                return "RUNWAY_CLOSED"
            return "RUNWAY_AVAILABLE"
        
        return None
    
    def _create_altitude_condition(
        self,
        rule: Rule,
        condition_type: str,
    ) -> ConditionEvaluator:
        """Crea condición de altitud desde regla KEX."""
        # Extraer parámetros de la regla
        params = self._extract_parameters(rule)
        
        # Determinar check_type
        check_type = "above" if condition_type == "ALTITUDE_MINIMUM" else "below"
        
        # Valor por defecto para altitud mínima
        altitude_value = params.get("altitude", 5000)
        
        # Crear instancia y almacenar parámetros para evaluación posterior
        condition = AltitudeCondition()
        condition.condition_id = rule.id
        condition.condition_type = check_type
        condition.altitude_value = altitude_value
        condition._params = params
        return condition
    
    def _create_separation_condition(
        self,
        rule: Rule,
        condition_type: str,
    ) -> ConditionEvaluator:
        """Crea condición de separación desde regla KEX."""
        params = self._extract_parameters(rule)
        
        # Valores por defecto estándar ATC
        min_vertical = params.get("min_vertical", 1000)  # feet
        min_horizontal = params.get("min_horizontal", 5)  # NM
        
        # Crear instancia y almacenar parámetros para evaluación posterior
        condition = SeparationCondition()
        condition.condition_id = rule.id
        condition.min_vertical_separation = min_vertical
        condition.min_horizontal_separation = min_horizontal
        condition._params = params
        return condition
    
    def _create_runway_condition(
        self,
        rule: Rule,
        condition_type: str,
    ) -> ConditionEvaluator:
        """Crea condición de pista desde regla KEX."""
        params = self._extract_parameters(rule)
        
        check_type = "available" if condition_type == "RUNWAY_AVAILABLE" else "occupied"
        runway_id = params.get("runway_id")
        
        # Crear instancia y almacenar parámetros para evaluación posterior
        condition = RunwayCondition()
        condition.condition_id = rule.id
        condition.check_type = check_type
        condition.runway_id = runway_id
        condition._params = params
        return condition
    
    def _extract_parameters(self, rule: Rule) -> Dict[str, Any]:
        """
        Extrae parámetros numéricos de la regla KEX.
        
        Busca valores en el texto de la regla.
        """
        import re
        
        params = {}
        
        # Combinar texto relevante
        full_text = f"{rule.trigger.description} {rule.formal_if_then.if_condition} {rule.formal_if_then.then_action}"
        
        # Extraer números con unidades
        # Altitudes: 5000 feet, FL240, etc.
        alt_matches = re.findall(r'(\d+)\s*(?:feet|ft)', full_text, re.IGNORECASE)
        if alt_matches:
            params["altitude"] = int(alt_matches[0])
        
        fl_matches = re.findall(r'FL\s*(\d+)', full_text, re.IGNORECASE)
        if fl_matches:
            params["altitude"] = int(fl_matches[0]) * 100
        
        # Distancias: 5 NM, 1000 feet vertical
        nm_matches = re.findall(r'(\d+(?:\.\d+)?)\s*NM', full_text, re.IGNORECASE)
        if nm_matches:
            params["min_horizontal"] = float(nm_matches[0])
        
        # Separación vertical
        vert_matches = re.findall(r'(\d+)\s*feet\s+(?:vertical|separation)', full_text, re.IGNORECASE)
        if vert_matches:
            params["min_vertical"] = int(vert_matches[0])
        
        # Runway ID
        rwy_matches = re.findall(r'(?:runway|rwy)\s*(\d{2}[LR]?)', full_text, re.IGNORECASE)
        if rwy_matches:
            params["runway_id"] = rwy_matches[0].upper()
        
        return params
    
    def create_knowledge_context(
        self,
        rules: List[Rule],
        entities: List[Entity],
        events: List[Event],
        source: str = "unknown",
    ) -> KnowledgeContext:
        """
        Crea contexto de conocimiento para uso en el pipeline.
        
        Args:
            rules: Reglas extraídas
            entities: Entidades extraídas
            events: Eventos extraídos
            source: Documento fuente
            
        Returns:
            KnowledgeContext
        """
        return KnowledgeContext(
            rules=rules,
            entities=entities,
            events=events,
            extraction_timestamp=datetime.utcnow(),
            source_document=source,
        )
