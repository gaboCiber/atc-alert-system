"""Schemas para integracion KEX - RuleEngine."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ExecutableRule(BaseModel):
    """Regla en formato ejecutable para el RuleEngine."""
    
    source_rule_id: str = Field(..., description="ID de la regla original en el KEX")
    rule_category: str = Field(..., description="Categoria: ALTITUDE, SEPARATION, RUNWAY, GENERIC, UNEVALUABLE")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Parametros estructurados para evaluadores especificos")
    condition_description: Optional[str] = Field(default=None, description="Descripcion textual de la condicion")
    required_state_fields: List[str] = Field(default_factory=list, description="Campos del TrafficState necesarios")
    reason_unexecutable: Optional[str] = Field(default=None, description="Razon por la que no se puede evaluar automaticamente")
    raw_trigger: Optional[str] = Field(default=None, description="Texto del trigger original del KEX")
    raw_constraint: Optional[str] = Field(default=None, description="Texto de la constraint original del KEX")
    severity: Optional[str] = Field(default=None, description="Severidad de la regla")
    safety_critical: bool = Field(default=False, description="Si es critica para seguridad")
    
    # Campos para clasificación LLM
    rule_type: Optional[str] = Field(default=None, description="Tipo de regla (prohibition, obligation, etc.)")
    modality: Optional[str] = Field(default=None, description="Modalidad (shall, may, etc.)")
    raw_formal_if_then: Optional[Dict[str, Any]] = Field(default=None, description="Representación formal if-then")
    raw_applicability: Optional[Dict[str, Any]] = Field(default=None, description="Ámbito de aplicación")
    explainability: Optional[str] = Field(default=None, description="Razón de la regla")


class LLMEvaluationResult(BaseModel):
    """Resultado estructurado de evaluación LLM para reglas genéricas."""

    is_violated: bool = Field(..., description="Si la regla está violada según el estado actual")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza en la evaluación (0-1)")
    explanation: str = Field(..., description="Explicación del razonamiento LLM")
    suggested_action: Optional[str] = Field(default=None, description="Acción sugerida si hay violación")
    severity_override: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = Field(
        default=None, description="Severidad sugerida por LLM"
    )
    extracted_values: Optional[Dict[str, Any]] = Field(
        default=None, description="Valores extraídos del estado para la evaluación"
    )


class RuleRelevance(BaseModel):
    """Resultado de relevancia de una regla para una instrucción."""

    rule_index: int = Field(..., ge=0, description="Índice 0-based de la regla en la lista candidata")
    is_relevant: bool = Field(..., description="Si la regla es relevante para la instrucción")
    reason: str = Field(..., max_length=100, description="Breve justificación de por qué aplica o no")


class RelevanceFilterResult(BaseModel):
    """Resultado del filtro batch de relevancia vía LLM."""

    relevances: List[RuleRelevance] = Field(..., description="Lista de relevancias por regla")
    summary: str = Field(..., max_length=200, description="Resumen de cuántas reglas son relevantes")
    relevant_count: int = Field(..., ge=0, description="Número total de reglas marcadas como relevantes")
