"""Schemas para integracion KEX - RuleEngine."""

from typing import List, Dict, Any, Optional
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
