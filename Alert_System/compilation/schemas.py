"""Schemas para compilación de reglas KEX a código Python."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field


class CompilationStatus(str, Enum):
    """Estado de compilación de una regla."""
    COMPILED = "compiled"
    FAILED = "failed"
    PENDING = "pending"
    NOT_COMPILABLE = "not_compilable"  # Regla subjetiva, no verificable con TrafficState


class RuleVerdict(BaseModel):
    """Veredicto de clasificación: ¿es la regla compilable con TrafficState?"""
    is_compilable: bool = Field(..., description="Si la regla puede evaluarse objetivamente con TrafficState")
    reason: str = Field(..., description="Razón de la clasificación")
    required_fields: List[str] = Field(
        default_factory=list,
        description="Campos de TrafficState necesarios para evaluar la regla"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confianza en la clasificación (0-1)"
    )


class CompiledRule(BaseModel):
    """Regla compilada por LLM, almacenada en disco."""
    
    source_rule_id: str = Field(..., description="ID de la regla KEX original")
    rule_category: str = Field(..., description="Categoría: GENERIC, ALTITUDE, etc.")
    condition_description: str = Field(..., description="Descripción original de la condición")
    compiled_code: str = Field(..., description="Código Python generado por LLM (función evaluate)")
    required_state_fields: List[str] = Field(
        default_factory=list,
        description="Campos del TrafficState que usa la función"
    )
    compilation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata: modelo LLM, timestamp, confianza compilación"
    )
    compilation_status: CompilationStatus = Field(
        default=CompilationStatus.PENDING,
        description="Estado de la compilación"
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Razón si la compilación falló"
    )
    raw_trigger: Optional[str] = Field(
        default=None,
        description="Texto del trigger original del KEX"
    )
    raw_constraint: Optional[str] = Field(
        default=None,
        description="Texto de la constraint original del KEX"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severidad de la regla original"
    )
    safety_critical: bool = Field(
        default=False,
        description="Si es crítica para seguridad"
    )


class CompilationManifest(BaseModel):
    """Manifiesto de todas las reglas compiladas."""
    
    version: str = Field(default="1.0", description="Versión del formato de manifiesto")
    compiled_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de compilación")
    model_used: str = Field(..., description="Modelo LLM usado para compilar")
    rules: Dict[str, CompiledRule] = Field(
        default_factory=dict,
        description="Reglas compiladas indexadas por source_rule_id"
    )
    total_compiled: int = Field(default=0, description="Total reglas compiladas exitosamente")
    total_failed: int = Field(default=0, description="Total reglas que fallaron compilación")
    total_fallback: int = Field(default=0, description="Total reglas que usan fallback LLM runtime")
    total_not_compilable: int = Field(default=0, description="Total reglas no compilables (subjetivas)")
    
    def add_rule(self, rule: CompiledRule) -> None:
        """Agrega una regla al manifiesto y actualiza contadores."""
        self.rules[rule.source_rule_id] = rule
        if rule.compilation_status == CompilationStatus.COMPILED:
            self.total_compiled += 1
        elif rule.compilation_status == CompilationStatus.FAILED:
            self.total_failed += 1
            self.total_fallback += 1
        elif rule.compilation_status == CompilationStatus.NOT_COMPILABLE:
            self.total_not_compilable += 1
            self.total_fallback += 1
