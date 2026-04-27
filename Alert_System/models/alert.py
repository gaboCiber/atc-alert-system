"""Modelos para alertas del sistema de alertas ATC."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AlertSeverity(str, Enum):
    """Niveles de severidad de alerta."""
    INFO = "info"           # Información, no requiere acción
    LOW = "low"             # Precaución, monitorear
    MEDIUM = "medium"       # Atención recomendada
    HIGH = "high"           # Acción requerida
    CRITICAL = "critical"   # Acción inmediata requerida


class AlertCategory(str, Enum):
    """Categorías de alertas ATC."""
    # Violaciones de altitud
    ALTITUDE_VIOLATION = "altitude_violation"
    MSA_VIOLATION = "msa_violation"
    FLIGHT_LEVEL_VIOLATION = "flight_level_violation"
    
    # Separación
    SEPARATION_LOSS = "separation_loss"
    SEPARATION_CONFLICT = "separation_conflict"
    LATERAL_SEPARATION = "lateral_separation"
    VERTICAL_SEPARATION = "vertical_separation"
    
    # Pista
    RUNWAY_CONFLICT = "runway_conflict"
    RUNWAY_INCURSION = "runway_incursion"
    RUNWAY_OCCUPIED = "runway_occupied"
    
    # Velocidad
    SPEED_VIOLATION = "speed_violation"
    OVERSPEED = "overspeed"
    UNDERSPEED = "underspeed"
    
    # Fase
    PHASE_VIOLATION = "phase_violation"
    WRONG_PHASE = "wrong_phase"
    
    # Procedimental
    PROCEDURAL_ERROR = "procedural_error"
    CLEARANCE_ERROR = "clearance_error"
    SEQUENCING_ERROR = "sequencing_error"
    
    # Emergencia
    EMERGENCY_DETECTED = "emergency_detected"
    MAYDAY_RECEIVED = "mayday_received"
    PAN_PAN_RECEIVED = "pan_pan_received"
    
    # Comunicación
    COMMS_LOSS = "comms_loss"
    READBACK_ERROR = "readback_error"
    
    # Sistema
    SYSTEM_ERROR = "system_error"
    PARSING_ERROR = "parsing_error"


class Violation(BaseModel):
    """Una violación específica de una regla."""
    
    # Identificación
    violation_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    rule_id: str = Field(..., description="ID de la regla violada (ej: RULE001)")
    
    # Tipo
    condition_type: str = Field(
        ...,
        description="Tipo de condición que falló: 'ALTITUDE_MINIMUM', 'SEPARATION_VERTICAL', etc."
    )
    severity: AlertSeverity = Field(..., description="Severidad de esta violación")
    
    # Detalles específicos
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros de la violación"
    )
    
    # Ejemplos de details:
    # - expected_minimum: 5000, actual: 4200
    # - conflicting_callsign: "UAL456", distance_nm: 2.3
    # - runway: "09L", occupied_by: "AAL123"
    
    # Explicación
    explanation: str = Field(..., description="Texto descriptivo de la violación")
    suggestion: Optional[str] = Field(None, description="Sugerencia para corregir")
    
    # Metadatos
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_detail(self, key: str, default: Any = None) -> Any:
        """Obtener un detalle específico."""
        return self.details.get(key, default)


class Alert(BaseModel):
    """Alerta completa generada por el sistema."""
    
    # Identificación
    alert_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Clasificación
    severity: AlertSeverity = Field(..., description="Severidad global de la alerta")
    category: AlertCategory = Field(..., description="Categoría de la alerta")
    
    # Afectados
    affected_callsigns: List[str] = Field(
        default_factory=list,
        description="Callsigns de aeronaves involucradas"
    )
    primary_callsign: Optional[str] = Field(
        None,
        description="Callsign principal al que iba dirigida la instrucción"
    )
    
    # Causa
    triggering_instruction_raw: str = Field(
        ...,
        description="Texto de la instrucción que disparó la alerta"
    )
    violations: List[Violation] = Field(
        default_factory=list,
        description="Violaciones detectadas"
    )
    
    # Presentación
    title: str = Field(..., description="Título corto de la alerta")
    explanation: str = Field(..., description="Explicación detallada para el ATCO")
    suggested_action: str = Field(..., description="Acción correctiva sugerida")
    
    # Estado proyectado (para diagnóstico)
    projected_state: Optional[Dict[str, Any]] = Field(
        None,
        description="Snapshot del estado proyectado que causó la alerta"
    )
    
    # Estado de la alerta
    acknowledged: bool = Field(False, description="Si el ATCO la reconoció")
    acknowledged_at: Optional[datetime] = Field(None)
    acknowledged_by: Optional[str] = Field(None, description="Operador que la reconoció")
    
    commit_decision: str = Field(
        "PENDING",
        description="Decisión del ATCO: 'PENDING', 'COMMIT', 'ROLLBACK'"
    )
    commit_decision_at: Optional[datetime] = Field(None)
    
    # Override
    force_committed: bool = Field(
        False,
        description="Si el ATCO forzó el commit ignorando la alerta"
    )
    force_commit_reason: Optional[str] = Field(None, description="Razón del override")
    
    # Metadatos
    sector_id: Optional[str] = Field(None, description="Sector donde ocurrió")
    source_system: str = Field("alert_pipeline", description="Sistema que generó la alerta")
    
    # Historial
    state_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historial de estados de la alerta"
    )
    
    def get_primary_violation(self) -> Optional[Violation]:
        """Obtener la violación más severa."""
        if not self.violations:
            return None
        
        severity_order = [
            AlertSeverity.CRITICAL,
            AlertSeverity.HIGH,
            AlertSeverity.MEDIUM,
            AlertSeverity.LOW,
            AlertSeverity.INFO,
        ]
        
        sorted_violations = sorted(
            self.violations,
            key=lambda v: severity_order.index(v.severity)
        )
        return sorted_violations[0]
    
    def is_resolved(self) -> bool:
        """¿La alerta está resuelta?"""
        return self.commit_decision in ["COMMIT", "ROLLBACK"] or self.acknowledged
    
    def is_critical(self) -> bool:
        """¿Es una alerta crítica?"""
        return self.severity == AlertSeverity.CRITICAL
    
    def requires_immediate_action(self) -> bool:
        """¿Requiere acción inmediata?"""
        return self.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
    
    def add_violation(self, violation: Violation) -> None:
        """Añadir una violación y actualizar severidad si es necesario."""
        self.violations.append(violation)
        # Actualizar severidad global si la nueva es más grave
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4,
        }
        if severity_order.get(violation.severity, 0) > severity_order.get(self.severity, 0):
            self.severity = violation.severity
    
    def acknowledge(self, operator_id: str) -> None:
        """Marcar alerta como reconocida."""
        self.acknowledged = True
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = operator_id
    
    def set_commit_decision(self, decision: str, reason: Optional[str] = None) -> None:
        """Establecer decisión de commit."""
        self.commit_decision = decision
        self.commit_decision_at = datetime.utcnow()
        if decision == "COMMIT" and reason:
            self.force_committed = True
            self.force_commit_reason = reason
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class AlertResult(BaseModel):
    """Resultado de evaluar una instrucción."""
    
    instruction: Dict[str, Any] = Field(description="Instrucción evaluada")
    status: str = Field(..., description="'OK', 'WARNING', 'ALERT'")
    alert: Optional[Alert] = Field(None, description="Alerta generada si la hay")
    violations_count: int = Field(0, description="Número de violaciones")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento")
    
    def has_alert(self) -> bool:
        """¿Se generó una alerta?"""
        return self.alert is not None
    
    def is_safe(self) -> bool:
        """¿Es seguro proceder?"""
        return self.status == "OK" or (self.status == "WARNING" and self.alert is not None and not self.alert.is_critical())
