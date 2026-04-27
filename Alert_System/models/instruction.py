"""Modelos para instrucciones ATC parseadas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class InstructionType(str, Enum):
    """Tipos de instrucciones ATC."""
    # Desconocido / sin clasificar
    UNKNOWN = "unknown"
    
    # Movimiento vertical
    DESCENT = "descent"
    CLIMB = "climb"
    MAINTAIN_ALTITUDE = "maintain_altitude"
    EXPEDITE_DESCENT = "expedite_descent"
    EXPEDITE_CLIMB = "expedite_climb"
    
    # Rumbo
    HEADING = "heading"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    PRESENT_HEADING = "present_heading"
    
    # Velocidad
    SPEED = "speed"
    MAINTAIN_SPEED = "maintain_speed"
    REDUCE_SPEED = "reduce_speed"
    INCREASE_SPEED = "increase_speed"
    NO_SPEED_RESTRICTION = "no_speed_restriction"
    
    # Clearances
    TAKEOFF_CLEARANCE = "takeoff_clearance"
    LANDING_CLEARANCE = "landing_clearance"
    APPROACH_CLEARANCE = "approach_clearance"
    
    # Movimiento en tierra
    TAXI = "taxi"
    TAXI_VIA = "taxi_via"
    HOLD_POSITION = "hold_position"
    HOLD_SHORT = "hold_short"
    LINE_UP = "line_up"
    LINE_UP_AND_WAIT = "line_up_and_wait"
    
    # Comunicaciones
    CONTACT = "contact"
    MONITOR = "monitor"
    SQUAWK = "squawk"
    IDENT = "ident"
    CHECK_STROBE = "check_strobe"
    
    # Emergencia
    PAN_PAN = "pan_pan"
    MAYDAY = "mayday"
    EMERGENCY_DESCENT = "emergency_descent"
    
    # Otros
    REPORT = "report"
    CLEARED_AS_FILED = "cleared_as_filed"
    DIRECT_TO = "direct_to"
    CLEARED_TO_LAND = "cleared_to_land"
    GO_AROUND = "go_around"
    MISSED_APPROACH = "missed_approach"


class Speaker(str, Enum):
    """Origen de la comunicación."""
    ATCO = "atco"
    PILOT = "pilot"


class ParsedInstruction(BaseModel):
    """Instrucción ATC parseada y estructurada."""
    
    # Texto
    raw_text: str = Field(..., description="Texto original del ASR")
    normalized_text: str = Field(..., description="Texto normalizado")
    
    # Origen
    speaker: Speaker = Field(..., description="Quién habló")
    
    # Callsign objetivo
    callsign: Optional[str] = Field(None, description="Callsign al que va dirigida")
    callsign_confidence: float = Field(1.0, ge=0, le=1, description="Confianza en el callsign")
    
    # Tipo y acción
    instruction_type: InstructionType = Field(..., description="Tipo de instrucción")
    action_verb: str = Field(..., description="Verbo de acción principal (ej: 'descend')")
    action_verb_confidence: float = Field(1.0, ge=0, le=1)
    
    # Parámetros extraídos
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros numéricos y valores"
    )
    
    # Ejemplos de parámetros:
    # - target_altitude: 24000 (FL240)
    # - heading: 90
    # - speed: 250
    # - runway: "09L"
    # - waypoint: "KORLI"
    
    # Entidades referenciadas
    entities: List[str] = Field(
        default_factory=list,
        description="IDs de entidades del KEX referenciadas"
    )
    
    # Modificadores
    temporal_marker: Optional[str] = Field(
        None,
        description="Cuándo ejecutar: 'immediately', 'when_ready', 'at_pilot_discretion'"
    )
    priority_marker: Optional[str] = Field(
        None,
        description="Prioridad: 'urgent', 'priority', 'expedite'"
    )
    
    # Metadatos
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    asr_confidence: float = Field(1.0, ge=0, le=1, description="Confianza del ASR")
    source_audio: Optional[str] = Field(None, description="Path al audio fuente")
    
    # Validación
    is_valid: bool = Field(True, description="Si la instrucción es válida/entendible")
    validation_errors: List[str] = Field(default_factory=list, description="Errores de validación")
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Obtener un parámetro específico."""
        return self.parameters.get(key, default)
    
    def has_parameter(self, key: str) -> bool:
        """Verificar si existe un parámetro."""
        return key in self.parameters
    
    def requires_immediate_action(self) -> bool:
        """¿Requiere acción inmediata?"""
        return self.temporal_marker == "immediately" or self.priority_marker in ["urgent", "expedite"]
    
    def is_clearance(self) -> bool:
        """¿Es un clearance (takeoff, landing, etc.)?"""
        clearance_types = [
            InstructionType.TAKEOFF_CLEARANCE,
            InstructionType.LANDING_CLEARANCE,
            InstructionType.APPROACH_CLEARANCE,
            InstructionType.CLEARED_TO_LAND,
            InstructionType.CLEARED_AS_FILED,
        ]
        return self.instruction_type in clearance_types
    
    def is_altitude_change(self) -> bool:
        """¿Cambia la altitud?"""
        return self.instruction_type in [
            InstructionType.DESCENT,
            InstructionType.CLIMB,
            InstructionType.EXPEDITE_DESCENT,
            InstructionType.EXPEDITE_CLIMB,
            InstructionType.MAINTAIN_ALTITUDE,
        ]
    
    def get_target_altitude(self) -> Optional[int]:
        """Obtener altitud objetivo en ft."""
        alt = self.parameters.get("target_altitude")
        if alt is not None:
            return int(alt)
        
        # Intentar extraer de texto tipo "FL240"
        flight_level = self.parameters.get("flight_level")
        if flight_level is not None:
            return int(flight_level) * 100
        
        return None
    
    def get_target_heading(self) -> Optional[int]:
        """Obtener rumbo objetivo."""
        heading = self.parameters.get("heading")
        if heading is not None:
            return int(heading)
        return None
    
    def get_target_speed(self) -> Optional[int]:
        """Obtener velocidad objetivo en nudos."""
        speed = self.parameters.get("speed") or self.parameters.get("target_speed")
        if speed is not None:
            return int(speed)
        return None
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
