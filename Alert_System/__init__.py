"""Sistema de Alertas ATC - Integración ASR + KEX en tiempo real.

Este módulo implementa un sistema de alertas que procesa instrucciones ATC
desde transcripción hasta generación de alertas estructuradas, utilizando
state projection para validar "what-if" antes de aplicar cambios.
"""

from .models import (
    Alert,
    AlertCategory,
    AlertResult,
    AlertSeverity,
    AircraftState,
    Clearances,
    FlightPhase,
    InstructionType,
    ParsedInstruction,
    Position,
    RunwayState,
    Speaker,
    TrafficState,
    Violation,
    WakeTurbulenceCategory,
)

__version__ = "0.1.0"
__all__ = [
    "Alert",
    "AlertCategory",
    "AlertResult",
    "AlertSeverity",
    "AircraftState",
    "Clearances",
    "FlightPhase",
    "InstructionType",
    "ParsedInstruction",
    "Position",
    "RunwayState",
    "Speaker",
    "TrafficState",
    "Violation",
    "WakeTurbulenceCategory",
]
