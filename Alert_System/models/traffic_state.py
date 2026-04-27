"""Modelos de estado dinámico del tráfico aéreo (runtime)."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlightPhase(str, Enum):
    """Fases de vuelo de una aeronave."""
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI = "taxi"


class WakeTurbulenceCategory(str, Enum):
    """Categorías de estela turbulenta."""
    LIGHT = "L"
    MEDIUM = "M"
    HEAVY = "H"
    SUPER = "S"


class Position(BaseModel):
    """Posición geográfica y altimétrica de una aeronave."""
    latitude: float = Field(..., description="Latitud en grados")
    longitude: float = Field(..., description="Longitud en grados")
    altitude: int = Field(..., description="Altitud en pies (ft)")
    heading: int = Field(..., ge=0, le=360, description="Rumbo en grados")
    speed: int = Field(..., ge=0, description="Velocidad en nudos (knots)")
    vertical_rate: Optional[int] = Field(None, description="Tasa de ascenso/descenso en ft/min")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Clearances(BaseModel):
    """Clearances activos de una aeronave."""
    altitude_assigned: Optional[int] = Field(None, description="Altitud asignada en ft")
    heading_assigned: Optional[int] = Field(None, ge=0, le=360, description="Rumbo asignado")
    runway_assigned: Optional[str] = Field(None, description="Pista asignada")
    route: Optional[str] = Field(None, description="Ruta asignada")
    squawk: Optional[str] = Field(None, description="Código transponder")
    speed_assigned: Optional[int] = Field(None, description="Velocidad asignada en nudos")


class AircraftState(BaseModel):
    """Estado completo de una aeronave en el sistema."""
    callsign: str = Field(..., description="Callsign único (ej: AAL123)")
    position: Position = Field(..., description="Posición actual")
    flight_phase: FlightPhase = Field(..., description="Fase de vuelo actual")
    
    # Clearances y restricciones
    clearances: Clearances = Field(default_factory=Clearances)
    restrictions: List[str] = Field(default_factory=list, description="Restricciones activas")
    
    # Características
    wake_turbulence: WakeTurbulenceCategory = Field(
        WakeTurbulenceCategory.MEDIUM,
        description="Categoría de estela turbulenta"
    )
    aircraft_type: Optional[str] = Field(None, description="Tipo de aeronave (ej: B738)")
    
    # Estado del sistema
    last_contact: datetime = Field(default_factory=datetime.utcnow)
    is_emergency: bool = Field(False, description="Si está en emergencia")
    emergency_type: Optional[str] = Field(None, description="Tipo de emergencia")
    
    # Historial reciente
    position_history: List[Position] = Field(
        default_factory=list,
        max_length=10,
        description="Últimas posiciones conocidas"
    )

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class RunwayOperationMode(str, Enum):
    """Modo de operación de pista."""
    LANDING = "landing"
    TAKEOFF = "takeoff"
    MIXED = "mixed"
    CLOSED = "closed"


class RunwayState(BaseModel):
    """Estado de una pista de aeropuerto."""
    runway_id: str = Field(..., description="Identificador de pista (ej: '09L')")
    occupied: bool = Field(False, description="Si está ocupada")
    occupied_by: Optional[str] = Field(None, description="Callsign que ocupa la pista")
    operation_mode: RunwayOperationMode = Field(RunwayOperationMode.MIXED)
    
    # Colas
    holding_short: List[str] = Field(
        default_factory=list,
        description="Callsigns esperando en holding short"
    )
    landing_queue: List[str] = Field(
        default_factory=list,
        description="Callsigns en cola de aterrizaje"
    )
    
    # Restricciones
    closed_until: Optional[datetime] = Field(None)
    closure_reason: Optional[str] = Field(None)


class TrafficState(BaseModel):
    """Estado global del tráfico aéreo en un sector."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sector_id: str = Field(..., description="Identificador del sector ATC")
    
    # Aeronaves
    aircrafts: Dict[str, AircraftState] = Field(
        default_factory=dict,
        description="Aeronaves indexadas por callsign"
    )
    
    # Infraestructura
    runways: Dict[str, RunwayState] = Field(
        default_factory=dict,
        description="Pistas indexadas por runway_id"
    )
    
    # Configuración del sector
    msa: Optional[int] = Field(None, description="Minimum Sector Altitude en ft")
    sector_boundary: Optional[List[tuple]] = Field(None, description="Polígono del sector")
    
    # Meteo
    qnh: Optional[int] = Field(None, description="Presión QNH en hPa")
    wind: Optional[Dict[str, Any]] = Field(None, description="Dirección y velocidad del viento")
    
    def get_aircraft(self, callsign: str) -> Optional[AircraftState]:
        """Obtener estado de una aeronave por callsign."""
        return self.aircrafts.get(callsign.upper())
    
    def add_aircraft(self, aircraft: AircraftState) -> None:
        """Añadir o actualizar una aeronave."""
        self.aircrafts[aircraft.callsign.upper()] = aircraft
    
    def remove_aircraft(self, callsign: str) -> None:
        """Remover una aeronave del estado."""
        self.aircrafts.pop(callsign.upper(), None)
    
    def get_runway(self, runway_id: str) -> Optional[RunwayState]:
        """Obtener estado de una pista por runway_id."""
        return self.runways.get(runway_id.upper())
    
    def add_runway(self, runway: RunwayState) -> None:
        """Añadir o actualizar una pista."""
        self.runways[runway.runway_id.upper()] = runway
    
    def remove_runway(self, runway_id: str) -> None:
        """Remover una pista del estado."""
        self.runways.pop(runway_id.upper(), None)
    
    def get_nearby_aircraft(
        self,
        callsign: str,
        max_distance_nm: float = 20.0
    ) -> List[AircraftState]:
        """Obtener aeronaves cercanas a una aeronave dada."""
        target = self.get_aircraft(callsign)
        if not target:
            return []
        
        nearby = []
        for ac in self.aircrafts.values():
            if ac.callsign.upper() != callsign.upper():
                # Simplified distance calculation (would use haversine in production)
                distance = TrafficState.calculate_distance(target.position, ac.position)
                if distance <= max_distance_nm:
                    nearby.append(ac)
        
        return nearby
    
    @staticmethod
    def calculate_distance(pos1: Position, pos2: Position) -> float:
        """Calcula distancia aproximada en NM entre dos posiciones.
        
        Método estático para uso general (no requiere instancia de TrafficState).
        """
        import math
        
        lat_diff = pos2.latitude - pos1.latitude
        lon_diff = pos2.longitude - pos1.longitude
        
        # 1 degree latitude ≈ 60 NM
        # 1 degree longitude varies with latitude, approximation:
        lat_nm = lat_diff * 60
        lon_nm = lon_diff * 60 * math.cos(math.radians(pos1.latitude))
        
        return math.sqrt(lat_nm**2 + lon_nm**2)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
