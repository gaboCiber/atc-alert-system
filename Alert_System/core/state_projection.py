"""State Projection - simulación "what-if" para el sistema de alertas ATC."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from Alert_System.models.instruction import InstructionType, ParsedInstruction
from Alert_System.models.traffic_state import (
    AircraftState,
    FlightPhase,
    Position,
    TrafficState,
)


@dataclass
class ProjectedWaypoint:
    """Waypoint proyectado en una trayectoria."""
    latitude: float
    longitude: float
    altitude: int
    estimated_time: float  # segundos desde el inicio
    speed: int
    vertical_rate: int


@dataclass
class ProjectedTrajectory:
    """Trayectoria proyectada de una aeronave."""
    callsign: str
    waypoints: List[ProjectedWaypoint] = field(default_factory=list)
    estimated_duration_sec: float = 0.0
    final_phase: FlightPhase = FlightPhase.CRUISE


@dataclass
class ProjectedSeparation:
    """Separación proyectada entre dos aeronaves en un punto futuro."""
    aircraft_1: str
    aircraft_2: str
    vertical_separation_ft: float
    horizontal_separation_nm: float
    time_to_conflict: Optional[float] = None  # segundos hasta pérdida de separación
    conflict_predicted: bool = False


@dataclass
class ProjectedState:
    """
    Estado proyectado del tráfico aéreo.
    
    Es una copia modificada del TrafficState que representa cómo
    quedaría el sistema si se aplicara la instrucción.
    """
    
    # Copia del estado base
    traffic_state: TrafficState
    
    # Instrucción que generó esta proyección
    source_instruction: ParsedInstruction
    
    # Trayectorias proyectadas
    trajectories: Dict[str, ProjectedTrajectory] = field(default_factory=dict)
    
    # Separaciones proyectadas vs otras aeronaves
    projected_separations: Dict[str, List[ProjectedSeparation]] = field(
        default_factory=dict
    )
    
    # Metadatos
    projection_timestamp: float = 0.0  # timestamp de cuando se hizo la proyección
    projection_horizon_min: int = 10  # minutos de proyección
    
    # Estado de la aeronave objetivo después de la instrucción
    target_aircraft_final: Optional[AircraftState] = None
    
    # Flags de validación
    is_valid_projection: bool = True
    projection_errors: List[str] = field(default_factory=list)
    
    def get_aircraft(self, callsign: str) -> Optional[AircraftState]:
        """Obtener aeronave del estado proyectado."""
        return self.traffic_state.get_aircraft(callsign)
    
    def get_trajectory(self, callsign: str) -> Optional[ProjectedTrajectory]:
        """Obtener trayectoria proyectada de una aeronave."""
        return self.trajectories.get(callsign)
    
    def has_conflicts(self) -> bool:
        """¿Hay conflictos de separación proyectados?"""
        for separations in self.projected_separations.values():
            for sep in separations:
                if sep.conflict_predicted:
                    return True
        return False
    
    def get_conflicts(self) -> List[ProjectedSeparation]:
        """Obtener todos los conflictos proyectados."""
        conflicts = []
        for separations in self.projected_separations.values():
            for sep in separations:
                if sep.conflict_predicted:
                    conflicts.append(sep)
        return conflicts


class StateProjector:
    """
    Proyecta el estado del tráfico aplicando una instrucción.
    
    Implementa el concepto de "State Projection" de la Propuesta 4:
    crear una copia simulada del mundo, aplicar la instrucción ahí,
    y ver qué pasaría antes de tocar el estado real.
    """
    
    # Tasas típicas de ascenso/descenso (ft/min)
    DESCENT_RATE_NORMAL = 1000
    DESCENT_RATE_EXPEDITE = 2000
    CLIMB_RATE_NORMAL = 1500
    CLIMB_RATE_EXPEDITE = 2500
    
    # Velocidades típicas por fase (knots)
    SPEED_GROUND = 30
    SPEED_TAKEOFF = 150
    SPEED_CLIMB = 250
    SPEED_CRUISE = 450
    SPEED_DESCENT = 280
    SPEED_APPROACH = 180
    SPEED_LANDING = 140
    
    def __init__(self):
        """Inicializa el proyector de estado."""
        pass
    
    def create_projection(
        self,
        traffic_state: TrafficState,
        instruction: ParsedInstruction,
        projection_minutes: int = 10,
    ) -> ProjectedState:
        """
        Crea una proyección del estado aplicando una instrucción.
        
        Args:
            traffic_state: Estado actual del tráfico
            instruction: Instrucción a proyectar
            projection_minutes: Minutos de proyección hacia adelante
            
        Returns:
            ProjectedState con el estado simulado
        """
        # Crear copia profunda del estado
        projected_traffic = deepcopy(traffic_state)
        
        # Aplicar la instrucción a la copia
        target_callsign = instruction.callsign
        if not target_callsign:
            return ProjectedState(
                traffic_state=projected_traffic,
                source_instruction=instruction,
                is_valid_projection=False,
                projection_errors=["No callsign in instruction"],
            )
        
        target_aircraft = projected_traffic.get_aircraft(target_callsign)
        if not target_aircraft:
            return ProjectedState(
                traffic_state=projected_traffic,
                source_instruction=instruction,
                is_valid_projection=False,
                projection_errors=[f"Aircraft {target_callsign} not found"],
            )
        
        # Aplicar cambios según el tipo de instrucción
        try:
            self._apply_instruction(target_aircraft, instruction)
        except Exception as e:
            return ProjectedState(
                traffic_state=projected_traffic,
                source_instruction=instruction,
                is_valid_projection=False,
                projection_errors=[f"Error applying instruction: {str(e)}"],
            )
        
        # Calcular trayectoria proyectada
        trajectory = self._calculate_trajectory(
            target_aircraft,
            instruction,
            projection_minutes,
        )
        
        # Calcular separaciones proyectadas
        separations = self._calculate_projected_separations(
            projected_traffic,
            target_callsign,
            trajectory,
        )
        
        return ProjectedState(
            traffic_state=projected_traffic,
            source_instruction=instruction,
            trajectories={target_callsign: trajectory},
            projected_separations={target_callsign: separations},
            projection_horizon_min=projection_minutes,
            target_aircraft_final=target_aircraft,
        )
    
    def _apply_instruction(
        self,
        aircraft: AircraftState,
        instruction: ParsedInstruction,
    ) -> None:
        """
        Aplica los cambios de la instrucción a la aeronave.
        
        Modifica el AircraftState en el estado proyectado.
        """
        instruction_type = instruction.instruction_type
        params = instruction.parameters
        
        # Altitud
        if instruction_type in [
            InstructionType.DESCENT,
            InstructionType.CLIMB,
            InstructionType.MAINTAIN_ALTITUDE,
            InstructionType.EXPEDITE_DESCENT,
            InstructionType.EXPEDITE_CLIMB,
        ]:
            target_altitude = instruction.get_target_altitude()
            if target_altitude is not None:
                aircraft.position.altitude = target_altitude
                aircraft.clearances.altitude_assigned = target_altitude
                
                # Actualizar fase según altitud
                self._update_flight_phase(aircraft, instruction_type)
        
        # Rumbo
        if instruction_type in [
            InstructionType.HEADING,
            InstructionType.TURN_LEFT,
            InstructionType.TURN_RIGHT,
        ]:
            target_heading = instruction.get_target_heading()
            if target_heading is not None:
                aircraft.position.heading = target_heading
                aircraft.clearances.heading_assigned = target_heading
        
        # Velocidad
        if instruction_type in [
            InstructionType.SPEED,
            InstructionType.MAINTAIN_SPEED,
            InstructionType.REDUCE_SPEED,
            InstructionType.INCREASE_SPEED,
        ]:
            target_speed = instruction.get_target_speed()
            if target_speed is not None:
                aircraft.position.speed = target_speed
                aircraft.clearances.speed_assigned = target_speed
        
        # Pista (para takeoff/landing clearance)
        if instruction_type in [
            InstructionType.TAKEOFF_CLEARANCE,
            InstructionType.LANDING_CLEARANCE,
        ]:
            runway = params.get("runway")
            if runway:
                aircraft.clearances.runway_assigned = runway
                if instruction_type == InstructionType.TAKEOFF_CLEARANCE:
                    aircraft.flight_phase = FlightPhase.TAKEOFF
                elif instruction_type == InstructionType.LANDING_CLEARANCE:
                    aircraft.flight_phase = FlightPhase.LANDING
    
    def _update_flight_phase(
        self,
        aircraft: AircraftState,
        instruction_type: InstructionType,
    ) -> None:
        """Actualiza la fase de vuelo según la instrucción."""
        altitude = aircraft.position.altitude
        
        # Determinar fase según altitud (simplificado)
        if altitude < 1000:
            if instruction_type == InstructionType.TAKEOFF_CLEARANCE:
                aircraft.flight_phase = FlightPhase.TAKEOFF
            else:
                aircraft.flight_phase = FlightPhase.GROUND
        elif altitude < 10000:
            if instruction_type == InstructionType.CLIMB:
                aircraft.flight_phase = FlightPhase.CLIMB
            elif instruction_type == InstructionType.DESCENT:
                aircraft.flight_phase = FlightPhase.APPROACH
            else:
                aircraft.flight_phase = FlightPhase.APPROACH
        elif altitude < 18000:
            if instruction_type == InstructionType.CLIMB:
                aircraft.flight_phase = FlightPhase.CLIMB
            elif instruction_type == InstructionType.DESCENT:
                aircraft.flight_phase = FlightPhase.DESCENT
            else:
                aircraft.flight_phase = FlightPhase.CRUISE
        else:
            aircraft.flight_phase = FlightPhase.CRUISE
    
    def _calculate_trajectory(
        self,
        aircraft: AircraftState,
        instruction: ParsedInstruction,
        minutes: int,
    ) -> ProjectedTrajectory:
        """
        Calcula la trayectoria proyectada de la aeronave.
        
        Genera waypoints estimados para los próximos minutos.
        """
        waypoints = []
        
        # Posición inicial
        current_lat = aircraft.position.latitude
        current_lon = aircraft.position.longitude
        current_alt = aircraft.position.altitude
        current_speed = aircraft.position.speed
        heading = aircraft.position.heading
        
        # Determinar tasa vertical según instrucción
        vertical_rate = 0
        if instruction.instruction_type == InstructionType.DESCENT:
            vertical_rate = -self.DESCENT_RATE_NORMAL
        elif instruction.instruction_type == InstructionType.EXPEDITE_DESCENT:
            vertical_rate = -self.DESCENT_RATE_EXPEDITE
        elif instruction.instruction_type == InstructionType.CLIMB:
            vertical_rate = self.CLIMB_RATE_NORMAL
        elif instruction.instruction_type == InstructionType.EXPEDITE_CLIMB:
            vertical_rate = self.CLIMB_RATE_EXPEDITE
        
        # Generar waypoints cada minuto
        import math
        
        for minute in range(1, minutes + 1):
            # Calcular distancia recorrida en 1 minuto (NM)
            distance_nm = current_speed / 60.0
            
            # Convertir a grados (aproximación)
            # 1 NM ≈ 1/60 grado latitud
            # longitud varía con latitud
            lat_change = (distance_nm / 60.0) * math.cos(math.radians(heading))
            lon_change = (distance_nm / 60.0) * math.sin(math.radians(heading)) / math.cos(math.radians(current_lat))
            
            current_lat += lat_change
            current_lon += lon_change
            
            # Actualizar altitud
            current_alt += vertical_rate
            if current_alt < 0:
                current_alt = 0
            
            # Velocidad según fase
            if current_alt < 1000:
                current_speed = self.SPEED_TAKEOFF if vertical_rate > 0 else self.SPEED_LANDING
            elif current_alt < 10000:
                current_speed = self.SPEED_CLIMB if vertical_rate > 0 else self.SPEED_DESCENT
            else:
                current_speed = self.SPEED_CRUISE
            
            waypoint = ProjectedWaypoint(
                latitude=round(current_lat, 6),
                longitude=round(current_lon, 6),
                altitude=int(current_alt),
                estimated_time=minute * 60,  # segundos
                speed=int(current_speed),
                vertical_rate=int(vertical_rate),
            )
            waypoints.append(waypoint)
        
        return ProjectedTrajectory(
            callsign=aircraft.callsign,
            waypoints=waypoints,
            estimated_duration_sec=minutes * 60,
            final_phase=aircraft.flight_phase,
        )
    
    def _calculate_projected_separations(
        self,
        traffic_state: TrafficState,
        target_callsign: str,
        trajectory: ProjectedTrajectory,
    ) -> List[ProjectedSeparation]:
        """
        Calcula separaciones proyectadas vs otras aeronaves.
        
        Evalúa en cada waypoint de la trayectoria.
        """
        separations = []
        
        target_ac = traffic_state.get_aircraft(target_callsign)
        if not target_ac:
            return separations
        
        # Obtener aeronaves cercanas
        nearby = traffic_state.get_nearby_aircraft(target_callsign, max_distance_nm=30)
        
        import math
        
        for other in nearby:
            # Para cada aeronave cercana, evaluar separación en cada waypoint
            min_vertical_sep = float('inf')
            min_horizontal_sep = float('inf')
            conflict_at_time = None
            
            for waypoint in trajectory.waypoints:
                # Calcular separación vertical
                vertical_sep = abs(waypoint.altitude - other.position.altitude)
                min_vertical_sep = min(min_vertical_sep, vertical_sep)
                
                # Calcular separación horizontal (aproximada)
                lat_diff = waypoint.latitude - other.position.latitude
                lon_diff = waypoint.longitude - other.position.longitude
                lat_nm = lat_diff * 60
                lon_nm = lon_diff * 60 * math.cos(math.radians(other.position.latitude))
                horizontal_sep = math.sqrt(lat_nm**2 + lon_nm**2)
                min_horizontal_sep = min(min_horizontal_sep, horizontal_sep)
                
                # Detectar conflicto (estándar ICAO: 1000ft vertical, 5NM horizontal)
                if vertical_sep < 1000 and horizontal_sep < 5:
                    if conflict_at_time is None:
                        conflict_at_time = waypoint.estimated_time
            
            separation = ProjectedSeparation(
                aircraft_1=target_callsign,
                aircraft_2=other.callsign,
                vertical_separation_ft=min_vertical_sep,
                horizontal_separation_nm=min_horizontal_sep,
                time_to_conflict=conflict_at_time,
                conflict_predicted=conflict_at_time is not None,
            )
            separations.append(separation)
        
        return separations
    
    def estimate_time_to_altitude(
        self,
        current_altitude: int,
        target_altitude: int,
        instruction_type: InstructionType,
    ) -> float:
        """
        Estima tiempo (minutos) para alcanzar una altitud.
        
        Args:
            current_altitude: Altitud actual en ft
            target_altitude: Altitud objetivo en ft
            instruction_type: Tipo de instrucción (determina tasa)
            
        Returns:
            Tiempo estimado en minutos
        """
        altitude_diff = abs(target_altitude - current_altitude)
        
        if altitude_diff == 0:
            return 0.0
        
        # Determinar tasa
        if instruction_type == InstructionType.EXPEDITE_DESCENT:
            rate = self.DESCENT_RATE_EXPEDITE
        elif instruction_type == InstructionType.EXPEDITE_CLIMB:
            rate = self.CLIMB_RATE_EXPEDITE
        elif instruction_type == InstructionType.DESCENT:
            rate = self.DESCENT_RATE_NORMAL
        else:
            rate = self.CLIMB_RATE_NORMAL
        
        # Calcular tiempo en minutos
        return altitude_diff / rate
