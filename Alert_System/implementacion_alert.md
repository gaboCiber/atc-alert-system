# Implementación del Alert System

Este documento describe la implementación del módulo `Alert_System`.
Se asume familiaridad con el modelo conceptual (`modelo_conceptual.md`);
aquí nos centramos en cómo se traduce ese modelo a código, componentes,
clases, y flujo de datos concreto.

---

## 1. Arquitectura del Paquete

```
Alert_System/
├── __init__.py
├── compile_rules_cli.py             # CLI para compilación batch de reglas
├── README.md
│
├── models/
│   ├── __init__.py                  # Exporta todos los modelos públicos
│   ├── traffic_state.py             # TrafficState, AircraftState, Position, RunwayState, etc.
│   ├── instruction.py               # ParsedInstruction, InstructionType, Speaker
│   └── alert.py                     # Alert, Violation, AlertResult, AlertSeverity, AlertCategory
│
├── core/
│   ├── __init__.py
│   ├── state_projection.py          # StateProjector, ProjectedState, ProjectedTrajectory
│   └── state_manager.py             # StateManager, StateTransaction, Transaction context manager
│
├── rule_engine/
│   ├── __init__.py
│   ├── engine.py                    # RuleEngine — registro y evaluación de condiciones
│   └── conditions.py                # ConditionEvaluator (ABC), 4 implementaciones concretas
│       # AltitudeCondition, SeparationCondition, RunwayCondition,
│       # GenericKexCondition (LLM runtime + keywords), CompiledCondition (exec)
│
├── pipeline/
│   ├── __init__.py
│   └── alert_pipeline.py            # AlertPipeline (856 líneas) — 8 pasos + evaluaciones
│
├── compilation/
│   ├── __init__.py
│   ├── schemas.py                   # CompiledRule, CompilationManifest, RuleVerdict,
│   │                                # ClassificationResponse, GeneratedCodeResponse,
│   │                                # ValidatedCodeResponse (validación Pydantic completa)
│   ├── compiler.py                  # RuleCompiler — clasifica + genera código + testea
│   ├── loader.py                    # CompiledRuleLoader — carga/salva reglas .py + manifest.json
│   ├── prompts.py                   # Prompts de compilación (TRAFFIC_STATE_SCHEMA,
│   │                                # COMPILATION_SYSTEM_PROMPT, CLASSIFICATION_*, etc.)
│   ├── validator.py                 # Validación estática por AST (sintaxis, imports, nombres)
│   ├── kex_data_processor.py        # Procesamiento de datos KEX para compilación
│   └── cli.py                       # Lógica CLI interna (invocada por compile_rules_cli.py)
│
├── integration/
│   ├── __init__.py
│   ├── schemas.py                   # ExecutableRule, LLMEvaluationResult, RuleRelevance,
│   │                                # RelevanceFilterResult
│   ├── kex_adapter.py               # KEXAdapter — Rule → ExecutableRule → ConditionEvaluator
│   ├── end_to_end_pipeline.py       # Integración extremo a extremo (KEX → AlertSystem)
│   ├── asr_adapter.py               # ASRAdapter — integración con reconocimiento de voz
│   ├── bert_atc_parser.py           # BERT NER + regex para parseo de instrucciones ATC
│   └── atc_compact_normalizer.py    # Normalizador compacto de texto ATC
│
├── config/
│   ├── rule_patterns.json           # Patrones de categorización de reglas (keywords → categoría)
│   └── evaluation_prompts.py        # Prompts para LLM runtime (GenericKexCondition)
│
├── compiled_rules/                  # Reglas compiladas (archivos .py + manifest.json)
│   ├── manifest.json
│   ├── RULE008.py
│   ├── RULE009.py
│   ├── RULE018.py
│   ├── RULE019.py
│   ├── RULE022.py
│   ├── RULE024.py
│   ├── RULE030.py
│   ├── RULE046.py
│   └── RULE052.py
│
└── demo/
    ├── __init__.py
    ├── README.md
    ├── VOICE_INTEGRATION_README.md
    ├── demo_cli.py                  # CLI de demostración del sistema completo
    ├── simple_parser.py             # Parser ATC simple (regex)
    ├── state_loader.py              # Cargador de estados iniciales desde JSON
    ├── audio_recorder.py            # Grabación de audio para entrada por voz
    ├── rule_filter.py               # RuleFilter — 3 capas: keywords, embeddings, LLM batch
    └── config/
        └── initial_state.json       # Estado de tráfico inicial para demo
```

**Paquete externo compartido**: `common/llm_client_factory.py` — contiene
`ModelConfig` y las funciones `create_instructor_client()` y
`create_raw_client()` usadas tanto por `Knowledge_Extractor` como por
`Alert_System`.

---

## 2. Schemas Pydantic (Modelos de Datos)

Los modelos Pydantic son la columna vertebral del sistema. Definen exactamente
qué datos circulan entre componentes y cómo se validan.

### 2.1 `models/traffic_state.py` — Estado del Tráfico

Enums que restringen los valores posibles:

```python
class FlightPhase(str, Enum):
    GROUND, TAKEOFF, CLIMB, CRUISE, DESCENT, APPROACH, LANDING, TAXI

class WakeTurbulenceCategory(str, Enum):
    LIGHT("L"), MEDIUM("M"), HEAVY("H"), SUPER("S")

class OccupantType(str, Enum):
    AIRCRAFT, VEHICLE, UNKNOWN

class RunwayOperationMode(str, Enum):
    LANDING, TAKEOFF, MIXED, CLOSED
```

#### Position

```python
class Position(BaseModel):
    latitude: float              # Grados decimales
    longitude: float             # Grados decimales
    altitude: int                # Pies (ft)
    heading: int                 # Grados (0-360), validado con ge=0, le=360
    speed: int                   # Nudos (knots), validado con ge=0
    vertical_rate: int | None    # ft/min (None si no disponible)
    timestamp: datetime = utcnow()
```

#### Clearances

```python
class Clearances(BaseModel):
    altitude_assigned: int | None     # ft
    heading_assigned: int | None      # grados (0-360)
    runway_assigned: str | None       # "09L", "27R", etc.
    route: str | None
    squawk: str | None
    speed_assigned: int | None        # nudos
```

#### AircraftState

```python
class AircraftState(BaseModel):
    callsign: str                                # "AAL123"
    position: Position
    flight_phase: FlightPhase
    clearances: Clearances = Clearances()
    restrictions: list[str] = []
    wake_turbulence: WakeTurbulenceCategory = MEDIUM
    aircraft_type: str | None                    # "B738"
    last_contact: datetime = utcnow()
    is_emergency: bool = False
    emergency_type: str | None
    position_history: list[Position] = []        # máx 10 (max_length=10)
    phase_history: list[PhaseTransition] = []    # historial de cambios de fase
    previous_phase: FlightPhase | None
    phase_transition_timestamp: datetime | None
    squawk_history: list[SquawkChange] = []
    squawk_assigned_timestamp: datetime | None
```

#### RunwayState

```python
class RunwayState(BaseModel):
    runway_id: str                  # "09L"
    occupied: bool = False
    occupied_by: str | None         # callsign que ocupa
    operation_mode: RunwayOperationMode = MIXED
    holding_short: list[str] = []   # callsigns esperando
    landing_queue: list[str] = []
    closed_until: datetime | None
    closure_reason: str | None
    occupant_type: OccupantType | None  # AIRCRAFT, VEHICLE, UNKNOWN
```

#### TrafficState (Estado Global)

```python
class TrafficState(BaseModel):
    timestamp: datetime = utcnow()
    sector_id: str                             # "SECTOR_A"
    aircrafts: dict[str, AircraftState] = {}    # indexado por callsign
    runways: dict[str, RunwayState] = {}        # indexado por runway_id
    msa: int | None                             # Minimum Sector Altitude (ft)
    sector_boundary: list[tuple] | None         # polígono del sector
    qnh: int | None                             # presión (hPa)
    wind: dict | None                           # {direction, speed}

    # Métodos auxiliares:
    def get_aircraft(callsign) -> AircraftState | None
    def add_aircraft(aircraft) -> None
    def remove_aircraft(callsign) -> None
    def get_runway(runway_id) -> RunwayState | None
    def get_nearby_aircraft(callsign, max_distance_nm=20.0) -> list[AircraftState]
    @staticmethod def calculate_distance(pos1, pos2) -> float  # distancia en NM
```

`calculate_distance` usa una aproximación plana (no haversine):

```python
lat_diff = (pos2.latitude - pos1.latitude) * 60          # NM
lon_diff = (pos2.longitude - pos1.longitude) * 60 * cos(lat1)  # NM
return sqrt(lat_diff**2 + lon_diff**2)
```

Error: hasta ~0.5% en distancias < 30NM a latitudes medias. Aceptable para
detección de conflictos de 5NM.

### 2.2 `models/instruction.py` — Instrucciones ATC

#### InstructionType

```python
class InstructionType(str, Enum):
    # 66 valores, organizados por categoría:
    # Vertical:     DESCENT, CLIMB, MAINTAIN_ALTITUDE, EXPEDITE_DESCENT, EXPEDITE_CLIMB
    # Heading:      HEADING, TURN_LEFT, TURN_RIGHT, PRESENT_HEADING
    # Speed:        SPEED, MAINTAIN_SPEED, REDUCE_SPEED, INCREASE_SPEED, NO_SPEED_RESTRICTION
    # Clearances:   TAKEOFF_CLEARANCE, LANDING_CLEARANCE, APPROACH_CLEARANCE
    # Ground:       TAXI, TAXI_VIA, HOLD_POSITION, HOLD_SHORT, LINE_UP, LINE_UP_AND_WAIT
    # Comms:        CONTACT, MONITOR, SQUAWK, IDENT, CHECK_STROBE
    # Emergency:    PAN_PAN, MAYDAY, EMERGENCY_DESCENT
    # Other:        REPORT, CLEARED_AS_FILED, DIRECT_TO, CLEARED_TO_LAND, GO_AROUND, MISSED_APPROACH
```

#### ParsedInstruction

```python
class ParsedInstruction(BaseModel):
    raw_text: str                            # Texto original del ASR
    normalized_text: str                     # Texto normalizado
    speaker: Speaker                         # ATCO | PILOT
    callsign: str | None                     # "AAL123"
    callsign_confidence: float = 1.0         # [0, 1]
    instruction_type: InstructionType
    action_verb: str                         # "descend", "climb", "turn"
    action_verb_confidence: float = 1.0
    parameters: dict[str, Any] = {}           # {target_altitude: 24000, heading: 90, ...}
    entities: list[str] = []                 # IDs de entidades KEX
    temporal_marker: str | None              # "immediately", "when_ready"
    priority_marker: str | None              # "urgent", "expedite"
    timestamp: datetime = utcnow()
    asr_confidence: float = 1.0
    source_audio: str | None
    is_valid: bool = True
    validation_errors: list[str] = []

    # Métodos de acceso:
    def get_parameter(key, default=None) -> Any
    def has_parameter(key) -> bool
    def requires_immediate_action() -> bool
    def is_clearance() -> bool
    def is_altitude_change() -> bool
    def get_target_altitude() -> int | None    # de parameters o FL
    def get_target_heading() -> int | None
    def get_target_speed() -> int | None
```

### 2.3 `models/alert.py` — Alertas y Violaciones

#### AlertSeverity

```python
class AlertSeverity(str, Enum):
    INFO = "info"          # No requiere acción
    LOW = "low"            # Precaución, monitorear
    MEDIUM = "medium"      # Atención recomendada
    HIGH = "high"          # Acción requerida
    CRITICAL = "critical"  # Acción inmediata
```

#### AlertCategory

```python
class AlertCategory(str, Enum):
    # Altitud:      ALTITUDE_VIOLATION, MSA_VIOLATION, FLIGHT_LEVEL_VIOLATION
    # Separación:   SEPARATION_LOSS, SEPARATION_CONFLICT, LATERAL_SEPARATION, VERTICAL_SEPARATION
    # Pista:        RUNWAY_CONFLICT, RUNWAY_INCURSION, RUNWAY_OCCUPIED
    # Velocidad:    SPEED_VIOLATION, OVERSPEED, UNDERSPEED
    # Fase:         PHASE_VIOLATION, WRONG_PHASE
    # Procedural:   PROCEDURAL_ERROR, CLEARANCE_ERROR, SEQUENCING_ERROR
    # Emergencia:   EMERGENCY_DETECTED, MAYDAY_RECEIVED, PAN_PAN_RECEIVED
    # Comms:        COMMS_LOSS, READBACK_ERROR
    # Sistema:      SYSTEM_ERROR, PARSING_ERROR
```

#### Violation

```python
class Violation(BaseModel):
    violation_id: str = uuid4()[:8]      # "VIO_a1b2c3d4"
    rule_id: str                          # "RULE001"
    condition_type: str                   # "ALTITUDE_MINIMUM", "SEPARATION_VERTICAL"
    severity: AlertSeverity
    details: dict[str, Any] = {}          # parámetros específicos
    explanation: str                      # texto descriptivo
    suggestion: str | None                # acción correctiva
    detected_at: datetime = utcnow()

    def get_detail(key, default=None) -> Any
```

#### Alert

```python
class Alert(BaseModel):
    alert_id: str = uuid4()[:12]          # "ALT_a1b2c3d4e5f6"
    timestamp: datetime = utcnow()
    severity: AlertSeverity               # severidad global
    category: AlertCategory               # inferida de las violaciones
    affected_callsigns: list[str] = []
    primary_callsign: str | None
    triggering_instruction_raw: str       # texto original
    violations: list[Violation] = []
    title: str                            # título corto
    explanation: str                      # explicación detallada
    suggested_action: str                 # acción correctiva
    projected_state: dict | None          # snapshot del estado proyectado
    acknowledged: bool = False
    commit_decision: str = "PENDING"      # PENDING | COMMIT | ROLLBACK
    force_committed: bool = False
    sector_id: str | None

    # Métodos:
    def get_primary_violation() -> Violation | None  # la más severa
    def is_resolved() -> bool
    def is_critical() -> bool
    def requires_immediate_action() -> bool
    def add_violation(violation) -> None             # actualiza severidad global
    def acknowledge(operator_id) -> None
    def set_commit_decision(decision, reason=None) -> None
```

#### AlertResult

```python
class AlertResult(BaseModel):
    instruction: dict[str, Any]    # instrucción evaluada
    status: str                    # "OK" | "WARNING" | "ALERT"
    alert: Alert | None
    violations_count: int = 0
    processing_time_ms: float

    def has_alert() -> bool
    def is_safe() -> bool          # OK o WARNING sin CRITICAL
```

### 2.4 `integration/schemas.py` — Schemas de Integración

#### ExecutableRule

```python
class ExecutableRule(BaseModel):
    """Formato intermedio entre KEX y RuleEngine."""
    source_rule_id: str                        # ID original en KEX
    rule_category: str                         # ALTITUDE | SEPARATION | RUNWAY | GENERIC | UNEVALUABLE
    parameters: dict | None = None             # parámetros estructurados
    condition_description: str | None = None   # descripción textual
    required_state_fields: list[str] = []
    reason_unexecutable: str | None = None
    raw_trigger: str | None = None
    raw_constraint: str | None = None
    severity: str | None = None
    safety_critical: bool = False

    # Campos para clasificación LLM (compilación):
    rule_type: str | None = None               # prohibition, obligation, etc.
    modality: str | None = None                # shall, may, etc.
    raw_formal_if_then: dict | None = None     # representación if-then
    raw_applicability: dict | None = None      # ámbito de aplicación
    explainability: str | None = None          # razón de la regla
```

#### LLMEvaluationResult

```python
class LLMEvaluationResult(BaseModel):
    """Resultado estructurado del LLM para evaluación en runtime."""
    is_violated: bool
    confidence: float                          # [0.0, 1.0]
    explanation: str
    suggested_action: str | None = None
    severity_override: Literal["LOW","MEDIUM","HIGH","CRITICAL"] | None = None
    extracted_values: dict | None = None
```

#### RuleRelevance y RelevanceFilterResult

```python
class RuleRelevance(BaseModel):
    rule_index: int                     # índice 0-based
    is_relevant: bool
    reason: str                         # max_length=100

class RelevanceFilterResult(BaseModel):
    relevances: list[RuleRelevance]
    summary: str                        # max_length=200
    relevant_count: int
```

### 2.5 `compilation/schemas.py` — Schemas de Compilación

#### Enums y Estados

```python
class CompilationStatus(str, Enum):
    COMPILED = "compiled"
    FAILED = "failed"
    PENDING = "pending"
    NOT_COMPILABLE = "not_compilable"   # regla subjetiva
```

#### RuleVerdict (Salida de Clasificación)

```python
class RuleVerdict(BaseModel):
    is_compilable: bool
    reason: str
    required_fields: list[str] = []
    confidence: float = 0.0  # [0, 1]
```

#### ClassificationResponse (Respuesta LLM para clasificación)

```python
class ClassificationResponse(BaseModel):
    is_compilable: bool
    reason: str
    required_fields: list[str] = []
    confidence: float = 0.0

    # Validador: required_fields debe pertenecer a {aircrafts, msa, sector_id,
    #   runway_state, weather, airspace_class, separation_minima, altitude_limits}
```

#### ValidatedCodeResponse (Respuesta LLM para código, con validación total)

```python
class ValidatedCodeResponse(BaseModel):
    code: str                           # código Python de evaluate()
    explanation: str = ""
    required_state_fields: list[str] = []

    # 6 validadores anidados (field_validator 'code'), ejecutados en orden:
    #
    # 1. validate_syntax:       ast.parse(code) — rechaza SyntaxError
    #
    # 2. validate_function_name: debe contener "def evaluate("
    #
    # 3. validate_function_signature: evalúa AST, verifica:
    #       - Función evaluate(traffic_state, callsign=None)
    #       - Primer parámetro es "traffic_state"
    #
    # 4. validate_no_forbidden_imports: escanea AST en busca de:
    #       - {os, subprocess, open, exec, eval, compile, __import__,
    #          globals, locals, socket, http, urllib, requests, sys}
    #       - Solo permite {math, datetime}
    #
    # 5. validate_no_forbidden_names: AST walk, Name nodes:
    #       - {os, subprocess, open, exec, eval, compile, __import__,
    #          globals, locals, memoryview, bytearray, socket}
    #
    # 6. validate_return_structure: AST walk Return + Dict:
    #       - Keys requeridas: {satisfied, details, explanation, severity}
```

#### CompiledRule

```python
class CompiledRule(BaseModel):
    source_rule_id: str
    rule_category: str
    condition_description: str
    compiled_code: str                   # código Python
    required_state_fields: list[str] = []
    compilation_metadata: dict = {}      # modelo, timestamp, intentos
    compilation_status: CompilationStatus = PENDING
    failure_reason: str | None = None
    raw_trigger: str | None = None
    raw_constraint: str | None = None
    severity: str | None = None
    safety_critical: bool = False
```

#### CompilationManifest

```python
class CompilationManifest(BaseModel):
    version: str = "1.0"
    compiled_at: datetime = utcnow()
    model_used: str
    rules: dict[str, CompiledRule] = {}    # indexado por source_rule_id
    total_compiled: int = 0
    total_failed: int = 0
    total_fallback: int = 0
    total_not_compilable: int = 0

    def add_rule(rule: CompiledRule) -> None:
        # Actualiza regla y contadores según compilation_status
```

---

## 3. Core — Proyección y Gestión de Estado

### 3.1 `core/state_projection.py` — StateProjector

Clase `StateProjector`. Implementa la proyección "what-if" del estado.

#### Constantes de simulación cinemática

```python
class StateProjector:
    DESCENT_RATE_NORMAL = 1000      # ft/min
    DESCENT_RATE_EXPEDITE = 2000
    CLIMB_RATE_NORMAL = 1500
    CLIMB_RATE_EXPEDITE = 2500

    SPEED_GROUND = 30    # kts
    SPEED_TAKEOFF = 150
    SPEED_CLIMB = 250
    SPEED_CRUISE = 450
    SPEED_DESCENT = 280
    SPEED_APPROACH = 180
    SPEED_LANDING = 140
```

#### Proyección principal: `create_projection(traffic_state, instruction, projection_minutes=10)`

```
ProjectedState = {
    traffic_state: deepcopy(TS) con llamada _apply_instruction()
    source_instruction: PI
    trajectories: _calculate_trajectory(aircraft, instruction, minutes, initial_altitude)
    projected_separations: _calculate_projected_separations(TS_copy, callsign, trajectory)
    target_aircraft_final: aircraft modificado
    is_valid_projection: True/False
    projection_errors: [str]
    projection_horizon_min: 10
}
```

**Algoritmo**:
1. `deepcopy(traffic_state)` — copia profunda completa.
2. Busca aeronave por `callsign`. Si no existe → proyección inválida.
3. Guarda altitud original.
4. `_apply_instruction(aircraft, instruction)` — modifica posición, clearances, fase.
5. `_calculate_trajectory(...)` — genera waypoints desde altitud original.
6. `_calculate_projected_separations(...)` — evalúa conflictos con vecinos.

#### `_apply_instruction(aircraft, instruction)`

Modifica el `AircraftState` in-place según `instruction_type`:

```python
def _apply_instruction(self, aircraft, instruction):
    # ALTITUD: modifica position.altitude, clearances.altitude_assigned,
    #          actualiza flight_phase vía _update_flight_phase()
    # RUMBO:   modifica position.heading, clearances.heading_assigned
    # VELOCIDAD: modifica position.speed, clearances.speed_assigned
    # PISTA:   modifica clearances.runway_assigned, flight_phase (TAKEOFF/LANDING)
```

#### `_update_flight_phase(aircraft, instruction_type)`

Determina FlightPhase según altitud resultante:

```python
def _update_flight_phase(self, aircraft, instruction_type):
    alt = aircraft.position.altitude
    if alt < 1000:     → GROUND/TAKEOFF
    elif alt < 10000:  → CLIMB/APPROACH
    elif alt < 18000:  → CLIMB/DESCENT/CRUISE
    else:              → CRUISE
```

#### `_calculate_trajectory(aircraft, instruction, minutes, initial_altitude)`

Genera `minutes` waypoints (1 por minuto):

```python
for minute in range(1, minutes + 1):
    distance_nm = current_speed / 60.0                     # NM en 1 minuto
    lat_change = (distance_nm / 60.0) * cos(radians(heading))
    lon_change = (distance_nm / 60.0) * sin(radians(heading)) / cos(radians(lat))
    current_lat += lat_change
    current_lon += lon_change
    current_alt += vertical_rate                            # ft/min
    if current_alt < 0: current_alt = 0
    # Ajusta speed según altitud (SPEED_CRUISE, SPEED_CLIMB, etc.)
    waypoints.append(ProjectedWaypoint(lat, lon, alt, minute*60, speed, vrate))
```

#### `_calculate_projected_separations(traffic_state, target_callsign, trajectory)`

Para cada waypoint de la trayectoria y cada aeronave cercana (< 30NM):

```python
for other in nearby:
    for waypoint in trajectory.waypoints:
        vertical_sep = abs(waypoint.altitude - other.position.altitude)
        waypoint_pos = Position(lat=..., lon=..., alt=..., heading=0, speed=0)
        horizontal_sep = TrafficState.calculate_distance(waypoint_pos, other.position)
        if vertical_sep < 1000 and horizontal_sep < 5:
            conflict_at_time = waypoint.estimated_time  # primer conflicto
```

#### Dataclasses de salida

```python
@dataclass
class ProjectedWaypoint:
    latitude: float; longitude: float; altitude: int
    estimated_time: float    # segundos desde inicio
    speed: int; vertical_rate: int

@dataclass
class ProjectedTrajectory:
    callsign: str
    waypoints: list[ProjectedWaypoint]
    estimated_duration_sec: float
    final_phase: FlightPhase

@dataclass
class ProjectedSeparation:
    aircraft_1: str; aircraft_2: str
    vertical_separation_ft: float
    horizontal_separation_nm: float
    time_to_conflict: float | None
    conflict_predicted: bool

@dataclass
class ProjectedState:
    traffic_state: TrafficState
    source_instruction: ParsedInstruction
    trajectories: dict[str, ProjectedTrajectory]
    projected_separations: dict[str, list[ProjectedSeparation]]
    projection_timestamp: float = 0.0
    projection_horizon_min: int = 10
    target_aircraft_final: AircraftState | None
    is_valid_projection: bool = True
    projection_errors: list[str] = ()

    def get_aircraft(callsign) -> AircraftState | None
    def get_trajectory(callsign) -> ProjectedTrajectory | None
    def has_conflicts() -> bool
    def get_conflicts() -> list[ProjectedSeparation]
```

### 3.2 `core/state_manager.py` — StateManager

Clase `StateManager`. Gestiona el estado real con soporte transaccional.

#### Estado interno

```python
class StateManager:
    _state: TrafficState                              # estado actual (real)
    _state_history: list[TrafficState] = []             # estados anteriores (máx 10)
    _transactions: dict[str, StateTransaction] = {}     # todas las transacciones
    _pending_transaction: StateTransaction | None = None
    _max_history = 10
```

#### StateTransaction

```python
@dataclass
class StateTransaction:
    transaction_id: str                     # "TXN_20260526_120000_a1b2c3d4"
    timestamp: datetime
    projected_state: ProjectedState
    status: str = "PENDING"                 # PENDING | COMMITTED | ROLLBACK
    atco_decision: str | None = None        # "COMMIT" | "ROLLBACK"
    atco_reason: str | None = None
    decision_timestamp: datetime | None
    has_alerts: bool = False
    alert_ids: list[str] = ()
    force_committed: bool = False
```

#### Métodos transaccionales

**`propose_change(projected_state)`** → crea transacción PENDING:

```python
def propose_change(self, projected_state, transaction_id=None):
    txn_id = transaction_id or f"TXN_{now}_{uuid4()[:8]}"
    has_alerts = projected_state.has_conflicts()  # True si conflict_predicted
    txn = StateTransaction(id=txn_id, timestamp=now(), projected=projected_state,
                           status="PENDING", has_alerts=has_alerts)
    self._transactions[txn_id] = txn
    self._pending_transaction = txn
    return txn
```

**`commit(transaction_id, force=False, reason=None)`** → aplica proyección:

```python
def commit(self, transaction_id=None, force=False, reason=None):
    txn = self._get_transaction(transaction_id)
    if not txn: return False
    if txn.has_alerts and not force: return False   # bloqueado por alertas
    self._save_to_history()                          # guarda estado previo
    self._state = txn.projected_state.traffic_state  # aplica proyección
    txn.status = "COMMITTED"
    txn.atco_decision = "COMMIT"
    txn.force_committed = force
    self._pending_transaction = None
    return True
```

**`rollback(transaction_id, reason=None)`** → descarta proyección:

```python
def rollback(self, transaction_id=None, reason=None):
    txn = self._get_transaction(transaction_id)
    if not txn: return False
    txn.status = "ROLLBACK"       # solo marca, no modifica _state
    txn.atco_decision = "ROLLBACK"
    self._pending_transaction = None
    return True
```

**`undo_last_commit()`** → restaura estado anterior del historial:

```python
def undo_last_commit(self):
    if not self._state_history: return False
    self._state = self._state_history.pop()
    return True
```

#### Transaction (Context Manager)

```python
class Transaction:
    def __init__(self, state_manager, projected_state, auto_commit=False): ...
    def __enter__(self) -> StateTransaction:  # propose_change()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:        → rollback()
        elif auto_commit:   → commit()
```

---

## 4. Rule Engine

### 4.1 `rule_engine/engine.py` — RuleEngine

Clase `RuleEngine`. Motor genérico de evaluación de condiciones.

#### Estado interno

```python
class RuleEngine:
    _evaluators: dict[str, Type[ConditionEvaluator]]     # registro de clases
    _evaluator_instances: dict[str, ConditionEvaluator]  # instancias activas
```

Constructor registra 3 evaluadores por defecto: ALTITUDE, SEPARATION, RUNWAY.

#### Métodos principales

```python
def register_evaluator(condition_type: str, evaluator_class) -> None
    # Almacena clase y crea instancia

def evaluate(condition_type, parameters, traffic_state, aircraft_callsign) -> ConditionResult
    # Busca evaluador, valida parámetros, ejecuta

def batch_evaluate(conditions: list[dict], traffic_state, callsign) -> list[ConditionResult]
    # Itera sobre condiciones [{type, parameters}] y llama evaluate()

def evaluate_all_violations(conditions, traffic_state, callsign) -> list[Violation]
    # batch_evaluate + filtrar no satisfechas

def check_rule(rule, traffic_state, callsign) -> dict
    # rule = {conditions, logic (ALL|ANY)}
    # Retorna {rule_id, passed, violations, severity, condition_results}
```

#### ConditionResult

```python
@dataclass
class ConditionResult:
    satisfied: bool                        # True = sin violación
    violation: Violation | None = None     # violación si no satisfecha
    details: dict = None                   # detalles adicionales
```

### 4.2 `rule_engine/conditions.py` — Evaluadores

#### ConditionEvaluator (ABC)

```python
class ConditionEvaluator(ABC):
    condition_type: str = ""

    def __init__(self):        self._rules: list[dict] = []
    def add_rule(self, rule):  self._rules.append(rule)
    def clear_rules(self):     self._rules.clear()

    @abstractmethod
    def evaluate(self, traffic_state, parameters, aircraft_callsign=None) -> ConditionResult

    @abstractmethod
    def evaluate_all(self, traffic_state, aircraft_callsign=None) -> list[Violation]

    def get_required_parameters(self) -> list[str]
    def validate_parameters(self, parameters) -> tuple[bool, list[str]]
```

#### AltitudeCondition

Evalúa condiciones de altitud contra `aircraft.position.altitude`.

**Parámetros**: `{check_type: "MINIMUM"|"MAXIMUM"|"MSA_CHECK", reference_value: int}`

- `MINIMUM`: violación si `altitude < reference_value` (severity HIGH)
- `MAXIMUM`: violación si `altitude > reference_value` (severity MEDIUM)
- `MSA_CHECK`: violación si `altitude < traffic_state.msa` (severity CRITICAL)

Si no hay reglas registradas (`_rules` vacío), `evaluate_all()` usa regla por
defecto `MSA_CHECK`.

#### SeparationCondition

Evalúa separación entre aeronaves. Usa separaciones pre-calculadas de la
proyección o las calcula en vivo.

**Parámetros**: `{separation_type: "VERTICAL"|"HORIZONTAL"|"BOTH", min_distance: float}`

Estándares ICAO embebidos: `VERTICAL_SEPARATION_STD = 1000` ft,
`HORIZONTAL_SEPARATION_STD = 5` NM.

`_check_vertical_separation`: itera `nearby_aircraft`, calcula
`abs(own_alt - other_alt)`, violación si < `min_separation`.

`_check_horizontal_separation`: itera `nearby_aircraft`, calcula
`TrafficState.calculate_distance(pos1, pos2)`, violación si < `min_separation`.

#### RunwayCondition

Evalúa condiciones de pista contra `traffic_state.runways[runway_id]`.

**Parámetros**: `{check_type: "OCCUPIED"|"HOLDING_SHORT_FULL"|"EXISTS", runway_id: str}`

- `OCCUPIED`: violación si `runway.occupied == True` (severity CRITICAL)
- `HOLDING_SHORT_FULL`: violación si `|runway.holding_short| >= max_holding` (severity MEDIUM)
- `EXISTS`: violación si `runway_id not in traffic_state.runways` (severity HIGH)

#### GenericKexCondition

Evaluador genérico para reglas KEX que no encajan en categorías predefinidas.
Almacena un `ExecutableRule` y lo evalúa mediante LLM runtime o keywords.

```python
class GenericKexCondition(ConditionEvaluator):
    condition_type = "GENERIC"
    _executable_rule: ExecutableRule | None = None
    condition_id: str = ""
    llm_config: Any = None
    _instructor_client = None   # lazy initialization
    _raw_client = None
```

**`evaluate(traffic_state, parameters, aircraft_callsign)`**:

1. Si `llm_config` está disponible: `_evaluate_with_llm()`
   - Construye resumen del estado (`_build_traffic_state_summary`)
   - Llama a `build_evaluation_prompt()` para construir system + user prompts
   - Envía al LLM con `response_model=LLMEvaluationResult`
   - Si `is_violated=True` y `confidence > 0.5` → violación con severidad del LLM

2. Fallback: `_evaluate_with_keywords()`
   - Si condición contiene "altitude" o "below": verifica `a.alt < TS.msa`
   - Si no hay match → satisfecho (no violación)

**`_evaluate_with_llm`** — llamada LLM con Instructor:

```python
response = self._instructor_client.chat.completions.create(
    model=llm_config.name,
    response_model=LLMEvaluationResult,
    max_retries=llm_config.max_retries,
    messages=[
        {"role": "system", "content": LLM_EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
)
```

#### CompiledCondition

Envoltura para funciones evaluadoras compiladas por LLM. Ejecuta código Python
con namespace restringido, con fallback a `GenericKexCondition`.

```python
class CompiledCondition(ConditionEvaluator):
    condition_type = "COMPILED"
    _compiled_rule: CompiledRule | None
    _evaluate_fn = None           # lazy: se carga con exec()
    _load_error: str | None
    condition_id: str = ""
    _llm_config: Any = None
    _fallback_condition: GenericKexCondition | None = None
```

**Namespace seguro** (`_get_safe_namespace`):

```python
{
    "math": math,
    "TrafficState": TrafficState, "AircraftState": AircraftState,
    "Position": Position, "FlightPhase": FlightPhase,
    "RunwayState": RunwayState, "Clearances": Clearances,
    "__builtins__": {
        "True", "False", "None", "int", "float", "str", "bool",
        "list", "dict", "tuple", "set", "len", "range", "enumerate",
        "isinstance", "abs", "min", "max", "round", "sorted",
        "any", "all", "zip", "map", "filter",
        "print", "ValueError", "TypeError", "KeyError",
        "AttributeError", "RuntimeError", "Exception",
    },
}
```

**`evaluate(traffic_state, parameters, aircraft_callsign)`**:

1. `_load_function()`: `exec(compiled_code, namespace)` → obtiene `evaluate_fn`
2. Si el código compilado no está disponible o falla → `_get_fallback_condition()`
   que crea un `GenericKexCondition` con el `ExecutableRule` correspondiente
3. Si `evaluate_fn` existe: `result_dict = evaluate_fn(actual_state, callsign=cs)`
4. `_dict_to_condition_result(result_dict, cs)`: convierte el dict a
   `ConditionResult`. Si `satisfied=False`, crea una `Violation` con los
   datos del dict.

---

## 5. Pipeline de Alertas

### 5.1 `pipeline/alert_pipeline.py` — AlertPipeline (856 líneas)

Clase `AlertPipeline`. Implementa el pipeline de 8 pasos.

#### Inicialización

```python
class AlertPipeline:
    def __init__(self, state_manager, rule_engine, llm_config=None,
                 rule_filter=None, filter_timeout=0.0, filter_top_k=30, verbose=False):
        self.state_manager = state_manager
        self.rule_engine = rule_engine
        self.state_projector = StateProjector()
        self.llm_config = llm_config
        self.rule_filter = rule_filter
        self.filter_timeout = filter_timeout
        self.filter_top_k = filter_top_k
        self.verbose = verbose
```

#### PipelineStep y PipelineResult

```python
@dataclass
class PipelineStep:
    step_number: int
    step_name: str
    status: str = "PENDING"       # PENDING | RUNNING | SUCCESS | FAILED | SKIPPED
    input_data: Any = None
    output_data: Any = None
    error_message: str | None = None
    execution_time_ms: float = 0.0

    def mark_success(output), mark_failed(error), mark_skipped(reason)

@dataclass
class PipelineResult:
    pipeline_id: str                    # "PL_20260526_120000_a1b2c3"
    timestamp: datetime
    raw_instruction: str
    parsed_instruction: ParsedInstruction | None
    steps: list[PipelineStep] = ()
    final_decision: str = "PENDING"     # COMMIT | ROLLBACK | PENDING
    atco_override: bool = False
    alerts_generated: list[Alert] = ()
    violations_found: list[Violation] = ()
    committed_state: TrafficState | None = None
    projected_state: ProjectedState | None = None
    total_execution_time_ms: float = 0.0
    has_errors: bool = False
```

#### Paso 1: Input Processing (`_step_1_input_processing`)

Si `pre_parsed` está disponible, lo usa directamente. Si no, llama a
`_simple_atc_parser(raw_instruction)`:

```python
def _simple_atc_parser(self, raw_instruction):
    # Regex para callsign: \b([a-z]{3}\d+)\b
    # Regex para FL: fl\s*(\d+) → parameters["target_altitude"] = match * 100
    # Regex para heading: (\d{3})
    # Regex para runway: runway\s+(\d+[lr]?)
    # Match keywords: "descend" → DESCENT, "climb" → CLIMB, etc.
```

#### Paso 2: Normalization (`_step_2_normalization`)

En la implementación actual es identidad (el `ParsedInstruction` ya está
normalizado). Slot de extensión para futuro.

#### Paso 3: State Projection (`_step_3_state_update`)

```python
current_state = self.state_manager.current_state
projected = self.state_projector.create_projection(current_state, parsed, projection_minutes=10)
```

#### Paso 4: Rule Evaluation (`_step_4_rule_evaluation`)

Evalúa 4 categorías de reglas:

```python
violations = []
violations += self._evaluate_altitude_rules(parsed, projected, callsign)
violations += self._evaluate_separation_rules(parsed, projected, callsign)
violations += self._evaluate_runway_rules(parsed, projected, callsign)
violations += self._evaluate_compiled_rules(parsed, projected, callsign)
```

**`_evaluate_altitude_rules`**: usa `rule_engine._evaluator_instances["ALTITUDE"]`.
Si no existe, fallback manual: verifica `altitude < msa`.

**`_evaluate_separation_rules`**: usa separaciones pre-calculadas de `ProjectedState`.
Si `conflict_predicted` en alguna, llama al evaluador SEPARATION.

**`_evaluate_runway_rules`**: solo si `instruction_type` es TAKEOFF o LANDING.
Usa evaluador RUNWAY o fallback manual.

**`_evaluate_compiled_rules`**: el más complejo. Itera evaluadores registrados
que no son ALTITUDE/SEPARATION/RUNWAY:

```python
# Separa en compiled (COMPILED_*) y generic (GENERIC_*)
for condition_type, evaluator in self.rule_engine._evaluator_instances.items():
    if condition_type.startswith("COMPILED_"):
        compiled_evaluators.append(...)
    elif condition_type.startswith("GENERIC_"):
        generic_evaluators[...] = ...

# Para genéricas: aplica RuleFilter si está disponible
if self.rule_filter:
    filtered_rules = self.rule_filter.filter_rules(
        rules, instruction_text, llm_config,
        timeout_seconds, top_k,
    )
    filtered_ids = {r.source_rule_id for r in filtered_rules}
    # Solo evalúa reglas en filtered_ids
```

#### Paso 5: Alert Generation (`_step_5_alert_generation`)

Convierte cada `Violation` en `Alert`:

```python
for violation in violations:
    category = self._infer_alert_category(violation.condition_type)
    alert = Alert(
        category=category,
        severity=violation.severity,
        affected_callsigns=...,
        primary_callsign=...,
        triggering_instruction_raw=parsed.raw_text,
        violations=[violation],
        title=f"Alert: {violation.condition_type}",
        explanation=violation.explanation,
        suggested_action="Review instruction",
    )
```

#### Paso 6: Alert Presentation (`_step_6_alert_presentation`)

Empaqueta para UI:

```python
presentation = {
    "alert_count": len(alerts),
    "violation_count": len(violations),
    "has_critical": any(a.severity == CRITICAL for a in alerts),
    "alerts": alerts,
}
```

#### Paso 7: ATCO Decision (`_step_7_atco_decision`)

Lógica automática:

```python
if has_critical:  decision = "ROLLBACK"
else:             decision = "COMMIT"
```

#### Paso 8: Final State Update (`_step_8_final_state_update`)

```python
if decision == "COMMIT":
    txn = self.state_manager.propose_change(projected)
    success = self.state_manager.commit(txn.transaction_id)
elif decision == "ROLLBACK":
    # No modificar estado
```

### 5.2 `pipeline/__init__.py`

Exporta `AlertPipeline`, `PipelineResult`, `PipelineStep`.

---

## 6. Compilación de Reglas

### 6.1 `compilation/compiler.py` — RuleCompiler

Clase `RuleCompiler`. Compila reglas KEX a código Python usando LLM.

#### Inicialización

```python
class RuleCompiler:
    def __init__(self, llm_config=None):
        self.llm_config = llm_config
        self._instructor_client = None   # lazy
        self.mode = None
```

#### `classify_rule(...)` → RuleVerdict

Envía al LLM la regla completa con `response_model=ClassificationResponse` y
convierte a `RuleVerdict`. Si no hay LLM, asume compilable con confianza 0.5.

#### `compile_rule(...)` → CompiledRule

Flujo completo:

1. `classify_rule()` → si `NOT_COMPILABLE`, retorna inmediatamente
2. `_generate_code()` → llama LLM con `response_model=ValidatedCodeResponse`
   (la validación Pydantic integrada verifica sintaxis, imports, firma, retorno)
3. `_test_code()` → ejecuta con `TrafficState` sintético
4. Si OK → `CompilationStatus.COMPILED`; si falla → `FAILED`

#### `_generate_code(...)` → código Python

```python
response = self._instructor_client.chat.completions.create(
    model=llm_config.name,
    response_model=ValidatedCodeResponse,
    max_retries=llm_config.max_retries,
    messages=[
        {"role": "system", "content": COMPILATION_SYSTEM_PROMPT},
        {"role": "user", "content": COMPILATION_USER_PROMPT_TEMPLATE.format(...)},
    ],
    temperature=0.1,
)
return response.code
```

#### `_test_code(code)` → (bool, error)

Crea `TrafficState` de prueba con 2 aeronaves (TEST123 a 6000ft, TEST456 a
4500ft) y MSA=5000. Ejecuta `evaluate(test_state, callsign="TEST123")` y
verifica:

1. No hay excepción
2. Resultado es `dict`
3. Keys requeridas presentes
4. Tipos correctos
5. `severity` válido
6. También funciona con `callsign=None`

#### `compile_batch(executable_rules, save_incrementally, output_dir)` → CompilationManifest

Itera sobre reglas, compila cada una, guarda `.py` + actualiza `manifest.json`
incrementalmente si `save_incrementally=True`.

### 6.2 `compilation/prompts.py` — Prompts

#### TRAFFIC_STATE_SCHEMA

Definición Pydantic/Python de todas las clases del modelo de datos, usada
en los prompts para que el LLM conozca la estructura exacta de `TrafficState`.

#### COMPILATION_SYSTEM_PROMPT

Instrucciones detalladas para el LLM compilador:

- "You are an expert ATC rule compiler"
- Formato de respuesta: JSON con `code`, `explanation`, `required_state_fields`
- Reglas de código: firma `def evaluate(traffic_state, callsign=None)`,
  retorno `{satisfied, details, explanation, severity}`, solo imports `math`/`datetime`
- Instrucciones de seguridad: sin `os`, `subprocess`, `exec`, `eval`
- Manejo de entity references (E002, E015): evaluar el SIGNIFICADO, no el ID
- Prohibición de hardcoding: nunca hardcodear callsigns o runway IDs
- Self-validation checklist al final

#### COMPILATION_USER_PROMPT_TEMPLATE

Template con la regla a compilar + schema TrafficState + 2 ejemplos:
- MSA altitude check
- Separation check

#### CLASSIFICATION_SYSTEM_PROMPT

Instrucciones para clasificar si una regla es compilable con TrafficState o
requiere juicio subjetivo. Incluye el schema completo de TrafficState.

#### CLASSIFICATION_USER_PROMPT_TEMPLATE

Template con la regla a clasificar.

### 6.3 `compilation/validator.py` — Validación Estática

#### `validate_code(code)` → (bool, issues)

Validación AST del código generado:

1. `ast.parse(code)` — verifica sintaxis
2. Busca `FunctionDef` con nombre `evaluate`
3. Verifica que no haya más de 3 funciones helper
4. Verifica signature: primer argumento `traffic_state`
5. Verifica imports: solo permite `math` y `datetime`
6. Verifica nombres prohibidos: `os`, `subprocess`, `open`, `exec`, `eval`, etc.
7. Verifica atributos prohibidos: `__import__`, `__builtins__`, etc.
8. Verifica llamadas a `exec`, `eval`, `compile`, `__import__`

#### `validate_return_structure(code)` → (bool, issues)

Busca todos los `Return` dentro de la función `evaluate` y verifica que los
dicts literales contengan las 4 keys requeridas.

### 6.4 `compilation/loader.py` — CompiledRuleLoader

Clase `CompiledRuleLoader`. Carga y guarda reglas compiladas.

#### Directorio

```python
DEFAULT_COMPILED_RULES_DIR = "Alert_System/compiled_rules/"
```

#### Métodos principales

```python
def load_manifest() -> CompilationManifest | None
    # Carga compiled_rules/manifest.json

def load_compiled_rule(rule_id) -> CompiledRule | None
    # Busca manifest primero, luego archivo .py

def load_all_compiled_rules() -> dict[str, CompiledRule]
    # Carga todas las reglas del directorio

def create_compiled_conditions() -> list[CompiledCondition]
    # Crea instancias de CompiledCondition para cada regla compilada

def register_in_engine(rule_engine) -> int
    # Registra cada CompiledCondition como "COMPILED_{rule_id}" en RuleEngine

def has_compiled_rule(rule_id) -> bool
def get_compiled_rule(rule_id) -> CompiledRule | None

def save_manifest(manifest) -> bool
def save_compiled_rule(compiled_rule) -> bool  # Guarda .py con header de metadata
def save_all(manifest) -> int
```

### 6.5 `compilation/schemas.py` — (ya descrito en sección 2.5)

---

## 7. Integración con KEX

### 7.1 `integration/kex_adapter.py` — KEXAdapter

Clase `KEXAdapter`. Convierte reglas del `Knowledge_Extractor` en
`ConditionEvaluator` del `Alert_System`.

#### Flujo de adaptación

```
Rule (KEX) → ExecutableRule (intermedio) → ConditionEvaluator
```

#### `adapt_rules(rules: list[Rule]) → list[ConditionEvaluator]`

Por cada regla KEX:

1. `compile_to_executable(rule)` → `ExecutableRule`
2. `_adapt_executable_rule(executable)` → `ConditionEvaluator | None`

#### `compile_to_executable(rule) → ExecutableRule`

1. `_categorize_rule(rule)`: clasifica usando patrones del JSON o fallback
   por keywords → `"ALTITUDE"|"SEPARATION"|"RUNWAY"|"GENERIC"|"UNEVALUABLE"`
2. Construye `ExecutableRule` con:
   - `source_rule_id`, `rule_category`
   - Textos: `raw_trigger`, `raw_constraint`
   - Metadatos: `severity`, `safety_critical`, `rule_type`, `modality`
   - Si es categoría conocida: `parameters` extraídos con regex
   - Si es GENERIC: `condition_description` para LLM runtime
   - Si es UNEVALUABLE: `reason_unexecutable`

#### `_categorize_rule(rule) → str`

1. **Patrones del JSON** (`config/rule_patterns.json`): busca keywords en el
   texto completo de la regla. Si encuentra match, retorna la categoría del patrón.
2. **Fallback por defecto**: secuencia de detección por palabras clave:
   - Altitud: "altitude", "msa", "flight level", "climb", "descend"
   - Separación: "separation", "distance", "conflict", "loss of separation"
   - Pista: "runway", "rwy", "occupied", "landing", "takeoff"
   - No evaluable: "pilot", "crew", "fatigue", "weather", "judgment", "discretion"
   - Por defecto: "GENERIC"

#### `_adapt_executable_rule(executable) → ConditionEvaluator`

Árbol de decisión:

```
category?
├─ UNEVALUABLE → return None (no se puede evaluar)
├─ GENERIC:
│   ├─ ¿Tiene versión compilada? loader.has_compiled_rule(id)
│   │   ├─ Sí → CompiledCondition(compiled_rule, llm_config)  [prioridad 1]
│   │   └─ No → GenericKexCondition(llm_config)               [prioridad 3]
│   └─ Almacena ExecutableRule en condition._executable_rule
├─ ALTITUDE → AltitudeCondition() con parámetros
├─ SEPARATION → SeparationCondition() con parámetros
└─ RUNWAY → RunwayCondition() con parámetros
```

### 7.2 `integration/schemas.py` — (ya descrito en sección 2.4)

### 7.3 `integration/asr_adapter.py` — ASRAdapter

Adaptador para sistemas de reconocimiento de voz. Convierte salida del ASR
en `ParsedInstruction`. Integra `bert_atc_parser` + `atc_compact_normalizer`.

### 7.4 `integration/bert_atc_parser.py` — BERT ATC Parser

Parser de instrucciones ATC usando BERT NER + regex. Extrae callsign,
intención (heading, altitude, etc.) y parámetros numéricos del texto ASR.

### 7.5 `integration/atc_compact_normalizer.py` — Normalizador

Normaliza texto ACR: expande abreviaturas, normaliza unidades, corrige
errores comunes del ASR.

### 7.6 `integration/end_to_end_pipeline.py`

Integración completa: `KEX → AlertSystem`. Probablemente orquesta la carga
de reglas desde el Knowledge Extractor, la compilación, y la configuración
del pipeline de alertas.

---

## 8. Configuración

### 8.1 `config/rule_patterns.json`

Patrones para categorizar reglas KEX:

```json
{
  "patterns": [
    {
      "name": "altitude_msa",
      "keywords": ["altitude", "msa", "minimum sector altitude", "flight level"],
      "category": "ALTITUDE",
      "required_state_fields": ["aircrafts", "msa"]
    },
    {
      "name": "separation",
      "keywords": ["separation", "distance", "conflict"],
      "category": "SEPARATION",
      "required_state_fields": ["aircrafts"]
    },
    {
      "name": "runway",
      "keywords": ["runway", "rwy", "occupied", "landing", "takeoff"],
      "category": "RUNWAY",
      "required_state_fields": ["runway_state"]
    }
  ]
}
```

### 8.2 `config/evaluation_prompts.py` — Prompts para LLM Runtime

#### LLM_EVALUATION_SYSTEM_PROMPT

Prompt de sistema para el evaluador LLM en tiempo real. Define:

- Rol: "expert Air Traffic Control (ATC) safety evaluator"
- Criterios de evaluación: analizar regla, examinar estado, determinar violación
- Formato de respuesta: JSON con `is_violated`, `confidence`, `explanation`, etc.
- Comportamiento conservador: si hay incertidumbre, marcar como violación potencial

#### LLM_EVALUATION_USER_PROMPT_TEMPLATE

Template con la regla a evaluar + resumen del estado de tráfico.

#### `build_evaluation_prompt(...)` → (system_prompt, user_prompt)

Función helper que construye ambos prompts a partir de los datos del estado
y la regla.

---

## 9. Demo y Utilidades

### 9.1 `demo/rule_filter.py` — RuleFilter

Sistema de prefiltrado de reglas genéricas con 3 capas. Ver implementación
detallada en modelo conceptual (Proceso 3.6).

#### Clase RuleFilter

```python
class RuleFilter:
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self._embedding_model = None          # SentenceTransformer (lazy)
        self._rule_embeddings: dict[str, np.ndarray] = {}
        self._rules_hash: str = ""            # para invalidar cache
```

#### FilterConfig

```python
@dataclass
class FilterConfig:
    use_keywords: bool = True
    use_embeddings: bool = True
    use_llm_batch: bool = True
    top_k: int = 30
    timeout_seconds: float | None = None
    embedding_cache_dir: str = ""    # "demo/cache/" por defecto
    verbose: bool = False
```

#### ATC_KEYWORDS

Diccionario de 18 categorías semánticas con palabras clave ATC:

```python
ATC_KEYWORDS = {
    "climb": {"climb", "ascend", "altitude", "flight level", "fl"},
    "descend": {"descend", "descent", "lower", "altitude", "flight level", "fl"},
    "altitude": {"altitude", "flight level", "fl", "msa", "below"},
    "speed": {"speed", "knots", "slow", "reduce speed", "mach"},
    "heading": {"heading", "turn", "left", "right", "course", "vector"},
    "runway": {"runway", "rwy", "taxi", "landing", "takeoff", "hold short"},
    "separation": {"separation", "distance", "nm", "conflict"},
    "emergency": {"emergency", "mayday", "pan", "squawk", "7700"},
    "clearance": {"clearance", "cleared", "permission", "authorize"},
    "weather": {"weather", "wind", "visibility", "ceiling", "turbulence"},
    "communication": {"readback", "read back", "confirm", "acknowledge"},
    "wake": {"wake", "turbulence", "heavy", "super"},
    "holding": {"hold", "holding", "pattern", "orbit", "delay"},
    "route": {"route", "direct", "waypoint", "fix", "intersection"},
    "phrasing": {"plain language", "phraseology", "standard"},
    "frequency": {"frequency", "freq", "radio", "contact", "monitor"},
    "lights": {"lights", "beacon", "strobe", "landing lights"},
}
```

#### `filter_rules(rules, instruction_text, llm_config, timeout_seconds, top_k)` → list[ExecutableRule]

Flujo completo de filtrado:

```python
candidates = rules
keywords = self._extract_keywords(instruction_text)

# Capa 1a: Keywords
candidates = self._keyword_filter(candidates, keywords)

# Capa 1b: Embeddings (re-rank)
self.load_or_compute_embeddings(rules)  # cache
candidates = self._embedding_rank(candidates, instruction_text, top_k)

# Capa 2: LLM Batch Relevance
candidates = self._llm_batch_filter(candidates, instruction_text, llm_config)

return candidates
```

**`_keyword_filter`**: extrae keywords de la instrucción, selecciona reglas
con al menos una keyword coincidente.

**`_embedding_rank`**: calcula similitud coseno entre embedding de instrucción
y cada regla, ordena descendente, toma top-k.

**`_llm_batch_filter`**: envía todas las reglas candidatas numeradas al LLM
con la instrucción, recibe `RelevanceFilterResult` con índices relevantes.
Verifica disponibilidad de Ollama antes de llamar.

#### **`load_or_compute_embeddings(rules)` → dict[str, ndarray]**

Cachea embeddings en `demo/cache/rule_embeddings.pkl`. Invalida por hash MD5
del contenido de las reglas.

### 9.2 `demo/simple_parser.py`

Parser ATC simple (simplificado del `_simple_atc_parser` del pipeline).

### 9.3 `demo/state_loader.py`

Carga estados de tráfico iniciales desde `demo/config/initial_state.json`.

### 9.4 `demo/demo_cli.py`

CLI interactivo de demostración del sistema completo.

### 9.5 `demo/audio_recorder.py`

Grabación de audio para entrada por voz en demo.

---

## 10. CLI

### 10.1 `compile_rules_cli.py` — CLI de Compilación

Punto de entrada `python -m Alert_System.compile_rules_cli`.

Usa `compilation/cli.py` para la lógica interna. Argumentos:

| Flag | Default | Descripción |
|------|---------|-------------|
| `--model` | `llama3.2` | Modelo LLM para compilar |
| `--base-url` | `http://localhost:11434/v1` | Endpoint API |
| `--provider` | `openai` | Proveedor LLM |
| `--rules-file` | (obligatorio) | JSON con reglas ExecutableRule |
| `--output-dir` | `compiled_rules/` | Directorio de salida |
| `--max-retries` | `3` | Reintentos Instructor |

---

## 11. Flujo de Datos Concreto

### Línea Offline: Compilación de Reglas

```
1. KEX produce List[Rule] (entidades, reglas, relaciones, etc.)
2. KEXAdapter.adapt_rules(rules):
   ├─ Por cada Rule:
   │   ├─ compile_to_executable() → ExecutableRule
   │   │   ├─ _categorize_rule() → ALTITUDE|SEPARATION|RUNWAY|GENERIC|UNEVALUABLE
   │   │   ├─ Extrae parámetros con regex (altitudes, distancias, rwy IDs)
   │   │   └─ Construye ExecutableRule con metadata y descripciones
   │   │
   │   └─ _adapt_executable_rule():
   │       ├─ ¿ALTITUDE? → AltitudeCondition(params)
   │       ├─ ¿SEPARATION? → SeparationCondition(params)
   │       ├─ ¿RUNWAY? → RunwayCondition(params)
   │       └─ ¿GENERIC?
   │           ├─ ¿CompiledRule en disco? → CompiledCondition(compiled_rule, llm_config)
   │           └─ ¿Sin compilar? → GenericKexCondition(llm_config) + ExecutableRule
   │
   └─ Retorna list[ConditionEvaluator] → registrados en RuleEngine


2b. (Alternativa) RuleCompiler.compile_batch(executable_rules):
    ├─ Por cada ExecutableRule:
    │   ├─ classify_rule() → RuleVerdict
    │   │   └─ ¿NOT_COMPILABLE? → marcar, continuar
    │   │
    │   ├─ _generate_code() → ValidatedCodeResponse.code
    │   │   ├─ COMPILATION_SYSTEM_PROMPT + user_prompt → LLM
    │   │   └─ Pydantic valida: sintaxis, imports, firma, retorno
    │   │
    │   ├─ _test_code() → ejecuta con TrafficState sintético
    │   │
    │   └─ CompiledRule(COMPILED|FAILED)
    │
    ├─ ¿save_incrementally?
    │   └─ CompiledRuleLoader.save_compiled_rule() → .py
    │       CompiledRuleLoader.save_manifest() → manifest.json
    │
    └─ Retorna CompilationManifest
```

### Línea Online: Pipeline de 8 Pasos

```
1. Instrucción entrante: "descend AAL123 to FL240"
2. AlertPipeline.process_instruction(raw_text):

   [Paso 1] _step_1_input_processing()
   ├─ _simple_atc_parser():
   │   ├─ regex \b([A-Z]{3}\d+)\b → callsign = "AAL123"
   │   ├─ keyword "descend" → instruction_type = DESCENT
   │   ├─ regex fl\s*(\d+) → parameters = {target_altitude: 24000}
   │   └─ → ParsedInstruction(raw, normalized, callsign, DESCENT, params)
   └─ step.mark_success(parsed)

   [Paso 2] _step_2_normalization()
   └─ step.mark_success(parsed)  [identidad]

   [Paso 3] _step_3_state_projection()
   ├─ state_manager.current_state → TrafficState real
   ├─ StateProjector.create_projection(state, instruction, 10min):
   │   ├─ deepcopy(traffic_state)
   │   ├─ _apply_instruction(AAL123, instruction):
   │   │   ├─ position.altitude = 24000
   │   │   ├─ clearances.altitude_assigned = 24000
   │   │   └─ flight_phase = CRUISE (alt > 18000)
   │   ├─ _calculate_trajectory(AAL123, ...):
   │   │   ├─ Desde altitud original, vrate = -1000 ft/min
   │   │   └─ 10 waypoints (1/min) con posición, altitud, velocidad
   │   ├─ _calculate_projected_separations(AAL123, trajectory):
   │   │   ├─ get_nearby_aircraft(AAL123, 30NM) → [UAL456, DAL789]
   │   │   ├─ Para cada waypoint y cada vecino:
   │   │   │   └─ ¿d_vert < 1000ft ∧ d_horiz < 5NM? → conflict_predicted
   │   │   └─ → [ProjectedSeparation(AAL123, UAL456, ...), ...]
   │   └─ → ProjectedState(traffic_state_copy, trajectories, separations)
   └─ step.mark_success(projected)

   [Paso 4] _step_4_rule_evaluation()
   ├─ _evaluate_altitude_rules():
   │   ├─ rule_engine._evaluator_instances["ALTITUDE"].evaluate_all(projected, "AAL123")
   │   │   └─ MSA_CHECK: 24000 >= MSA(5000) → OK
   │   └─ → []
   │
   ├─ _evaluate_separation_rules():
   │   ├─ projected.projected_separations["AAL123"] → [conflict?]
   │   └─ → separations con conflict_predicted → Violation(SEPARATION_VERTICAL, CRITICAL)
   │
   ├─ _evaluate_runway_rules(): (no es TAKEOFF/LANDING) → []
   │
   └─ _evaluate_compiled_rules():
       ├─ compiled_evaluators = [COMPILED_RULE008, COMPILED_RULE009, ...]
       ├─ generic_evaluators = [GENERIC_RULE022, GENERIC_RULE024, ...]
       │
       ├─ Evaluar COMPILED_RULE008.evaluate(projected, {}, "AAL123"):
       │   ├─ exec(compiled_code, safe_namespace)
       │   ├─ evaluate_fn(projected.traffic_state, callsign="AAL123")
       │   └─ → ConditionResult(satisfied=True, ...)
       │
       ├─ ¿RuleFilter disponible?
       │   ├─ keyword_filter: "descend" → keywords {descend, altitude}
       │   ├─ embedding_rank: top-30 reglas por similitud coseno
       │   └─ llm_batch_filter: LLM determina relevancia
       │
       └─ Evaluar GENERIC_RULE022.evaluate(projected, {}, "AAL123"):
           ├─ _evaluate_with_llm() → LLM evaluation
           │   └─ → ConditionResult(satisfied=False, violation=Violation(...))
           └─ → [Violation, Violation, ...]

   └─ step.mark_success([Violation, ...])

   [Paso 5] _step_5_alert_generation()
   ├─ Por cada Violation:
   │   ├─ _infer_alert_category("SEPARATION_VERTICAL") → SEPARATION_LOSS
   │   └─ Alert(category=SEPARATION_LOSS, severity=CRITICAL, violations=[...], ...)
   └─ step.mark_success([Alert, ...])

   [Paso 6] _step_6_alert_presentation()
   └─ {alert_count: 1, violation_count: 2, has_critical: true, alerts: [...]}

   [Paso 7] _step_7_atco_decision()
   ├─ has_critical = true → decision = "ROLLBACK"
   └─ step.mark_success("ROLLBACK")

   [Paso 8] _step_8_final_state_update()
   ├─ decision = "ROLLBACK" → no modificar estado
   └─ step.mark_success(None)

3. → PipelineResult(final_decision="ROLLBACK", alerts=[Alert(CRITICAL)], ...)
```

---

## 12. Formatos de Archivo

### `compiled_rules/{RULE_ID}.py` — Regla Compilada

```python
"""Regla compilada: Aircraft must maintain altitude above 5000ft MSL"""
# Rule ID: RULE008
# Category: ALTITUDE
# Compiled with: llama3.2
# Compiled at: 2026-05-25T14:30:00

def evaluate(traffic_state, callsign=None):
    aircraft = traffic_state.get_aircraft(callsign) if callsign else None
    if callsign and not aircraft:
        return {"satisfied": True, "details": {"error": "not found"},
                "explanation": "Aircraft not found", "severity": "INFO"}

    msa = traffic_state.msa or 5000
    aircrafts_to_check = [aircraft] if aircraft else list(traffic_state.aircrafts.values())

    for ac in aircrafts_to_check:
        alt = ac.position.altitude
        if alt < msa:
            return {
                "satisfied": False,
                "details": {"callsign": ac.callsign, "altitude": alt, "msa": msa},
                "explanation": f"{ac.callsign} at {alt}ft below MSA {msa}ft",
                "severity": "HIGH"
            }
    return {"satisfied": True, "details": {}, "explanation": "All above MSA", "severity": "INFO"}
```

### `compiled_rules/manifest.json`

```json
{
  "version": "1.0",
  "compiled_at": "2026-05-25T14:35:00",
  "model_used": "llama3.2",
  "rules": {
    "RULE008": {
      "source_rule_id": "RULE008",
      "rule_category": "ALTITUDE",
      "condition_description": "Aircraft must maintain altitude above 5000ft MSL",
      "compiled_code": "def evaluate(traffic_state, callsign=None):\n    ...",
      "required_state_fields": ["aircrafts", "msa"],
      "compilation_metadata": {
        "model": "llama3.2",
        "timestamp": "2026-05-25T14:30:00",
        "attempts": 1
      },
      "compilation_status": "compiled",
      "raw_trigger": "When an aircraft is below 5000ft",
      "raw_constraint": "maintain altitude above 5000ft",
      "severity": "HIGH",
      "safety_critical": true
    }
  },
  "total_compiled": 9,
  "total_failed": 2,
  "total_fallback": 3,
  "total_not_compilable": 1
}
```

---

## 13. Dependencias Externas

| Librería | Uso en Alert_System |
|----------|---------------------|
| `pydantic` | Schemas de datos con validación automática (todos los modelos) |
| `instructor` | Structured outputs del LLM (compilación, evaluación runtime, clasificación) |
| `openai` | Cliente HTTP para LLMs (Ollama, OpenAI, OpenRouter) |
| `sentence-transformers` | Embeddings para RuleFilter (all-MiniLM-L6-v2) |
| `numpy` | Similitud coseno vectorizada en RuleFilter |
| `nltk` | (declarada, no usada directamente en Alert_System) |
| `common` (local) | `llm_client_factory.py` — creación de clientes LLM |
| `Knowledge_Extractor` (local) | `Rule`, `Entity`, `Event` — tipos importados por KEXAdapter |

---

## 14. Observaciones sobre la Implementación

1. **Parser ATC dual**: el pipeline usa un parser regex simple
   (`_simple_atc_parser`) para demo/pruebas, mientras que el parser real vive
   en `integration/` (BERT NER + regex + normalizador). En producción, el
   pipeline debe recibir `pre_parsed` desde el `ASRAdapter`.

2. **El `exec()` del `CompiledCondition` es el punto más riesgoso**: aunque
   el namespace está restringido y el código pasa validación AST+Pydantic,
   `exec()` sigue siendo inherentemente peligroso. Las defensas son:
   - Validación de imports prohibidos (solo `math`, `datetime`)
   - Validación de nombres prohibidos (no `os`, `subprocess`, etc.)
   - Namespace con builtins limitados (sin `open`, `exec`, `eval`, `__import__`)
   - Prueba sintética antes de aceptar la regla
   - Fallback a LLM runtime si la ejecución falla

3. **`RuleFilter` asíncrono incompleto**: las 3 capas se ejecutan de forma
   secuencial y bloqueante. No hay soporte para `asyncio` o `ThreadPoolExecutor`.
   La capa LLM batch es especialmente lenta porque espera la respuesta completa
   del LLM.

4. **Caché de embeddings frágil**: el caché se invalida por hash MD5 del texto
   completo de las reglas. Si se agrega o modifica una regla, se recalculan
   todos los embeddings. Para conjuntos grandes (>1000 reglas), esto puede
   ser costoso.

5. **Proyección de vecinos estáticos**: `_calculate_projected_separations`
   asume que las aeronaves vecinas permanecen en su posición actual durante
   todo el horizonte de proyección. No proyecta sus trayectorias, lo que puede
   subestimar o sobreestimar conflictos.

6. **`PipelineState` no implementado**: a diferencia del Knowledge Extractor
   que tiene un estado de pipeline explícito, el Alert Pipeline usa variables
   locales y el `PipelineResult` como única salida. No hay un objeto de estado
   compartido entre invocaciones del pipeline.

7. **Lógica de decisión simplificada**: el Paso 7 (ATCO Decision) usa una
   regla fija "CRITICAL → ROLLBACK, sino → COMMIT". En producción, esto
   debería ser interactivo (el ATCO ve las alertas y decide) o al menos
   parametrizable (ej: "HIGH → ROLLBACK" o "solo CRITICAL → ROLLBACK").

8. **`GenericKexCondition` como singleton de regla**: cada instancia de
   `GenericKexCondition` almacena exactamente un `ExecutableRule` en
   `_executable_rule`. No soporta múltiples reglas por instancia. Esto
   significa que cada regla genérica requiere su propia instancia del
   evaluador, lo que aumenta el uso de memoria.

9. **El `KEXAdapter` mezcla dos responsabilidades**: (a) categorizar reglas
   KEX y (b) crear evaluadores. La categorización usa `rule_patterns.json`
   con fallback por defecto, pero la lógica de fallback está hardcodeada en
   `_categorize_rule()`, lo que dificulta extender las categorías sin
   modificar el código.

10. **Sin paralelismo en evaluación batch**: `RuleEngine.batch_evaluate()`
    itera secuencialmente sobre las condiciones. No hay paralelización con
    `ThreadPoolExecutor` (útil para reglas LLM, que son I/O-bound) ni con
    `ProcessPoolExecutor` (útil para reglas compiladas, que son CPU-bound).

11. **Respuesta a fallos en cadena**: si el `CompiledCondition` falla y el
    fallback `GenericKexCondition` también falla, no hay un segundo fallback
    ni una excepción que propague el error completo. El pipeline continúa con
    una violación "fantasma" o un resultado incompleto.

12. **Acoplamiento al RuleEngine**: el pipeline accede directamente a
    `rule_engine._evaluator_instances`, violando el encapsulamiento del
    RuleEngine. Las evaluaciones específicas (altitud, separación, pista,
    compiladas) deberían estar abstraídas detrás de métodos públicos del
    RuleEngine en lugar de acceder a su diccionario interno.
