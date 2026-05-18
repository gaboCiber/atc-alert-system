"""Prompts para compilación de reglas KEX a código Python evaluador."""

TRAFFIC_STATE_SCHEMA = """
class FlightPhase(str, Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI = "taxi"

class OccupantType(str, Enum):
    AIRCRAFT = "aircraft"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

class PhaseTransition(BaseModel):
    from_phase: FlightPhase
    to_phase: FlightPhase
    timestamp: datetime
    reason: str|None

class SquawkChange(BaseModel):
    from_squawk: str|None
    to_squawk: str
    timestamp: datetime
    changed_by: str|None  # "ATC" o "PILOT"

class Position(BaseModel):
    latitude: float       # Latitud en grados
    longitude: float      # Longitud en grados
    altitude: int         # Altitud en pies (ft)
    heading: int          # Rumbo en grados (0-360)
    speed: int            # Velocidad en nudos (knots)
    vertical_rate: int|None  # Tasa ascenso/descenso en ft/min

class Clearances(BaseModel):
    altitude_assigned: int|None
    heading_assigned: int|None
    runway_assigned: str|None
    route: str|None
    squawk: str|None
    speed_assigned: int|None

class AircraftState(BaseModel):
    callsign: str
    position: Position
    flight_phase: FlightPhase
    clearances: Clearances
    restrictions: list[str]
    wake_turbulence: str  # L, M, H, S
    aircraft_type: str|None
    is_emergency: bool
    emergency_type: str|None
    # Nuevos campos para Fase 1:
    phase_history: list[PhaseTransition]  # Historial de transiciones de fase
    previous_phase: FlightPhase|None      # Fase anterior
    phase_transition_timestamp: datetime|None
    squawk_history: list[SquawkChange]    # Historial de cambios de squawk
    squawk_assigned_timestamp: datetime|None

class RunwayState(BaseModel):
    runway_id: str
    occupied: bool
    occupied_by: str|None
    operation_mode: str  # landing, takeoff, mixed, closed
    holding_short: list[str]
    landing_queue: list[str]
    closed_until: str|None
    closure_reason: str|None
    # Nuevo campo para Fase 1:
    occupant_type: OccupantType|None  # AIRCRAFT, VEHICLE, UNKNOWN

class TrafficState(BaseModel):
    timestamp: datetime
    sector_id: str
    aircrafts: dict[str, AircraftState]  # indexadas por callsign
    runways: dict[str, RunwayState]      # indexadas por runway_id
    msa: int|None                        # Minimum Sector Altitude en ft
    qnh: int|None                        # Presión QNH en hPa
    wind: dict|None                      # Dirección y velocidad del viento

    # Métodos disponibles:
    # get_aircraft(callsign) -> AircraftState|None
    # get_runway(runway_id) -> RunwayState|None
    # get_nearby_aircraft(callsign, max_distance_nm=20.0) -> list[AircraftState]
    # calculate_distance(pos1, pos2) -> float  # distancia en NM
"""

COMPILATION_SYSTEM_PROMPT = """You are an expert ATC (Air Traffic Control) rule compiler. Your task is to convert natural language ATC rules into Python evaluation functions.

You must respond with a structured JSON object containing:
- "code": The complete Python function (string)
- "explanation": Brief explanation of what code does (string, optional)
- "required_state_fields": List of TrafficState fields used (list of strings, optional)

CRITICAL VALIDATION REQUIREMENTS:
Your code will be automatically validated before acceptance. Ensure:
1. VALID SYNTAX: Code must parse without syntax errors
2. CORRECT FUNCTION: Must contain `def evaluate(traffic_state, callsign=None):`
3. NO FORBIDDEN IMPORTS: Only allowed imports are `math` and `datetime`
4. NO FORBIDDEN OPERATIONS: Cannot use os, subprocess, open, exec, eval, etc.
5. CORRECT RETURN STRUCTURE: Must return dict with keys: "satisfied", "details", "explanation", "severity"

CODE GENERATION RULES:
1. Generate ONLY the `evaluate` function — no imports, no classes, no code outside function.
2. The function signature MUST be exactly: `def evaluate(traffic_state, callsign=None):`
3. The function MUST return a dict with these exact keys:
   - "satisfied" (bool): True if rule is satisfied (no violation), False if violated
   - "details" (dict): Relevant values extracted from traffic_state
   - "explanation" (str): Human-readable explanation of result
   - "severity" (str): One of "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"
4. You may use: `math` module (already imported in namespace)
5. You may NOT use: os, subprocess, open, exec, eval, import, __import__, globals, locals, or any file/network/system operations.
6. Handle edge cases: aircraft not found, None values, missing fields.
7. If callsign is None, evaluate the rule for ALL aircraft in traffic_state (return worst result).
8. Use `traffic_state.get_aircraft(callsign)` to get a specific aircraft.
9. Use `traffic_state.aircrafts` dict to iterate all aircraft.
10. Use `traffic_state.get_nearby_aircraft(callsign)` for separation checks.
11. Use `TrafficState.calculate_distance(pos1, pos2)` for distance in NM.
12. **CRITICAL**: Entity references (like E002, E015) represent CONCEPTS, not technical codes:
    - E002 = "Standard Phraseology" → Check if communications follow proper patterns
    - E015 = "RTF" → Check if radiotelephony is used appropriately
    - NEVER match entity IDs directly (e.g., squawk == "E002")
    - Generate code that evaluates the MEANING of these concepts
13. **CRITICAL - NO HARDCODING**:
    - NEVER hardcode specific callsigns (e.g., "Big Jet 345", "AAL123", "UAL456")
    - NEVER hardcode specific runway IDs (e.g., "09L", "18", "27R")
    - Use the `callsign` parameter or iterate over `traffic_state.aircrafts.values()`
    - If the rule explicitly requires a specific entity (e.g., "Big Jet 345 must maintain squawk 3456"), use `.upper()` for normalization
    - Example CORRECT: `target = traffic_state.get_aircraft(callsign.upper()) if callsign else None`
    - Example WRONG: `if aircraft.callsign == "Big Jet 345":` (without .upper())
14. Keep the function concise but correct. Prioritize correctness over brevity.
15. The function MUST contain `def evaluate(` exactly as specified.

SELF-VALIDATION CHECKLIST before responding:
□ No syntax errors
□ Function signature is correct
□ No forbidden imports
□ No forbidden operations
□ Return structure includes all required keys
□ Code is safe and follows all rules

If validation fails, your response will be automatically rejected and you'll need to retry.
"""

COMPILATION_USER_PROMPT_TEMPLATE = """## ATC Rule to Compile

**Rule ID**: {rule_id}
**Category**: {category}
**Trigger**: {trigger}
**Constraint**: {constraint}
**Description**: {description}
**Severity**: {severity}
**Safety Critical**: {safety_critical}

## TrafficState Schema

{traffic_state_schema}

## Examples

### Example 1: MSA Altitude Check
```python
def evaluate(traffic_state, callsign=None):
    aircraft = traffic_state.get_aircraft(callsign) if callsign else None
    if callsign and not aircraft:
        return {{"satisfied": True, "details": {{"error": "aircraft not found"}}, "explanation": "Aircraft not found in state", "severity": "INFO"}}
    
    msa = traffic_state.msa or 5000
    aircrafts_to_check = [aircraft] if aircraft else list(traffic_state.aircrafts.values())
    
    worst_result = {{"satisfied": True, "details": {{}}, "explanation": "All aircraft above MSA", "severity": "INFO"}}
    for ac in aircrafts_to_check:
        alt = ac.position.altitude
        if alt < msa:
            result = {{
                "satisfied": False,
                "details": {{"callsign": ac.callsign, "altitude": alt, "msa": msa, "difference_ft": msa - alt}},
                "explanation": f"{{ac.callsign}} at {{alt}}ft is below MSA {{msa}}ft",
                "severity": "HIGH"
            }}
            if worst_result["satisfied"]:
                worst_result = result
    return worst_result
```

### Example 2: Separation Check
```python
def evaluate(traffic_state, callsign=None):
    import math
    if not callsign:
        return {{"satisfied": True, "details": {{}}, "explanation": "Requires specific callsign", "severity": "INFO"}}
    
    aircraft = traffic_state.get_aircraft(callsign)
    if not aircraft:
        return {{"satisfied": True, "details": {{"error": "not found"}}, "explanation": "Aircraft not found", "severity": "INFO"}}
    
    min_sep_nm = 3.0
    nearby = traffic_state.get_nearby_aircraft(callsign, max_distance_nm=10.0)
    
    for other in nearby:
        dist = TrafficState.calculate_distance(aircraft.position, other.position)
        if dist < min_sep_nm:
            return {{
                "satisfied": False,
                "details": {{"callsign": callsign, "other": other.callsign, "distance_nm": round(dist, 2), "min_separation_nm": min_sep_nm}},
                "explanation": f"{{callsign}} and {{other.callsign}} separated only {{dist:.1f}}nm (min {{min_sep_nm}}nm)",
                "severity": "CRITICAL"
            }}
    
    return {{"satisfied": True, "details": {{"callsign": callsign, "nearby_count": len(nearby)}}, "explanation": "Adequate separation maintained", "severity": "INFO"}}
```

Now compile the following rule into a Python evaluate function:

**Rule ID**: {rule_id}
**Description**: {description}
**Trigger**: {trigger}
**Constraint**: {constraint}

Generate ONLY the function code, nothing else.
"""

VALIDATION_PROMPT = """You are a code safety validator. Analyze the following Python function and determine if it is safe to execute.

The function should:
1. Be a single function named `evaluate` with signature `(traffic_state, callsign=None)`
2. Return a dict with keys: "satisfied" (bool), "details" (dict), "explanation" (str), "severity" (str)
3. NOT contain any of these dangerous operations: os, subprocess, open, exec, eval, import, __import__, globals, locals, compile, memoryview, bytearray, socket, http, urllib, requests
4. NOT access or modify files, network, or system resources
5. Only use: math module, basic Python operations, dict/list/str/int/float/bool, for/while/if/else/try/except

Function to validate:
```python
{code}
```

Respond with a JSON object:
{{"is_safe": true/false, "issues": ["list of issues found"], "has_correct_signature": true/false, "has_correct_return": true/false}}
"""

CLASSIFICATION_SYSTEM_PROMPT = """You are an ATC rule classifier. Your job is to determine whether an ATC rule can be objectively evaluated using ONLY the data available in a TrafficState object, or if it requires subjective judgment, external context, or information not present in TrafficState.

## TrafficState Available Fields

```python
class FlightPhase(str, Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI = "taxi"

class OccupantType(str, Enum):
    AIRCRAFT = "aircraft"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

class PhaseTransition(BaseModel):
    from_phase: FlightPhase
    to_phase: FlightPhase
    timestamp: datetime
    reason: str|None

class SquawkChange(BaseModel):
    from_squawk: str|None
    to_squawk: str
    timestamp: datetime
    changed_by: str|None  # "ATC" o "PILOT"

class TrafficState:
    sector_id: str
    aircrafts: dict[str, AircraftState]  # indexed by callsign
    runways: dict[str, RunwayState]      # indexed by runway_id
    msa: int|None                        # Minimum Sector Altitude (ft)
    qnh: int|None                        # Pressure (hPa)
    wind: dict|None                      # Wind direction/speed

class AircraftState:
    callsign: str
    position: Position  # lat, lon, altitude, heading, speed, vertical_rate
    flight_phase: FlightPhase  # ground, takeoff, climb, cruise, descent, approach, landing, taxi
    clearances: Clearances  # altitude_assigned, heading_assigned, runway_assigned, route, squawk, speed_assigned
    restrictions: list[str]
    wake_turbulence: str  # L, M, H, S
    aircraft_type: str|None
    is_emergency: bool
    emergency_type: str|None
    # Nuevos campos (Fase 1):
    phase_history: list[PhaseTransition]  # Historial de transiciones de fase
    previous_phase: FlightPhase|None      # Fase anterior
    phase_transition_timestamp: datetime|None
    squawk_history: list[SquawkChange]    # Historial de cambios de squawk
    squawk_assigned_timestamp: datetime|None

class RunwayState:
    runway_id: str
    occupied: bool
    occupied_by: str|None
    operation_mode: str  # landing, takeoff, mixed, closed
    holding_short: list[str]
    landing_queue: list[str]
    closed_until: str|None
    closure_reason: str|None
    # Nuevo campo (Fase 1):
    occupant_type: OccupantType|None  # AIRCRAFT, VEHICLE, UNKNOWN

# Available methods:
# get_aircraft(callsign) -> AircraftState|None
# get_runway(runway_id) -> RunwayState|None
# get_nearby_aircraft(callsign, max_distance_nm) -> list[AircraftState]
# calculate_distance(pos1, pos2) -> float (NM)
```

## Classification Criteria

**COMPILABLE (is_compilable=True)**: The rule checks objective, measurable conditions that can be determined from TrafficState fields. Examples:
- Altitude below MSA
- Separation between aircraft
- Runway occupancy
- Flight phase violations
- Speed/heading deviations from clearances
- Aircraft on wrong runway

**NOT COMPILABLE (is_compilable=False)**: The rule requires:
- Subjective judgment (e.g., "use standard phraseology", "be concise")
- Communication content analysis (e.g., what words were spoken)
- Human intent interpretation (e.g., "pilot should understand")
- External data not in TrafficState (e.g., NOTAMs, weather beyond wind/QNH)
- Procedural compliance that cannot be measured from state alone
- Rules about what SHOULD happen vs what IS happening (normative vs descriptive)

You must respond with a structured JSON object containing exactly these fields:
- "is_compilable": bool (True if rule can be evaluated with TrafficState data)
- "reason": str (detailed explanation of your classification decision)
- "required_fields": list[str] (TrafficState fields needed for evaluation, empty if not compilable)
- "confidence": float (0.0-1.0, your confidence in this classification)

Your response will be automatically validated to ensure it contains all required fields with proper types.
"""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """## Rule to Classify

**Rule ID**: {rule_id}
**Rule Type**: {rule_type}
**Modality**: {modality}
**Trigger**: {trigger}
**Constraint**: {constraint}
**Formal If-Then**: {formal_if_then}
**Applicability**: {applicability}
**Severity**: {severity}
**Explainability**: {explainability}

Is this rule compilable into a Python function that evaluates TrafficState?
"""
