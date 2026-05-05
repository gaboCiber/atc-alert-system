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

class RunwayState(BaseModel):
    runway_id: str
    occupied: bool
    occupied_by: str|None
    operation_mode: str  # landing, takeoff, mixed, closed
    holding_short: list[str]
    landing_queue: list[str]
    closed_until: str|None
    closure_reason: str|None

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

CRITICAL RULES:
1. Generate ONLY the `evaluate` function — no imports, no classes, no code outside the function.
2. The function signature MUST be exactly: `def evaluate(traffic_state, callsign=None):`
3. The function MUST return a dict with these exact keys:
   - "satisfied" (bool): True if rule is satisfied (no violation), False if violated
   - "details" (dict): Relevant values extracted from traffic_state
   - "explanation" (str): Human-readable explanation of the result
   - "severity" (str): One of "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"
4. You may use: `math` module (already imported in namespace)
5. You may NOT use: os, subprocess, open, exec, eval, import, __import__, globals, locals, or any file/network/system operations.
6. Handle edge cases: aircraft not found, None values, missing fields.
7. If callsign is None, evaluate the rule for ALL aircraft in traffic_state (return worst result).
8. Use `traffic_state.get_aircraft(callsign)` to get a specific aircraft.
9. Use `traffic_state.aircrafts` dict to iterate all aircraft.
10. Use `traffic_state.get_nearby_aircraft(callsign)` for separation checks.
11. Use `TrafficState.calculate_distance(pos1, pos2)` for distance in NM.
12. Keep the function concise but correct. Prioritize correctness over brevity.
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
