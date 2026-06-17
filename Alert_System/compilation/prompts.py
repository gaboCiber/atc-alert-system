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

PARSED_INSTRUCTION_SCHEMA = """
# INSTRUCTION PARAMETER (available as optional 3rd parameter of evaluate())
# The instruction parameter contains the parsed ATC communication that triggered this evaluation.
# Use it for phraseology checks, read-back verification, and instruction type matching.

class InstructionType(str, Enum):
    UNKNOWN = "unknown"
    # Vertical movement
    DESCENT = "descent"
    CLIMB = "climb"
    MAINTAIN_ALTITUDE = "maintain_altitude"
    EXPEDITE_DESCENT = "expedite_descent"
    EXPEDITE_CLIMB = "expedite_climb"
    # Heading
    HEADING = "heading"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    PRESENT_HEADING = "present_heading"
    # Speed
    SPEED = "speed"
    MAINTAIN_SPEED = "maintain_speed"
    REDUCE_SPEED = "reduce_speed"
    INCREASE_SPEED = "increase_speed"
    NO_SPEED_RESTRICTION = "no_speed_restriction"
    # Clearances
    TAKEOFF_CLEARANCE = "takeoff_clearance"
    LANDING_CLEARANCE = "landing_clearance"
    APPROACH_CLEARANCE = "approach_clearance"
    # Ground movement
    TAXI = "taxi"
    TAXI_VIA = "taxi_via"
    HOLD_POSITION = "hold_position"
    HOLD_SHORT = "hold_short"
    LINE_UP = "line_up"
    LINE_UP_AND_WAIT = "line_up_and_wait"
    # Communications
    CONTACT = "contact"
    MONITOR = "monitor"
    SQUAWK = "squawk"
    IDENT = "ident"
    CHECK_STROBE = "check_strobe"
    # Emergency
    PAN_PAN = "pan_pan"
    MAYDAY = "mayday"
    EMERGENCY_DESCENT = "emergency_descent"
    # Other
    REPORT = "report"
    CLEARED_AS_FILED = "cleared_as_filed"
    DIRECT_TO = "direct_to"
    CLEARED_TO_LAND = "cleared_to_land"
    GO_AROUND = "go_around"
    MISSED_APPROACH = "missed_approach"

class Speaker(str, Enum):
    ATCO = "atco"
    PILOT = "pilot"

class ParsedInstruction:
    raw_text: str              # Raw ATC text: "AAL123 descend to FL240"
    normalized_text: str       # Normalized version
    speaker: Speaker           # Who spoke: ATCO or PILOT
    callsign: str|None         # Target aircraft callsign: "AAL123"
    instruction_type: InstructionType  # DESCENT, HEADING, TAKEOFF_CLEARANCE, etc.
    action_verb: str           # Main action: "descend", "turn", "hold", "contact"
    parameters: dict           # Extracted values:
                               #   target_altitude: int (ft), heading: int (deg),
                               #   speed: int (kts), runway: str,
                               #   waypoint: str, flight_level: int,
                               #   frequency: str (e.g. "118.5")
    temporal_marker: str|None  # "immediately", "when_ready", "at_pilot_discretion"
    priority_marker: str|None  # "urgent", "priority", "expedite"
    entities: list[str]        # KEX entity IDs referenced
    is_valid: bool             # Whether the instruction was parsed correctly

    # Useful methods:
    # get_target_altitude() -> int|None
    # get_target_heading() -> int|None
    # get_target_speed() -> int|None
    # is_clearance() -> bool  (takeoff/landing/approach clearance?)
    # is_altitude_change() -> bool
    # requires_immediate_action() -> bool
    # has_parameter(key) -> bool
"""

COMPILATION_SYSTEM_PROMPT = """You are an expert ATC (Air Traffic Control) rule compiler. Your task is to convert natural language ATC rules into Python evaluation functions.

You must respond with a structured JSON object containing:
- "code": The complete Python function (string)
- "explanation": Brief explanation of what code does (string, optional)
- "required_state_fields": List of TrafficState fields used (list of strings, optional)

CRITICAL VALIDATION REQUIREMENTS:
Your code will be automatically validated before acceptance. Ensure:
1. VALID SYNTAX: Code must parse without syntax errors
2. CORRECT FUNCTION: Must contain `def evaluate(traffic_state, callsign=None, instruction=None):`
3. NO FORBIDDEN IMPORTS: Only allowed imports are `math` and `datetime`
4. NO FORBIDDEN OPERATIONS: Cannot use os, subprocess, open, exec, eval, etc.
5. CORRECT RETURN STRUCTURE: Must return dict with keys: "satisfied", "details", "explanation", "severity"

CODE GENERATION RULES:
1. Generate ONLY the `evaluate` function — no imports, no classes, no code outside function.
2. The function signature MUST be exactly: `def evaluate(traffic_state, callsign=None, instruction=None):`
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
16. **USING INSTRUCTION PARAMETER (3rd parameter)**:
    - `instruction` is an optional `ParsedInstruction` with the ATC communication data
    - If `instruction` is None, evaluate using `traffic_state` only
    - `instruction.raw_text`: Full raw text of the ATC communication
    - `instruction.instruction_type`: Type of instruction (DESCENT, HEADING, TAKEOFF_CLEARANCE, CONTACT, etc.)
    - `instruction.action_verb`: Main action verb ("descend", "turn", "hold", "contact")
    - `instruction.speaker`: Who spoke (ATCO or PILOT)
    - `instruction.parameters`: Dict with extracted values:
      - `target_altitude`: int (ft), `heading`: int (deg), `speed`: int (kts)
      - `runway`: str, `frequency`: str, `waypoint`: str
    - `instruction.temporal_marker`: "immediately", "when_ready", or None
    - `instruction.is_clearance()`: True if takeoff/landing/approach clearance
    - `instruction.is_altitude_change()`: True if instruction changes altitude
    - Use for: phraseology checks, read-back verification, instruction type matching,
      clearance content validation, frequency assignment checks
17. **SAFE PARAMETER ACCESS**: When accessing `instruction.parameters`, ALWAYS use
    `.get("key", default)` instead of `["key"]` to avoid KeyError:
    - CORRECT: `instruction.parameters.get("target_altitude", 0)`
    - CORRECT: `instruction.parameters.get("heading", None)`
    - WRONG: `instruction.parameters["target_altitude"]` (will crash if key missing)
    - WRONG: `instruction.parameters.get("target_altitude")` without default (returns None,
      which is fine if you check for None, but prefer an explicit default)
18. **SAFE INSTRUCTION ACCESS**: Always handle `instruction is None`:
    ```python
    if instruction is None:
        return {"satisfied": True, "details": {}, "explanation": "No instruction data", "severity": "INFO"}
    ```
    Or use safe patterns like:
    ```python
    params = instruction.parameters if instruction else {}
    heading = params.get("heading", None)
    ```
19. **PHRASEOLOGY PATTERNS** (when using instruction):
    - "cleared for take-off" → `instruction.instruction_type == TAKEOFF_CLEARANCE`
    - "hold position" → `instruction.instruction_type == HOLD_POSITION`
    - "contact [frequency]" → `instruction_type == CONTACT` and `params.get("frequency")`
    - Heading ending in 0 → `params.get("heading")` should be followed by "degrees"
    - First contact → Check if instruction contains callsign + position/altitude in raw_text
20. **READ-BACK CHECKS** (when using instruction):
    - For pilot read-backs: `instruction.speaker == PILOT`
    - Check if instruction text matches standard read-back patterns
    - Altitude read-back: instruction should repeat the assigned altitude
    - Heading read-back: instruction should repeat the assigned heading
    - Squawk read-back: instruction should repeat the squawk code

SELF-VALIDATION CHECKLIST before responding:
□ No syntax errors
□ Function signature is correct (traffic_state, callsign=None, instruction=None)
□ No forbidden imports
□ No forbidden operations
□ Return structure includes all required keys
□ Code handles instruction=None safely
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

## Instruction Parameter (Optional 3rd Parameter)

The `instruction` parameter is available for rules that need communication data:

```python
instruction.raw_text: str              # "AAL123 descend to FL240"
instruction.instruction_type: str      # DESCENT, HEADING, TAKEOFF_CLEARANCE, etc.
instruction.action_verb: str           # "descend", "turn", "hold", "contact"
instruction.speaker: str              # "atco" or "pilot"
instruction.parameters: dict           # {{"target_altitude": 24000, "heading": 90}}
instruction.callsign: str|None         # "AAL123"
instruction.temporal_marker: str|None  # "immediately"
instruction.is_clearance() -> bool
instruction.is_altitude_change() -> bool
```

When the rule checks communication content, phraseology, or read-backs, use `instruction`.
When `instruction is None`, evaluate using `traffic_state` only and return satisfied=True.

## Examples

### Example 1: MSA Altitude Check
```python
def evaluate(traffic_state, callsign=None, instruction=None):
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
def evaluate(traffic_state, callsign=None, instruction=None):
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

### Example 3: Read-back Check (using instruction parameter with safe access)
```python
def evaluate(traffic_state, callsign=None, instruction=None):
    if instruction is None:
        return {{"satisfied": True, "details": {{}}, "explanation": "No instruction data available", "severity": "INFO"}}
    
    params = instruction.parameters if instruction else {{}}
    target = params.get("target_altitude", 0)
    hdg = params.get("heading", None)
    spd = params.get("speed", None)
    
    # Check that pilot read back the altitude instruction
    if instruction.instruction_type == "descent" and instruction.speaker == "pilot":
        if target > 0 and str(target) not in instruction.raw_text:
            return {{
                "satisfied": False,
                "details": {{"instruction": instruction.raw_text, "expected_altitude": target}},
                "explanation": f"Pilot did not read back altitude {{target}}ft in: {{instruction.raw_text}}",
                "severity": "MEDIUM"
            }}
    
    return {{"satisfied": True, "details": {{"check": "readback_ok"}}, "explanation": "Instruction read-back satisfactory", "severity": "INFO"}}
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

**COMPILABLE WITH INSTRUCTION PARAMETER (is_compilable=True)**: Rules that check communication content
CAN be compiled using the optional `instruction` parameter. The `instruction` parameter provides
access to the parsed ATC communication data. Examples:
- Phraseology rules: check instruction.raw_text, instruction.action_verb
- Read-back rules: check instruction.speaker, instruction.instruction_type
- Clearance content rules: check instruction.instruction_type, instruction.parameters
- Frequency assignment: check instruction.parameters.get("frequency")
- First contact rules: check instruction.raw_text for callsign + position
NOTE: These would NOT be compilable without the instruction parameter, but WITH it they ARE.

**NOT COMPILABLE (is_compilable=False)**: The rule requires:
- Subjective judgment (e.g., "be concise", "use professional tone", "sound confident")
- Human intent interpretation (e.g., "pilot should understand")
- External data not in TrafficState or ParsedInstruction (e.g., TCAS Resolution Advisories, NOTAMs, RVSM approval status, specific geographical holding points, weather beyond wind/QNH)
- Voice tone analysis (e.g., "tone of voice was urgent")
- Real-time coordination status with other sectors
- Procedural compliance that cannot be measured from state or instruction alone

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

IMPORTANT: An optional `instruction` parameter (ParsedInstruction) is available at evaluation time.
This parameter contains the parsed ATC communication (raw_text, instruction_type, action_verb,
parameters, speaker). Rules about communication content, phraseology, read-backs, and
type-specific validations CAN be compiled if they can be checked using instruction data.

Is this rule compilable into a Python function that evaluates TrafficState (and optionally instruction)?
"""
