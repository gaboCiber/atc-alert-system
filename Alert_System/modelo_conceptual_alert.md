# Modelo Conceptual del Alert System

## Vista General

El sistema resuelve un problema de **detección de violaciones de seguridad en
tiempo real para Control de Tráfico Aéreo (ATC)**. Dada una instrucción ATC
entrante (proveniente de un sistema ASR de voz o texto) y el estado actual del
tráfico aéreo en un sector, el sistema debe determinar si ejecutar esa
instrucción causaría una violación a las reglas de seguridad aeronáutica, y en
caso afirmativo, generar alertas y decidir si aceptar o rechazar la instrucción.

El sistema se descompone en **5 procesos fundamentales** que operan en dos
líneas de tiempo distintas:

**Línea Offline (compilación única, previa a la operación)**:
1. **Compilación de Reglas**: Transformar reglas de seguridad abstractas (extraídas
   por el Knowledge Extractor) en código Python ejecutable que pueda evaluar
   condiciones contra un estado de tráfico.

**Línea Online (por cada instrucción, en tiempo real)**:
2. **Proyección de Estado**: Simular el efecto de la instrucción sobre el estado
   actual sin modificar el estado real (operación "what-if").
3. **Evaluación de Reglas**: Evaluar el conjunto de reglas de seguridad contra
   el estado proyectado.
4. **Generación de Alertas**: Convertir las violaciones detectadas en alertas
   estructuradas para el controlador (ATCO).
5. **Actualización Transaccional del Estado**: Decidir si aplicar la instrucción
   al estado real (commit) o rechazarla (rollback), con soporte para override
   del ATCO.

Cada proceso está diseñado como un transformador puro de datos, sin efectos
secundarios, lo que permite composición, testing aislado y paralelización.

---

## Proceso 1: Compilación de Reglas (Offline)

### Problema Formal

Sea una regla extraída por el Knowledge Extractor `R` que contiene una
descripción en lenguaje natural de una restricción operacional ATC, junto con
metadatos como tipo de regla, modalidad, trigger, constraint, y representación
formal if-then.

Se necesita transformar `R` en una **función evaluadora** `f_R` que pueda
ejecutarse contra un `TrafficState` concreto, retornando si la regla se viola
o no en ese estado.

El problema se descompone en 3 etapas secuenciales:

#### Subproceso 1.1: Clasificación de Compilabilidad

No todas las reglas aeronáuticas pueden evaluarse objetivamente con datos del
`TrafficState`. Algunas requieren juicio subjetivo, contexto externo, o
información que el sistema no posee (condiciones meteorológicas más allá del
viento/QNH, intención del piloto, fatiga, visibilidad, etc.).

Se define un clasificador:

```
LLM_classify: R → (is_compilable ∈ {0,1}, confidence ∈ [0,1], required_fields ⊆ F_TS, reason ∈ String)
```

donde `F_TS` es el conjunto de campos disponibles en `TrafficState`:

```
F_TS = {aircrafts.{callsign, position.{altitude, heading, speed, latitude, longitude, vertical_rate},
        flight_phase, clearances.{altitude_assigned, heading_assigned, runway_assigned, speed_assigned},
        restrictions, wake_turbulence, aircraft_type, is_emergency,
        phase_history, squawk_history},
        runways.{occupied, occupied_by, operation_mode, holding_short, landing_queue, occupant_type},
        msa, qnh, wind, sector_id}
```

La clasificación sigue estos criterios:

| Categoría | `is_compilable` | Ejemplos |
|-----------|:---:|---|
| Altitud, separación, pista, velocidad, fase | 1 | "Aircraft must not descend below MSA", "Maintain 5NM separation" |
| Emergencias detectables | 1 | "Squawk 7700 indicates emergency" |
| Clearances | 1 | "Runway must be clear before landing clearance" |
| Fraseología, comunicación | 0 | "Use standard phraseology", "Read back instructions" |
| Juicio humano | 0 | "Pilot should use best judgment", "Weather permitting" |
| Datos externos no disponibles | 0 | "Visibility must be above 5km", "NOTAMs apply" |

Si `is_compilable = 0`, la regla se marca para **fallback por LLM runtime**
(evaluación vía LLM en tiempo real) o se descarta si ni siquiera eso es posible.

#### Subproceso 1.2: Generación de Código

Para reglas clasificadas como compilables, un LLM genera una función Python
`evaluate(traffic_state, callsign=None)` que implementa la lógica de la regla:

```
LLM_gen: (R, schema_TS) → code
```

La función generada debe cumplir **restricciones formales estrictas** validadas
por Pydantic y AST:

1. **Sintaxis válida**: `V_syntax(code) = AST_parse(code) ∈ {0,1}`
2. **Firma correcta**: `V_signature = (def evaluate(traffic_state, callsign=None) ∈ code)`
3. **Sin imports prohibidos**: `V_forbidden = ⋀_{f ∈ F_forbidden} f ∉ imports(code)`
   ```
   F_forbidden = {os, subprocess, open, exec, eval, compile, __import__,
                  globals, locals, socket, http, urllib, requests, sys}
   F_allowed = {math, datetime}
   ```
4. **Sin nombres prohibidos**: `V_names = ⋀_{f ∈ F_forbidden} f ∉ name_references(code)`
5. **Estructura de retorno correcta**: `V_return = (K_required ⊆ keys(return_dict))`
   ```
   K_required = {satisfied ∈ {0,1}, details ∈ dict, explanation ∈ String, severity ∈ Severity}
   Severity = {INFO, LOW, MEDIUM, HIGH, CRITICAL}
   ```

La validación completa: `V_total = V_syntax ∧ V_forbidden ∧ V_signature ∧ V_return`

En la práctica, `V_forbidden`, `V_signature`, y `V_return` se implementan como
validadores Pydantic anidados en el response_model de Instructor, que reintenta
automáticamente si la validación falla.

#### Subproceso 1.3: Testing Sintético

El código generado se ejecuta contra un `TrafficState` de prueba para verificar
que corre sin errores y produce la estructura de retorno esperada:

```
T(code) → (success ∈ {0,1}, error ∈ String)
```

El estado de prueba contiene dos aeronaves sintéticas (TEST123, TEST456) con
posiciones, altitudes y fases de vuelo conocidas. Se ejecuta la función
`evaluate(test_state, callsign="TEST123")` y se verifica:

1. No hay excepciones en tiempo de ejecución.
2. El resultado es un diccionario con todas las claves requeridas.
3. Cada clave tiene el tipo correcto (bool, dict, str, str).
4. El campo `severity` es uno de los 5 valores permitidos.
5. La función también funciona con `callsign=None` (evaluación para todas las
   aeronaves).

Si todas las validaciones pasan, la regla se considera exitosamente compilada
y se persiste en disco como archivo `.py` + `manifest.json`.

### Estados de Compilación

Cada regla compilada puede terminar en uno de estos estados:

| Estado | Significado | Acción en Runtime |
|--------|-------------|-------------------|
| `COMPILED` | Compilación exitosa, código válido | Usar `CompiledCondition` (ejecución directa) |
| `NOT_COMPILABLE` | Regla subjetiva, no evaluable con TrafficState | Fallback a `GenericKexCondition` con LLM runtime |
| `FAILED` | Error de compilación (sintaxis, imports, test) | Fallback a `GenericKexCondition` con LLM runtime |
| `PENDING` | No compilada aún | N/A |

### Persistencia

Las reglas compiladas exitosamente se almacenan como:

- **`compiled_rules/{RULE_ID}.py`**: archivo individual con el código generado,
  incluyendo un header con metadata (modelo LLM, timestamp, categoría).
- **`compiled_rules/manifest.json`**: manifiesto JSON con todas las reglas
  compiladas, sus metadatos, y contadores.

---

## Proceso 2: Proyección de Estado (State Projection)

### Problema Formal

Sea un `TrafficState` actual `TS_t` que representa el estado real y verificado
del tráfico aéreo en el instante `t`:

```
TS_t = (sector_id, timestamp, aircrafts, runways, msa, qnh, wind)
```

donde `aircrafts` es un mapeo `callsign → AircraftState` que contiene para cada
aeronave su posición (`Position`), fase de vuelo (`FlightPhase`), clearances
activos (`Clearances`), y otros atributos.

Sea una instrucción parseada `PI = (callsign, instruction_type, parameters, ...)`
donde `parameters` contiene los valores específicos de la instrucción
(`target_altitude`, `heading`, `speed`, `runway`, etc.).

Se necesita crear un **estado proyectado** `TS'` que represente cómo quedaría
el sistema si se aplicara `PI` a `TS_t`, sin modificar `TS_t`.

```
Π: (TS_t, PI) → ProjectedState
```

#### Componentes del Estado Proyectado

```
ProjectedState = {
    traffic_state: TS'            # Copia modificada del TrafficState
    source_instruction: PI        # Instrucción que originó la proyección
    trajectories: Dict[callsign, ProjectedTrajectory]  # Trayectorias futuras
    projected_separations: Dict[callsign, [ProjectedSeparation]]  # Separaciones
    target_aircraft_final: AircraftState  # Estado final de la aeronave objetivo
    is_valid_projection: bool     # Si la proyección es válida
    projection_errors: [String]   # Errores si no es válida
    projection_horizon_min: int   # Horizonte de proyección en minutos (default: 10)
}
```

#### Subproceso 2.1: Copia y Aplicación de Instrucción

Se crea una copia profunda (`deepcopy`) del `TrafficState` para evitar cualquier
efecto colateral sobre el estado real:

```
TS' = deepcopy(TS_t)
```

Luego, se localiza la aeronave objetivo en `TS'.aircrafts[callsign]` y se le
aplican las modificaciones correspondientes según `instruction_type`:

**Instrucciones de altitud** (`DESCENT`, `CLIMB`, `MAINTAIN_ALTITUDE`,
`EXPEDITE_DESCENT`, `EXPEDITE_CLIMB`):

```
TS'.aircrafts[cs].position.altitude = parameters.target_altitude
TS'.aircrafts[cs].clearances.altitude_assigned = parameters.target_altitude
```

**Instrucciones de rumbo** (`HEADING`, `TURN_LEFT`, `TURN_RIGHT`):

```
TS'.aircrafts[cs].position.heading = parameters.heading
TS'.aircrafts[cs].clearances.heading_assigned = parameters.heading
```

**Instrucciones de velocidad** (`SPEED`, `REDUCE_SPEED`, `INCREASE_SPEED`,
`MAINTAIN_SPEED`):

```
TS'.aircrafts[cs].position.speed = parameters.speed
TS'.aircrafts[cs].clearances.speed_assigned = parameters.speed
```

**Clearances de pista** (`TAKEOFF_CLEARANCE`, `LANDING_CLEARANCE`):

```
TS'.aircrafts[cs].clearances.runway_assigned = parameters.runway
```

La fase de vuelo se actualiza mediante una función `Φ` que depende de la
altitud resultante y el tipo de instrucción:

```
Φ(altitude, instruction_type):
    if altitude < 1000:
        if instruction_type == TAKEOFF_CLEARANCE → TAKEOFF
        else → GROUND
    elif altitude < 10000:
        if instruction_type == CLIMB → CLIMB
        elif instruction_type == DESCENT → APPROACH
        else → APPROACH
    elif altitude < 18000:
        if instruction_type == CLIMB → CLIMB
        elif instruction_type == DESCENT → DESCENT
        else → CRUISE
    else → CRUISE
```

#### Subproceso 2.2: Cálculo de Trayectoria Proyectada

Para la aeronave objetivo, se genera una trayectoria de `N` waypoints (uno por
minuto hasta `projection_horizon_min`), modelando el movimiento cinemático:

```
Trj: (aircraft, instruction, N) → [w_1, w_2, ..., w_N]
```

Cada waypoint `w_i` es una tupla:

```
w_i = (latitude, longitude, altitude, estimated_time_sec, speed, vertical_rate)
```

La generación es iterativa, partiendo de la posición actual de la aeronave:

**Parámetros cinemáticos**:

La tasa vertical (`vrate`) depende del tipo de instrucción:

```
vrate(instruction_type):
    DESCENT         → -1000 ft/min
    EXPEDITE_DESCENT → -2000 ft/min
    CLIMB           → +1500 ft/min
    EXPEDITE_CLIMB  → +2500 ft/min
    otherwise       → 0 ft/min
```

**Actualización de posición** (por minuto, geometría simplificada):

```
distance_nm = current_speed / 60    # NM recorridos en 1 minuto

lat_change = (distance_nm / 60) × cos(heading × π/180)
lon_change = (distance_nm / 60) × sin(heading × π/180) / cos(current_lat × π/180)

lat_i = lat_{i-1} + lat_change
lon_i = lon_{i-1} + lon_change
alt_i = max(0, alt_{i-1} + vrate)
```

La velocidad se ajusta según la altitud y fase de vuelo:

```
speed(altitude, vertical_rate):
    altitude < 1000:
        vertical_rate > 0 → SPEED_TAKEOFF (150 kts)
        vertical_rate < 0 → SPEED_LANDING (140 kts)
    altitude < 10000:
        vertical_rate > 0 → SPEED_CLIMB (250 kts)
        vertical_rate < 0 → SPEED_DESCENT (280 kts)
    else → SPEED_CRUISE (450 kts)
```

Este es un modelo simplificado que asume:
- Trayectoria en línea recta (sin virajes ni cambios de rumbo intermedios).
- Velocidad constante por segmentos de altitud.
- Tasa vertical constante durante todo el ascenso/descenso.
- Sin efectos de viento, temperatura, o restricciones de performance.

**Nota importante**: la trayectoria comienza desde la **altitud original**
(antes de aplicar la instrucción) para mostrar la transición completa. La
altitud final en el último waypoint coincidirá con la altitud objetivo (si
el horizonte es suficiente para completar el cambio).

#### Subproceso 2.3: Cálculo de Separaciones Proyectadas

Para la aeronave objetivo y cada aeronave cercana (dentro de 30NM), se calcula
la separación vertical y horizontal en cada waypoint de la trayectoria:

```
Sep: (TS', callsign, Trj) → [ProjectedSeparation_1, ..., ProjectedSeparation_M]
```

Cada `ProjectedSeparation` contiene:

```
ProjectedSeparation = {
    aircraft_1: String,           # callsign objetivo
    aircraft_2: String,           # callsign de la otra aeronave
    vertical_separation_ft: float,  # mínima separación vertical en la trayectoria
    horizontal_separation_nm: float,  # mínima separación horizontal en la trayectoria
    time_to_conflict: float|null,   # segundos hasta la primera pérdida de separación
    conflict_predicted: bool        # si se predice algún conflicto ICAO
}
```

Para cada waypoint `w_i` y cada aeronave vecina `a_j`:

```
d_vert(w_i, a_j) = |w_i.altitude - a_j.position.altitude|
```

```
d_horiz(w_i, a_j) = √((Δlat × 60)² + (Δlon × 60 × cos(lat_cs × π/180))²)
```

donde:
```
Δlat = w_i.latitude - a_j.position.latitude
Δlon = w_i.longitude - a_j.position.longitude
```

El estándar ICAO para separación mínima utilizado es:
- **Vertical**: 1000 pies
- **Horizontal**: 5 NM

Un conflicto se predice si existe al menos un waypoint donde **ambas**
separaciones estén por debajo de sus umbrales simultáneamente:

```
conflict_predicted(a_cs, a_j) = ∃ w_i : d_vert(w_i, a_j) < 1000 ∧ d_horiz(w_i, a_j) < 5
```

La separación de las aeronaves vecinas se considera **estática** (no se
proyecta su movimiento). Esto es una limitación importante del modelo actual:
asume que las otras aeronaves mantienen su posición, rumbo y velocidad
constantes, lo que puede subestimar o sobreestimar conflictos. Una versión
más sofisticada proyectaría también las trayectorias de las aeronaves vecinas
usando su rumbo y velocidad actuales.

---

## Proceso 3: Evaluación de Reglas

### Problema Formal

Dado un estado proyectado `TS'` (que ya incorpora los efectos de la instrucción
bajo evaluación) y un conjunto de reglas `R_set = {R_1, ..., R_n}`, determinar
cuáles reglas se violan en `TS'`.

Cada regla `R_k` está asociada a un **evaluador de condición** `E_k` que
implementa la lógica específica de evaluación:

```
E_k: (TS', params_k, callsign) → ConditionResult
```

donde:

```
ConditionResult = {
    satisfied: bool,              # true si la condición se cumple (sin violación)
    violation: Violation | null,   # violación si no está satisfecha
    details: dict                  # detalles adicionales de la evaluación
}
```

y:

```
Violation = {
    violation_id: String,
    rule_id: String,
    condition_type: String,       # tipo de condición que falló
    severity: AlertSeverity,      # INFO, LOW, MEDIUM, HIGH, CRITICAL
    details: dict,                # parámetros específicos de la violación
    explanation: String,          # texto descriptivo
    suggestion: String | null     # acción correctiva sugerida
    detected_at: datetime
}
```

### Arquitectura de Evaluadores

El sistema implementa 4 tipos de evaluadores con un mecanismo de prioridad:

| Prioridad | Evaluador | Origen | Mecanismo |
|:---:|---|---|---|
| 1 | `CompiledCondition` | Código Python generado por LLM (offline) | `exec()` en namespace restringido |
| 2 | Evaluadores Nativos | `AltitudeCondition`, `SeparationCondition`, `RunwayCondition` | Lógica hardcodeada |
| 3 | `GenericKexCondition (LLM)` | Reglas genéricas sin compilación | LLM runtime con Instructor |
| 4 | `GenericKexCondition (Keywords)` | Fallback cuando LLM no está disponible | Matching por keywords |

#### Subproceso 3.1: Evaluación Nativa de Altitud (`AltitudeCondition`)

Evalúa condiciones que involucran la altitud de una aeronave.

**Tipos de chequeo**:

| `check_type` | Condición de violación | Severidad |
|---|---|---|
| `MINIMUM` | `a_cs.altitude < reference_value` | `HIGH` |
| `MAXIMUM` | `a_cs.altitude > reference_value` | `MEDIUM` |
| `MSA_CHECK` | `a_cs.altitude < TS.msa` | `CRITICAL` |

Si no hay reglas de altitud registradas, por defecto se evalúa `MSA_CHECK`.

#### Subproceso 3.2: Evaluación Nativa de Separación (`SeparationCondition`)

Evalúa condiciones de separación entre aeronaves usando las separaciones
pre-calculadas en la proyección.

**Tipos de chequeo**:

| `separation_type` | Condición de violación |
|---|---|
| `VERTICAL` | `∃ a_j : |a_cs.alt - a_j.alt| < min_distance` |
| `HORIZONTAL` | `∃ a_j : dist(a_cs, a_j) < min_distance` |
| `BOTH` | `∃ a_j : (vert < 1000 ∧ horiz < 5)` |

Si no hay reglas registradas, por defecto se evalúa `BOTH` con estándares ICAO
(1000ft vertical, 5NM horizontal).

#### Subproceso 3.3: Evaluación Nativa de Pista (`RunwayCondition`)

Evalúa condiciones relacionadas con el estado de las pistas.

**Tipos de chequeo**:

| `check_type` | Condición de violación |
|---|---|
| `OCCUPIED` | `TS.runways[runway_id].occupied = true` |
| `HOLDING_SHORT_FULL` | `|TS.runways[runway_id].holding_short| ≥ max_holding` |
| `EXISTS` | `runway_id ∉ TS.runways` |

#### Subproceso 3.4: Evaluación Compilada (`CompiledCondition`)

Ejecuta código Python generado por el compilador de reglas en un namespace
restringido:

```
CompiledCondition.evaluate(TS', params, cs):
    evaluate_fn = exec(compiled_code, safe_namespace)
    result_dict = evaluate_fn(TS', callsign=cs)
    return dict_to_condition_result(result_dict)
```

El namespace seguro (`safe_namespace`) contiene únicamente:

```
safe_namespace = {
    "math": math,
    "TrafficState": TrafficState,
    "AircraftState": AircraftState,
    "Position": Position,
    "FlightPhase": FlightPhase,
    "RunwayState": RunwayState,
    "Clearances": Clearances,
    "__builtins__": {True, False, None, int, float, str, bool,
                     list, dict, tuple, set, len, range, abs,
                     min, max, round, sorted, any, all, ...}
}
```

**Manejo de errores**: Si la ejecución del código compilado falla por cualquier
razón (error de sintaxis en runtime, excepción no capturada, referencia a
nombre inexistente), el `CompiledCondition` hace **fallback automático** a un
`GenericKexCondition` con LLM runtime, que intenta evaluar la misma regla
usando el LLM.

#### Subproceso 3.5: Evaluación Genérica con LLM (`GenericKexCondition`)

Para reglas que no encajan en categorías predefinidas (ni altitud, ni
separación, ni pista, ni compiladas), se utiliza un evaluador genérico que
puede operar en dos modos:

**Modo LLM Runtime** (prioritario): construye un resumen legible del estado
del tráfico y lo envía al LLM junto con la descripción de la regla. El LLM
responde con una estructura `LLMEvaluationResult`:

```
LLM: (rule_description, traffic_summary, aircraft_summary,
      msa_value, runway_status, separation_summary)
   → LLMEvaluationResult = {
        is_violated: bool,
        confidence: float ∈ [0,1],
        explanation: String,
        suggested_action: String | null,
        severity_override: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL" | null,
        extracted_values: dict | null
      }
```

Se considera violación si `is_violated = true ∧ confidence > 0.5`.

**Fallback por Keywords** (si LLM no está disponible o falla): analiza la
descripción de la regla buscando palabras clave como "altitude", "below",
"msa", etc., y si encuentra coincidencias, verifica condiciones simples como
`a_cs.altitude < TS.msa`.

Este evaluador es el cuello de botella del sistema: cada regla genérica requiere
una llamada LLM por instrucción. Para mitigarlo, se emplea un sistema de
**prefiltrado** (RuleFilter) que reduce el número de reglas a evaluar.

#### Subproceso 3.6: Prefiltrado de Reglas Genéricas (RuleFilter)

El `RuleFilter` aplica hasta 3 capas consecutivas de filtrado para reducir el
conjunto de reglas genéricas que requieren evaluación costosa (LLM runtime):

**Capa 1a — Keywords**: extrae palabras clave ATC del texto de la instrucción
entrante y selecciona solo las reglas que contienen al menos una keyword
coincidente:

```
rules_kw = {r ∈ rules | keywords(r) ∩ keywords(instruction) ≠ ∅}
```

Las keywords ATC están organizadas por categorías semánticas:
`{climb, descend, altitude, speed, heading, runway, separation, emergency,
clearance, weather, communication, wake, holding, route, ...}`.

**Capa 1b — Embeddings (re-rank)**: usando modelo `all-MiniLM-L6-v2`
(sentence-transformers), se calcula la similitud coseno entre el embedding de
la instrucción y el embedding de cada regla, y se seleccionan las top-k
reglas con mayor similitud:

```
emb_inst = encode(instruction_text)
emb_r = encode(rule_text)
similarity(r) = emb_inst · emb_r / (‖emb_inst‖ × ‖emb_r‖)
rules_emb = top_k(rules_kw, k, key=similarity)
```

Los embeddings de las reglas se precalculan y cachean en disco
(`demo/cache/rule_embeddings.pkl`) para evitar recalcularlos en cada
instrucción. El caché se invalida cuando cambia el contenido de las reglas
(verificado por hash MD5).

**Capa 2 — LLM Batch Relevance**: envía un lote de reglas candidatas al LLM
con la instrucción actual, y el LLM determina cuáles son relevantes:

```
rules_llm = {r ∈ rules_emb | LLM_relevance(r, instruction) = true}
```

Esta capa solo se ejecuta si el LLM está disponible (verificación de
disponibilidad de Ollama vía API). Si el LLM no responde, se omiten todas
las reglas candidatas de la capa 2.

Si alguna capa falla o no reduce el conjunto, se mantienen todas las reglas
del paso anterior. El timeout configurable (`filter_timeout`) aborta el
filtrado si se excede el tiempo límite.

### Reglas Nativas vs Compiladas vs Genéricas: Flujo de Decisión

El `KEXAdapter` orquesta la selección del evaluador apropiado para cada regla:

```
1. ¿La regla es de categoría conocida (ALTITUDE, SEPARATION, RUNWAY)?
   └─ Sí → Crear evaluador nativo correspondiente
   └─ No → ¿Es GENERIC?
       ├─ Sí → ¿Tiene versión compilada en disco?
       │   ├─ Sí → Crear CompiledCondition (con fallback a LLM runtime)
       │   └─ No → Crear GenericKexCondition (LLM runtime)
       └─ No (UNEVALUABLE) → Descartar (no se puede evaluar automáticamente)
```

---

## Proceso 4: Pipeline de Alertas (8 Pasos)

### Problema Formal

Dada una instrucción ATC entrante (texto crudo del ASR), ejecutar una secuencia
de transformaciones que culminan en una decisión de commit/rollback sobre el
estado del tráfico.

El pipeline es una **composición secuencial** de 8 funciones:

```
P = step_8 ∘ step_7 ∘ step_6 ∘ step_5 ∘ step_4 ∘ step_3 ∘ step_2 ∘ step_1
```

Cada paso es un transformador puro: recibe entrada, produce salida, y reporta
estado (`SUCCESS`, `FAILED`, `SKIPPED`). Si un paso falla, el pipeline se
detiene y retorna un resultado de error.

#### Paso 1: Input Processing (`π`)

Transforma el texto crudo de la instrucción en un `ParsedInstruction`
estructurado:

```
π: String → ParsedInstruction
```

La implementación por defecto usa un parser regex simple:

```
callsign   = regex_search(raw, \b[A-Z]{3}\d+\b)        # ej: "AAL123"
type       = match_keyword(raw, T_inst_patterns)         # ej: "descend" → DESCENT
params     = extract_values(raw, type)                   # ej: "FL240" → {target_altitude: 24000}
```

Los patrones de matching para `type` son secuenciales:
- `"descend"` o `"descent"` → `DESCENT`, extrae `FL\d+` como `target_altitude`
- `"climb"` → `CLIMB`, extrae `FL\d+` como `target_altitude`
- `"heading"` o `"turn"` → `HEADING`, extrae `\d{3}` como `heading`
- `"cleared for takeoff"` → `TAKEOFF_CLEARANCE`, extrae `runway \d+[LR]?`
- `"cleared to land"` → `LANDING_CLEARANCE`, extrae `runway \d+[LR]?`

Este parser es voluntariamente simple (para demo/prototipo). En producción,
se reemplazaría por el parser completo del `ASRAdapter` que combina BERT NER,
regex, y normalizador compacto.

#### Paso 2: Normalization

Normaliza la instrucción a un formato interno estandarizado. En la
implementación actual, el `ParsedInstruction` ya está normalizado desde el
paso 1, por lo que este paso es una identidad:

```
normalize(PI) = PI
```

Está diseñado como un slot de extensión para futuras transformaciones
(conversión de unidades, resolución de ambigüedades, etc.).

#### Paso 3: State Projection (`Π`)

Crea el estado proyectado aplicando la instrucción al estado actual:

```
Π(TS_t, PI) → ProjectedState
```

Implementado por `StateProjector.create_projection()`. Este paso es crítico
porque:

1. **Aísla el estado real**: ninguna modificación se aplica al `TS_t` real.
2. **Calcula consecuencias**: trayectorias y separaciones futuras.
3. **Es determinista**: dada la misma entrada, siempre produce la misma salida.

Si la proyección falla (callsign no encontrado, error al aplicar instrucción),
el pipeline se detiene con estado `FAILED`.

#### Paso 4: Rule Evaluation (`E`)

Evalúa todas las reglas de seguridad contra el estado proyectado:

```
E(PI, ProjectedState) → [Violation]
```

Se evalúan secuencialmente 4 categorías de reglas:

```
violations = []
violations += evaluate_altitude_rules(PI, projected, cs)
violations += evaluate_separation_rules(PI, projected, cs)
violations += evaluate_runway_rules(PI, projected, cs)
violations += evaluate_compiled_rules(PI, projected, cs)  # incluye genéricas
```

Las reglas de altitud, separación y pista se evalúan usando los evaluadores
nativos registrados en el `RuleEngine`. Las reglas compiladas y genéricas se
evalúan según la jerarquía de evaluadores descrita en el Proceso 3.

#### Paso 5: Alert Generation

Convierte cada violación en una alerta estructurada:

```
gen_alert(violations, PI) → [Alert]
```

Cada `Alert` contiene:

```
Alert = {
    alert_id: String,
    severity: AlertSeverity,      # severidad máxima entre las violaciones
    category: AlertCategory,      # inferida del condition_type
    affected_callsigns: [String],  # extraídos de los detalles de violación
    primary_callsign: String,     # callsign objetivo de la instrucción
    triggering_instruction_raw: String,  # texto original de la instrucción
    violations: [Violation],      # violaciones que componen esta alerta
    title: String,                # título descriptivo
    explanation: String,          # explicación detallada
    suggested_action: String,     # acción correctiva sugerida
    projected_state: dict | null, # snapshot del estado proyectado
    acknowledged: bool,           # si el ATCO la reconoció
    commit_decision: "PENDING" | "COMMIT" | "ROLLBACK"
}
```

La categoría de la alerta se infiere del `condition_type` de la violación:

| El `condition_type` contiene... | Categoría asignada |
|---|---|
| `"altitude"` o `"msa"` | `ALTITUDE_VIOLATION` |
| `"separation"` | `SEPARATION_LOSS` |
| `"runway"` | `RUNWAY_CONFLICT` |
| `"speed"` | `SPEED_VIOLATION` |
| otro | `PROCEDURAL_ERROR` |

#### Paso 6: Alert Presentation

Prepara las alertas para su presentación al ATCO. En la implementación actual,
empaqueta las alertas en una estructura de presentación:

```
present(alerts, violations) → {
    alert_count: int,
    violation_count: int,
    has_critical: bool,
    alerts: [Alert]
}
```

Este paso está diseñado como interfaz para una futura UI. En producción, aquí
se formatearían las alertas para visualización en pantalla, codificación por
colores, o integración con sistemas de display ATC.

#### Paso 7: ATCO Decision

Decide si la instrucción debe aceptarse (COMMIT) o rechazarse (ROLLBACK).

En la implementación automática actual, la lógica es:

```
decision(alerts):
    if ∃ a ∈ alerts : a.severity = CRITICAL → ROLLBACK
    else → COMMIT
```

Esta lógica es intencionalmente conservadora: cualquier alerta crítica (como
MSA violation, pérdida de separación, runway incursion) causa un rechazo
automático de la instrucción.

En producción, este paso sería **interactivo**: el ATCO vería las alertas y
decidiría si confirmar (COMMIT) o rechazar (ROLLBACK) la instrucción, con
opción de `force_committed` para override en casos justificados.

#### Paso 8: Final State Update

Aplica la decisión al estado real del tráfico:

```
update(decision, projected):
    if decision = "COMMIT":
        txn = state_manager.propose_change(projected)
        state_manager.commit(txn.transaction_id)
        return state_manager.current_state
    elif decision = "ROLLBACK":
        # No modificar estado, descartar proyección
        return null
```

Si es COMMIT, el estado proyectado (`ProjectedState.traffic_state`) se convierte
en el nuevo estado real. Si es ROLLBACK, el estado real permanece intacto y la
proyección se descarta.

---

## Proceso 5: Gestión Transaccional del Estado

### Problema Formal

El `StateManager` mantiene el estado real del tráfico y gestiona las
transiciones de estado como transacciones atómicas con soporte para
commit/rollback.

```
StateManager = {
    _state: TrafficState,              # estado actual (real)
    _state_history: [TrafficState],     # historial de estados anteriores
    _transactions: Dict[String, StateTransaction],  # transacciones por ID
    _pending_transaction: StateTransaction | null,  # transacción pendiente
    _max_history: int = 10              # profundidad máxima del historial
}
```

### Modelo Transaccional

Cada instrucción procesada genera una transacción:

```
StateTransaction = {
    transaction_id: String,
    timestamp: datetime,
    projected_state: ProjectedState,
    status: "PENDING" | "COMMITTED" | "ROLLBACK",
    atco_decision: "COMMIT" | "ROLLBACK" | null,
    atco_reason: String | null,
    has_alerts: bool,
    alert_ids: [String],
    force_committed: bool
}
```

El ciclo de vida de una transacción es:

```
1. propose_change(projected) → txn (PENDING)
2. commit(txn_id) → txn.status := COMMITTED, _state := projected.traffic_state
   rollback(txn_id) → txn.status := ROLLBACK, _state unchanged
3. Solo se puede hacer commit si:
   - No hay alertas (has_alerts = false), O
   - force = true (override del ATCO)
```

### Commit (Aceptar Instrucción)

```
commit(txn_id, force=false, reason=null):
    txn = transactions[txn_id]
    if txn.has_alerts and not force → return false  # rechazar
    _save_to_history()  # guardar estado actual antes de modificar
    _state = txn.projected_state.traffic_state
    txn.status = "COMMITTED"
    txn.atco_decision = "COMMIT"
    return true
```

### Rollback (Rechazar Instrucción)

```
rollback(txn_id, reason=null):
    txn = transactions[txn_id]
    txn.status = "ROLLBACK"    # solo cambiar estado de la transacción
    txn.atco_decision = "ROLLBACK"
    # _state NO se modifica
    return true
```

### Historial y Deshacer

El sistema mantiene un historial de hasta 10 estados anteriores. Permite
deshacer el último commit:

```
undo_last_commit():
    if _state_history is empty → return false
    _state = _state_history.pop()
    return true
```

También permite buscar un estado por timestamp:

```
get_state_at_timestamp(t):
    return argmin_{state in history} |state.timestamp - t|
```

### Context Manager Transaction

El sistema expone un context manager `Transaction` para uso procedural:

```
with Transaction(state_manager, projected, auto_commit=True) as txn:
    # Dentro del bloque: la transacción está PENDING
    # Si el bloque termina sin excepción y auto_commit=True → commit automático
    # Si hay excepción → rollback automático
```

---

## Diagrama de Flujo Integrado

```
Instrucción ATC (texto crudo)
    │
    ├─[Paso 1] Input Processing ─────────────────────────→ ParsedInstruction
    │                                                           │
    ├─[Paso 2] Normalization ──────────────────────────────────→ PI normalizado
    │                                                           │
    ├─[Paso 3] State Projection ───────────────────────────────→ ProjectedState
    │   ├─ deepcopy(TrafficState)
    │   ├─ apply_instruction(aircraft, PI)
    │   ├─ calculate_trajectory(aircraft, PI)
    │   └─ calculate_separations(target, nearby)
    │                                                           │
    ├─[Paso 4] Rule Evaluation ────────────────────────────────→ [Violation]
    │   ├─ evaluate_altitude_rules() ──── AltitudeCondition
    │   ├─ evaluate_separation_rules() ── SeparationCondition
    │   ├─ evaluate_runway_rules() ────── RunwayCondition
    │   └─ evaluate_compiled_rules() ──── CompiledCondition
    │       └─ [RuleFilter] ──→ [GenericKexCondition (LLM)]
    │                                                           │
    ├─[Paso 5] Alert Generation ───────────────────────────────→ [Alert]
    │   └─ violations → Alert(severity, category, explanation)
    │                                                           │
    ├─[Paso 6] Alert Presentation ─────────────────────────────→ Presentación UI
    │   └─ alerts → {count, has_critical, alerts}
    │                                                           │
    ├─[Paso 7] ATCO Decision ─────────────────────────────────→ "COMMIT" | "ROLLBACK"
    │   └─ CRITICAL? → ROLLBACK, sino → COMMIT
    │                                                           │
    └─[Paso 8] Final State Update ────────────────────────────→ TrafficState_{t+1}
        ├─ COMMIT  → state_manager.commit()  [estado real actualizado]
        └─ ROLLBACK → state_manager.rollback()  [estado real intacto]


Línea Offline (previa a la operación):

Knowledge Extractor → [Rule] → KEXAdapter
    │
    ├─ Classify: RuleVerdict(is_compilable, reason, fields)
    │   └─ is_compilable?
    │       ├─ Sí → LLM genera código → validación Pydantic → test sintético
    │       │        └─ pasa? → persistir como .py + manifest.json
    │       │        └─ falla? → marcar como FAILED (fallback runtime)
    │       └─ No (subjetiva) → marcar NOT_COMPILABLE (fallback runtime)
    │
    └─ Evaluadores resultantes:
        ├─ CompiledCondition (código compilado + fallback LLM)
        ├─ AltitudeCondition | SeparationCondition | RunwayCondition (nativos)
        └─ GenericKexCondition (LLM runtime + fallback keywords)
```

---

## Modelo de Datos Completo

### TrafficState (Estado del Tráfico)

```
TrafficState:
    timestamp: datetime
    sector_id: String
    aircrafts: Dict[callsign → AircraftState]   # aeronaves activas en el sector
    runways: Dict[runway_id → RunwayState]       # pistas del aeropuerto/sector
    msa: int | null                              # Minimum Sector Altitude (ft)
    qnh: int | null                              # presión barométrica (hPa)
    wind: dict | null                            # {direction, speed}
```

### AircraftState (Estado de una Aeronave)

```
AircraftState:
    callsign: String                              # identificador único
    position: Position                            # {lat, lon, alt, heading, speed, vertical_rate}
    flight_phase: FlightPhase                     # {GROUND, TAKEOFF, CLIMB, CRUISE, DESCENT, APPROACH, LANDING, TAXI}
    clearances: Clearances                        # {altitude_assigned, heading_assigned, runway_assigned, route, squawk, speed_assigned}
    restrictions: [String]                        # restricciones activas
    wake_turbulence: WakeTurbulenceCategory       # {LIGHT, MEDIUM, HEAVY, SUPER}
    aircraft_type: String | null                  # código OACI (ej: "B738")
    is_emergency: bool
    emergency_type: String | null
    position_history: [Position]                  # últimas 10 posiciones
    phase_history: [PhaseTransition]              # historial de cambios de fase
    previous_phase: FlightPhase | null            # fase anterior
    phase_transition_timestamp: datetime | null
    squawk_history: [SquawkChange]                # historial de cambios de squawk
    squawk_assigned_timestamp: datetime | null
```

### ParsedInstruction (Instrucción Parseada)

```
ParsedInstruction:
    raw_text: String                              # texto original del ASR
    normalized_text: String                        # texto normalizado
    speaker: Speaker                               # {ATCO, PILOT}
    callsign: String | null                        # callsign objetivo
    instruction_type: InstructionType              # {DESCENT, CLIMB, HEADING, SPEED, TAKEOFF_CLEARANCE, ...}
    action_verb: String                            # verbo de acción principal
    parameters: Dict                               # {target_altitude, heading, speed, runway, ...}
    entities: [String]                             # IDs de entidades KEX referenciadas
    temporal_marker: String | null                 # {immediately, when_ready, at_pilot_discretion}
    priority_marker: String | null                 # {urgent, priority, expedite}
    asr_confidence: float                          # confianza del ASR (0-1)
    is_valid: bool
    validation_errors: [String]
```

### ProjectedState (Estado Proyectado)

```
ProjectedState:
    traffic_state: TrafficState                        # copia modificada
    source_instruction: ParsedInstruction               # instrucción que originó la proyección
    trajectories: Dict[callsign → ProjectedTrajectory] # trayectorias calculadas
    projected_separations: Dict[callsign → [ProjectedSeparation]]
    target_aircraft_final: AircraftState | null
    is_valid_projection: bool
    projection_errors: [String]
    projection_horizon_min: int (default: 10)

ProjectedTrajectory:
    callsign: String
    waypoints: [ProjectedWaypoint]                     # uno por minuto
    estimated_duration_sec: float
    final_phase: FlightPhase

ProjectedWaypoint:
    latitude: float
    longitude: float
    altitude: int
    estimated_time: float                              # segundos desde el inicio
    speed: int
    vertical_rate: int

ProjectedSeparation:
    aircraft_1: String
    aircraft_2: String
    vertical_separation_ft: float
    horizontal_separation_nm: float
    time_to_conflict: float | null
    conflict_predicted: bool
```

### Alert y Violation

```
Alert:
    alert_id: String
    severity: AlertSeverity               # {INFO, LOW, MEDIUM, HIGH, CRITICAL}
    category: AlertCategory               # {ALTITUDE_VIOLATION, SEPARATION_LOSS, RUNWAY_CONFLICT, ...}
    affected_callsigns: [String]
    primary_callsign: String | null
    triggering_instruction_raw: String
    violations: [Violation]
    title: String
    explanation: String
    suggested_action: String
    projected_state: dict | null
    acknowledged: bool
    commit_decision: "PENDING" | "COMMIT" | "ROLLBACK"
    force_committed: bool

Violation:
    violation_id: String
    rule_id: String                      # "RULE001", "MSA_RULE", etc.
    condition_type: String               # "ALTITUDE_MINIMUM", "SEPARATION_VERTICAL", etc.
    severity: AlertSeverity
    details: dict                        # parámetros específicos de la violación
    explanation: String
    suggestion: String | null
    detected_at: datetime
```

### ExecutableRule y CompiledRule (Formato Intermedio)

```
ExecutableRule:                          # Puente entre KEX y RuleEngine
    source_rule_id: String               # ID en el KEX
    rule_category: String                # ALTITUDE, SEPARATION, RUNWAY, GENERIC, UNEVALUABLE
    parameters: dict | null              # parámetros estructurados
    condition_description: String | null # descripción textual
    required_state_fields: [String]      # campos de TrafficState necesarios
    raw_trigger: String | null
    raw_constraint: String | null
    severity: String | null
    safety_critical: bool
    rule_type: String | null             # prohibition, obligation, etc.
    modality: String | null              # shall, may, etc.

CompiledRule:                            # Regla compilada por LLM
    source_rule_id: String
    rule_category: String
    condition_description: String
    compiled_code: String                # código Python de evaluate()
    required_state_fields: [String]
    compilation_metadata: dict
    compilation_status: CompilationStatus  # COMPILED, FAILED, etc.
    raw_trigger: String | null
    raw_constraint: String | null
    severity: String | null
    safety_critical: bool
```

---

## Arquitectura Híbrida de Evaluación: Árbol de Decisión Completo

```
Para cada regla R_k en el conjunto de reglas:

  ¿R_k es de categoría conocida?
  │
  ├─ ALTITUDE ──────────────────────────→ AltitudeCondition.evaluate()
  │                                         evalúa contra TS'.aircrafts[cs].position.altitude
  │
  ├─ SEPARATION ─────────────────────────→ SeparationCondition.evaluate()
  │                                         evalúa contra separaciones pre-calculadas
  │
  ├─ RUNWAY ─────────────────────────────→ RunwayCondition.evaluate()
  │                                         evalúa contra TS'.runways[rw_id]
  │
  └─ GENERIC:
      │
      ¿Existe archivo compilado (.py) para R_k?
      │
      ├─ SÍ → CompiledCondition.evaluate()
      │        ├─ exec(compiled_code, safe_namespace)
      │        ├─ ¿Ejecución exitosa?
      │        │   ├─ Sí → retornar ConditionResult
      │        │   └─ No → FALLBACK a GenericKexCondition (LLM)
      │        └─ ¿Fallback disponible?
      │            ├─ Sí → LLM evalúa la regla
      │            └─ No → retornar error
      │
      └─ NO → GenericKexCondition.evaluate()
               ├─ ¿LLM disponible?
               │   ├─ Sí → LLM runtime evalúa con Instructor
               │   │         ¿Violación? → ConditionResult(satisfied=false)
               │   │         ¿Sin violación? → ConditionResult(satisfied=true)
               │   └─ No → FALLBACK a keywords
               └─ Keywords:
                     ¿"altitude" o "below" en descripción?
                       ├─ Sí → ¿a_cs.altitude < TS.msa?
                       │         ├─ Sí → violación
                       │         └─ No → satisfecho
                       └─ No → satisfecho (por defecto)
```

---

## Modelo Computacional como Sistema de Transición

El sistema completo puede modelarse como un autómata de estados finitos donde:

### Estado del Sistema en el instante `t`

```
Σ_t = (TS_t, Pending_t, Alerts_t, Txn_t)
```

donde:
- `TS_t`: TrafficState real en el instante `t`
- `Pending_t`: transacción pendiente (si existe)
- `Alerts_t`: alertas activas no resueltas
- `Txn_t`: historial de transacciones

### Transición de Estado

En cada instante `t`, llega una instrucción `I_t` y se ejecuta:

```
1. PI_t = π(I_t)                                    # Parseo
2. Proj_t = Π(TS_t, PI_t)                            # Proyección
3. V_t = ∪_{k} E_k(Proj_t, params_k, cs_t)           # Evaluación
4. A_t = gen_alert(V_t)                               # Generación de alertas
5. d_t = decision(A_t)                                # Decisión
6. TS_{t+1} = state_update(TS_t, Proj_t, d_t)        # Actualización
```

La función de transición de estado `δ`:

```
δ(Σ_t, I_t) = Σ_{t+1}
```

donde:

```
state_update(TS_t, Proj_t, d_t):
    if d_t = "COMMIT":
        return Proj_t.traffic_state   # el estado proyectado se vuelve real
    if d_t = "ROLLBACK":
        return TS_t                   # el estado real no cambia
```

### Propiedades del Sistema

1. **Atomicidad**: cada instrucción se procesa como una transacción. O bien
   se aplica completamente (COMMIT) o no se aplica en absoluto (ROLLBACK).
   No hay estados intermedios.

2. **Aislamiento (Snapshot Isolation)**: la proyección se hace sobre una copia
   del estado (`deepcopy`), lo que garantiza que el estado real nunca se
   modifica durante la evaluación. Solo el commit final modifica el estado.

3. **Determinismo**: dada la misma secuencia de instrucciones y el mismo estado
   inicial, el sistema produce exactamente la misma secuencia de estados y
   alertas (asumiendo que los evaluadores LLM son deterministas para la misma
   entrada, lo cual no es estrictamente cierto en la práctica).

4. **Monotonicidad de alertas**: una vez que una alerta se genera con severidad
   `s`, puede aumentar de severidad (por violaciones adicionales) pero nunca
   disminuir.

5. **Transparencia transaccional**: cada transacción registra su decisión,
   timestamp, y razón, permitiendo auditoría completa de todas las decisiones
   del sistema.

---

## Observaciones sobre el Modelo

1. **Separación de líneas temporales**: la compilación offline (Proceso 1) y
   el pipeline online (Procesos 2-5) están completamente desacoplados. Las
   reglas se compilan una vez y se reutilizan en todas las evaluaciones
   posteriores. Esto permite que la compilación use modelos LLM costosos sin
   afectar la latencia del pipeline online.

2. **Arquitectura híbrida LLM + código**: es el punto más distintivo del
   sistema. Combina 4 mecanismos de evaluación (nativo, compilado, LLM runtime,
   keywords) con prioridades decrecientes. Esto maximiza la cobertura de reglas
   (incluso reglas subjetivas pueden evaluarse aproximadamente) mientras
   minimiza llamadas LLM costosas (las reglas nativas y compiladas son
   esencialmente instantáneas).

3. **Seguridad del `exec()`**: el mayor riesgo del sistema es la ejecución de
   código generado por LLM. Las defensas son:
   - Namespace restringido (solo modelos Pydantic, `math`, builtins limitados).
   - Validación estática por AST antes de ejecutar (sintaxis, imports,
     nombres prohibidos, estructura de retorno).
   - Prueba sintética con datos de test antes de aceptar la regla.
   - Fallback a LLM runtime si la ejecución compilada falla.

4. **Costo computacional del LLM runtime**: el `GenericKexCondition` con LLM
   es el cuello de botella del sistema. Cada regla genérica requiere una
   llamada LLM por instrucción. Con ~100 reglas genéricas y ~1000 instrucciones
   por hora, el costo es prohibitivo. El `RuleFilter` (keywords → embeddings →
   LLM batch) es esencial para reducir el número real de evaluaciones LLM.

5. **Limitaciones del modelo de proyección**:
   - La trayectoria asume línea recta y velocidad/tasa vertical constantes
     (sin considerar maniobras, virajes, o restricciones ATC intermedias).
   - Las aeronaves vecinas son estáticas en la proyección (no se simula su
     movimiento concurrente).
   - No considera el viento, temperatura, o performance específica de cada
     tipo de aeronave.

6. **Compensaciones conocidas**:
   - **Velocidad vs cobertura**: las reglas nativas y compiladas son rápidas
     pero cubren solo reglas predecibles. Las genéricas con LLM cubren cualquier
     regla pero son lentas.
   - **Precisión vs latencia**: el RuleFilter reduce latencia pero puede
     descartar reglas relevantes (falsos negativos en el filtrado).
   - **Seguridad vs flexibilidad**: el namespace restringido del `exec()`
     limita lo que el código compilado puede hacer, lo que es seguro pero
     también restrictivo.

7. **Extensibilidad**: el diseño basado en evaluadores (`ConditionEvaluator`)
   permite agregar nuevos tipos de condiciones implementando la interfaz
   abstracta y registrándolos en el `RuleEngine`. No requiere modificar el
   pipeline ni el `StateManager`.

8. **Auditabilidad**: cada paso del pipeline registra su resultado,
   incluyendo las violaciones detectadas, las alertas generadas, y la decisión
   final con timestamp. El `StateManager` mantiene un historial completo de
   transacciones y estados anteriores, permitiendo reconstruir la secuencia
   completa de decisiones.
