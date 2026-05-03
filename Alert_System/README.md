# Alert System - Sistema de Alertas ATC

Sistema de detección de violaciones de seguridad en tiempo real para control de tráfico aéreo (ATC). Evalúa instrucciones ATC contra reglas de seguridad y genera alertas cuando se detectan violaciones.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         Alert System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Pipeline   │───▶│ Rule Engine  │───▶│   Alerts     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Core        │    │ Conditions   │    │  Models      │      │
│  │  - State     │    │  - Altitude  │    │  - Violation │      │
│  │  - Projection│    │  - Separation│    │  - Alert     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────┐                            │
│                    │ Integration  │                            │
│                    │  - KEX       │                            │
│                    │  - ASR       │                            │
│                    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Componentes

### 1. Pipeline (`pipeline/`)

**AlertPipeline**: Orquestador principal del sistema.

**Flujo de procesamiento:**
1. **Input Processing**: Recibe instrucción ATC (texto o pre-parsed)
2. **State Projection**: Simula estado futuro si se aplica la instrucción
3. **Rule Evaluation**: Evalúa reglas de seguridad contra estado proyectado
4. **Alert Generation**: Genera alertas si hay violaciones
5. **ATCO Decision**: Determina si se debe alertar al controlador
6. **Commit/Rollback**: Actualiza estado o revierte cambios

**Archivos:**
- `pipeline.py`: AlertPipeline principal
- `pipeline_state.py`: Estado del pipeline (commits, rollbacks)

### 2. Core (`core/`)

**StateProjection**: Simulación "what-if" de instrucciones ATC.

**Clases:**
- `ProjectedState`: Estado del tráfico después de aplicar instrucción
- `StateProjector`: Crea proyecciones aplicando instrucciones
- `ProjectedTrajectory`: Trayectoria futura de una aeronave
- `ProjectedSeparation`: Separación proyectada entre aeronaves

**Archivos:**
- `state_projection.py`: Proyección de estado
- `instruction_parser.py`: Parsing de instrucciones ATC

### 3. Rule Engine (`rule_engine/`)

**RuleEngine**: Motor de evaluación de reglas de seguridad.

**Evaluadores de Condiciones:**
- `AltitudeCondition`: Verifica límites de altitud (MSA, mínimos, máximos)
- `SeparationCondition`: Verifica separación vertical/horizontal entre aeronaves
- `RunwayCondition`: Verifica disponibilidad de pistas
- `GenericKexCondition`: Evaluador genérico para reglas arbitrarias del KEX

**Archivos:**
- `rule_engine.py`: Motor de reglas principal
- `conditions.py`: Implementaciones de condiciones evaluables

### 4. Models (`models/`)

**Modelos de datos:**

**Instrucciones:**
- `ParsedInstruction`: Instrucción ATC parseada con parámetros
- `InstructionType`: Enum de tipos (CLIMB, DESCENT, TURN, etc.)
- `Speaker`: Enum de hablantes (ATCO, PILOT)

**Estado de Tráfico:**
- `TrafficState`: Estado completo del sector
- `AircraftState`: Estado de una aeronave
- `RunwayState`: Estado de una pista
- `Position`: Posición geográfica

**Alertas:**
- `Alert`: Alerta generada
- `Violation`: Violación de regla detectada
- `AlertSeverity`: Severidad (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- `AlertCategory`: Categoría (ALTITUDE, SEPARATION, RUNWAY, etc.)

**Archivos:**
- `instruction.py`: Modelos de instrucciones
- `traffic_state.py`: Modelos de estado de tráfico
- `alert.py`: Modelos de alertas

### 5. Integration (`integration/`)

**Adaptadores para integración con sistemas externos:**

**KEXAdapter**: Convierte reglas del Knowledge Extractor a evaluadores.
- `compile_to_executable()`: Transforma Rule KEX → ExecutableRule
- `_categorize_rule()`: Detecta tipo de regla (ALTITUDE, SEPARATION, RUNWAY, GENERIC, UNEVALUABLE)
- `_adapt_executable_rule()`: Convierte ExecutableRule → ConditionEvaluator

**ASRAdapter**: Convierte transcripciones de voz a instrucciones.
- `transcribe_to_instruction()`: Texto → ParsedInstruction

**Normalizadores:**
- `atc_compact_normalizer`: Normaliza frases ATC compactas

**Archivos:**
- `kex_adapter.py`: Adaptador KEX
- `asr_adapter.py`: Adaptador ASR
- `schemas.py`: ExecutableRule (formato intermedio)
- `end_to_end_pipeline.py`: Pipeline completo integrado

### 6. Config (`config/`)

**Configuración del sistema:**

- `rule_patterns.json`: Patrones de mapeo de reglas KEX a categorías ejecutables
  - 12 patrones predefinidos (ALTITUDE, SEPARATION, RUNWAY, SPEED, WEATHER, etc.)
  - Extensible sin modificar código

## Flujo de Trabajo

### Ejemplo: Instrucción de Descenso

```python
from Alert_System.pipeline.pipeline import AlertPipeline
from Alert_System.models.instruction import ParsedInstruction

# 1. Crear pipeline
pipeline = AlertPipeline(sector_id="KJFK", msa=5000)

# 2. Procesar instrucción
instruction = ParsedInstruction(
    raw_text="AAL123 descend to 4000",
    normalized_text="AAL123 descend to 4000",
    speaker=Speaker.ATCO,
    callsign="AAL123",
    instruction_type=InstructionType.DESCENT,
    action_verb="descend",
    parameters={"target_altitude": 4000},
)

# 3. Evaluar
result = pipeline.process_instruction(instruction)

# 4. Revisar alertas
if result.alerts:
    for alert in result.alerts:
        print(f"⚠️ {alert.severity}: {alert.message}")
```

### Ejemplo: Integración con KEX

```python
from Knowledge_Extractor import Rule
from Alert_System.integration.kex_adapter import KEXAdapter

# 1. Cargar reglas del KEX
kex_rules = [...]  # Reglas extraídas por KEX

# 2. Crear adaptador
adapter = KEXAdapter()

# 3. Compilar a formato ejecutable
executable_rules = [adapter.compile_to_executable(r) for r in kex_rules]

# 4. Adaptar a evaluadores
evaluators = [adapter._adapt_executable_rule(e) for e in executable_rules]

# 5. Registrar en RuleEngine
rule_engine.register_evaluator("ALTITUDE", evaluators[0])
```

## Arquitectura Híbrida KEX-RuleEngine

El sistema soporta reglas conocidas y reglas arbitrarias:

### Reglas Conocidas
- **ALTITUDE**: Mínimos de altitud, MSA, restricciones de flight level
- **SEPARATION**: Separación vertical/horizontal entre aeronaves
- **RUNWAY**: Disponibilidad de pistas, ocupación

### Reglas Genéricas
- **GENERIC**: Reglas arbitrarias que no encajan en categorías predefinidas
- **UNEVALUABLE**: Reglas que requieren juicio humano o datos externos

### Flujo Híbrido

```
KEX Rule (texto estructurado)
    ↓
KEXAdapter.compile_to_executable()
    ↓
ExecutableRule (categorizada)
    ├─ ALTITUDE → AltitudeCondition
    ├─ SEPARATION → SeparationCondition
    ├─ RUNWAY → RunwayCondition
    ├─ GENERIC → GenericKexCondition
    └─ UNEVALUABLE → None (ignorada)
    ↓
RuleEngine.evaluate()
    ↓
Violations
```

## Tests

**Ubicación**: `tests/alert_system/test_pipeline.py`

**Suites de tests:**
- `TestPipelineResult`: Resultados del pipeline
- `TestAlertPipeline`: Pipeline principal
- `TestSimpleATCParser`: Parser de instrucciones
- `TestAltitudeRuleEvaluation`: Reglas de altitud
- `TestSeparationRuleEvaluation`: Reglas de separación
- `TestRunwayRuleEvaluation`: Reglas de pista
- `TestMultipleRuleEvaluation`: Múltiples reglas
- `TestEndToEndScenarios`: Escenarios end-to-end (7 tests)
- `TestKEXAdapterHybrid`: Integración hibrida KEX (12 tests)

**Ejecutar tests:**
```bash
uv run python -m pytest tests/alert_system/test_pipeline.py -v
```

## Configuración

### Variables de Entorno
No requeridas actualmente.

### Archivos de Configuración
- `config/rule_patterns.json`: Patrones de reglas KEX

## Dependencias

Ver `requirements.txt` en el proyecto principal.

## Futuras Mejoras

### Decisiones Pendientes

1. **LLM en GenericKexCondition**
   - ¿Llamada síncrona o asíncrona?
   - ¿Cache de interpretaciones LLM?
   - ¿Evaluación estricta o fuzzy?

2. **Extensibilidad**
   - API runtime para registrar nuevos patrones
   - Plugins para tipos de reglas custom

3. **Performance**
   - Caching de proyecciones
   - Evaluación paralela de reglas
   - Optimización de StateProjector

## Contacto

Para preguntas o contribuciones, contactar al equipo de desarrollo.
