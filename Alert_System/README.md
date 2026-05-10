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

### 2. Compilation System (`compilation/`)

**Sistema de compilación de reglas KEX a código Python evaluador usando LLM con Instructor.**

**RuleCompiler**: Compilador principal que utiliza LLM para generar código Python evaluador de reglas ATC.

**Características Principales:**
- **Migración a Instructor**: Usa Instructor para manejo automático de reintentos y validación JSON schema
- **Validación Integrada**: Los modelos Pydantic incluyen validadores para sintaxis, imports prohibidos, estructura de retorno
- **Clasificación Inteligente**: Determina si una regla es compilable con TrafficState o requiere juicio subjetivo
- **Generación de Código**: Produce funciones `evaluate()` que operan sobre objetos TrafficState
- **Testing Automático**: Ejecuta código generado con TrafficState de prueba antes de aceptarlo

**Flujo de Compilación:**
1. **Clasificación**: Determina si la regla es objetivamente evaluablen con datos TrafficState
2. **Generación con Instructor**: LLM genera código Python con validación automática via Pydantic
3. **Validación Pydantic**: Verificación automática de sintaxis, imports prohibidos, estructura de retorno
4. **Testing**: Ejecución del código generado con estado de prueba
5. **Guardado**: Almacenamiento de reglas compiladas con metadata completa

**Modelos Pydantic:**
- `ClassificationResponse`: Respuesta estructurada para clasificación de reglas
- `GeneratedCodeResponse`: Respuesta básica para generación de código
- `ValidatedCodeResponse`: Respuesta con validación completa integrada (sintaxis, imports, estructura)
- `CompiledRule`: Regla compilada lista para ejecución
- `CompilationManifest`: Manifiesto de lote de compilación

**Archivos:**
- `compiler.py`: RuleCompiler principal con migración a Instructor
- `schemas.py`: Modelos Pydantic para respuestas estructuradas
- `validator.py`: Validación estática de código (legacy, parcialmente reemplazado por Pydantic)
- `prompts.py`: Prompts optimizados para salida estructurada
- `loader.py`: Cargador de reglas compiladas
- `kex_data_processor.py`: Procesamiento de datos KEX con resolución de entidades
- `compile_rules_cli.py`: CLI para compilación por lotes

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

### Ejemplo: Compilación de Reglas KEX

```bash
# Compilar todas las reglas con configuración por defecto
uv run python Alert_System/compile_rules_cli.py "Knowledge_Extractor/output/kex_data"

# Compilar máximo 5 reglas con modelo específico
uv run python Alert_System/compile_rules_cli.py "Knowledge_Extractor/output/kex_data" \
  --max-rules 5 \
  --model llama3.2 \
  --provider ollama \
  --output compiled_rules

# Compilar con proveedor OpenAI
uv run python Alert_System/compile_rules_cli.py "Knowledge_Extractor/output/kex_data" \
  --model gpt-4 \
  --provider openai \
  --api-key tu-api-key
```

## CLI de Compilación

### `compile_rules_cli.py`

**Herramienta de línea de comandos para compilación por lotes de reglas KEX a código Python evaluador.**

**Uso:**
```bash
compile_rules_cli.py [-h] [--output OUTPUT] [--max-rules MAX_RULES] \
  [--model MODEL] [--provider PROVIDER] input_dir
```

**Argumentos:**
- `input_dir`: Carpeta de entrada con archivos KEX (obligatorio)

**Opciones:**
- `-h, --help`: Muestra ayuda y sale
- `-o OUTPUT, --output OUTPUT`: Carpeta de salida para reglas compiladas
- `-n MAX_RULES, --max-rules MAX_RULES`: Número máximo de reglas a procesar (default: todas)
- `-m MODEL, --model MODEL`: Modelo LLM a usar (default: llama3.2)
- `-p PROVIDER, --provider PROVIDER`: Proveedor LLM (ollama, openai, etc.)

**Características:**
- **Resolución de Entidades**: Convierte IDs de entidades (E001, R002) a sus descripciones
- **Clasificación Automática**: Determina si reglas son compilables con TrafficState
- **Guardado Incremental**: Guarda cada regla exitosa inmediatamente
- **Manifiesto**: Genera JSON con estadísticas de compilación
- **Testing**: Ejecuta prueba básica de reglas compiladas

**Ejemplo de Salida:**
```
🚀 Compilando reglas KEX...
📁 Entrada: Knowledge_Extractor/output/data
📁 Salida: compiled_rules
🤖 Modelo: llama3.2 (ollama)
📊 Máximo reglas: 5

🔨 Compilando 5 reglas con guardado incremental...
✅ RULE001: ALTITUDE - compiled successfully
✅ RULE002: SEPARATION - compiled successfully
🚫 RULE003: not_compilable - requiere juicio subjetivo

📊 Resumen:
  Compiled: 2
  Not compilable: 1
  Failed: 0
  Total: 3
```

**Integración con RuleEngine:**
Las reglas compiladas pueden ser cargadas automáticamente en el RuleEngine:

```python
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.rule_engine import RuleEngine

# Cargar reglas compiladas
loader = CompiledRuleLoader(compiled_rules_dir="compiled_rules")
manifest = loader.load_manifest()

# Registrar en RuleEngine
engine = RuleEngine()
for rule_id, compiled_rule in manifest.rules.items():
    if compiled_rule.compilation_status == CompilationStatus.COMPILED:
        engine.register_evaluator(rule_id, compiled_rule.compiled_code)
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

### Cambios Recientes Implementados

✅ **Migración a Instructor (2025)**
- Reemplazo de cliente raw por Instructor con manejo automático de reintentos
- Validación integrada via modelos Pydantic con validadores automáticos
- Eliminación de lógica manual de reintentos y parsing JSON
- Mejor manejo de errores y logging

✅ **CLI de Compilación (2025)**
- Herramienta completa para compilación por lotes
- Resolución automática de entidades a descripciones
- Guardado incremental y manifiestos
- Soporte para múltiples proveedores LLM

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

4. **Mejoras en Compilation System**
   - Testing más exhaustivo de código generado
   - Métricas de calidad de compilación
   - Optimización de prompts para mejores resultados

## Contacto

Para preguntas o contribuciones, contactar al equipo de desarrollo.
