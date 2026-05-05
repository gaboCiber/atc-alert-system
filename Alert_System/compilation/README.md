# LLM Rule Compiler

Compilación offline de reglas KEX genéricas a código Python evaluador usando LLM.

## Arquitectura

```
KEX Rule → ExecutableRule → [LLM Compiler] → CompiledRule (.py) → CompiledCondition → RuleEngine
                                      ↓ (falla)
                              GenericKexCondition (LLM runtime)
```

## Flujo

1. **Compilación (offline)**: Script CLI toma reglas JSON, las envía al LLM, genera funciones `evaluate()` y las guarda en archivos `.py`
2. **Carga (startup)**: Sistema carga archivos compilados, los envuelve en `CompiledCondition`, y los registra en `RuleEngine`
3. **Evaluación (runtime)**: Condiciones compiladas se ejecutan como código Python nativo — tan rápido como `AltitudeCondition`
4. **Fallback**: Si la ejecución falla, usa `GenericKexCondition` con LLM runtime

## Uso

### Compilar reglas

```bash
# Compilar con modelo default
uv run python -m Alert_System.compilation.cli sample_rules.json

# Compilar con modelo específico
uv run python -m Alert_System.compilation.cli sample_rules.json --model llama3.2:latest

# Forzar re-compilación
uv run python -m Alert_System.compilation.cli sample_rules.json --force
```

### En código

```python
from Alert_System.compilation.compiler import RuleCompiler
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.integration.kex_adapter import KEXAdapter

# Configurar LLM
from common.llm_client_factory import ModelConfig
llm_config = ModelConfig(name="llama3.2:latest", provider="ollama", base_url="http://localhost:11434")

# Compilar reglas
compiler = RuleCompiler(llm_config=llm_config)
manifest = compiler.compile_batch(executable_rules)

# Guardar reglas compiladas
loader = CompiledRuleLoader(llm_config=llm_config)
saved = loader.save_all(manifest)

# Cargar y registrar en RuleEngine
loader.load_all_compiled_rules()
conditions = loader.create_compiled_conditions()
for condition in conditions:
    rule_engine.add_rule(condition)
```

## Seguridad

- **Validación estática**: AST checking para imports prohibidos, nombres peligrosos
- **Namespace restringido**: Solo `math` y tipos básicos permitidos
- **Timeout**: Ejecución con timeout para evitar loops infinitos
- **Fallback**: Si el código compilado falla, usa LLM runtime automáticamente

## Estructura de archivos

```
Alert_System/
├── compilation/
│   ├── schemas.py          # CompiledRule, CompilationManifest
│   ├── prompts.py          # Prompts para LLM
│   ├── validator.py        # Validación estática de código
│   ├── compiler.py         # RuleCompiler (LLM → código)
│   ├── loader.py           # Carga reglas compiladas
│   └── cli.py              # CLI para compilación offline
├── compiled_rules/         # Directorio de reglas compiladas
│   ├── manifest.json       # Metadata de todas las reglas
│   ├── RULE_ALT_001.py     # Regla compilada individual
│   └── ...
└── rule_engine/
    └── conditions.py        # CompiledCondition wrapper
```

## Formato de regla compilada

Cada regla compilada genera una función `evaluate(traffic_state, callsign=None)`:

```python
def evaluate(traffic_state, callsign=None):
    """
    Evalúa la regla contra el estado del tráfico.
    
    Returns:
        dict: {
            "satisfied": bool,     # True si no hay violación
            "details": dict,       # Valores relevantes
            "explanation": str,    # Explicación del resultado
            "severity": str        # INFO, LOW, MEDIUM, HIGH, CRITICAL
        }
    """
    aircraft = traffic_state.get_aircraft(callsign)
    if not aircraft:
        return {"satisfied": True, "details": {}, "explanation": "Aircraft not found", "severity": "INFO"}
    
    msa = traffic_state.msa or 5000
    if aircraft.position.altitude < msa:
        return {
            "satisfied": False,
            "details": {"altitude": aircraft.position.altitude, "msa": msa},
            "explanation": f"Aircraft below MSA",
            "severity": "HIGH"
        }
    
    return {"satisfied": True, "details": {"altitude": aircraft.position.altitude}, "explanation": "OK", "severity": "INFO"}
```

## Tests

```bash
# Ejecutar todos los tests de compilación
uv run python -m pytest tests/alert_system/test_compilation.py -v

# Tests específicos
uv run python -m pytest tests/alert_system/test_compilation.py::TestCodeValidator -v
uv run python -m pytest tests/alert_system/test_compilation.py::TestCompiledCondition -v
```

## Prioridad de evaluación

El `KEXAdapter` sigue esta prioridad:

1. **Condiciones específicas** (ALTITUDE, SEPARATION, RUNWAY) → evaluadores nativos
2. **Reglas compiladas** → `CompiledCondition` (código Python nativo)
3. **Reglas genéricas sin compilar** → `GenericKexCondition` (LLM runtime)

## Ventajas

- **Rendimiento**: Código compilado se ejecuta tan rápido como condiciones nativas
- **Seguridad**: Validación estática y sandboxing del código generado
- **Flexibilidad**: LLM puede generar lógica compleja que sería difícil codificar manualmente
- **Robustez**: Fallback automático a LLM runtime si algo falla
- **Versionado**: Reglas compiladas se commitean como código del proyecto
