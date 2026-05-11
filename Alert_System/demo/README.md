# ATC Alert System - Demo CLI

CLI interactivo (REPL) para simular instrucciones ATC contra el motor de reglas del Alert System.

## Uso

```bash
# Desde el root del proyecto:
python -m Alert_System.demo.demo_cli

# Con estado inicial desde JSON:
python -m Alert_System.demo.demo_cli --state Alert_System/demo/config/initial_state.json
```

## Comandos principales

### Estado
- `load <path>` — carga estado desde JSON
- `save <path>` — guarda estado actual a JSON
- `show state` — muestra estado completo del tráfico
- `show aircrafts` — lista aeronaves
- `show runways` — lista pistas
- `show config` — configuración actual

### CRUD de entidades
- `add aircraft <callsign> --altitude <ft> --heading <deg> --speed <kts> --phase <phase>`
- `add runway <id> --occupied [true|false] --mode [landing|takeoff|mixed|closed]`
- `update aircraft <callsign> --altitude <ft> ...`
- `remove aircraft <callsign>` / `remove runway <id>`

### Simulación de instrucciones
- `instr "AAL123 climb to 5000"` — parsea y ejecuta instrucción
- `manual` — ingresa instrucción campo por campo

### Configuración de reglas
- `set compiled on|off` — activa/desactiva reglas compiladas del KEX
- `set generic on|off` — activa/desactiva reglas genéricas (LLM)

### Transacciones
- `commit` — aplica cambio pendiente
- `rollback` — rechaza cambio pendiente
- `undo` — deshace último commit

## Ejemplo de sesión

```
atc-demo> load Alert_System/demo/config/initial_state.json
[✓] Estado cargado desde: Alert_System/demo/config/initial_state.json
    Sector: TEST_SECTOR_01, Aeronaves: 3, Pistas: 2

atc-demo> show state
--------------------------------------------------
  SECTOR: TEST_SECTOR_01
  MSA: 5000 ft
  QNH: 1013 hPa
  ...

atc-demo> instr "AAL123 descend to 4000"
[>] Parseando: 'AAL123 descend to 4000'
[i] Callsign: AAL123, Tipo: descent, Acción: descend
[i] Parámetros: {'target_altitude': 4000}

[>] Ejecutando pipeline...
--------------------------------------------------
  ESTADO PROYECTADO:
    AAL123: ALT=4000 HDG=90 SPD=250
--------------------------------------------------

  ALERTAS GENERADAS: 1
    🔴 [CRITICAL] altitude_violation
       Altitude 4000ft below MSA 5000ft
       Sugerencia: Review instruction

  Decisión automática del sistema: ROLLBACK
  Tiempo de ejecución: 12.3 ms
```

## Estado inicial JSON

El archivo `config/initial_state.json` define el estado inicial con:
- `sector_id`, `msa`, `qnh`, `wind`
- `aircrafts`: dict de `AircraftState`
- `runways`: dict de `RunwayState`

Puedes editarlo o crear tu propio JSON para escenarios custom.
