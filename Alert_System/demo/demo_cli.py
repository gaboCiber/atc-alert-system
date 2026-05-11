#!/usr/bin/env python3
"""CLI interactivo de demo para el Alert System ATC.

Permite:
  - Cargar estado inicial desde JSON
  - Añadir/modificar/eliminar aeronaves y pistas
  - Simular instrucciones ATC (parser simple o manual)
  - Evaluar contra reglas y ver alertas generadas
  - Decidir COMMIT / ROLLBACK de cada instrucción

Uso:
  python -m Alert_System.demo.demo_cli
  python -m Alert_System.demo.demo_cli --state config/initial_state.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Asegurar que el root del proyecto esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Alert_System.core.state_manager import StateManager
from Alert_System.core.state_projection import StateProjector
from Alert_System.models.alert import AlertSeverity
from Alert_System.models.instruction import InstructionType, ParsedInstruction, Speaker
from Alert_System.models.traffic_state import (
    AircraftState,
    Clearances,
    FlightPhase,
    Position,
    RunwayOperationMode,
    RunwayState,
    TrafficState,
    WakeTurbulenceCategory,
)
from Alert_System.pipeline.alert_pipeline import AlertPipeline
from Alert_System.rule_engine.engine import RuleEngine

from Alert_System.demo.simple_parser import SimpleATCParser
from Alert_System.demo.state_loader import TrafficStateLoader
from common.llm_client_factory import ModelConfig

# Try importing KEX integration, gracefully handle missing
HAS_KEX = False
try:
    from Alert_System.integration.kex_adapter import KEXAdapter
    from Knowledge_Extractor import Rule

    HAS_KEX = True
except Exception:
    pass


def print_banner():
    print("=" * 70)
    print("   ATC Alert System - Interactive Demo CLI")
    print("   Simulador de instrucciones ATC con motor de reglas")
    print("=" * 70)
    print()


def print_help():
    print("""
Comandos disponibles:

  Estado y configuración:
    load <path>              Carga estado inicial desde JSON
    save <path>              Guarda estado actual a JSON
    show state               Muestra estado actual del tráfico
    show aircrafts           Lista todas las aeronaves
    show runways             Lista todas las pistas
    show config              Muestra configuración actual

  Gestión de estado:
    add aircraft <callsign> --altitude <ft> --heading <deg> --speed <kts> ...
    add runway <id> --occupied [true|false] --mode [landing|takeoff|mixed|closed]
    update aircraft <callsign> --altitude <ft> ...
    remove aircraft <callsign>
    remove runway <id>

  Simulación de instrucciones:
    instr <texto>            Parsea texto ATC y simula (ej: 'AAL123 climb to 5000')
    manual                   Modo manual: ingresa campo por campo
    exec <texto>             Alias de 'instr'

  Configuración de reglas:
    set compiled on|off      Activa/desactiva reglas compiladas del KEX
    set generic on|off       Activa/desactiva reglas genéricas (LLM)
    set llm <modelo>         Configura modelo LLM (default: llama3.2:latest)
    set timeout <seg|none>    Tiempo máximo prefiltrado (none=ilimitado, default: none)
    set topk <n>             Máx reglas tras embeddings (default: 30)
    set verbose on|off       Muestra debug del prefiltrado (default: off)
    load rules <path>        Carga reglas desde archivo JSON

  Transacciones:
    commit                   Aplica cambio pendiente
    rollback                 Rechaza cambio pendiente
    undo                     Deshace último commit

  Otros:
    help                     Muestra esta ayuda
    clear                    Limpia la pantalla
    exit / quit              Salir del demo

Ejemplos:
  > load config/initial_state.json
  > add aircraft JBU456 --altitude 8000 --heading 180 --speed 220 --phase approach
  > instr "AAL123 descend to 4000"
  > manual
  > show state
""")


class DemoCLI:
    """CLI interactivo REPL para demo del Alert System."""

    def __init__(self, initial_state_path: Optional[str] = None):
        self.parser = SimpleATCParser()
        self.state_manager: Optional[StateManager] = None
        self.pipeline: Optional[AlertPipeline] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.state_projector = StateProjector()

        # Configuración de reglas
        self.use_compiled = False
        self.use_generic = False
        self.llm_model = os.environ.get("ATC_LLM_MODEL", "llama3.2:latest")
        self.llm_config = ModelConfig(
            name=self.llm_model,
            provider="ollama",
            base_url="http://localhost:11434",
            api_key="ollama",
            max_retries=2,
            timeout=30,
        )
        self.filter_timeout: Optional[float] = None  # None = sin timeout
        self.filter_top_k = 30
        self.rule_filter = None
        self.verbose = False
        self.loaded_rules: List[Any] = []

        # Historial de resultados
        self.last_result = None

        # Inicializar con estado por defecto o desde archivo
        if initial_state_path:
            self._load_state(initial_state_path)
        else:
            self._init_default_state()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _init_default_state(self):
        """Inicializa con un estado vacío por defecto."""
        state = TrafficState(sector_id="DEMO")
        self.state_manager = StateManager(state)
        self.rule_engine = RuleEngine()
        self.pipeline = AlertPipeline(
            self.state_manager,
            self.rule_engine,
            llm_config=self.llm_config,
            rule_filter=self.rule_filter,
            filter_timeout=self.filter_timeout,
            filter_top_k=self.filter_top_k,
            verbose=self.verbose,
        )
        print("[i] Estado inicial vacío creado. Usa 'load <path>' para cargar uno.")

    def _load_state(self, path: str):
        """Carga estado desde JSON."""
        try:
            state = TrafficStateLoader.from_file(path)
            self.state_manager = StateManager(state)
            self.rule_engine = RuleEngine()
            self.pipeline = AlertPipeline(
                self.state_manager,
                self.rule_engine,
                llm_config=self.llm_config,
                rule_filter=self.rule_filter,
                filter_timeout=self.filter_timeout,
                filter_top_k=self.filter_top_k,
                verbose=self.verbose,
            )
            print(f"[✓] Estado cargado desde: {path}")
            print(f"    Sector: {state.sector_id}, Aeronaves: {len(state.aircrafts)}, Pistas: {len(state.runways)}")
        except Exception as e:
            print(f"[✗] Error cargando estado: {e}")
            self._init_default_state()

    # ------------------------------------------------------------------
    # Loop principal REPL
    # ------------------------------------------------------------------

    def run(self):
        print_banner()
        print("Escribe 'help' para ver los comandos disponibles.\n")

        while True:
            try:
                raw = input("atc-demo> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[+] Saliendo...")
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("exit", "quit", "q"):
                print("[+] Saliendo...")
                break

            elif cmd == "help" or cmd == "h":
                print_help()

            elif cmd == "clear":
                os.system("clear" if os.name != "nt" else "cls")

            elif cmd == "load":
                self._cmd_load(args)

            elif cmd == "save":
                self._cmd_save(args)

            elif cmd in ("show", "s"):
                self._cmd_show(args)

            elif cmd == "add":
                self._cmd_add(args)

            elif cmd == "update":
                self._cmd_update(args)

            elif cmd == "remove" or cmd == "rm" or cmd == "del":
                self._cmd_remove(args)

            elif cmd in ("instr", "exec", "i", "e"):
                self._cmd_instr(args, raw)

            elif cmd == "manual" or cmd == "m":
                self._cmd_manual()

            elif cmd == "set":
                self._cmd_set(args)

            elif cmd == "commit" or cmd == "c":
                self._cmd_commit()

            elif cmd == "rollback" or cmd == "r":
                self._cmd_rollback()

            elif cmd == "undo" or cmd == "u":
                self._cmd_undo()

            else:
                print(f"[✗] Comando desconocido: '{cmd}'. Escribe 'help' para ver opciones.")

    # ------------------------------------------------------------------
    # Comandos: Estado
    # ------------------------------------------------------------------

    def _cmd_load(self, args: List[str]):
        if not args:
            print("[✗] Uso: load <path>")
            return
        path = args[0]
        self._load_state(path)

    def _cmd_save(self, args: List[str]):
        if not self.state_manager:
            print("[✗] No hay estado cargado")
            return
        path = args[0] if args else "demo_state.json"
        try:
            loader = TrafficStateLoader()
            loader.save(self.state_manager.current_state, path)
            print(f"[✓] Estado guardado en: {path}")
        except Exception as e:
            print(f"[✗] Error guardando: {e}")

    def _cmd_show(self, args: List[str]):
        if not self.state_manager:
            print("[✗] No hay estado cargado")
            return

        sub = args[0].lower() if args else "state"
        state = self.state_manager.current_state

        if sub in ("state", "s"):
            print(f"\n{'─' * 50}")
            print(f"  SECTOR: {state.sector_id}")
            print(f"  MSA: {state.msa or 'N/A'} ft")
            print(f"  QNH: {state.qnh or 'N/A'} hPa")
            print(f"  Wind: {state.wind or 'N/A'}")
            print(f"  Timestamp: {state.timestamp}")
            print(f"{'─' * 50}")

            print(f"\n  AERONAVES ({len(state.aircrafts)}):")
            for cs, ac in state.aircrafts.items():
                pos = ac.position
                clr = ac.clearances
                clr_str = ""
                if clr.altitude_assigned:
                    clr_str += f" A:{clr.altitude_assigned}"
                if clr.heading_assigned:
                    clr_str += f" H:{clr.heading_assigned}"
                if clr.runway_assigned:
                    clr_str += f" R:{clr.runway_assigned}"
                print(
                    f"    {cs:8} | ALT:{pos.altitude:5} HDG:{pos.heading:3} SPD:{pos.speed:3} "
                    f"PH:{ac.flight_phase.value:<8} {clr_str}"
                )

            print(f"\n  PISTAS ({len(state.runways)}):")
            for rw_id, rw in state.runways.items():
                occ = f"OCC:{rw.occupied_by}" if rw.occupied else "FREE"
                print(
                    f"    {rw_id:5} | {occ:<12} MODE:{rw.operation_mode.value:<8} "
                    f"QUEUE:{len(rw.landing_queue)} HOLD:{len(rw.holding_short)}"
                )
            print()

        elif sub in ("aircrafts", "aircraft", "ac", "a"):
            print(f"\n  AERONAVES ({len(state.aircrafts)}):")
            for cs, ac in state.aircrafts.items():
                pos = ac.position
                print(
                    f"    {cs}: ALT={pos.altitude} HDG={pos.heading} SPD={pos.speed} "
                    f"LAT={pos.latitude:.3f} LON={pos.longitude:.3f} "
                    f"PHASE={ac.flight_phase.value} TYPE={ac.aircraft_type or 'N/A'}"
                )
            print()

        elif sub in ("runways", "runway", "rw", "r"):
            print(f"\n  PISTAS ({len(state.runways)}):")
            for rw_id, rw in state.runways.items():
                print(
                    f"    {rw_id}: OCC={rw.occupied} BY={rw.occupied_by or 'N/A'} "
                    f"MODE={rw.operation_mode.value} "
                    f"LAND_Q={rw.landing_queue} HOLD={rw.holding_short}"
                )
            print()

        elif sub in ("config", "cfg"):
            engine_rules = len(self.rule_engine._evaluator_instances) if self.rule_engine else 0
            print(f"\n  CONFIGURACIÓN:")
            print(f"    Reglas compiladas: {'ON' if self.use_compiled else 'OFF'}")
            print(f"    Reglas genéricas:  {'ON' if self.use_generic else 'OFF'}")
            print(f"    Reglas cargadas:   {engine_rules}")
            print(f"    Modelo LLM:        {self.llm_model}")
            timeout_str = f"{self.filter_timeout}s" if self.filter_timeout is not None else "Sin límite"
            print(f"    Timeout filtro:    {timeout_str}")
            print(f"    Top-K embeddings:  {self.filter_top_k}")
            print(f"    Verbose:           {'ON' if self.verbose else 'OFF'}")
            print(f"    Sector:            {state.sector_id}")
            if self.rule_engine:
                compiled = [k for k in self.rule_engine._evaluator_instances if k.startswith("COMPILED_")]
                generic = [k for k in self.rule_engine._evaluator_instances if k.startswith("GENERIC_")]
                if compiled:
                    print(f"    Compiled:          {', '.join(compiled)}")
                if generic:
                    print(f"    Genéricas:         {len(generic)} reglas")
            print()

        else:
            print(f"[✗] Subcomando desconocido: '{sub}'")

    # ------------------------------------------------------------------
    # Comandos: CRUD
    # ------------------------------------------------------------------

    def _cmd_add(self, args: List[str]):
        if len(args) < 2:
            print("[✗] Uso: add aircraft <callsign> [--altitude <ft>] [--heading <deg>] ...")
            print("       add runway <id> [--occupied true] [--mode mixed] ...")
            return

        entity_type = args[0].lower()
        entity_id = args[1]
        kwargs = self._parse_kwargs(args[2:])

        state = self.state_manager.current_state

        if entity_type in ("aircraft", "ac"):
            ac = self._build_aircraft_from_kwargs(entity_id, kwargs)
            state.add_aircraft(ac)
            print(f"[✓] Aeronave {entity_id} añadida")

        elif entity_type in ("runway", "rw"):
            rw = self._build_runway_from_kwargs(entity_id, kwargs)
            state.add_runway(rw)
            print(f"[✓] Pista {entity_id} añadida")

        else:
            print(f"[✗] Tipo desconocido: '{entity_type}'")

    def _cmd_update(self, args: List[str]):
        if len(args) < 2:
            print("[✗] Uso: update aircraft <callsign> [--altitude <ft>] ...")
            return

        entity_type = args[0].lower()
        entity_id = args[1]
        kwargs = self._parse_kwargs(args[2:])
        state = self.state_manager.current_state

        if entity_type in ("aircraft", "ac"):
            ac = state.get_aircraft(entity_id)
            if not ac:
                print(f"[✗] Aeronave '{entity_id}' no encontrada")
                return
            self._update_aircraft(ac, kwargs)
            print(f"[✓] Aeronave {entity_id} actualizada")

        elif entity_type in ("runway", "rw"):
            rw = state.get_runway(entity_id)
            if not rw:
                print(f"[✗] Pista '{entity_id}' no encontrada")
                return
            self._update_runway(rw, kwargs)
            print(f"[✓] Pista {entity_id} actualizada")

        else:
            print(f"[✗] Tipo desconocido: '{entity_type}'")

    def _cmd_remove(self, args: List[str]):
        if len(args) < 2:
            print("[✗] Uso: remove aircraft <callsign> | remove runway <id>")
            return

        entity_type = args[0].lower()
        entity_id = args[1]
        state = self.state_manager.current_state

        if entity_type in ("aircraft", "ac"):
            if state.get_aircraft(entity_id):
                state.remove_aircraft(entity_id)
                print(f"[✓] Aeronave {entity_id} eliminada")
            else:
                print(f"[✗] Aeronave '{entity_id}' no encontrada")

        elif entity_type in ("runway", "rw"):
            if state.get_runway(entity_id):
                state.remove_runway(entity_id)
                print(f"[✓] Pista {entity_id} eliminada")
            else:
                print(f"[✗] Pista '{entity_id}' no encontrada")

        else:
            print(f"[✗] Tipo desconocido: '{entity_type}'")

    # ------------------------------------------------------------------
    # Comandos: Simulación
    # ------------------------------------------------------------------

    def _cmd_instr(self, args: List[str], raw: str):
        """Ejecuta una instrucción ATC parseada automáticamente."""
        if not self.pipeline:
            print("[✗] Pipeline no inicializado")
            return

        # Extraer el texto después del comando (todo después del primer espacio)
        split_raw = raw.split(None, 1)
        text = split_raw[1] if len(split_raw) > 1 else ""
        if not text:
            print("[✗] Uso: instr <texto de instrucción ATC>")
            print("     Ej: instr 'AAL123 climb to 5000'")
            return

        # Quitar comillas si existen
        text = text.strip("'\"")

        print(f"\n[>] Parseando: '{text}'")
        parsed = self.parser.parse(text)

        if not parsed.callsign:
            print("[✗] No se pudo detectar callsign en la instrucción")
            return

        print(f"[i] Callsign: {parsed.callsign}, Tipo: {parsed.instruction_type.value}, Acción: {parsed.action_verb}")
        if parsed.parameters:
            print(f"[i] Parámetros: {parsed.parameters}")

        self._execute_instruction(text, parsed)

    def _cmd_manual(self):
        """Modo manual: ingresa campo por campo."""
        if not self.pipeline:
            print("[✗] Pipeline no inicializado")
            return

        print("\n[ Modo Manual - Instrucción ATC ]")
        print("(Deja en blanco para usar defaults)\n")

        callsign = input("  Callsign: ").strip().upper() or None
        raw_text = input("  Texto raw: ").strip() or "manual instruction"

        print("\n  Tipos disponibles:")
        for t in InstructionType:
            print(f"    {t.value}")
        itype_str = input("  Tipo de instrucción: ").strip().lower() or "unknown"
        try:
            instruction_type = InstructionType(itype_str)
        except ValueError:
            instruction_type = InstructionType.UNKNOWN

        action_verb = input("  Verbo de acción: ").strip().lower() or instruction_type.value

        # Parámetros
        params: Dict[str, Any] = {}
        print("\n  Parámetros (clave=valor, vacío para terminar):")
        while True:
            param = input("    > ").strip()
            if not param:
                break
            if "=" in param:
                k, v = param.split("=", 1)
                # Intentar convertir a int/float
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                params[k.strip()] = v

        parsed = ParsedInstruction(
            raw_text=raw_text,
            normalized_text=raw_text,
            speaker=Speaker.ATCO,
            callsign=callsign,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=params,
        )

        self._execute_instruction(raw_text, parsed)

    def _execute_instruction(self, raw_text: str, parsed: ParsedInstruction):
        """Ejecuta la instrucción en el pipeline y muestra resultados."""
        print(f"\n[>] Ejecutando pipeline...")

        result = self.pipeline.process_instruction(raw_text, pre_parsed=parsed)
        self.last_result = result

        # Mostrar proyección
        if result.projected_state:
            ps = result.projected_state
            print(f"\n{'─' * 50}")
            print("  ESTADO PROYECTADO:")
            if ps.target_aircraft_final:
                ac = ps.target_aircraft_final
                print(f"    {ac.callsign}: ALT={ac.position.altitude} HDG={ac.position.heading} SPD={ac.position.speed}")
            if ps.projection_errors:
                print(f"    Errores de proyección: {ps.projection_errors}")
            print(f"{'─' * 50}")

        # Mostrar alertas
        print(f"\n  ALERTAS GENERADAS: {len(result.alerts_generated)}")
        if result.alerts_generated:
            for alert in result.alerts_generated:
                icon = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟠" if alert.severity == AlertSeverity.HIGH else "🟡"
                print(f"    {icon} [{alert.severity.value.upper()}] {alert.category.value}")
                print(f"       {alert.explanation}")
                if alert.suggested_action:
                    print(f"       Sugerencia: {alert.suggested_action}")
        else:
            print("    (Ninguna)")

        # Mostrar violaciones
        if result.violations_found:
            print(f"\n  VIOLACIONES: {len(result.violations_found)}")
            for v in result.violations_found:
                print(f"    - {v.condition_type}: {v.explanation}")

        # Decisión
        print(f"\n  Decisión automática del sistema: {result.final_decision}")
        print(f"  Tiempo de ejecución: {result.total_execution_time_ms:.1f} ms")

        # Si no se hizo commit/rollback automático, preguntar
        if result.final_decision == "PENDING" and self.state_manager.has_pending_transaction():
            print(f"\n  [?] ¿Aplicar cambio? (commit / rollback): ", end="")
            decision = input().strip().lower()
            if decision in ("commit", "c", "yes", "y"):
                self._cmd_commit()
            else:
                self._cmd_rollback()

        print()

    # ------------------------------------------------------------------
    # Comandos: Configuración de reglas
    # ------------------------------------------------------------------

    def _cmd_set(self, args: List[str]):
        if len(args) < 2:
            print("[✗] Uso: set compiled on|off  /  set generic on|off  /  set llm <modelo>  /  set timeout <seg|none>  /  set topk <n>  /  set verbose on|off")
            return

        key = args[0].lower()
        value = args[1]

        if key == "llm":
            self.llm_model = value
            self.llm_config = ModelConfig(
                name=self.llm_model,
                provider="ollama",
                base_url="http://localhost:11434",
                api_key="ollama",
                max_retries=2,
                timeout=30,
            )
            print(f"[✓] Modelo LLM configurado: {self.llm_model}")
            return

        if key == "timeout":
            if value.lower() in ("none", "null", "inf"):
                self.filter_timeout = None
                print("[✓] Timeout de filtro configurado: Sin límite")
            else:
                try:
                    self.filter_timeout = float(value)
                    print(f"[✓] Timeout de filtro configurado: {self.filter_timeout}s")
                except ValueError:
                    print(f"[✗] Valor inválido para timeout: '{value}'")
            return

        if key == "topk":
            try:
                self.filter_top_k = int(value)
                print(f"[✓] Top-K de embeddings configurado: {self.filter_top_k}")
            except ValueError:
                print(f"[✗] Valor inválido para topk: '{value}'")
            return

        if key == "verbose":
            self.verbose = value.lower() in ("on", "true", "1", "yes")
            if self.pipeline is not None:
                self.pipeline.verbose = self.verbose
            if self.rule_filter is not None:
                self.rule_filter.config.verbose = self.verbose
            print(f"[✓] Verbose: {'ON' if self.verbose else 'OFF'}")
            return

        enabled = value.lower() in ("on", "true", "1", "yes")

        if key == "compiled":
            self.use_compiled = enabled
            print(f"[✓] Reglas compiladas: {'ON' if enabled else 'OFF'}")
            if enabled:
                self._try_load_compiled_rules()

        elif key == "generic":
            self.use_generic = enabled
            print(f"[✓] Reglas genéricas: {'ON' if enabled else 'OFF'}")
            if enabled:
                self._try_load_generic_rules()

        else:
            print(f"[✗] Opción desconocida: '{key}'")

    def _try_load_compiled_rules(self):
        """Intenta cargar reglas compiladas desde compiled_rules/."""
        try:
            compiled_dir = os.path.join(os.path.dirname(__file__), "..", "compiled_rules")
            if not os.path.exists(compiled_dir):
                print("[i] No se encontró directorio compiled_rules/")
                return
            from Alert_System.compilation.loader import CompiledRuleLoader

            loader = CompiledRuleLoader(compiled_dir)
            count = loader.register_in_engine(self.rule_engine)
            print(f"[✓] {count} reglas compiladas cargadas en el RuleEngine")
        except Exception as e:
            print(f"[i] No se pudieron cargar reglas compiladas: {e}")

    def _try_load_generic_rules(self):
        """Intenta cargar reglas genéricas desde rules.json, excluyendo las ya compiladas."""
        try:
            rules_path = os.path.join(os.path.dirname(__file__), "..", "..", "rules.json")
            if not os.path.exists(rules_path):
                print("[i] No se encontró archivo rules.json")
                return 0
            with open(rules_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)
            if isinstance(rules_data, list):
                raw_rules = rules_data
            elif isinstance(rules_data, dict):
                raw_rules = rules_data.get("rules", [])
            else:
                raw_rules = []
            if not raw_rules:
                print("[i] No hay reglas en rules.json")
                return 0

            # Cargar manifest de reglas compiladas para filtrar duplicados
            compiled_ids = self._get_compiled_rule_ids()
            if compiled_ids:
                print(f"[i] IDs compilados encontrados: {', '.join(sorted(compiled_ids))}")

            from Alert_System.rule_engine.conditions import GenericKexCondition
            from Alert_System.integration.schemas import ExecutableRule
            count = 0
            skipped = 0
            for idx, rule_dict in enumerate(raw_rules):
                try:
                    rule_id = rule_dict.get("rule_data", {}).get("id", f"GENERIC_{idx}")
                    # Saltar si esta regla ya está compilada
                    if rule_id in compiled_ids:
                        skipped += 1
                        continue
                    executable = ExecutableRule(
                        source_rule_id=rule_id,
                        rule_category=rule_dict.get("rule_data", {}).get("rule_type", "GENERIC"),
                        condition_description=rule_dict.get("rule_data", {}).get("explainability", ""),
                        raw_trigger=rule_dict.get("rule_data", {}).get("trigger", {}).get("description", ""),
                        raw_constraint=rule_dict.get("rule_data", {}).get("constraint", {}).get("description", ""),
                        severity=rule_dict.get("rule_data", {}).get("severity", "MEDIUM"),
                        safety_critical=rule_dict.get("rule_data", {}).get("safety_critical", False),
                    )
                    condition = GenericKexCondition(llm_config=self.llm_config)
                    condition._executable_rule = executable
                    condition.condition_id = executable.source_rule_id
                    self.rule_engine.register_evaluator(
                        f"GENERIC_{executable.source_rule_id}", type(condition)
                    )
                    self.rule_engine._evaluator_instances[f"GENERIC_{executable.source_rule_id}"] = condition
                    count += 1
                except Exception:
                    pass
            print(f"[✓] {count} reglas genéricas cargadas en el RuleEngine")
            if skipped:
                print(f"[i] {skipped} reglas omitidas (ya compiladas)")

            # Crear RuleFilter y precalcular embeddings de reglas cargadas
            if count > 0:
                try:
                    from Alert_System.demo.rule_filter import RuleFilter, FilterConfig
                    loaded_rules = [
                        ev._executable_rule
                        for ev in self.rule_engine._evaluator_instances.values()
                        if hasattr(ev, "_executable_rule") and ev._executable_rule
                    ]
                    self.rule_filter = RuleFilter(
                        FilterConfig(
                            top_k=self.filter_top_k,
                            embedding_cache_dir=os.path.join(
                                os.path.dirname(__file__), "cache"
                            ),
                        )
                    )
                    self.rule_filter.load_or_compute_embeddings(loaded_rules)
                    print(f"[✓] Embeddings precalculados para {len(loaded_rules)} reglas")
                except Exception as emb_err:
                    print(f"[i] No se pudieron precalcular embeddings: {emb_err}")

            # Actualizar pipeline con rule_filter
            if self.pipeline is not None:
                self.pipeline.rule_filter = self.rule_filter
                self.pipeline.llm_config = self.llm_config
                self.pipeline.filter_timeout = self.filter_timeout
                self.pipeline.filter_top_k = self.filter_top_k
            if self.rule_filter:
                self.rule_filter.config.verbose = self.verbose

            return count
        except Exception as e:
            print(f"[i] No se pudieron cargar reglas genéricas: {e}")
            return 0

    def _get_compiled_rule_ids(self) -> set:
        """Lee manifest.json y devuelve IDs de reglas con compilation_status='compiled'."""
        compiled_ids = set()
        try:
            compiled_dir = os.path.join(os.path.dirname(__file__), "..", "compiled_rules")
            manifest_path = os.path.join(compiled_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                return compiled_ids
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            rules = manifest.get("rules", {})
            for rule_id, rule_data in rules.items():
                if rule_data.get("compilation_status") == "compiled":
                    compiled_ids.add(rule_data.get("source_rule_id", rule_id))
        except Exception:
            pass
        return compiled_ids

    # ------------------------------------------------------------------
    # Comandos: Transacciones
    # ------------------------------------------------------------------

    def _cmd_commit(self):
        if not self.state_manager.has_pending_transaction():
            print("[✗] No hay transacción pendiente")
            return
        txn = self.state_manager.get_pending_transaction()
        success = self.state_manager.commit(txn.transaction_id)
        if success:
            print("[✓] Cambio aplicado (COMMIT)")
        else:
            print("[✗] Commit fallido (¿hay alertas críticas? Usa force si es necesario)")

    def _cmd_rollback(self):
        if not self.state_manager.has_pending_transaction():
            print("[✗] No hay transacción pendiente")
            return
        txn = self.state_manager.get_pending_transaction()
        success = self.state_manager.rollback(txn.transaction_id)
        if success:
            print("[✓] Cambio descartado (ROLLBACK)")
        else:
            print("[✗] Rollback fallido")

    def _cmd_undo(self):
        success = self.state_manager.undo_last_commit()
        if success:
            print("[✓] Último commit deshecho")
        else:
            print("[✗] No hay commits para deshacer")

    # ------------------------------------------------------------------
    # Helpers: construcción desde kwargs
    # ------------------------------------------------------------------

    def _parse_kwargs(self, args: List[str]) -> Dict[str, Any]:
        """Parsea argumentos tipo --key value o --key."""
        kwargs = {}
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                key = arg[2:].replace("-", "_")
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    val = args[i + 1]
                    # Intentar convertir tipos
                    if val.lower() in ("true", "yes"):
                        val = True
                    elif val.lower() in ("false", "no"):
                        val = False
                    else:
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    kwargs[key] = val
                    i += 2
                else:
                    kwargs[key] = True
                    i += 1
            else:
                i += 1
        return kwargs

    def _build_aircraft_from_kwargs(self, callsign: str, kwargs: Dict[str, Any]) -> AircraftState:
        """Construye AircraftState desde kwargs del CLI."""
        pos = Position(
            latitude=kwargs.get("latitude", kwargs.get("lat", 0.0)),
            longitude=kwargs.get("longitude", kwargs.get("lon", 0.0)),
            altitude=kwargs.get("altitude", kwargs.get("alt", 0)),
            heading=kwargs.get("heading", kwargs.get("hdg", 0)),
            speed=kwargs.get("speed", kwargs.get("spd", 0)),
            vertical_rate=kwargs.get("vertical_rate", kwargs.get("vs", None)),
        )

        phase_str = kwargs.get("phase", kwargs.get("flight_phase", "cruise"))
        try:
            flight_phase = FlightPhase(phase_str)
        except ValueError:
            flight_phase = FlightPhase.CRUISE

        wake_str = kwargs.get("wake", kwargs.get("wake_turbulence", "M"))
        try:
            wake_turbulence = WakeTurbulenceCategory(wake_str)
        except ValueError:
            wake_turbulence = WakeTurbulenceCategory.MEDIUM

        clearances = Clearances(
            altitude_assigned=kwargs.get("altitude_assigned", kwargs.get("clr_alt", None)),
            heading_assigned=kwargs.get("heading_assigned", kwargs.get("clr_hdg", None)),
            runway_assigned=kwargs.get("runway_assigned", kwargs.get("clr_rw", None)),
            speed_assigned=kwargs.get("speed_assigned", kwargs.get("clr_spd", None)),
        )

        return AircraftState(
            callsign=callsign.upper(),
            position=pos,
            flight_phase=flight_phase,
            clearances=clearances,
            restrictions=kwargs.get("restrictions", []),
            wake_turbulence=wake_turbulence,
            aircraft_type=kwargs.get("aircraft_type", kwargs.get("type", None)),
            is_emergency=kwargs.get("emergency", kwargs.get("is_emergency", False)),
        )

    def _build_runway_from_kwargs(self, runway_id: str, kwargs: Dict[str, Any]) -> RunwayState:
        """Construye RunwayState desde kwargs del CLI."""
        mode_str = kwargs.get("mode", kwargs.get("operation_mode", "mixed"))
        try:
            operation_mode = RunwayOperationMode(mode_str)
        except ValueError:
            operation_mode = RunwayOperationMode.MIXED

        return RunwayState(
            runway_id=runway_id.upper(),
            occupied=kwargs.get("occupied", False),
            occupied_by=kwargs.get("occupied_by", None),
            operation_mode=operation_mode,
            holding_short=kwargs.get("holding_short", []),
            landing_queue=kwargs.get("landing_queue", []),
        )

    def _update_aircraft(self, ac: AircraftState, kwargs: Dict[str, Any]):
        """Actualiza un AircraftState existente con kwargs."""
        if "altitude" in kwargs or "alt" in kwargs:
            ac.position.altitude = kwargs.get("altitude", kwargs.get("alt", ac.position.altitude))
        if "heading" in kwargs or "hdg" in kwargs:
            ac.position.heading = kwargs.get("heading", kwargs.get("hdg", ac.position.heading))
        if "speed" in kwargs or "spd" in kwargs:
            ac.position.speed = kwargs.get("speed", kwargs.get("spd", ac.position.speed))
        if "latitude" in kwargs or "lat" in kwargs:
            ac.position.latitude = kwargs.get("latitude", kwargs.get("lat", ac.position.latitude))
        if "longitude" in kwargs or "lon" in kwargs:
            ac.position.longitude = kwargs.get("longitude", kwargs.get("lon", ac.position.longitude))
        if "vertical_rate" in kwargs or "vs" in kwargs:
            ac.position.vertical_rate = kwargs.get("vertical_rate", kwargs.get("vs", ac.position.vertical_rate))
        if "phase" in kwargs:
            try:
                ac.flight_phase = FlightPhase(kwargs["phase"])
            except ValueError:
                pass
        if "type" in kwargs or "aircraft_type" in kwargs:
            ac.aircraft_type = kwargs.get("aircraft_type", kwargs.get("type", ac.aircraft_type))

    def _update_runway(self, rw: RunwayState, kwargs: Dict[str, Any]):
        """Actualiza un RunwayState existente con kwargs."""
        if "occupied" in kwargs:
            rw.occupied = bool(kwargs["occupied"])
        if "occupied_by" in kwargs:
            rw.occupied_by = kwargs["occupied_by"]
        if "mode" in kwargs or "operation_mode" in kwargs:
            try:
                rw.operation_mode = RunwayOperationMode(kwargs.get("mode", kwargs.get("operation_mode")))
            except ValueError:
                pass


def main():
    parser = argparse.ArgumentParser(description="ATC Alert System - Demo CLI")
    parser.add_argument("--state", "-s", help="Path al JSON de estado inicial")
    args = parser.parse_args()

    cli = DemoCLI(initial_state_path=args.state)
    cli.run()


if __name__ == "__main__":
    main()
