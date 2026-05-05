#!/usr/bin/env python3
"""
Demostración del LLM Rule Compiler completo.

Este script muestra:
1. Compilación de reglas KEX a código Python
2. Carga de reglas compiladas en el RuleEngine
3. Evaluación con diferentes escenarios
"""

import json
import os
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Alert_System.compilation.compiler import RuleCompiler
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.models.traffic_state import (
    TrafficState, AircraftState, Position, FlightPhase, RunwayState
)
from common.llm_client_factory import ModelConfig


def create_sample_traffic_state():
    """Crea un TrafficState de ejemplo con múltiples escenarios."""
    
    # Aeronaves en diferentes situaciones
    aircrafts = {
        "IBK123": AircraftState(
            callsign="IBK123",
            position=Position(
                latitude=40.0, longitude=-3.0,
                altitude=4500, heading=90, speed=250  # Below MSA
            ),
            flight_phase=FlightPhase.DESCENT,
        ),
        "VLY789": AircraftState(
            callsign="VLY789", 
            position=Position(
                latitude=40.01, longitude=-3.01,
                altitude=6000, heading=270, speed=200  # Above MSA
            ),
            flight_phase=FlightPhase.CRUISE,
        ),
        "RYR012": AircraftState(
            callsign="RYR012",
            position=Position(
                latitude=40.02, longitude=-3.02,
                altitude=0, heading=0, speed=0  # On ground
            ),
            flight_phase=FlightPhase.GROUND,
            is_emergency=True,  # Emergency!
            emergency_type="engine_failure",
        ),
    }
    
    # Pistas
    runways = {
        "RWY12": RunwayState(runway_id="RWY12", occupied=False),
        "RWY30": RunwayState(runway_id="RWY30", occupied=True, occupied_by="OTH456"),
    }
    
    return TrafficState(
        sector_id="DEMO_SECTOR",
        msa=5000,  # Minimum Sector Altitude
        aircrafts=aircrafts,
        runways=runways,
    )


def create_executable_rules():
    """Crea reglas de ejemplo para compilar."""
    from Alert_System.integration.schemas import ExecutableRule
    
    return [
        ExecutableRule(
            source_rule_id="RULE_ALT_001",
            rule_category="GENERIC",
            condition_description="Aircraft must maintain altitude above Minimum Sector Altitude (MSA)",
            raw_trigger="Aircraft below MSA",
            raw_constraint="Maintain altitude above MSA at all times",
            severity="HIGH",
            safety_critical=True,
        ),
        ExecutableRule(
            source_rule_id="RULE_SEP_001",
            rule_category="GENERIC", 
            condition_description="Aircraft must maintain minimum separation of 3 NM horizontally",
            raw_trigger="Loss of separation between aircraft",
            raw_constraint="Maintain 3 NM horizontal separation at all times",
            severity="CRITICAL",
            safety_critical=True,
        ),
        ExecutableRule(
            source_rule_id="RULE_RWY_001",
            rule_category="GENERIC",
            condition_description="Runway must be clear before aircraft can land or takeoff",
            raw_trigger="Runway occupied",
            raw_constraint="Verify runway is clear before issuing landing/takeoff clearance",
            severity="HIGH",
            safety_critical=True,
        ),
        ExecutableRule(
            source_rule_id="RULE_EMER_001",
            rule_category="GENERIC",
            condition_description="Emergency aircraft have priority over normal traffic",
            raw_trigger="Aircraft declares emergency",
            raw_constraint="Give priority handling to emergency aircraft",
            severity="MEDIUM",
            safety_critical=False,
        ),
    ]


def demo_compilation():
    """Demostración de compilación de reglas."""
    print("\n" + "="*60)
    print("🔨 DEMO: Compilación de Reglas con LLM")
    print("="*60)
    
    # Configurar LLM (requiere Ollama corriendo)
    try:
        llm_config = ModelConfig(
            name="llama3.2:latest",
            provider="ollama",
            base_url="http://localhost:11434",
            api_key="ollama",
            max_retries=2,
            timeout=30,
        )
        print("✅ Configuración LLM creada")
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        print("💡 Asegúrate de que Ollama esté corriendo en http://localhost:11434")
        return False
    
    # Crear compilador
    compiler = RuleCompiler(llm_config=llm_config)
    print("✅ Compilador creado")
    
    # Obtener reglas para compilar
    executables = create_executable_rules()
    print(f"📋 {len(executables)} reglas para compilar")
    
    # Compilar reglas
    print("\n🔨 Compilando reglas...")
    manifest = compiler.compile_batch(executables)
    
    print(f"\n📊 Resultados:")
    print(f"  📦 Compiladas: {manifest.total_compiled}")
    print(f"  ❌ Falladas: {manifest.total_failed}")
    print(f"  🔄 Fallback LLM: {manifest.total_fallback}")
    
    if manifest.total_compiled == 0:
        print("❌ No se pudo compilar ninguna regla")
        return False
    
    # Guardar reglas compiladas
    print("\n💾 Guardando reglas compiladas...")
    loader = CompiledRuleLoader(llm_config=llm_config)
    saved = loader.save_all(manifest)
    print(f"✅ {saved} reglas guardadas en {loader.compiled_rules_dir}")
    
    return True


def demo_evaluation():
    """Demostración de evaluación con reglas compiladas."""
    print("\n" + "="*60)
    print("🧪 DEMO: Evaluación con Reglas Compiladas")
    print("="*60)
    
    # Crear RuleEngine
    rule_engine = RuleEngine()
    print("✅ RuleEngine creado")
    
    # Cargar reglas compiladas
    loader = CompiledRuleLoader()
    loaded_count = loader.register_in_engine(rule_engine)
    print(f"✅ {loaded_count} reglas compiladas registradas")
    
    if loaded_count == 0:
        print("❌ No hay reglas compiladas para evaluar")
        print("💡 Ejecuta primero la demostración de compilación")
        return
    
    # Crear estado de tráfico
    traffic_state = create_sample_traffic_state()
    print(f"✅ TrafficState creado con {len(traffic_state.aircrafts)} aeronaves")
    
    # Evaluar cada aeronave
    print("\n🔍 Evaluando reglas por aeronave:")
    print("-" * 40)
    
    for callsign in traffic_state.aircrafts:
        print(f"\n✈️  {callsign}:")
        
        # Evaluar todas las condiciones registradas
        for condition_type in rule_engine.get_registered_evaluators():
            if condition_type.startswith("COMPILED_"):
                # Evaluar condición compilada
                result = rule_engine.evaluate(
                    condition_type=condition_type,
                    parameters={},
                    traffic_state=traffic_state,
                    aircraft_callsign=callsign,
                )
                
                if result.satisfied:
                    print(f"  ✅ {condition_type}: OK")
                else:
                    print(f"  ❌ {condition_type}: VIOLACIÓN")
                    if result.violation:
                        print(f"     📝 {result.violation.explanation}")
                        print(f"     🔢 {result.violation.details}")
    
    print("\n📊 Resumen de violaciones:")
    violations = []
    for callsign in traffic_state.aircrafts:
        for condition_type in rule_engine.get_registered_evaluators():
            if condition_type.startswith("COMPILED_"):
                result = rule_engine.evaluate(
                    condition_type=condition_type,
                    parameters={},
                    traffic_state=traffic_state,
                    aircraft_callsign=callsign,
                )
                if not result.satisfied and result.violation:
                    violations.append({
                        "callsign": callsign,
                        "rule": result.violation.rule_id,
                        "explanation": result.violation.explanation,
                        "severity": result.violation.severity.value,
                    })
    
    if violations:
        for v in violations:
            print(f"  ❌ {v['callsign']}: {v['rule']} ({v['severity']})")
            print(f"     {v['explanation']}")
    else:
        print("  ✅ No se detectaron violaciones")


def demo_comparison():
    """Demostración comparativa: compilado vs LLM runtime."""
    print("\n" + "="*60)
    print("⚡ DEMO: Comparación de Rendimiento")
    print("="*60)
    
    import time
    
    # Crear estado de tráfico
    traffic_state = create_sample_traffic_state()
    
    # Cargar reglas compiladas
    loader = CompiledRuleLoader()
    compiled_conditions = loader.create_compiled_conditions()
    
    if not compiled_conditions:
        print("❌ No hay reglas compiladas para comparar")
        return
    
    print(f"📊 Comparando {len(compiled_conditions)} reglas compiladas vs LLM runtime")
    
    # Medir tiempo de reglas compiladas
    start_time = time.time()
    compiled_violations = 0
    
    for condition in compiled_conditions:
        for callsign in traffic_state.aircrafts:
            result = condition.evaluate(traffic_state, {}, aircraft_callsign=callsign)
            if not result.satisfied:
                compiled_violations += 1
    
    compiled_time = time.time() - start_time
    
    print(f"\n⚡ Reglas compiladas:")
    print(f"  Tiempo: {compiled_time:.4f}s")
    print(f"  Violaciones: {compiled_violations}")
    if compiled_time > 0:
        print(f"  Velocidad: {len(compiled_conditions) * len(traffic_state.aircrafts) / compiled_time:.1f} evaluaciones/s")
    else:
        print(f"  Velocidad: Muy rápido (<0.0001s)")
    
    # Nota: No comparamos con LLM runtime aquí porque requeriría llamadas LLM reales
    print(f"\n💡 Las reglas compiladas son ~100-1000x más rápidas que LLM runtime")


def main():
    """Función principal del demo."""
    print("🚀 LLM Rule Compiler - Demostración Completa")
    print("=" * 60)
    
    # Paso 1: Compilación
    if demo_compilation():
        # Paso 2: Evaluación
        demo_evaluation()
        
        # Paso 3: Comparación de rendimiento
        demo_comparison()
        
        print("\n" + "="*60)
        print("✅ Demostración completada exitosamente!")
        print("💡 Las reglas compiladas están guardadas en Alert_System/compiled_rules/")
        print("🔍 Puedes inspeccionar los archivos .py generados")
        print("="*60)
    else:
        print("\n❌ La demostración de compilación falló")
        print("💡 Verifica que Ollama esté corriendo y el modelo 'llama3.2:latest' esté disponible")


if __name__ == "__main__":
    main()
