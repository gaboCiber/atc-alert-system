#!/usr/bin/env python3
"""
Demo simplificado para probar reglas compiladas existentes en el RuleEngine.

Uso:
    uv run python demo_test_compiled.py
    uv run python demo_test_compiled.py --rules-dir ./mi_directorio
"""

import argparse
import sys
from pathlib import Path

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.models.traffic_state import (
    TrafficState, AircraftState, Position, FlightPhase, RunwayState
)


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
        "OTH456": AircraftState(
            callsign="OTH456",
            position=Position(
                latitude=40.03, longitude=-3.03,
                altitude=100, heading=180, speed=50  # On runway
            ),
            flight_phase=FlightPhase.TAKEOFF,
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


def test_compiled_rules(rules_dir=None):
    """Prueba reglas compiladas existentes."""
    print("\n" + "="*60)
    print("🧪 DEMO: Probando Reglas Compiladas en RuleEngine")
    print("="*60)
    
    # Crear RuleEngine
    rule_engine = RuleEngine()
    print("✅ RuleEngine creado")
    
    # Cargar reglas compiladas desde directorio
    loader = CompiledRuleLoader(compiled_rules_dir=rules_dir)
    
    # Cargar manifest
    manifest = loader.load_manifest()
    if manifest:
        print(f"📋 Manifest encontrado: {manifest.model_used}")
        print(f"   Reglas totales: {len(manifest.rules)}")
        print(f"   Compiladas: {manifest.total_compiled}")
        print(f"   Fallidas: {manifest.total_failed}")
    else:
        print("⚠️ No se encontró manifest.json")
    
    # Cargar todas las reglas compiladas
    all_rules = loader.load_all_compiled_rules()
    print(f"📦 {len(all_rules)} reglas compiladas cargadas")
    
    if not all_rules:
        print("❌ No hay reglas compiladas para probar")
        print("💡 Ejecuta primero: uv run python -m Alert_System.compilation.cli sample_rules.json")
        return
    
    # Mostrar reglas disponibles
    print("\n📋 Reglas disponibles:")
    for rule_id, rule in all_rules.items():
        status = "✅" if rule.compilation_status.value == "compiled" else "❌"
        print(f"  {status} {rule_id}: {rule.condition_description[:60]}...")
    
    # Registrar reglas en el RuleEngine
    print(f"\n🔧 Registrando reglas en RuleEngine...")
    loaded_count = loader.register_in_engine(rule_engine)
    print(f"✅ {loaded_count} reglas registradas")
    
    if loaded_count == 0:
        print("❌ No se pudieron registrar las reglas")
        return
    
    # Mostrar evaluadores registrados
    evaluators = rule_engine.get_registered_evaluators()
    compiled_evaluators = [e for e in evaluators if e.startswith("COMPILED_")]
    print(f"\n🔍 Evaluadores registrados: {len(compiled_evaluators)}")
    for evaluator in compiled_evaluators:
        print(f"  - {evaluator}")
    
    # Crear estado de tráfico
    traffic_state = create_sample_traffic_state()
    print(f"\n✈️ TrafficState creado con {len(traffic_state.aircrafts)} aeronaves")
    
    # Evaluar cada aeronave contra todas las reglas compiladas
    print(f"\n🔍 Evaluación de reglas por aeronave:")
    print("-" * 50)
    
    total_evaluations = 0
    total_violations = 0
    
    for callsign in traffic_state.aircrafts:
        print(f"\n✈️  {callsign}:")
        aircraft = traffic_state.aircrafts[callsign]
        print(f"   Altitud: {aircraft.position.altitude}ft | Fase: {aircraft.flight_phase.value}")
        print(f"   Emergencia: {'SÍ' if aircraft.is_emergency else 'NO'}")
        
        for condition_type in compiled_evaluators:
            result = rule_engine.evaluate(
                condition_type=condition_type,
                parameters={},
                traffic_state=traffic_state,
                aircraft_callsign=callsign,
            )
            
            total_evaluations += 1
            
            if result.satisfied:
                print(f"  ✅ {condition_type}: OK")
            else:
                total_violations += 1
                print(f"  ❌ {condition_type}: VIOLACIÓN")
                if result.violation:
                    print(f"     📝 {result.violation.explanation}")
                    print(f"     🔢 {result.violation.details}")
    
    # Resumen final
    print(f"\n📊 Resumen de evaluación:")
    print(f"  Evaluaciones totales: {total_evaluations}")
    print(f"  Violaciones detectadas: {total_violations}")
    print(f"  Tasa de cumplimiento: {((total_evaluations - total_violations) / total_evaluations * 100):.1f}%")
    
    # Detalles de violaciones
    if total_violations > 0:
        print(f"\n🚨 Detalles de violaciones:")
        for callsign in traffic_state.aircrafts:
            for condition_type in compiled_evaluators:
                result = rule_engine.evaluate(
                    condition_type=condition_type,
                    parameters={},
                    traffic_state=traffic_state,
                    aircraft_callsign=callsign,
                )
                if not result.satisfied and result.violation:
                    print(f"  ❌ {callsign} - {result.violation.rule_id} ({result.violation.severity.value})")
                    print(f"     {result.violation.explanation}")
    
    print(f"\n✅ Demo completada exitosamente!")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Probar reglas compiladas en RuleEngine")
    parser.add_argument(
        "--rules-dir",
        default=None,
        help="Directorio de reglas compiladas (default: Alert_System/compiled_rules/)"
    )
    
    args = parser.parse_args()
    
    # Directorio default si no se especifica
    if not args.rules_dir:
        args.rules_dir = str(Path(__file__).parent / "Alert_System" / "compiled_rules")
    
    print("🚀 Demo de Reglas Compiladas - RuleEngine")
    print("=" * 60)
    print(f"📁 Directorio de reglas: {args.rules_dir}")
    
    # Verificar que el directorio exista
    if not Path(args.rules_dir).exists():
        print(f"❌ El directorio no existe: {args.rules_dir}")
        print("💡 Ejecuta primero: uv run python -m Alert_System.compilation.cli sample_rules.json")
        return
    
    test_compiled_rules(args.rules_dir)


if __name__ == "__main__":
    main()
