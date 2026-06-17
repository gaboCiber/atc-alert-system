#!/usr/bin/env python3
"""
Demo que compila reglas reales del KEX con guardado incremental.

Este script:
1. Lee reglas reales del Knowledge_Extractor
2. Las procesa con KEXAdapter
3. Las compila una por una, guardando cada una en disco inmediatamente
4. Muestra progreso en tiempo real
"""

import json
import sys
import argparse
from pathlib import Path

# Agregar el proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Alert_System.integration.kex_adapter import KEXAdapter
from Alert_System.compilation.compiler import RuleCompiler
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.models.traffic_state import (
    TrafficState, AircraftState, Position, FlightPhase, RunwayState
)
from common.llm_client_factory import ModelConfig


def load_real_kex_rules(input_dir: str):
    """Carga reglas reales usando el procesador organizado."""
    from Alert_System.compilation.kex_data_processor import KEXFileProcessor
    
    # Directorio de salida KEX
    kex_dir = Path(input_dir)
    
    if not kex_dir.exists():
        print(f"❌ Directorio KEX no encontrado: {kex_dir}")
        return []
    
    print(f"📂 Procesando reglas desde: {kex_dir}")
    
    # Usar el procesador organizado
    processor = KEXFileProcessor(str(kex_dir))
    accumulator = processor.process_all_files()  # Procesar todos los archivos para tener más reglas
    
    # Obtener reglas completas con referencias resueltas
    complete_rules = accumulator.get_complete_rules()
    
    # Formatear para compatibilidad con el código existente
    formatted_rules = []
    for rule_data in complete_rules:
        formatted_rules.append({
            'source_file': 'processed_files',
            'chunk_index': 0,
            'rule_data': rule_data
        })
    
    print(f"\n📊 Resumen de carga:")
    print(f"  Reglas completas: {len(formatted_rules)}")
    print(f"  Con referencias resueltas: ✅")
    
    return formatted_rules


def process_kex_rules_with_incremental_save(kex_rules, max_rules=None, model="llama3.2:latest", provider="ollama", output_dir="compiled_kex_rules", start_rule_index=1):
    """Procesa reglas KEX con guardado incremental."""
    rules_to_process = len(kex_rules) if max_rules is None else min(max_rules, len(kex_rules))
    print(f"\n🔄 Procesando {rules_to_process} reglas KEX {'(todas)' if max_rules is None else f'(límite: {max_rules})'}...")
    
    # Usar argumentos pasados como parámetros
    
    # Configurar LLM
    try:
        llm_config = ModelConfig(
            name=model,
            provider=provider,
            base_url="http://localhost:11434" if provider == "ollama" else None,
            api_key="ollama" if provider == "ollama" else None,
            max_retries=2,
            timeout=60,
        )
        print("✅ Configuración LLM creada")
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        return None, None
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Directorio de salida: {output_path}")
    
    # Convertir reglas KEX a ExecutableRule
    adapter = KEXAdapter(llm_config=llm_config)
    executables = []
    
    for i, rule_info in enumerate(kex_rules[:rules_to_process]):
        rule_data = rule_info['rule_data']
        
        try:
            # Las reglas ya vienen procesadas con referencias resueltas
            # Solo necesitamos convertirlas a ExecutableRule
            
            # Extraer información básica
            rule_id = rule_data.get('id', f'KEX_RULE_{i}')
            rule_type = rule_data.get('rule_type', 'unknown')
            modality = rule_data.get('modality', 'unknown')
            
            # Crear objeto Rule simplificado para KEXAdapter
            from Knowledge_Extractor.schemas.kex_schemas import Rule, RuleType, Modality, DeonticStrength, Severity
            
            # Mapear tipos a enums
            rule_type_enum = RuleType.PROHIBITION if rule_type == 'prohibition' else RuleType.OBLIGATION
            modality_enum = Modality.SHALL_NOT if modality == 'shall_not' else Modality.MUST
            deontic_enum = DeonticStrength.STRONG
            severity_enum = Severity.HIGH if rule_data.get('severity') == 'high' else Severity.MEDIUM
            
            rule = Rule(
                id=rule_id,
                rule_type=rule_type_enum,
                modality=modality_enum,
                deontic_strength=deontic_enum,
                trigger=rule_data.get('trigger', {}),
                constraint=rule_data.get('constraint', {}),
                formal_if_then=rule_data.get('formal_if_then', {}),
                applicability=rule_data.get('applicability', {}),
                severity=severity_enum,
                safety_critical=rule_data.get('safety_critical', True),
                explainability=rule_data.get('explainability', f'Rule {rule_id} from KEX data'),
                linked_entities=rule_data.get('linked_entities', []),
                linked_relationships=rule_data.get('linked_relationships', [])
            )
            
            # Convertir a ExecutableRule
            executable = adapter.compile_to_executable(rule)
            executables.append(executable)
            
            print(f"📋 {executable.source_rule_id}: {executable.rule_category}")
            
        except Exception as e:
            print(f"⚠️ Error procesando {rule_data.get('id')}: {e}")
            import traceback
            traceback.print_exc()
    
    if not executables:
        print("❌ No se pudieron procesar reglas")
        return None, None
    
    print(f"\n🔨 Compilando {len(executables)} reglas con guardado incremental...")
    
    # Compilar con guardado incremental
    compiler = RuleCompiler(llm_config=llm_config)
    manifest = compiler.compile_batch(
        executables, 
        save_incrementally=True, 
        output_dir=str(output_path),
        start_rule_index=start_rule_index
    )
    
    # Mostrar resultados
    print(f"\n✅ Compilación completada:")
    print(f"  📦 Compiladas: {manifest.total_compiled}")
    print(f"  ❌ Falladas: {manifest.total_failed}")
    print(f"  📁 Archivos en: {output_path}")
    
    # Listar archivos generados
    py_files = list(output_path.glob("*.py"))
    print(f"\n📄 Archivos .py generados ({len(py_files)}):")
    for py_file in py_files:
        size = py_file.stat().st_size
        print(f"  - {py_file.name} ({size} bytes)")
    
    return output_path, manifest


def test_compiled_rules(output_dir):
    """Prueba las reglas compiladas."""
    print(f"\n🧪 Probando reglas compiladas desde {output_dir}...")
    
    # Crear RuleEngine
    rule_engine = RuleEngine()
    
    # Cargar reglas compiladas
    loader = CompiledRuleLoader(compiled_rules_dir=str(output_dir))
    loaded_count = loader.register_in_engine(rule_engine)
    
    print(f"✅ {loaded_count} reglas registradas en RuleEngine")
    
    if loaded_count == 0:
        print("❌ No hay reglas para probar")
        return
    
    # Crear estado de tráfico de prueba
    traffic_state = TrafficState(
        sector_id="KEX_TEST_SECTOR",
        msa=5000,
        aircrafts={
            "TEST001": AircraftState(
                callsign="TEST001",
                position=Position(latitude=40.0, longitude=-3.0, altitude=4500, heading=90, speed=250),  # Below MSA
                flight_phase=FlightPhase.DESCENT,
            ),
            "TEST002": AircraftState(
                callsign="TEST002",
                position=Position(latitude=40.01, longitude=-3.01, altitude=6000, heading=270, speed=200),  # Above MSA
                flight_phase=FlightPhase.CRUISE,
            ),
        },
    )
    
    # Evaluar reglas
    print(f"\n🔍 Evaluando reglas:")
    violations = 0
    
    for condition_type in rule_engine.get_registered_evaluators():
        if condition_type.startswith("COMPILED_"):
            for callsign in traffic_state.aircrafts:
                result = rule_engine.evaluate(
                    condition_type=condition_type,
                    parameters={},
                    traffic_state=traffic_state,
                    aircraft_callsign=callsign,
                    instruction=None,
                )
                
                status = "✅" if result.satisfied else "❌"
                print(f"  {status} {callsign} - {condition_type}")
                
                if not result.satisfied and result.violation:
                    violations += 1
                    print(f"     📝 {result.violation.explanation}")
    
    print(f"\n📊 Resumen de evaluación:")
    print(f"  Violaciones detectadas: {violations}")
    print(f"  Total evaluaciones: {len(traffic_state.aircrafts) * loaded_count}")


import argparse

def parse_arguments():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Compila reglas KEX con resolución de entidades y clasificación LLM")
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Carpeta de entrada con archivos KEX (obligatorio)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="compiled_kex_rules",
        help="Carpeta de salida para reglas compiladas"
    )
    
    parser.add_argument(
        "--max-rules", "-n",
        type=int,
        default=None,
        help="Número máximo de reglas a procesar (default: todas)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.2:latest",
        help="Modelo LLM a usar"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="ollama",
        help="Proveedor LLM (ollama, openai, etc.)"
    )

    parser.add_argument(
        "--start-rule-index", "-i",
        type=str,
        default=1,
        help="Rule index"
    )
    
    return parser.parse_args()

def main():
    """Función principal del demo."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Configuración
    kex_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    max_rules = args.max_rules
    
    print(f"🚀 Demo KEX Real - Compilación con Resolución y Clasificación")
    print("=" * 70)
    print(f"📂 Entrada: {kex_dir}")
    print(f"📁 Salida: {output_dir}")
    print(f"🤖 Modelo: {args.model} ({args.provider})")
    print(f"📊 Máximo reglas: {max_rules}")
    print()
    
    # Paso 1: Cargar reglas
    kex_rules = load_real_kex_rules(args.input_dir)
    
    if not kex_rules:
        print("❌ No se encontraron reglas KEX")
        return
    
    # Paso 2: Compilar con guardado incremental
    result = process_kex_rules_with_incremental_save(
        kex_rules, 
        max_rules=args.max_rules,
        model=args.model,
        provider=args.provider,
        output_dir=args.output,
        start_rule_index=args.start_rule_index
    )
    
    if not result:
        print("❌ Falló el procesamiento de reglas")
        return
    
    output_dir, manifest = result
    
    # Paso 3: Probar reglas compiladas
    test_compiled_rules(output_dir)
    
    print(f"\n✅ Demo completada!")
    print(f"💾 Reglas compiladas guardadas en: {output_dir}")
    print(f"🔍 Puedes inspeccionar los archivos .py generados")


if __name__ == "__main__":
    main()
