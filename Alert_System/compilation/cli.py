"""CLI para compilación offline de reglas KEX a código Python."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .schemas import CompilationStatus


def main():
    """Punto de entrada principal del CLI de compilación."""
    parser = argparse.ArgumentParser(
        description="Compila reglas KEX a código Python evaluador usando LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Compilar reglas desde archivo JSON con modelo default
  python -m Alert_System.compilation.cli rules.json
  
  # Compilar con modelo específico
  python -m Alert_System.compilation.cli rules.json --model llama3.2:latest
  
  # Forzar re-compilación de todas las reglas
  python -m Alert_System.compilation.cli rules.json --force
  
  # Directorio de salida personalizado
  python -m Alert_System.compilation.cli rules.json --output-dir ./my_compiled_rules
        """
    )
    
    parser.add_argument(
        "rules_file",
        help="Archivo JSON con reglas KEX a compilar",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="Modelo LLM a usar para compilación (default: llama3.2:latest)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        help="Proveedor LLM (default: ollama)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="URL base del proveedor LLM (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida para reglas compiladas (default: Alert_System/compiled_rules/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar re-compilación de todas las reglas (incluyendo las ya compiladas)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Intentos máximos por regla (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout en segundos para llamadas LLM (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo mostrar qué reglas se compilarían, sin ejecutar",
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo de reglas existe
    rules_path = Path(args.rules_file)
    if not rules_path.exists():
        print(f"❌ Archivo no encontrado: {args.rules_file}")
        sys.exit(1)
    
    # Cargar reglas
    print(f"📂 Cargando reglas desde {args.rules_file}...")
    rules = load_rules(rules_path)
    
    if not rules:
        print("❌ No se encontraron reglas para compilar")
        sys.exit(1)
    
    print(f"📊 Se encontraron {len(rules)} reglas")
    
    # Filtrar reglas que ya están compiladas (si no --force)
    if not args.force:
        from Alert_System.compilation.loader import CompiledRuleLoader
        output_dir = args.output_dir or get_default_output_dir()
        loader = CompiledRuleLoader(compiled_rules_dir=output_dir)
        loader.load_all_compiled_rules()
        
        uncompiled = []
        for rule in rules:
            rule_id = get_rule_id(rule)
            if not loader.has_compiled_rule(rule_id):
                uncompiled.append(rule)
            else:
                print(f"  ⏭️  {rule_id}: ya compilada (usar --force para re-compilar)")
        
        if not uncompiled:
            print("✅ Todas las reglas ya están compiladas")
            sys.exit(0)
        
        rules = uncompiled
        print(f"📊 {len(rules)} reglas pendientes de compilación")
    
    if args.dry_run:
        print("\n🔍 Dry run - Reglas que se compilarían:")
        for rule in rules:
            rule_id = get_rule_id(rule)
            desc = get_rule_description(rule)
            print(f"  - {rule_id}: {desc[:80]}...")
        sys.exit(0)
    
    # Crear configuración LLM
    from common.llm_client_factory import ModelConfig
    
    llm_config = ModelConfig(
        name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key="ollama" if args.provider == "ollama" else "",
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    
    print(f"\n🧠 Modelo: {args.model}")
    print(f"🔗 Provider: {args.provider}")
    print(f"📁 Output: {args.output_dir or 'default'}")
    print(f"🔄 Max retries: {args.max_retries}")
    
    # Compilar reglas
    from Alert_System.compilation.compiler import RuleCompiler
    from Alert_System.compilation.schemas import ExecutableRule
    
    compiler = RuleCompiler(llm_config=llm_config)
    
    # Convertir reglas a ExecutableRule
    from Alert_System.integration.kex_adapter import KEXAdapter
    adapter = KEXAdapter(llm_config=llm_config)
    
    executables = []
    for rule in rules:
        try:
            # Si es un dict (desde JSON), crear ExecutableRule directamente
            if isinstance(rule, dict):
                executable = ExecutableRule(
                    source_rule_id=rule.get("id", "UNKNOWN"),
                    rule_category=rule.get("category", "GENERIC"),
                    condition_description=rule.get("description", ""),
                    raw_trigger=rule.get("trigger", ""),
                    raw_constraint=rule.get("constraint", ""),
                    severity=rule.get("severity", "MEDIUM"),
                    safety_critical=rule.get("safety_critical", False),
                    required_state_fields=rule.get("required_state_fields", []),
                )
            else:
                # Si es un objeto Rule del KEX
                executable = adapter.compile_to_executable(rule)
            executables.append(executable)
        except Exception as e:
            print(f"⚠️ Error preparando regla: {e}")
    
    if not executables:
        print("❌ No hay reglas ejecutables para compilar")
        sys.exit(1)
    
    # Compilar batch con guardado incremental
    output_dir = args.output_dir or get_default_output_dir()
    print(f"\n🔨 Compilando {len(executables)} reglas...")
    print(f"📁 Directorio de salida: {output_dir}")
    
    manifest = compiler.compile_batch(
        executables, 
        save_incrementally=True, 
        output_dir=output_dir
    )
    
    # El guardado ya se hizo incrementalmente durante la compilación
    from Alert_System.compilation.loader import CompiledRuleLoader
    loader = CompiledRuleLoader(compiled_rules_dir=output_dir, llm_config=llm_config)
    
    # Verificar cuántas se guardaron realmente
    saved_count = 0
    for rule_id, rule in manifest.rules.items():
        if rule.compilation_status == CompilationStatus.COMPILED:
            rule_file = loader.compiled_rules_dir / f"{rule_id}.py"
            if rule_file.exists():
                saved_count += 1
    
    saved = saved_count
    
    print(f"\n✅ Compilación completada:")
    print(f"  📦 Compiladas: {manifest.total_compiled}")
    print(f"  ❌ Fallidas (fallback LLM): {manifest.total_failed}")
    print(f"  💾 Guardadas en disco: {saved}")
    print(f"  📁 Directorio: {output_dir}")
    
    # Mostrar detalles de fallidas
    if manifest.total_failed > 0:
        print(f"\n❌ Reglas que fallaron compilación:")
        for rule_id, rule in manifest.rules.items():
            if rule.compilation_status.value == "failed":
                print(f"  - {rule_id}: {rule.failure_reason}")


def load_rules(rules_path: Path) -> List:
    """Carga reglas desde archivo JSON."""
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Soportar diferentes formatos
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Puede tener key "rules" o ser un solo dict
            if "rules" in data:
                return data["rules"]
            return [data]
        else:
            return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parseando JSON: {e}")
        return []


def get_rule_id(rule) -> str:
    """Obtiene el ID de una regla."""
    if isinstance(rule, dict):
        return rule.get("id", rule.get("source_rule_id", "UNKNOWN"))
    return getattr(rule, "id", getattr(rule, "source_rule_id", "UNKNOWN"))


def get_rule_description(rule) -> str:
    """Obtiene la descripción de una regla."""
    if isinstance(rule, dict):
        return rule.get("description", rule.get("condition_description", ""))
    return getattr(rule, "description", getattr(rule, "condition_description", ""))


def get_default_output_dir() -> str:
    """Obtiene el directorio default para reglas compiladas."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "compiled_rules"
    )


if __name__ == "__main__":
    main()
