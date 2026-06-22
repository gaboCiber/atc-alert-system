"""Cargador de reglas para E7_integrated: nativas + compiladas + genéricas."""

import json
from pathlib import Path
from typing import Any, Set

from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.rule_engine.conditions import GenericKexCondition, CompiledCondition
from Alert_System.compilation.loader import CompiledRuleLoader
from Alert_System.demo.rule_filter import RuleFilter, FilterConfig as ASFilterConfig
from Alert_System.integration.schemas import ExecutableRule

from config import FilterConfig, to_model_config


def _to_as_filter_config(filter_config: FilterConfig) -> ASFilterConfig:
    return ASFilterConfig(
        use_keywords=filter_config.use_keywords,
        use_embeddings=filter_config.use_embeddings,
        use_llm_batch=filter_config.use_llm_batch,
        top_k=filter_config.top_k,
        timeout_seconds=filter_config.timeout_seconds,
        embedding_cache_dir=filter_config.embedding_cache_dir,
        verbose=filter_config.verbose,
    )


def _get_compiled_rule_ids(compiled_rules_dir: Path) -> Set[str]:
    manifest = compiled_rules_dir / "manifest.json"
    # if manifest.exists():
    #     try:
    #         with open(manifest, "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #         return set(data.get("rules", {}).keys())
    #     except Exception:
    #         pass
    return {p.stem for p in compiled_rules_dir.glob("RULE*.py")}


def load_all_rules(
    rule_engine: RuleEngine,
    llm_config: Any,
    filter_config: FilterConfig,
    compiled_rules_dir: Path,
    rules_json_path: Path,
    verbose: bool = True,
    skip_generic: bool = False,
) -> RuleFilter:
    """
    Carga todas las reglas en el RuleEngine:
    1. Nativas (ALTITUDE, SEPARATION, RUNWAY) — ya registradas por defecto
    2. Compiladas (COMPILED_RULEXXX) — desde directorio E4
    3. Genéricas (GENERIC_RULEXXX) — desde rules.json (excluyendo compiladas)

    Retorna RuleFilter configurado para prefiltrado de genéricas.
    """
    if verbose:
        print("\n📦 Cargando reglas en RuleEngine...")

    if verbose:
        print(f"   ✅ Nativas: {rule_engine.get_registered_evaluators()}")

    compiled_ids = _load_compiled_rules(rule_engine, llm_config, compiled_rules_dir, verbose)
    generic_count = 0
    if not skip_generic:
        generic_count = _load_generic_rules(
            rule_engine, llm_config, rules_json_path, compiled_ids, verbose
        )
    elif verbose:
        print("   ⏭️  Reglas genéricas omitidas (--skip-generic)")

    rule_filter = RuleFilter(_to_as_filter_config(filter_config))

    if generic_count > 0:
        loaded_rules = [
            ev._executable_rule
            for ev in rule_engine._evaluator_instances.values()
            if hasattr(ev, "_executable_rule") and ev._executable_rule
        ]
        if loaded_rules:
            rule_filter.load_or_compute_embeddings(loaded_rules)
            if verbose:
                print(f"   ✅ Embeddings precalculados para {len(loaded_rules)} reglas genéricas")

    if verbose:
        evaluators = rule_engine.get_registered_evaluators()
        native = [e for e in evaluators if e in ("ALTITUDE", "SEPARATION", "RUNWAY")]
        compiled = [e for e in evaluators if e.startswith("COMPILED_")]
        generic = [e for e in evaluators if e.startswith("GENERIC_")]
        print(f"   📊 Total: {len(evaluators)} evaluadores")
        print(f"      Nativas: {len(native)} | Compiladas: {len(compiled)} | Genéricas: {len(generic)}")

    return rule_filter


def _load_compiled_rules(
    rule_engine: RuleEngine,
    llm_config: Any,
    compiled_rules_dir: Path,
    verbose: bool,
) -> Set[str]:
    """Carga reglas compiladas desde E4 vía CompiledRuleLoader."""
    if not compiled_rules_dir.exists():
        if verbose:
            print(f"   ⚠️  Directorio compiladas no existe: {compiled_rules_dir}")
        return set()

    loader = CompiledRuleLoader(str(compiled_rules_dir), llm_config=llm_config)
    loaded = loader.register_in_engine(rule_engine)

    if verbose:
        print(f"   ✅ Compiladas cargadas: {loaded}")

    return _get_compiled_rule_ids(compiled_rules_dir)


def _load_generic_rules(
    rule_engine: RuleEngine,
    llm_config: Any,
    rules_json_path: Path,
    compiled_ids: Set[str],
    verbose: bool,
) -> int:
    """Carga reglas genéricas desde rules.json como GenericKexCondition."""
    if not rules_json_path.exists():
        if verbose:
            print(f"   ⚠️  rules.json no encontrado: {rules_json_path}")
        return 0

    with open(rules_json_path, "r", encoding="utf-8") as f:
        rules_data = json.load(f)

    if isinstance(rules_data, list):
        raw_rules = rules_data
    elif isinstance(rules_data, dict):
        raw_rules = rules_data.get("rules", [])
    else:
        raw_rules = []

    if not raw_rules:
        if verbose:
            print("   ⚠️  rules.json vacío")
        return 0

    loaded = 0
    skipped = 0
    for idx, rule_dict in enumerate(raw_rules):
        try:
            rule_data = rule_dict.get("rule_data", {})
            rule_id = rule_data.get("id", f"GENERIC_{idx}")
            if rule_id in compiled_ids:
                skipped += 1
                continue

            executable = ExecutableRule(
                source_rule_id=rule_id,
                rule_category=rule_data.get("rule_type", "GENERIC"),
                condition_description=rule_data.get("explainability", ""),
                raw_trigger=rule_data.get("trigger", {}).get("description", ""),
                raw_constraint=rule_data.get("constraint", {}).get("description", ""),
                severity=rule_data.get("severity", "MEDIUM"),
                safety_critical=rule_data.get("safety_critical", False),
            )
            condition = GenericKexCondition(llm_config=llm_config)
            condition._executable_rule = executable
            condition.condition_id = executable.source_rule_id

            condition_type = f"GENERIC_{executable.source_rule_id}"
            rule_engine.register_evaluator(condition_type, type(condition))
            rule_engine._evaluator_instances[condition_type] = condition
            loaded += 1
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Error registrando genérica idx={idx}: {e}")

    if verbose:
        print(f"   ✅ Genéricas cargadas: {loaded} (omitidas por compilación: {skipped})")

    return loaded
