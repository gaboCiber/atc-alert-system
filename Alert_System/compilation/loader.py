"""Cargador de reglas compiladas desde disco al RuleEngine."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import CompiledRule, CompilationManifest, CompilationStatus


# Directorio default para reglas compiladas
DEFAULT_COMPILED_RULES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "compiled_rules"
)


class CompiledRuleLoader:
    """
    Carga reglas compiladas desde disco y las registra en el RuleEngine.
    
    Lee el manifest.json y los archivos .py del directorio de reglas compiladas,
    crea instancias de CompiledCondition y las registra en el RuleEngine.
    """
    
    def __init__(
        self,
        compiled_rules_dir: str = DEFAULT_COMPILED_RULES_DIR,
        llm_config: Any = None,
    ):
        """
        Inicializa el cargador.
        
        Args:
            compiled_rules_dir: Directorio donde están las reglas compiladas
            llm_config: Configuración LLM para fallback en CompiledCondition
        """
        self.compiled_rules_dir = Path(compiled_rules_dir)
        self.llm_config = llm_config
        self._manifest: Optional[CompilationManifest] = None
        self._compiled_rules: Dict[str, CompiledRule] = {}
    
    def load_manifest(self) -> Optional[CompilationManifest]:
        """
        Carga el manifest.json del directorio de reglas compiladas.
        
        Returns:
            CompilationManifest o None si no existe
        """
        manifest_path = self.compiled_rules_dir / "manifest.json"
        
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._manifest = CompilationManifest(**data)
            return self._manifest
            
        except Exception as e:
            print(f"⚠️ Error loading manifest: {e}")
            return None
    
    def load_compiled_rule(self, rule_id: str) -> Optional[CompiledRule]:
        """
        Carga una regla compilada específica desde su archivo .py.
        
        Args:
            rule_id: ID de la regla a cargar
            
        Returns:
            CompiledRule o None si no existe
        """
        # Primero buscar en el manifest
        if self._manifest and rule_id in self._manifest.rules:
            rule_from_manifest = self._manifest.rules[rule_id]
        else:
            rule_from_manifest = None
        
        # Buscar el archivo .py
        rule_file = self.compiled_rules_dir / f"{rule_id}.py"
        
        if not rule_file.exists():
            # Si tenemos la regla en el manifest con código, usar esa
            if rule_from_manifest and rule_from_manifest.compiled_code:
                return rule_from_manifest
            return None
        
        try:
            with open(rule_file, "r", encoding="utf-8") as f:
                code = f.read()
            
            # Si tenemos metadata del manifest, combinar
            if rule_from_manifest:
                rule = rule_from_manifest.model_copy(update={"compiled_code": code})
            else:
                # Crear CompiledRule básico solo con el código
                rule = CompiledRule(
                    source_rule_id=rule_id,
                    rule_category="GENERIC",
                    condition_description="Loaded from compiled file",
                    compiled_code=code,
                    compilation_status=CompilationStatus.COMPILED,
                    compilation_metadata={"loaded_from": str(rule_file)},
                )
            
            self._compiled_rules[rule_id] = rule
            return rule
            
        except Exception as e:
            print(f"⚠️ Error loading compiled rule {rule_id}: {e}")
            return None
    
    def load_all_compiled_rules(self) -> Dict[str, CompiledRule]:
        """
        Carga todas las reglas compiladas del directorio.
        
        Returns:
            Dict de CompiledRule indexadas por rule_id
        """
        # Cargar manifest primero
        if not self._manifest:
            self.load_manifest()
        
        # Si tenemos manifest, usar sus rule_ids
        if self._manifest:
            for rule_id, rule_data in self._manifest.rules.items():
                if rule_data.compilation_status == CompilationStatus.COMPILED:
                    loaded = self.load_compiled_rule(rule_id)
                    if loaded:
                        self._compiled_rules[rule_id] = loaded
        else:
            # Sin manifest: escanear archivos .py
            if self.compiled_rules_dir.exists():
                for rule_file in self.compiled_rules_dir.glob("*.py"):
                    if rule_file.name.startswith("_"):
                        continue
                    rule_id = rule_file.stem
                    loaded = self.load_compiled_rule(rule_id)
                    if loaded:
                        self._compiled_rules[rule_id] = loaded
        
        return self._compiled_rules
    
    def create_compiled_conditions(self) -> List[Any]:
        """
        Crea instancias de CompiledCondition para todas las reglas compiladas.
        
        Returns:
            Lista de CompiledCondition listas para registrar en RuleEngine
        """
        from Alert_System.rule_engine.conditions import CompiledCondition
        
        if not self._compiled_rules:
            self.load_all_compiled_rules()
        
        conditions = []
        for rule_id, compiled_rule in self._compiled_rules.items():
            if compiled_rule.compilation_status == CompilationStatus.COMPILED:
                try:
                    condition = CompiledCondition(
                        compiled_rule=compiled_rule,
                        llm_config=self.llm_config,
                    )
                    conditions.append(condition)
                except Exception as e:
                    print(f"⚠️ Error creating CompiledCondition for {rule_id}: {e}")
        
        return conditions
    
    def register_in_engine(self, rule_engine: Any) -> int:
        """
        Registra todas las reglas compiladas en el RuleEngine.
        
        Args:
            rule_engine: Instancia de RuleEngine
            
        Returns:
            Número de reglas registradas exitosamente
        """
        conditions = self.create_compiled_conditions()
        
        loaded_count = 0
        for condition in conditions:
            try:
                # Registrar cada CompiledCondition como un tipo único
                condition_type = f"COMPILED_{condition.condition_id}"
                rule_engine.register_evaluator(condition_type, type(condition))
                
                # Guardar referencia a la instancia específica
                rule_engine._evaluator_instances[condition_type] = condition
                loaded_count += 1
                print(f"✅ Registered compiled rule {condition.condition_id} as {condition_type}")
            except Exception as e:
                print(f"⚠️ Error registering compiled rule {condition.condition_id}: {e}")
        
        return loaded_count
    
    def has_compiled_rule(self, rule_id: str) -> bool:
        """Verifica si existe una regla compilada para el rule_id dado."""
        if not self._compiled_rules:
            self.load_all_compiled_rules()
        
        return (
            rule_id in self._compiled_rules
            and self._compiled_rules[rule_id].compilation_status == CompilationStatus.COMPILED
        )
    
    def get_compiled_rule(self, rule_id: str) -> Optional[CompiledRule]:
        """Obtiene una regla compilada por su rule_id."""
        if not self._compiled_rules:
            self.load_all_compiled_rules()
        
        return self._compiled_rules.get(rule_id)
    
    def save_manifest(self, manifest: CompilationManifest) -> bool:
        """
        Guarda el manifest en disco.
        
        Args:
            manifest: CompilationManifest a guardar
            
        Returns:
            True si se guardó exitosamente
        """
        self.compiled_rules_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = self.compiled_rules_dir / "manifest.json"
        
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(manifest.model_dump_json(indent=2))
            
            self._manifest = manifest
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving manifest: {e}")
            return False
    
    def save_compiled_rule(self, compiled_rule: CompiledRule) -> bool:
        """
        Guarda una regla compilada en su archivo .py.
        
        Args:
            compiled_rule: CompiledRule a guardar
            
        Returns:
            True si se guardó exitosamente
        """
        self.compiled_rules_dir.mkdir(parents=True, exist_ok=True)
        
        rule_file = self.compiled_rules_dir / f"{compiled_rule.source_rule_id}.py"
        
        try:
            # Escribir header con metadata
            header = f'"""Regla compilada: {compiled_rule.condition_description}"""\n'
            header += f"# Rule ID: {compiled_rule.source_rule_id}\n"
            header += f"# Category: {compiled_rule.rule_category}\n"
            header += f"# Compiled with: {compiled_rule.compilation_metadata.get('model', 'unknown')}\n"
            header += f"# Compiled at: {compiled_rule.compilation_metadata.get('compiled_at', 'unknown')}\n\n"
            
            with open(rule_file, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(compiled_rule.compiled_code)
                f.write("\n")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving compiled rule {compiled_rule.source_rule_id}: {e}")
            return False
    
    def save_all(self, manifest: CompilationManifest) -> int:
        """
        Guarda el manifest y todas las reglas compiladas en disco.
        
        Args:
            manifest: CompilationManifest con todas las reglas
            
        Returns:
            Número de reglas guardadas exitosamente
        """
        # Guardar manifest
        self.save_manifest(manifest)
        
        # Guardar cada regla compilada
        saved_count = 0
        for rule_id, compiled_rule in manifest.rules.items():
            if compiled_rule.compilation_status == CompilationStatus.COMPILED:
                if self.save_compiled_rule(compiled_rule):
                    saved_count += 1
        
        return saved_count
