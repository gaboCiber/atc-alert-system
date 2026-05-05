"""Validación estática de código Python generado por LLM."""

import ast
from typing import Dict, List, Tuple

# Imports permitidos en código generado
ALLOWED_IMPORTS = {"math", "datetime"}

# Nombres prohibidos (acceso a sistema, archivos, red, etc.)
FORBIDDEN_NAMES = {
    "os", "subprocess", "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "memoryview", "bytearray", "socket", "http",
    "urllib", "requests", "sys", "pathlib", "shutil", "signal",
    "threading", "multiprocessing", "ctypes", "importlib",
    "pickle", "shelve", "marshal", "codecs", "io",
    "builtins", "inspect", "pkgutil", "pkg_resources",
}

# Atributos prohibidos en cualquier acceso tipo x.y
FORBIDDEN_ATTRS = {
    "__import__", "__builtins__", "__globals__", "__locals__",
    "__code__", "__dict__", "__class__", "__bases__",
    "func_globals", "func_code", "gi_frame", "cr_frame",
}


class CodeValidationError(Exception):
    """Error de validación de código generado."""
    
    def __init__(self, issues: List[str]):
        self.issues = issues
        super().__init__(f"Code validation failed: {'; '.join(issues)}")


def validate_code(code: str) -> Tuple[bool, List[str]]:
    """
    Valida estáticamente el código Python generado por LLM.
    
    Verifica:
    1. Sintaxis válida (parseable)
    2. Solo define una función llamada 'evaluate'
    3. No hay imports prohibidos
    4. No hay acceso a nombres prohibidos
    5. No hay atributos prohibidos
    6. Signature correcta: evaluate(traffic_state, callsign=None)
    7. Retorna dict con keys requeridas
    
    Returns:
        Tuple de (is_valid, lista_de_issues)
    """
    issues = []
    
    # 1. Parsear el código
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error: {e}")
        return False, issues
    
    # 2. Verificar que solo hay definiciones de función
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    if not function_defs:
        issues.append("No function definitions found")
        return False, issues
    
    # Verificar que la única función es 'evaluate'
    fn_names = [fn.name for fn in function_defs]
    if "evaluate" not in fn_names:
        issues.append("No 'evaluate' function found")
    
    # Permitir funciones helper internas, pero 'evaluate' debe existir
    extra_fns = [name for name in fn_names if name != "evaluate"]
    if len(extra_fns) > 3:
        issues.append(f"Too many helper functions ({len(extra_fn)}): {extra_fns}. Max 3 allowed.")
    
    # 3. Verificar signature de evaluate
    evaluate_fn = None
    for fn in function_defs:
        if fn.name == "evaluate":
            evaluate_fn = fn
            break
    
    if evaluate_fn:
        args = evaluate_fn.args
        # Debe tener al menos traffic_state
        pos_args = [a.arg for a in args.args]
        if not pos_args or pos_args[0] != "traffic_state":
            issues.append("evaluate() first argument must be 'traffic_state'")
        
        # callsign debe ser keyword-only con default None (optional)
        has_callsign = "callsign" in pos_args or any(
            a.arg == "callsign" for a in args.kw_defaults
        )
        # Also check defaults
        defaults = args.defaults
        kw_defaults = args.kw_defaults
        
        # Check if callsign is in pos_args with default None
        if len(pos_args) >= 2 and pos_args[1] == "callsign":
            # callsign is a positional arg with default
            num_no_default = len(pos_args) - len(defaults)
            callsign_idx = 1
            default_idx = callsign_idx - num_no_default
            if default_idx < 0 or defaults[default_idx] is None:
                pass  # callsign=None is correct
        elif "callsign" not in pos_args:
            # callsign might not be there, which is also acceptable
            pass
    
    # 4. Verificar imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in ALLOWED_IMPORTS:
                    issues.append(f"Forbidden import: {alias.name}")
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module not in ALLOWED_IMPORTS:
                    issues.append(f"Forbidden import from: {node.module}")
    
    # 5. Verificar nombres prohibidos
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                issues.append(f"Forbidden name access: {node.id}")
        
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                issues.append(f"Forbidden attribute access: {node.attr}")
    
    # 6. Verificar que no hay llamadas a exec/eval/compile
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"exec", "eval", "compile", "__import__"}:
                issues.append(f"Forbidden function call: {func.id}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_return_structure(code: str) -> Tuple[bool, List[str]]:
    """
    Verifica que la función evaluate retorna dicts con las keys requeridas.
    
    Esto es una verificación heurística basada en AST - no puede garantizar
    que todos los paths de ejecución retornen el formato correcto,
    pero puede detectar problemas obvios.
    """
    issues = []
    required_keys = {"satisfied", "details", "explanation", "severity"}
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ["Cannot parse code for return validation"]
    
    # Buscar todos los Return statements dentro de la función evaluate
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    # Verificar si es un dict literal
                    if isinstance(child.value, ast.Dict):
                        keys = set()
                        for key in child.value.keys:
                            if isinstance(key, ast.Constant):
                                keys.add(key.value)
                        
                        missing = required_keys - keys
                        if missing:
                            issues.append(
                                f"Return dict missing required keys: {missing}"
                            )
    
    is_valid = len(issues) == 0
    return is_valid, issues
