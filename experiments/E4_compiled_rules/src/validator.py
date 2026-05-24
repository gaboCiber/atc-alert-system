import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "Alert_System"))

ALLOWED_IMPORTS = {"math", "datetime"}
FORBIDDEN_IMPORTS = {
    "os", "subprocess", "socket", "http", "urllib", "requests", "sys",
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "importlib", "pathlib", "io", "builtins", "pickle", "shelve",
}
FORBIDDEN_NAMES = {
    "os", "subprocess", "socket", "http", "urllib", "requests", "sys",
    "open", "exec", "eval", "compile", "__import__", "globals", "locals",
    "__builtins__", "eval", "exec", "compile", "open", "file",
}


@dataclass
class ValidationResult:
    rule_id: str
    model_name: str
    passed: bool
    syntax_valid: bool = False
    function_exists: bool = False
    correct_signature: bool = False
    no_forbidden_imports: bool = False
    no_forbidden_names: bool = False
    return_structure_valid: bool = False
    failure_reasons: list = field(default_factory=list)


def _find_function_def(code: str, func_name: str = "evaluate") -> ast.FunctionDef:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def _check_forbidden_imports(tree: ast.AST) -> list:
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    forbidden.append(f"forbidden_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                forbidden.append(f"forbidden_import_from:{node.module}")
    return forbidden


def _check_forbidden_names(tree: ast.AST) -> list:
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                forbidden.append(f"forbidden_name:{node.id}")
    return forbidden


def _check_return_structure(tree: ast.AST) -> tuple:
    required_keys = {"satisfied", "details", "explanation", "severity"}
    return_stmts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = {k.value if isinstance(k, ast.Constant) else k.id for k in node.value.keys}
            return_stmts.append(keys)

    if not return_stmts:
        return False, list(required_keys)

    all_have_keys = all(required_keys.issubset(keys) for keys in return_stmts)
    missing_keys = required_keys - next(
        (keys for keys in return_stmts if required_keys.issubset(keys)), set()
    )
    return all_have_keys, list(missing_keys)


def validate_code(code: str, rule_id: str, model_name: str) -> ValidationResult:
    result = ValidationResult(rule_id=rule_id, model_name=model_name, passed=False)

    try:
        tree = ast.parse(code)
        result.syntax_valid = True
    except SyntaxError as e:
        result.failure_reasons.append(f"syntax_error:{e}")
        return result

    func_node = _find_function_def(code)
    if func_node is None:
        result.failure_reasons.append("no_evaluate_function")
        return result
    result.function_exists = True

    args = func_node.args
    if len(args.args) == 0 or args.args[0].arg != "traffic_state":
        result.failure_reasons.append("invalid_signature")
        return result
    result.correct_signature = True

    forbidden_imports = _check_forbidden_imports(tree)
    if forbidden_imports:
        result.failure_reasons.extend(forbidden_imports)
        result.no_forbidden_imports = False
        return result
    result.no_forbidden_imports = True

    forbidden_names = _check_forbidden_names(tree)
    if forbidden_names:
        result.failure_reasons.extend(forbidden_names)
        result.no_forbidden_names = False
        return result
    result.no_forbidden_names = True

    return_valid, missing_keys = _check_return_structure(tree)
    if not return_valid:
        result.failure_reasons.append(f"missing_return_keys:{missing_keys}")
        result.return_structure_valid = False
        return result
    result.return_structure_valid = True

    result.passed = True
    return result


def validate_rules(
    rule_ids: list,
    compiled_code: dict,
    model_name: str,
) -> dict:
    results = {}
    for rule_id, code in compiled_code.items():
        if rule_id not in rule_ids:
            continue
        results[rule_id] = validate_code(code, rule_id, model_name)
    return results