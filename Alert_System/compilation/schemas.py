"""Schemas para compilación de reglas KEX a código Python."""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, validator


class CompilationStatus(str, Enum):
    """Estado de compilación de una regla."""
    COMPILED = "compiled"
    FAILED = "failed"
    PENDING = "pending"
    NOT_COMPILABLE = "not_compilable"  # Regla subjetiva, no verificable con TrafficState


class RuleVerdict(BaseModel):
    """Veredicto de clasificación: ¿es la regla compilable con TrafficState?"""
    is_compilable: bool = Field(..., description="Si la regla puede evaluarse objetivamente con TrafficState")
    reason: str = Field(..., description="Razón de la clasificación")
    required_fields: List[str] = Field(
        default_factory=list,
        description="Campos de TrafficState necesarios para evaluar la regla"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confianza en la clasificación (0-1)"
    )


# Modelos para respuestas estructuradas del LLM con Instructor
class ClassificationResponse(BaseModel):
    """Respuesta estructurada del LLM para clasificación de reglas."""
    is_compilable: bool = Field(..., description="Si la regla puede evaluarse objetivamente con TrafficState")
    reason: str = Field(..., description="Razón detallada de la clasificación")
    required_fields: List[str] = Field(
        default_factory=list,
        description="Campos de TrafficState necesarios para evaluar la regla"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confianza en la clasificación (0-1)"
    )
    
    @validator('required_fields')
    def validate_required_fields(cls, v):
        """Validar que los campos requeridos sean válidos."""
        valid_fields = [
            'aircrafts', 'msa', 'sector_id', 'runway_state', 'weather',
            'airspace_class', 'separation_minima', 'altitude_limits'
        ]
        for field in v:
            if field not in valid_fields:
                # Permitir campos personalizados pero advertir
                pass
        return v


class GeneratedCodeResponse(BaseModel):
    """Respuesta estructurada del LLM para generación de código Python."""
    code: str = Field(..., description="Código Python de la función evaluate")
    explanation: str = Field(
        default="",
        description="Explicación del código generado"
    )
    required_state_fields: List[str] = Field(
        default_factory=list,
        description="Campos de TrafficState que usa el código"
    )
    
    @validator('code')
    def validate_code_structure(cls, v):
        """Validar que el código contenga la función evaluate."""
        if 'def evaluate(' not in v:
            raise ValueError("El código debe contener una función 'def evaluate('")
        return v
    
    @validator('required_state_fields')
    def validate_required_fields(cls, v):
        """Validar que los campos requeridos sean válidos."""
        valid_fields = [
            'aircrafts', 'msa', 'sector_id', 'runway_state', 'weather',
            'airspace_class', 'separation_minima', 'altitude_limits'
        ]
        for field in v:
            if field not in valid_fields:
                # Permitir campos personalizados pero advertir
                pass
        return v


class ValidatedCodeResponse(BaseModel):
    """Respuesta de código con validación completa integrada."""
    code: str = Field(..., description="Código Python de la función evaluate")
    explanation: str = Field(
        default="",
        description="Explicación del código generado"
    )
    required_state_fields: List[str] = Field(
        default_factory=list,
        description="Campos de TrafficState que usa el código"
    )
    
    @validator('code')
    def validate_syntax(cls, v):
        """Validar sintaxis del código."""
        try:
            import ast
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")
        return v
    
    @validator('code')
    def validate_function_name(cls, v):
        """Validar que contenga función evaluate."""
        if 'def evaluate(' not in v:
            raise ValueError("El código debe contener una función 'def evaluate('")
        return v
    
    @validator('code')
    def validate_function_signature(cls, v):
        """Validar signature de la función evaluate."""
        import ast
        try:
            tree = ast.parse(v)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate':
                    args = node.args.args
                    if not args or args[0].arg != 'traffic_state':
                        raise ValueError("La función evaluate debe tener 'traffic_state' como primer parámetro")
                    break
            else:
                raise ValueError("No se encontró la función evaluate")
        except Exception as e:
            if "must have" in str(e):
                raise  # Re-lanzar el error de validación
        return v
    
    @validator('code')
    def validate_no_forbidden_imports(cls, v):
        """Validar que no haya imports prohibidos."""
        import ast
        forbidden_imports = {
            "os", "subprocess", "open", "exec", "eval", "compile", "__import__",
            "globals", "locals", "socket", "http", "urllib", "requests", "sys"
        }
        allowed_imports = {"math", "datetime"}
        
        try:
            tree = ast.parse(v)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module not in allowed_imports:
                            raise ValueError(f"Import prohibido: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split(".")[0]
                        if module not in allowed_imports:
                            raise ValueError(f"Import prohibido desde: {node.module}")
        except Exception as e:
            if "prohibido" in str(e):
                raise  # Re-lanzar el error de validación
        return v
    
    @validator('code')
    def validate_no_forbidden_names(cls, v):
        """Validar que no haya acceso a nombres prohibidos."""
        import ast
        forbidden_names = {
            "os", "subprocess", "open", "exec", "eval", "compile", "__import__",
            "globals", "locals", "memoryview", "bytearray", "socket"
        }
        
        try:
            tree = ast.parse(v)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden_names:
                    raise ValueError(f"Acceso prohibido: {node.id}")
        except Exception as e:
            if "prohibido" in str(e):
                raise  # Re-lanzar el error de validación
        return v
    
    @validator('code')
    def validate_return_structure(cls, v):
        """Validar que la función retorne dict con keys requeridas."""
        import ast
        required_keys = {"satisfied", "details", "explanation", "severity"}
        
        try:
            tree = ast.parse(v)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'evaluate':
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value:
                            if isinstance(child.value, ast.Dict):
                                keys = set()
                                for key in child.value.keys:
                                    if isinstance(key, ast.Constant):
                                        keys.add(key.value)
                                
                                missing = required_keys - keys
                                if missing:
                                    raise ValueError(f"Retorno missing keys: {missing}")
                            break
                    break
        except Exception as e:
            if "missing" in str(e):
                raise  # Re-lanzar el error de validación
        return v
    
    @validator('required_state_fields')
    def validate_required_fields(cls, v):
        """Validar que los campos requeridos sean válidos."""
        valid_fields = [
            'aircrafts', 'msa', 'sector_id', 'runway_state', 'weather',
            'airspace_class', 'separation_minima', 'altitude_limits'
        ]
        for field in v:
            if field not in valid_fields:
                # Permitir campos personalizados
                pass
        return v


class CompiledRule(BaseModel):
    """Regla compilada por LLM, almacenada en disco."""
    
    source_rule_id: str = Field(..., description="ID de la regla KEX original")
    rule_category: str = Field(..., description="Categoría: GENERIC, ALTITUDE, etc.")
    condition_description: str = Field(..., description="Descripción original de la condición")
    compiled_code: str = Field(..., description="Código Python generado por LLM (función evaluate)")
    required_state_fields: List[str] = Field(
        default_factory=list,
        description="Campos del TrafficState que usa la función"
    )
    compilation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata: modelo LLM, timestamp, confianza compilación"
    )
    compilation_status: CompilationStatus = Field(
        default=CompilationStatus.PENDING,
        description="Estado de la compilación"
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Razón si la compilación falló"
    )
    raw_trigger: Optional[str] = Field(
        default=None,
        description="Texto del trigger original del KEX"
    )
    raw_constraint: Optional[str] = Field(
        default=None,
        description="Texto de la constraint original del KEX"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severidad de la regla original"
    )
    safety_critical: bool = Field(
        default=False,
        description="Si es crítica para seguridad"
    )


class CompilationManifest(BaseModel):
    """Manifiesto de todas las reglas compiladas."""
    
    version: str = Field(default="1.0", description="Versión del formato de manifiesto")
    compiled_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp de compilación")
    model_used: str = Field(..., description="Modelo LLM usado para compilar")
    rules: Dict[str, CompiledRule] = Field(
        default_factory=dict,
        description="Reglas compiladas indexadas por source_rule_id"
    )
    total_compiled: int = Field(default=0, description="Total reglas compiladas exitosamente")
    total_failed: int = Field(default=0, description="Total reglas que fallaron compilación")
    total_fallback: int = Field(default=0, description="Total reglas que usan fallback LLM runtime")
    total_not_compilable: int = Field(default=0, description="Total reglas no compilables (subjetivas)")
    
    def add_rule(self, rule: CompiledRule) -> None:
        """Agrega una regla al manifiesto y actualiza contadores."""
        self.rules[rule.source_rule_id] = rule
        if rule.compilation_status == CompilationStatus.COMPILED:
            self.total_compiled += 1
        elif rule.compilation_status == CompilationStatus.FAILED:
            self.total_failed += 1
            self.total_fallback += 1
        elif rule.compilation_status == CompilationStatus.NOT_COMPILABLE:
            self.total_not_compilable += 1
            self.total_fallback += 1
