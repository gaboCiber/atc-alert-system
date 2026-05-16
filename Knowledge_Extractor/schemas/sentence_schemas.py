from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Tuple

class LogicalChunk(BaseModel):
    """Representa un bloque lógico de oraciones contiguas."""
    indices: Tuple[int, int] = Field(
        ...,
        description="Rango contiguo de índices [inicio, fin] (inclusive). Puede incluir -1 si es contexto de página anterior."
    )
    
    @field_validator('indices')
    @classmethod
    def validate_indices(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Valida que el rango sea válido (inicio <= fin). Permite -1 como índice especial."""
        if v[0] < -1 or v[1] < -1:  # ← Permite -1 como índice válido
            raise ValueError(f"Los índices deben ser >= -1")
        if v[0] > v[1]:
            raise ValueError(f"Índice inicio ({v[0]}) > fin ({v[1]})")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"indices": [0, 4]},
                {"indices": [5, 5]},
                {"indices": [-1, -1]},  # Contexto solo
                {"indices": [-1, 0]},   # Contexto + primera oración
                {"indices": [-1, 4]}    # Contexto extendido
            ]
        }
    }


class SegmentationOutput(BaseModel):
    """Output completo del LLM para segmentación lógica."""
    chunks: List[LogicalChunk] = Field(
        ...,
        description="Lista de bloques lógicos que cubren todas las oraciones"
    )
    
    @field_validator('chunks')
    @classmethod
    def validate_coverage_and_order(cls, v: List[LogicalChunk], info: ValidationInfo) -> List[LogicalChunk]:
        """Valida cobertura con soporte para índice -1."""
        if not v:
            return v
        
        sorted_chunks = sorted(v, key=lambda x: x.indices[0])
        has_context = info.context.get("has_context", False) if info.context else False
        first_actual_index = info.context.get("first_actual_index", 0) if info.context else 0
        last_actual_index = info.context.get("last_actual_index", 0) if info.context else 0
        
        # Validar primer chunk
        first_chunk_start = sorted_chunks[0].indices[0]
        if has_context:
            # Si hay contexto, DEBE empezar en -1 (siempre disponible)
            if first_chunk_start != -1:
                raise ValueError(
                    f"Con contexto, primer chunk DEBE empezar en -1, "
                    f"pero empieza en {first_chunk_start}"
                )
        else:
            # Sin contexto (primera página), debe empezar en 0
            if first_chunk_start != 0:
                raise ValueError(
                    f"Sin contexto, primer chunk debe empezar en 0, "
                    f"pero empieza en {first_chunk_start}"
                )
        
        # Validar último chunk - SIEMPRE debe terminar en last_actual_index
        last_chunk_end = sorted_chunks[-1].indices[1]
        if last_chunk_end != last_actual_index:
            raise ValueError(
                f"Último chunk debe terminar en {last_actual_index}, "
                f"pero termina en {last_chunk_end}"
            )
        
        # Validar continuidad (sin gaps)
        prev_end = -2 if has_context else -1  # Preparar para -1
        for chunk in sorted_chunks:
            start, end = chunk.indices
            if start != prev_end + 1:
                raise ValueError(
                    f"Gap detectado: después de {prev_end}, "
                    f"siguiente empieza en {start}"
                )
            prev_end = end
        
        return sorted_chunks
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunks": [
                        {"indices": [0, 4]},
                        {"indices": [5, 5]}
                    ]
                },
                {
                    "chunks": [
                        {"indices": [-1, -1]},  # Contexto solo
                        {"indices": [0, 2]},    # Oraciones actuales
                        {"indices": [3, 3]}
                    ]
                },
                {
                    "chunks": [
                        {"indices": [-1, 0]},   # Contexto + primera
                        {"indices": [1, 4]}     # Resto
                    ]
                }
            ]
        }
    }