from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Tuple

class LogicalChunk(BaseModel):
    """Representa un bloque lógico de oraciones contiguas."""
    indices: Tuple[int, int] = Field(
        ...,
        description="Rango contiguo de índices [inicio, fin] (inclusive)"
    )
    
    @field_validator('indices')
    @classmethod
    def validate_indices(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Valida que el rango sea válido (inicio <= fin)."""
        if v[0] > v[1]:
            raise ValueError(f"El índice de inicio ({v[0]}) no puede ser mayor que el índice final ({v[1]})")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"indices": [0, 4]},
                {"indices": [5, 5]}
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
        """Valida que los chunks sean contiguos, no se solapen y cubran sin huecos."""
        if not v:
            return v
        
        # Ordenar por inicio (por si acaso)
        sorted_chunks = sorted(v, key=lambda x: x.indices[0])
        
        # 1. Validar orden y solapamiento
        prev_end = -1
        for chunk in sorted_chunks:
            start, end = chunk.indices
            
            if start <= prev_end:
                raise ValueError(f"Chunks solapados o no ordenados: [{start}, {end}] se solapa con final anterior {prev_end}")
            
            if start > prev_end + 1:
                raise ValueError(f"Hay un hueco entre chunks: final anterior {prev_end}, inicio actual {start}")
            
            prev_end = end
        
        # 2. Validar cobertura total con contexto
        if info.context:
            total_sentences = info.context.get("total_sentences")
            if total_sentences is not None:
                actual_last_index = sorted_chunks[-1].indices[1]
                expected_last_index = total_sentences - 1
                
                if actual_last_index != expected_last_index:
                    raise ValueError(
                        f"La segmentación es incompleta. El último índice debe ser "
                        f"{expected_last_index}, pero recibimos {actual_last_index}. "
                        f"Faltan las líneas {actual_last_index + 1} a {expected_last_index}."
                    )
                
                # Validar que empiece desde 0
                first_index = sorted_chunks[0].indices[0]
                if first_index != 0:
                    raise ValueError(
                        f"La segmentación debe empezar en el índice 0, pero empieza en {first_index}."
                    )
        
        return sorted_chunks
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "chunks": [
                        {"indices": [0, 4]},
                        {"indices": [5, 5]}
                    ]
                }
            ]
        }
    }