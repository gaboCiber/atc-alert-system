from pydantic import BaseModel, Field, field_validator
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
    def validate_coverage_and_order(cls, v: List[LogicalChunk]) -> List[LogicalChunk]:
        """Valida que los chunks sean contiguos, no se solapen y cubran sin huecos."""
        if not v:
            return v
        
        # Ordenar por inicio (por si acaso)
        sorted_chunks = sorted(v, key=lambda x: x.indices[0])
        
        # Verificar que no haya solapamientos y que sean contiguos
        prev_end = -1
        for chunk in sorted_chunks:
            start, end = chunk.indices
            
            if start <= prev_end:
                raise ValueError(f"Chunks solapados o no ordenados: [{start}, {end}] se solapa con final anterior {prev_end}")
            
            if start > prev_end + 1:
                raise ValueError(f"Hay un hueco entre chunks: final anterior {prev_end}, inicio actual {start}")
            
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
                }
            ]
        }
    }