"""
Adaptador ASR para el Alert System.

Convierte TranscriptionResult del módulo ASR a ParsedInstruction
para el pipeline de alertas.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

# ASR imports
from ASR.transcription.base import TranscriptionResult

# Alert System imports
from Alert_System.models.instruction import (
    ParsedInstruction,
    InstructionType,
    Speaker,
)
from Alert_System.integration.atc_compact_normalizer import ATCCompactNormalizer, normalize_to_compact


@dataclass
class TranscriptionContext:
    """Contexto de una transcripción para el pipeline de alertas."""
    raw_text: str
    normalized_text: str
    confidence: Optional[float]
    timestamp: datetime
    model_name: str
    metadata: Dict[str, Any]


class ASRAdapter:
    """
    Adaptador que convierte salida del ASR a ParsedInstruction.
    
    Responsabilidades:
    1. Recibir TranscriptionResult del ASR
    2. Normalizar texto ATC
    3. Extraer callsign, tipo de instrucción, parámetros
    4. Crear ParsedInstruction para el pipeline
    """
    
    def __init__(self, normalizer: Optional[ATCCompactNormalizer] = None):
        """
        Inicializa el adaptador.
        
        Args:
            normalizer: Normalizador compacto opcional
        """
        self.normalizer = normalizer or ATCCompactNormalizer()
        
        # Mapeo de verbos a tipos de instrucción
        self._verb_to_instruction_type = {
            # Altitude
            "climb": InstructionType.CLIMB,
            "descend": InstructionType.DESCENT,
            "maintain": InstructionType.MAINTAIN_ALTITUDE,
            "altitude": InstructionType.MAINTAIN_ALTITUDE,
            "level": InstructionType.MAINTAIN_ALTITUDE,
            "flight level": InstructionType.MAINTAIN_ALTITUDE,
            
            # Heading
            "turn": InstructionType.HEADING,
            "heading": InstructionType.HEADING,
            "fly heading": InstructionType.HEADING,
            "left": InstructionType.TURN_LEFT,
            "right": InstructionType.TURN_RIGHT,
            
            # Speed
            "speed": InstructionType.SPEED,
            "reduce": InstructionType.REDUCE_SPEED,
            "increase": InstructionType.INCREASE_SPEED,
            
            # Clearance (no CLEARED_DIRECT, usando TAKEOFF_CLEARANCE como genérico)
            "cleared": InstructionType.TAKEOFF_CLEARANCE,
            "proceed": InstructionType.TAKEOFF_CLEARANCE,
            "direct": InstructionType.TAKEOFF_CLEARANCE,
            
            # Approach/Landing
            "approach": InstructionType.APPROACH_CLEARANCE,
            "land": InstructionType.LANDING_CLEARANCE,
            "cleared to land": InstructionType.LANDING_CLEARANCE,
            
            # Takeoff
            "takeoff": InstructionType.TAKEOFF_CLEARANCE,
            "take off": InstructionType.TAKEOFF_CLEARANCE,
            "cleared for takeoff": InstructionType.TAKEOFF_CLEARANCE,
            
            # Hold
            "hold": InstructionType.HOLD_POSITION,
            "holding": InstructionType.HOLD_POSITION,
            
            # Handoff
            "contact": InstructionType.CONTACT,
            "frequency": InstructionType.CONTACT,
        }
        
        # Patrones regex para extracción
        self._callsign_pattern = re.compile(
            r'\b([A-Z]{2,3}\d{1,4}[A-Z]?)\b|\b([A-Z]{3}\d{1,3})\b',
            re.IGNORECASE
        )
        self._altitude_pattern = re.compile(
            r'FL(\d{2,3})|(?:to|at|maintain|level)\s+(\d{2,5})\s*(?:feet|ft)?|(\d{2,5})\s*(?:feet|ft)',
            re.IGNORECASE
        )
        self._heading_pattern = re.compile(
            r'(?:heading|turn)\s*(?:left|right)?\s*(\d{3})',
            re.IGNORECASE
        )
        self._runway_pattern = re.compile(
            r'(?:runway|rwy)\s*(\d{2}[LR]?)',
            re.IGNORECASE
        )
    
    def adapt(
        self,
        transcription: TranscriptionResult,
        speaker: Speaker = Speaker.ATCO,
    ) -> ParsedInstruction:
        """
        Convierte TranscriptionResult a ParsedInstruction.
        
        Args:
            transcription: Resultado del ASR
            speaker: Quién habla (ATCO o PILOT)
            
        Returns:
            ParsedInstruction lista para el pipeline
        """
        raw_text = transcription.text
        
        # Normalizar texto a formato compacto ATC
        normalized_text = self.normalizer.normalize(raw_text)
        
        # Extraer callsign del texto normalizado (formato AAL123)
        callsign = self._extract_callsign(normalized_text)
        
        # Detectar tipo de instrucción
        instruction_type, action_verb = self._detect_instruction_type(normalized_text)
        
        # Extraer parámetros
        parameters = self._extract_parameters(normalized_text, instruction_type)
        
        # Crear contexto
        context = TranscriptionContext(
            raw_text=raw_text,
            normalized_text=normalized_text,
            confidence=transcription.confidence,
            timestamp=datetime.utcnow(),
            model_name=transcription.model_name,
            metadata=transcription.metadata,
        )
        
        return ParsedInstruction(
            raw_text=raw_text,
            normalized_text=normalized_text,
            speaker=speaker,
            callsign=callsign,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=parameters,
            context={
                "asr_model": transcription.model_name,
                "confidence": transcription.confidence,
                "timestamp": context.timestamp.isoformat(),
            },
        )
    
    def adapt_batch(
        self,
        transcriptions: List[TranscriptionResult],
        speaker: Speaker = Speaker.ATCO,
    ) -> List[ParsedInstruction]:
        """
        Convierte múltiples transcripciones.
        
        Args:
            transcriptions: Lista de resultados del ASR
            speaker: Quién habla
            
        Returns:
            Lista de ParsedInstructions
        """
        return [self.adapt(t, speaker) for t in transcriptions]
    
    def _extract_callsign(self, text: str) -> Optional[str]:
        """Extrae callsign del texto normalizado."""
        match = self._callsign_pattern.search(text)
        if match:
            callsign = match.group(1) or match.group(2)
            return callsign.upper() if callsign else None
        return None
    
    def _detect_instruction_type(self, text: str) -> tuple[InstructionType, str]:
        """
        Detecta el tipo de instrucción y verbo principal.
        
        Returns:
            Tupla de (InstructionType, action_verb)
        """
        text_lower = text.lower()
        
        # Buscar verbos conocidos
        for verb, instruction_type in self._verb_to_instruction_type.items():
            if verb in text_lower:
                return instruction_type, verb.split()[0]  # Primer palabra del verbo
        
        # Default: UNKNOWN
        return InstructionType.UNKNOWN, "unknown"
    
    def _extract_parameters(
        self,
        text: str,
        instruction_type: InstructionType,
    ) -> Dict[str, Any]:
        """Extrae parámetros según el tipo de instrucción."""
        params = {}
        
        # Altitude
        if instruction_type in [InstructionType.CLIMB, InstructionType.DESCENT, InstructionType.MAINTAIN_ALTITUDE]:
            alt_match = self._altitude_pattern.search(text)
            if alt_match:
                # Grupo 1: FLXXX (ej: FL240 → 24000)
                if alt_match.group(1):
                    altitude = int(alt_match.group(1)) * 100
                # Grupo 2: Altitud en contexto (ej: "descend to 4000")
                elif alt_match.group(2):
                    altitude = int(alt_match.group(2))
                # Grupo 3: Altitud con feet/ft (ej: "5000 feet")
                elif alt_match.group(3):
                    altitude = int(alt_match.group(3))
                else:
                    altitude = 0
                params["target_altitude"] = altitude
        
        # Heading
        if instruction_type == InstructionType.HEADING:
            hdg_match = self._heading_pattern.search(text)
            if hdg_match:
                params["heading"] = int(hdg_match.group(1))
            # Detectar dirección
            if "left" in text.lower():
                params["direction"] = "left"
            elif "right" in text.lower():
                params["direction"] = "right"
        
        # Runway
        rwy_match = self._runway_pattern.search(text)
        if rwy_match:
            params["runway"] = rwy_match.group(1).upper()
        
        # Speed (extraer números de 3 dígitos que podrían ser velocidad)
        speed_matches = re.findall(r'\b(\d{3})\s*knots?\b', text, re.IGNORECASE)
        if speed_matches:
            params["target_speed"] = int(speed_matches[0])
        
        return params
