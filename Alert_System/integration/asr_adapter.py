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
from Alert_System.integration.bert_atc_parser import BertATCParser

# ASR imports
from ASR.normalization.text_normalizer import ATCTextNormalizer

# Mapa palabra->digito para extraccion de parametros desde formato expandido
_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "niner": "9",
}


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
    2. Normalizar texto ATC (expandido → compacto)
    3. Extraer callsign, tipo de instrucción, parámetros
    4. Crear ParsedInstruction para el pipeline
    """
    
    def __init__(
        self,
        normalizer: Optional[ATCCompactNormalizer] = None,
        bert_parser: Optional[BertATCParser] = None,
    ):
        """
        Inicializa el adaptador.
        
        Args:
            normalizer: Normalizador compacto opcional
            bert_parser: Parser BERT NER opcional. Si se provee, se intenta
                        usar primero para extraer entidades, con fallback a
                        la logica regex tradicional.
        """
        self.normalizer = normalizer or ATCCompactNormalizer()
        self.bert_parser = bert_parser
        self.asr_normalizer = ATCTextNormalizer(
            expand_callsigns=True,
            expand_numbers=True,
            expand_icao=False,
            normalize_terminology=True,
            remove_punctuation=True,
            lowercase=True,
            convert_compound_numbers=True,
            split_concatenated_terms=True
        )
        
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
            "hdg": InstructionType.HEADING,
            "left": InstructionType.TURN_LEFT,
            "right": InstructionType.TURN_RIGHT,
            
            # Speed (compound forms before generic "speed")
            "reduce speed": InstructionType.REDUCE_SPEED,
            "increase speed": InstructionType.INCREASE_SPEED,
            "speed": InstructionType.SPEED,
            "reduce": InstructionType.REDUCE_SPEED,
            "increase": InstructionType.INCREASE_SPEED,
            
            # Approach/Landing (must be before generic "cleared")
            "cleared to land": InstructionType.LANDING_CLEARANCE,
            "cleared for approach": InstructionType.APPROACH_CLEARANCE,
            "approach": InstructionType.APPROACH_CLEARANCE,
            "contact approach": InstructionType.APPROACH_CLEARANCE,
            "land": InstructionType.LANDING_CLEARANCE,
            
            # Takeoff / generic clearance
            "cleared for takeoff": InstructionType.TAKEOFF_CLEARANCE,
            "cleared for take off": InstructionType.TAKEOFF_CLEARANCE,
            "cleared for": InstructionType.TAKEOFF_CLEARANCE,
            "cleared": InstructionType.TAKEOFF_CLEARANCE,
            "proceed": InstructionType.TAKEOFF_CLEARANCE,
            "direct": InstructionType.TAKEOFF_CLEARANCE,
            "takeoff": InstructionType.TAKEOFF_CLEARANCE,
            "take off": InstructionType.TAKEOFF_CLEARANCE,
            
            # Hold
            "hold": InstructionType.HOLD_POSITION,
            "holding": InstructionType.HOLD_POSITION,
            
            # Handoff
            "contact": InstructionType.CONTACT,
            "frequency": InstructionType.CONTACT,
        }
        
        # Patrones regex para extracción
        # Pattern mejorado: busca callsigns al inicio del texto o después de aerolínea
        self._callsign_pattern = re.compile(
            r'^([A-Z]{2,3}\d{1,4}[A-Z]?)\b|\b([A-Z]{3}\d{1,3})\b(?!\s*(?:fl|hdg|rwy))',
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
        self._heading_compact_pattern = re.compile(
            r'HDG(\d{3})',
            re.IGNORECASE
        )
        self._runway_pattern = re.compile(
            r'(?:runway|rwy)\s*(\d{2}[LR]?)',
            re.IGNORECASE
        )
        self._runway_compact_pattern = re.compile(
            r'RWY(\d{2}[LR]?)',
            re.IGNORECASE
        )
    
    def adapt(
        self,
        transcription: TranscriptionResult,
        speaker: Speaker = Speaker.ATCO,
    ) -> ParsedInstruction:
        """
        Convierte TranscriptionResult a ParsedInstruction.
        
        Pipeline:
            1. Si hay BertATCParser configurado, intentar extraccion via BERT NER
            2. Si BERT no esta disponible o falla, usar logica regex tradicional
        
        Args:
            transcription: Resultado del ASR
            speaker: Quién habla (ATCO o PILOT)
            
        Returns:
            ParsedInstruction lista para el pipeline
        """
        raw_text = transcription.text
        
        # --- Rama 1: BERT NER (si esta configurado y disponible) ---
        if self.bert_parser is not None:
            parsed = self._adapt_with_bert(raw_text, speaker, transcription)
            if parsed is not None:
                return parsed
        
        # --- Rama 2: fallback regex tradicional ---
        return self._adapt_fallback(raw_text, speaker, transcription)
    
    def _adapt_with_bert(
        self,
        raw_text: str,
        speaker: Speaker,
        transcription: TranscriptionResult,
    ) -> Optional[ParsedInstruction]:
        """
        Intenta parsear usando BERT NER.
        
        Returns:
            ParsedInstruction si BERT tuvo exito, None si falla (usa fallback)
        """
        result = self.bert_parser.parse(raw_text)
        
        if not result["success"]:
            return None
        
        # Expandir y compactar texto completo (para metadata y params suplementarios)
        asr_normalized = self.asr_normalizer.normalize(raw_text)
        compact_full = self.normalizer.normalize(asr_normalized)
        
        # --- Callsign (ya compacto desde BertATCParser) ---
        callsign = result["callsign"]
        
        # --- Instruction type desde command entity ---
        command_text = result["command_entity"] or ""
        instruction_type, action_verb = self._detect_instruction_type(command_text)
        
        # Si no se detecto tipo desde el command, intentar con texto completo
        if instruction_type == InstructionType.UNKNOWN:
            instruction_type, action_verb = self._detect_instruction_type(compact_full)
        
        # --- Parametros desde value entity ---
        parameters: Dict[str, Any] = {}
        
        # 1. Intentar extraer desde value compact (formato "FL330", "RWY34L")
        value_compact = result.get("value_compact")
        if value_compact:
            parameters = self._extract_parameters(value_compact, instruction_type)
        
        # 2. Suplementar con extraccion desde texto completo compacto
        full_params = self._extract_parameters(compact_full, instruction_type)
        for k, v in full_params.items():
            if k not in parameters:
                parameters[k] = v
        
        # 3. Si hay value entity expandido, intentar extraer tambien desde ahi
        value_entity = result.get("value_entity")
        if value_entity and not parameters:
            expanded_params = self._extract_parameters_from_expanded(
                value_entity, instruction_type
            )
            parameters.update(expanded_params)
        
        # --- Construir ParsedInstruction ---
        return ParsedInstruction(
            raw_text=raw_text,
            normalized_text=compact_full,
            speaker=speaker,
            callsign=callsign,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=parameters,
            context={
                "asr_model": transcription.model_name,
                "confidence": transcription.confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "asr_normalized": asr_normalized,
                "bert_callsign_score": result["callsign_score"],
                "bert_command_score": result["command_score"],
                "bert_value_score": result["value_score"],
                "bert_callsign_entity": result["callsign_entity"],
                "bert_command_entity": result["command_entity"],
                "bert_value_entity": result["value_entity"],
            },
        )
    
    def _adapt_fallback(
        self,
        raw_text: str,
        speaker: Speaker,
        transcription: TranscriptionResult,
    ) -> ParsedInstruction:
        """Parseo via regex tradicional (fallback cuando BERT no esta disponible)."""
        asr_normalized = self.asr_normalizer.normalize(raw_text)
        compact_text = self.normalizer.normalize(asr_normalized)
        
        callsign = self._extract_callsign(compact_text)
        instruction_type, action_verb = self._detect_instruction_type(compact_text)
        parameters = self._extract_parameters(compact_text, instruction_type)
        if not parameters:
            expanded_params = self._extract_parameters_from_expanded(
                asr_normalized, instruction_type
            )
            parameters.update(expanded_params)
        
        context = TranscriptionContext(
            raw_text=raw_text,
            normalized_text=compact_text,
            confidence=transcription.confidence,
            timestamp=datetime.utcnow(),
            model_name=transcription.model_name,
            metadata=transcription.metadata,
        )
        
        return ParsedInstruction(
            raw_text=raw_text,
            normalized_text=compact_text,
            speaker=speaker,
            callsign=callsign,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=parameters,
            context={
                "asr_model": transcription.model_name,
                "confidence": transcription.confidence,
                "timestamp": context.timestamp.isoformat(),
                "asr_normalized": asr_normalized,
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
        
        # Heading (formato expandido: heading 270, turn left 270)
        if instruction_type in [InstructionType.HEADING, InstructionType.TURN_LEFT, InstructionType.TURN_RIGHT]:
            hdg_match = self._heading_pattern.search(text)
            if hdg_match:
                params["heading"] = int(hdg_match.group(1))
            # Heading (formato compacto: HDG270)
            hdg_compact = self._heading_compact_pattern.search(text)
            if hdg_compact and "heading" not in params:
                params["heading"] = int(hdg_compact.group(1))
            # Detectar dirección
            if "left" in text.lower():
                params["direction"] = "left"
            elif "right" in text.lower():
                params["direction"] = "right"
        
        # Runway (formato expandido: runway 34L, rwy 34L)
        rwy_match = self._runway_pattern.search(text)
        if rwy_match:
            params["runway"] = rwy_match.group(1).upper()
        
        # Runway (formato compacto: RWY34L, RWY30R)
        rwy_compact = self._runway_compact_pattern.search(text)
        if rwy_compact:
            params["runway"] = rwy_compact.group(1).upper()
        
        # Speed (mejorado para diferentes patrones)
        if instruction_type in [InstructionType.SPEED, InstructionType.REDUCE_SPEED, InstructionType.INCREASE_SPEED]:
            # Buscar "speed XXX" o "XXX knots"
            speed_patterns = [
                re.search(r'speed\s+(\d{3})', text, re.IGNORECASE),
                re.search(r'(\d{3})\s*knots?', text, re.IGNORECASE),
                re.search(r'maintain\s+speed\s+(\d{3})', text, re.IGNORECASE),
                re.search(r'reduce\s+speed\s+to\s+(\d{3})', text, re.IGNORECASE),
                re.search(r'increase\s+speed\s+to\s+(\d{3})', text, re.IGNORECASE),
            ]
            
            for pattern_match in speed_patterns:
                if pattern_match:
                    params["target_speed"] = int(pattern_match.group(1))
                    break
        
        return params
    
    def _extract_parameters_from_expanded(
        self,
        text: str,
        instruction_type: InstructionType,
    ) -> Dict[str, Any]:
        """
        Extrae parámetros desde texto en formato expandido (números como palabras).
        
        Útil cuando el value entity del BERT NER está en formato expandido
        y no pasó por el compact normalizer.
        
        Ejemplos:
            "flight level three three zero" -> target_altitude: 33000
            "runway three four left" -> runway: "34L"
            "heading two seven zero" -> heading: 270
            "two five zero knots" -> target_speed: 250
        """
        params: Dict[str, Any] = {}
        if not text:
            return params
        
        text_lower = text.lower()
        
        # Altitud: "flight level X X X" o "flight level X X"
        fl_match = re.search(
            r"flight\s+level\s+("
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
            r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)){1,2}"
            r")",
            text_lower,
        )
        if fl_match:
            digits = self._words_to_digits(fl_match.group(1))
            if len(digits) in (2, 3):
                params["flight_level"] = int(digits)
                params["target_altitude"] = int(digits) * 100
        
        # Altitud directa: "X thousand feet" o solo numero
        alt_match = re.search(
            r"(\d+)\s*(?:thousand\s+)?feet",
            text_lower,
        )
        if alt_match and "target_altitude" not in params:
            altitude = int(alt_match.group(1))
            if "thousand" in text_lower:
                altitude *= 1000
            params["target_altitude"] = altitude
        
        # Altitud desde digitos expandidos: "descend to three zero zero zero"
        if "target_altitude" not in params:
            alt_words = re.search(
                r"(?:(?:descend|climb)\s+to|at|maintain|level)\s+("
                r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
                r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner))+"
                r")",
                text_lower,
            )
            if alt_words:
                digits = self._words_to_digits(alt_words.group(1))
                if digits and len(digits) >= 3:
                    params["target_altitude"] = int(digits)
        
        # Runway: "runway X X left/right" o "runway X X"
        rwy_match = re.search(r"runway\s+(.+)", text_lower)
        if rwy_match:
            rwy_text = rwy_match.group(1)
            digits = self._words_to_digits(rwy_text)
            # Detectar left/right
            suffix = ""
            if "left" in rwy_text:
                suffix = "L"
            elif "right" in rwy_text:
                suffix = "R"
            if digits:
                params["runway"] = f"{digits}{suffix}"
        
        # Heading: "heading X X X"
        hdg_match = re.search(
            r"heading\s+("
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
            r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)){2}"
            r")",
            text_lower,
        )
        if hdg_match:
            digits = self._words_to_digits(hdg_match.group(1))
            if len(digits) == 3:
                params["heading"] = int(digits)
        
        # Speed: "X knots", "speed X X X"
        spd_match = re.search(r"(\d+)\s*knots?", text_lower)
        if spd_match:
            params["target_speed"] = int(spd_match.group(1))
        
        # Speed en palabras: "two five zero knots"
        spd_word_match = re.search(
            r"("
            r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
            r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)){1,2}"
            r")\s*knots?",
            text_lower,
        )
        if spd_word_match and "target_speed" not in params:
            digits = self._words_to_digits(spd_word_match.group(1))
            if digits:
                params["target_speed"] = int(digits)
        
        # Speed en palabras sin "knots": "speed two eight zero"
        if "target_speed" not in params:
            spd_plain = re.search(
                r"(?:speed|reduce speed to|increase speed to)\s+("
                r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)"
                r"(?:\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)){1,2}"
                r")",
                text_lower,
            )
            if spd_plain:
                digits = self._words_to_digits(spd_plain.group(1))
                if digits:
                    params["target_speed"] = int(digits)
        
        return params
    
    @staticmethod
    def _words_to_digits(text: str) -> str:
        """Convierte palabras numericas a digitos. Ej: 'three four zero' -> '340'."""
        digits = []
        for word in text.split():
            word_clean = word.strip().rstrip(".,")
            digit = _WORD_TO_DIGIT.get(word_clean)
            if digit is not None:
                digits.append(digit)
        return "".join(digits)
