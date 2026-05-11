"""Parser simple de instrucciones ATC para el demo CLI.

Soporta comandos tipo:
  - "AAL123 climb to 5000"
  - "AAL123 descend to FL240"
  - "AAL123 heading 090"
  - "AAL123 speed 250"
  - "AAL123 cleared for takeoff runway 09L"
  - "AAL123 cleared to land runway 27R"
  - "AAL123 taxi to runway 09L"
  - "AAL123 hold position"
"""

import re
from typing import Dict, Any, Optional

from Alert_System.models.instruction import ParsedInstruction, InstructionType, Speaker


class SimpleATCParser:
    """Parser regex-based para instrucciones ATC comunes."""

    # Patrón de callsign: 2-4 letras + 1-4 dígitos (ej: AAL123, UAL4567, BA123)
    CALLSIGN_PATTERN = re.compile(r'\b([A-Z]{2,4}\d{1,4})\b', re.IGNORECASE)

    # Patrones de altitud
    ALTITUDE_PATTERNS = [
        (re.compile(r'(?:climb|descend|descent)\s+(?:to\s+)?FL\s*(\d{3})', re.IGNORECASE), "flight_level"),
        (re.compile(r'(?:climb|descend|descent)\s+(?:to\s+)?(\d{4,5})', re.IGNORECASE), "altitude"),
        (re.compile(r'(?:maintain|level)\s+(?:altitude\s+)?(\d{4,5})', re.IGNORECASE), "altitude"),
        (re.compile(r'(?:maintain|level)\s+FL\s*(\d{3})', re.IGNORECASE), "flight_level"),
    ]

    # Patrones de heading
    HEADING_PATTERN = re.compile(r'(?:heading|turn\s+(?:left|right)?|fly\s+heading)\s+(\d{3})', re.IGNORECASE)

    # Patrones de velocidad
    SPEED_PATTERN = re.compile(r'(?:speed|reduce\s+speed|increase\s+speed|maintain\s+speed)\s+(\d{3})', re.IGNORECASE)

    # Patrones de pista
    RUNWAY_PATTERN = re.compile(r'runway\s+(\d{2}[LR]?)', re.IGNORECASE)

    def parse(self, raw_text: str) -> ParsedInstruction:
        """Parsea una instrucción ATC y retorna un ParsedInstruction."""
        text = raw_text.strip()
        lower_text = text.lower()

        # Extraer callsign
        callsign = self._extract_callsign(text)

        # Determinar tipo de instrucción y parámetros
        instruction_type, action_verb, parameters = self._determine_instruction(lower_text, text)

        # Si no se detectó callsign pero hay parámetros con callsign en ellos
        if not callsign and "callsign" in parameters:
            callsign = parameters.pop("callsign")

        return ParsedInstruction(
            raw_text=text,
            normalized_text=text,
            speaker=Speaker.ATCO,
            callsign=callsign.upper() if callsign else None,
            instruction_type=instruction_type,
            action_verb=action_verb,
            parameters=parameters,
        )

    def _extract_callsign(self, text: str) -> Optional[str]:
        """Extrae el callsign del texto."""
        match = self.CALLSIGN_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None

    def _determine_instruction(
        self, lower_text: str, original_text: str
    ) -> tuple:
        """Determina el tipo de instrucción, verbo y parámetros."""
        parameters: Dict[str, Any] = {}

        # DETECTAR ALTITUD
        for pattern, key in self.ALTITUDE_PATTERNS:
            match = pattern.search(original_text)
            if match:
                value = int(match.group(1))
                if key == "flight_level":
                    parameters["target_altitude"] = value * 100
                    parameters["flight_level"] = value
                else:
                    parameters["target_altitude"] = value

                # Determinar si es climb o descent
                if "climb" in lower_text:
                    return InstructionType.CLIMB, "climb", parameters
                elif "descend" in lower_text or "descent" in lower_text:
                    return InstructionType.DESCENT, "descend", parameters
                else:
                    return InstructionType.MAINTAIN_ALTITUDE, "maintain", parameters

        # DETECTAR HEADING
        heading_match = self.HEADING_PATTERN.search(original_text)
        if heading_match:
            parameters["heading"] = int(heading_match.group(1))
            if "turn left" in lower_text:
                return InstructionType.TURN_LEFT, "turn_left", parameters
            elif "turn right" in lower_text:
                return InstructionType.TURN_RIGHT, "turn_right", parameters
            else:
                return InstructionType.HEADING, "heading", parameters

        # DETECTAR VELOCIDAD
        speed_match = self.SPEED_PATTERN.search(original_text)
        if speed_match:
            parameters["speed"] = int(speed_match.group(1))
            if "reduce" in lower_text:
                return InstructionType.REDUCE_SPEED, "reduce_speed", parameters
            elif "increase" in lower_text:
                return InstructionType.INCREASE_SPEED, "increase_speed", parameters
            else:
                return InstructionType.SPEED, "speed", parameters

        # DETECTAR CLEARANCES DE PISTA
        rw_match = self.RUNWAY_PATTERN.search(original_text)
        runway = rw_match.group(1).upper() if rw_match else None
        if runway:
            parameters["runway"] = runway

        if "cleared for takeoff" in lower_text:
            return InstructionType.TAKEOFF_CLEARANCE, "takeoff", parameters
        elif "cleared to land" in lower_text or "land" in lower_text:
            return InstructionType.LANDING_CLEARANCE, "land", parameters
        elif "taxi" in lower_text:
            return InstructionType.TAXI, "taxi", parameters
        elif "line up" in lower_text and "wait" in lower_text:
            return InstructionType.LINE_UP_AND_WAIT, "line_up_and_wait", parameters
        elif "line up" in lower_text:
            return InstructionType.LINE_UP, "line_up", parameters
        elif "hold short" in lower_text:
            return InstructionType.HOLD_SHORT, "hold_short", parameters
        elif "hold position" in lower_text:
            return InstructionType.HOLD_POSITION, "hold_position", parameters

        # DETECTAR GO-AROUND
        if "go around" in lower_text or "go-around" in lower_text:
            return InstructionType.GO_AROUND, "go_around", parameters

        # DETECTAR DIRECT TO
        if "direct to" in lower_text:
            # Extraer waypoint si existe
            words = lower_text.split()
            if "to" in words:
                idx = words.index("to")
                if idx + 1 < len(words):
                    parameters["waypoint"] = words[idx + 1].upper()
            return InstructionType.DIRECT_TO, "direct_to", parameters

        # DETECTAR SQUAWK
        if "squawk" in lower_text:
            sq_match = re.search(r'squawk\s+(\d{4})', lower_text)
            if sq_match:
                parameters["squawk"] = sq_match.group(1)
            return InstructionType.SQUAWK, "squawk", parameters

        # DETECTAR EMERGENCIA
        if "mayday" in lower_text:
            return InstructionType.MAYDAY, "mayday", parameters
        elif "pan pan" in lower_text:
            return InstructionType.PAN_PAN, "pan_pan", parameters
        elif "emergency descent" in lower_text:
            return InstructionType.EMERGENCY_DESCENT, "emergency_descent", parameters

        # DETECTAR CONTACT / MONITOR
        if "contact" in lower_text:
            freq_match = re.search(r'(\d{3}\.\d{1,3})', original_text)
            if freq_match:
                parameters["frequency"] = freq_match.group(1)
            return InstructionType.CONTACT, "contact", parameters
        elif "monitor" in lower_text:
            return InstructionType.MONITOR, "monitor", parameters

        # DEFAULT
        return InstructionType.UNKNOWN, "unknown", parameters
