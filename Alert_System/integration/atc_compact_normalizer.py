"""
Normalizador para convertir texto ATC expandido a formato compacto.

Usa los diccionarios inversos de ASR.normalization.terminology para
convertir "american one two three descend to flight level two four zero"
a "AAL123 descend to FL240".
"""

import re
from typing import Dict, List, Optional

# Importar diccionarios existentes del ASR
from ASR.normalization.terminology import (
    nato_to_letter,
    word_to_number,
    airlines_icao,
)


class ATCCompactNormalizer:
    """
    Convierte texto ATC expandido (palabras) a formato compacto (códigos/números).
    """
    
    def __init__(self):
        # Invertir airlines_icao: nombre_aerolinea -> código_ICAO
        self.airline_to_icao: Dict[str, str] = {}
        for code, name in airlines_icao.items():
            # Mapear nombre completo y variaciones
            self.airline_to_icao[name.lower()] = code
            # Mapear primera palabra también
            first_word = name.split()[0].lower()
            if first_word not in self.airline_to_icao:
                self.airline_to_icao[first_word] = code
        
        # Mapeos adicionales para variaciones comunes
        self.airline_to_icao.update({
            "american airline": "AAL",  # American Airlines
            "american airlines": "AAL",
            "united airline": "UAL",  # United Airlines  
            "united airlines": "UAL",
            "delta airline": "DAL",   # Delta Airlines
            "delta airlines": "DAL",
            "british airway": "BAW",  # British Airways
            "british airways": "BAW",
            "continental airline": "COA",  # Continental
            "continental airlines": "COA",
            "southwest airline": "SWA",   # Southwest
            "southwest airlines": "SWA",
            "jetblue": "JBU",  # JetBlue
            "spirit airline": "NK",  # Spirit
            "spirit airlines": "NK",
            "frontier airline": "FFT",  # Frontier
            "frontier airlines": "FFT",
            "alaska airline": "ASA",  # Alaska
            "alaska airlines": "ASA",
        })
        
        # Compilar patrones regex
        self._flight_level_pattern = re.compile(
            r'flight\s+level\s+((?:zero|one|two|three|four|five|six|seven|eight|nine|niner)\s*)+',
            re.IGNORECASE
        )
        self._runway_pattern = re.compile(
            r'runway\s+((?:zero|one|two|three|four|five|six|seven|eight|nine|niner)\s*)+'
            r'(left|right|center|L|R|C)?\b',
            re.IGNORECASE
        )
        self._heading_pattern = re.compile(
            r'heading\s+((?:zero|one|two|three|four|five|six|seven|eight|nine)\s*)+',
            re.IGNORECASE
        )
        self._altitude_pattern = re.compile(
            r'(\d+)\s*feet',
            re.IGNORECASE
        )
    
    def normalize(self, text: str) -> str:
        """
        Convierte texto expandido a formato compacto ATC.
        
        Args:
            text: Texto expandido (ej: "american one two three descend to flight level two four zero")
            
        Returns:
            Texto compacto (ej: "AAL123 descend to FL240")
        """
        if not text:
            return text
        
        result = text.lower()
        
        # 1. Convertir flight level X → FLXXX
        result = self._compact_flight_level(result)
        
        # 2. Convertir runway X → RWYXX
        result = self._compact_runway(result)
        
        # 3. Convertir heading X → HDGXXX
        result = self._compact_heading(result)
        
        # 4. Convertir callsign (american one two three → AAL123)
        result = self._compact_callsign(result)
        
        # 5. Limpiar espacios extra
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def _compact_flight_level(self, text: str) -> str:
        """Convierte 'flight level two four zero' a 'FL240'."""
        def replace_fl(match):
            words = match.group(0)
            # Extraer solo las palabras de números
            digits = []
            for word in words.lower().split():
                if word in word_to_number:
                    digits.append(word_to_number[word])
            if digits:
                return f"FL{''.join(digits)}"
            return match.group(0)
        
        return self._flight_level_pattern.sub(replace_fl, text)
    
    def _compact_runway(self, text: str) -> str:
        """Convierte 'runway zero four left' a 'RWY04L'."""
        def replace_rwy(match):
            words = match.group(0)
            digits = []
            suffix = ""
            for word in words.lower().split():
                if word in word_to_number:
                    digits.append(word_to_number[word])
                elif word in ('left', 'l'):
                    suffix = 'L'
                elif word in ('right', 'r'):
                    suffix = 'R'
                elif word in ('center', 'c'):
                    suffix = 'C'
            if digits:
                return f"RWY{''.join(digits)}{suffix}"
            return match.group(0)
        
        return self._runway_pattern.sub(replace_rwy, text)
    
    def _compact_heading(self, text: str) -> str:
        """Convierte 'heading two seven zero' a 'HDG270'."""
        def replace_hdg(match):
            words = match.group(0)
            digits = []
            for word in words.lower().split():
                if word in word_to_number:
                    digits.append(word_to_number[word])
            if digits:
                return f"HDG{''.join(digits)}"
            return match.group(0)
        
        return self._heading_pattern.sub(replace_hdg, text)
    
    def _compact_callsign(self, text: str) -> str:
        """Convierte 'american one two three' a 'AAL123'."""
        words = text.split()
        if len(words) < 2:
            return text
        
        # Buscar aerolínea en las primeras palabras
        for i in range(min(3, len(words))):
            # Probar con nombre completo (hasta 3 palabras)
            possible_name = ' '.join(words[:i+1])
            if possible_name in self.airline_to_icao:
                # Encontramos la aerolínea, ahora extraer números
                icao_code = self.airline_to_icao[possible_name]
                remaining_words = words[i+1:]
                
                # Convertir números a dígitos
                digits = []
                for word in remaining_words:
                    if word in word_to_number:
                        digits.append(word_to_number[word])
                    elif word in nato_to_letter:
                        # Letra del alfabeto NATO (ej: "alpha" → "A")
                        digits.append(nato_to_letter[word])
                    else:
                        # Palabra no reconocida, terminar
                        break
                
                if digits:
                    return f"{icao_code}{''.join(digits)} {' '.join(remaining_words[len(digits):])}"
        
        return text


# Instancia global para uso directo
default_compact_normalizer = ATCCompactNormalizer()


def normalize_to_compact(text: str) -> str:
    """Función de conveniencia para normalizar a formato compacto."""
    return default_compact_normalizer.normalize(text)
