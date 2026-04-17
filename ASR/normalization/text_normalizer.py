"""
Normalizador de textos ATC para evaluación ASR.
Convierte transcripciones a formato comparable para métricas WER.
"""

import re
import string
from typing import Optional, List, Dict, Any
from .terminology import (
    expand_callsign,
    expand_number,
    expand_icao_spelling,
    number_to_word,
    atc_terminology,
    extract_callsigns,
    airlines_icao,
    iata_to_icao,
)


class ATCTextNormalizer:
    """
    Normalizador de textos aeronáuticos para comparación ASR.
    
    Pipeline de normalización:
    1. Separar términos concatenados (ILS16R → ils 16R)
    2. Minúsculas
    3. Expansión de callsigns (JBU1676 → jetblue one six seven six)
    4. Expansión de números (FL340 → flight level three four zero)
    5. Conversión de números compuestos (eight thirty five → eight three five)
    6. Expansión opcional de ICAO (BEMOL → bravo echo mike oscar lima)
    7. Normalización de terminología ATC
    8. Limpieza de puntuación
    
    Args:
        expand_callsigns: Si expandir callsigns de aeronaves
        expand_numbers: Si expandir números dígito por dígito
        expand_icao: Si expandir palabras ICAO a alfabeto NATO
        normalize_terminology: Si reemplazar abreviaturas ATC
        remove_punctuation: Si eliminar signos de puntuación
        lowercase: Si convertir a minúsculas
        convert_compound_numbers: Si convertir números compuestos (>10) a dígitos individuales
        split_concatenated_terms: Si separar términos ATC concatenados con números
    """
    
    def __init__(
        self,
        expand_callsigns: bool = True,
        expand_numbers: bool = True,
        expand_icao: bool = False,
        normalize_terminology: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        convert_compound_numbers: bool = True,
        split_concatenated_terms: bool = True,
    ):
        self.expand_callsigns = expand_callsigns
        self.expand_numbers = expand_numbers
        self.expand_icao = expand_icao
        self.normalize_terminology = normalize_terminology
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.convert_compound_numbers = convert_compound_numbers
        self.split_concatenated_terms = split_concatenated_terms
        
        # Reverse mapping for compound number conversion
        self._word_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'niner': '9'  # Aviation pronunciation
        }
        
        # Compilar regex para eficiencia
        self._flight_level_pattern = re.compile(r'\b(FL|FLIGHT LEVEL)\s*(\d{2,3})\b', re.IGNORECASE)
        self._frequency_pattern = re.compile(r'\b(\d{2,3}\.\d{2,3})\b')
        self._squawk_pattern = re.compile(r'\b(SQUAWK|SQ)\s*(\d{4})\b', re.IGNORECASE)
        self._heading_pattern = re.compile(r'\b(HDG|HEADING)\s*(\d{3})\b', re.IGNORECASE)
        self._altitude_pattern = re.compile(r'\b(ALT|ALTITUDE)\s*(\d{2,5})\b', re.IGNORECASE)
        
        # Patrón para runway numbers (16R, 16L, 30R, etc.)
        self._runway_pattern = re.compile(r'\b(runway\s+)?(\d{1,2})(L|R)\b', re.IGNORECASE)
        
        # Patrón para términos ATC concatenados con números (ILS16R, VOR15, etc.)
        self._concatenated_terms_pattern = re.compile(
            r'\b(ILS|VOR|NDB|DME|GS|LOC|VORDME|VORTAC)(\d{1,3})(L|R|C)?\b',
            re.IGNORECASE
        )
        
        # Patrones de callsign más específicos
        self._callsign_icao_pattern = re.compile(r'\b([A-Z]{3})(\d{1,4})\b')
        self._callsign_iata_pattern = re.compile(r'\b([A-Z]{2})(\d{1,4})\b')
        
        # Patrón para callsigns parcialmente reconocidos (ej: W1676 -> whiskey one six seven six)
        self._callsign_partial_pattern = re.compile(r'\b([A-Z])(\d{1,4})\b')
    
    def normalize(self, text: str) -> str:
        """
        Aplica el pipeline completo de normalización.
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Separar términos ATC concatenados (ILS16R → ils 16R)
        if self.split_concatenated_terms:
            text = self._split_concatenated_terms(text)
        
        # 2. Expansión de callsigns (antes de expandir números)
        if self.expand_callsigns:
            text = self._expand_callsigns_in_text(text)
        
        # 3. Expansión de terminología ATC
        if self.normalize_terminology:
            text = self._normalize_atc_terminology(text)
        
        # 4. Expansión de números (después de callsigns)
        if self.expand_numbers:
            text = self._expand_numbers_in_text(text)
        
        # 5. Conversión de números compuestos a dígitos individuales
        if self.convert_compound_numbers:
            text = self._convert_compound_numbers(text)
        
        # 6. Expansión de ICAO (waypoints, etc.)
        if self.expand_icao:
            text = self._expand_icao_waypoints(text)
        
        # 7. Limpieza de puntuación
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # 8. Normalización final de espacios
        text = self._normalize_whitespace(text)

        # 9. Minúsculas al final
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _expand_callsigns_in_text(self, text: str) -> str:
        """
        Expande callsigns de aeronaves en el texto.
        
        Busca patrones como:
        - JBU1676 → jetblue one six seven six
        - NKS236 → spirit two three six
        """
        # Primero buscar códigos ICAO (3 letras)
        def replace_icao_callsign(match):
            code = match.group(1).upper()
            numbers = match.group(2)
            airline_name = airlines_icao.get(code)
            if airline_name:
                expanded_nums = self._expand_number_string(numbers)
                return f"{airline_name} {expanded_nums}"
            return match.group(0)
        
        text = self._callsign_icao_pattern.sub(replace_icao_callsign, text)
        
        # Luego códigos IATA (2 letras)
        def replace_iata_callsign(match):
            iata_code = match.group(1).upper()
            numbers = match.group(2)
            icao_code = iata_to_icao.get(iata_code)
            if icao_code:
                airline_name = airlines_icao.get(icao_code)
                if airline_name:
                    expanded_nums = self._expand_number_string(numbers)
                    return f"{airline_name} {expanded_nums}"
            return match.group(0)
        
        text = self._callsign_iata_pattern.sub(replace_iata_callsign, text)
        
        # Expandir callsigns parcialmente reconocidos (W1676 -> whiskey one six seven six)
        # Esto captura casos donde el ASR no reconoció bien el código de aerolínea
        text = self._expand_partial_callsigns(text)
        
        # Expandir números sueltos que parecen callsigns (31236 -> three one two three six)
        # Esto captura casos donde el ASR no reconoció ninguna letra del callsign
        text = self._expand_callsign_numbers(text)
        
        return text
    
    def _expand_callsign_numbers(self, text: str) -> str:
        """
        Expande números sueltos de 3-5 dígitos que parecen callsigns.
        
        Casos como:
        - 31236 -> three one two three six
        - 1676 -> one six seven six
        
        Esto captura callsigns donde el ASR no reconoció la parte de letras.
        """
        # Patrón: números de 3-5 dígitos (tamaño típico de callsign)
        pattern = re.compile(r'\b(\d{3,5})\b')
        
        def expand_match(match):
            numbers = match.group(1)
            return self._expand_number_string(numbers)
        
        return pattern.sub(expand_match, text)
    
    def _expand_partial_callsigns(self, text: str) -> str:
        """
        Expande callsigns parcialmente reconocidos.
        
        Casos como:
        - W1676 -> whiskey one six seven six
        - B6 -> bravo six
        
        Esto permite mayor fidelidad al comparar con ground truth,
        ya que los números del callsign se expanden correctamente.
        """
        def replace_partial_callsign(match):
            letter = match.group(1).upper()
            numbers = match.group(2)
            
            # Expandir la letra a NATO
            letter_nato = self._get_nato_letter(letter)
            
            # Expandir los números
            expanded_nums = self._expand_number_string(numbers)
            
            return f"{letter_nato} {expanded_nums}"
        
        # Solo expandir si no coincide con un código conocido completo
        # (para no duplicar expansiones)
        def should_expand(match):
            full_match = match.group(0)
            # Verificar si ya fue expandido como callsign conocido
            # Si el texto contiene la versión expandida, no re-expandir
            return True  # Por defecto expandir todos
        
        text = self._callsign_partial_pattern.sub(replace_partial_callsign, text)
        return text
    
    def _expand_number_string(self, number_str: str) -> str:
        """Expande un string de números a palabras."""
        words = []
        for char in number_str:
            if char.isdigit():
                words.append(number_to_word.get(char, char))
            else:
                words.append(char)
        return ' '.join(words)
    
    def _normalize_atc_terminology(self, text: str) -> str:
        """
        Normaliza abreviaturas comunes de ATC.
        """
        # Flight level: fl340 → flight level three four zero
        def expand_flight_level(match):
            numbers = match.group(2)
            expanded = self._expand_number_string(numbers)
            return f"flight level {expanded}"
        
        text = self._flight_level_pattern.sub(expand_flight_level, text)
        
        # Heading: heading 270 → heading two seven zero
        def expand_heading(match):
            numbers = match.group(2)
            expanded = self._expand_number_string(numbers)
            return f"heading {expanded}"
        
        text = self._heading_pattern.sub(expand_heading, text)
        
        # Frequencies: 133.85 → one three three decimal eight five
        def expand_frequency(match):
            freq = match.group(1)
            parts = freq.split('.')
            if len(parts) == 2:
                whole = self._expand_number_string(parts[0])
                decimal = self._expand_number_string(parts[1])
                return f"{whole} decimal {decimal}"
            return self._expand_number_string(freq)
        
        text = self._frequency_pattern.sub(expand_frequency, text)
        
        # Squawk codes
        def expand_squawk(match):
            numbers = match.group(2)
            expanded = self._expand_number_string(numbers)
            return f"squawk {expanded}"
        
        text = self._squawk_pattern.sub(expand_squawk, text)
        
        # Runway numbers: 16R → one six right, 16L → one six left
        def expand_runway(match):
            runway_prefix = match.group(1) if match.group(1) else ""
            number = match.group(2)
            direction = match.group(3).upper()
            
            # Expandir número
            expanded_number = self._expand_number_string(number)
            # Mapear L/R a left/right
            direction_word = "left" if direction == "L" else "right"
            
            if runway_prefix:
                return f"runway {expanded_number} {direction_word}"
            else:
                return f"{expanded_number} {direction_word}"
        
        text = self._runway_pattern.sub(expand_runway, text)
        
        # Otras abreviaturas comunes
        replacements = {
            r'\btfc\b': 'traffic',
            r'\bwx\b': 'weather',
            r'\bdep\b': 'departure',
            r'\barr\b': 'arrival',
            r'\bapp\b': 'approach',
            r'\btwr\b': 'tower',
            r'\bgnd\b': 'ground',
            r'\bctr\b': 'center',
            r'\bhab\b': 'havana',
            r'\bpax\b': 'passengers',
            r'\brad\b': 'radar',
            r'\bnav\b': 'navigation',
            r'\bhdg\b': 'heading',
            r'\balt\b': 'altitude',
            r'\bspd\b': 'speed',
            r'\bpos\b': 'position',
            r'\bvis\b': 'visibility',
            r'\bnm\b': 'nautical miles',
            r'\bft\b': 'feet',
            r'\brwy\b': 'runway',
            r'\btaxiwy\b': 'taxiway',
            r'\bclnc\b': 'clearance',
            r'\batis\b': 'atis',
            r'\bnotam\b': 'notam',
            r'\bmetar\b': 'metar',
            r'\btaf\b': 'taf',
            r'\bniner\b': 'nine',  # Aviation pronunciation
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _expand_numbers_in_text(self, text: str) -> str:
        """
        Expande números restantes en el texto.
        Expande TODOS los números sin límite de dígitos.
        """
        def expand_number_match(match):
            number = match.group(0)
            return self._expand_number_string(number)
        
        # Expandir TODOS los números sin límite de dígitos
        pattern = r'\b\d+\b'
        text = re.sub(pattern, expand_number_match, text)
        
        return text
    
    def _expand_icao_waypoints(self, text: str) -> str:
        """
        Expande waypoints/navaids de 4-5 letras a alfabeto NATO.
        Ej: BORDO → bravo oscar romeo delta oscar
        """
        # Buscar palabras de 4-5 letras mayúsculas que podrían ser waypoints
        def expand_waypoint(match):
            word = match.group(0)
            if len(word) >= 4 and word.isalpha() and word.isupper():
                # Expandir a NATO
                letters = [number_to_word.get(c, c.lower()) if c.isdigit() 
                          else self._get_nato_letter(c) for c in word]
                return ' '.join(letters)
            return word
        
        # Patrón: palabras de 4-5 letras mayúsculas
        text = re.sub(r'\b[A-Z]{4,5}\b', expand_waypoint, text)
        
        return text
    
    def _get_nato_letter(self, letter: str) -> str:
        """Obtiene la palabra NATO para una letra."""
        from .terminology import nato_alphabet
        return nato_alphabet.get(letter.upper(), letter.lower())
    
    def _remove_punctuation(self, text: str) -> str:
        """Reemplaza signos de puntuación con espacios.
        
        Esto permite que patrones como '8-0-0' se conviertan en '8 0 0'
        y luego se expandan a 'eight zero zero'.
        """
        # Reemplazar toda puntuación con espacios
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        return text.translate(translator)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normaliza espacios múltiples."""
        return ' '.join(text.split())
    
    def _split_concatenated_terms(self, text: str) -> str:
        """
        Separa términos ATC concatenados con números.
        
        Ejemplos:
        - ILS16R → ils 16R
        - VOR15 → vor 15
        - DME340 → dme 340
        """
        def split_term(match):
            term = match.group(1).lower()
            numbers = match.group(2)
            direction = match.group(3) if match.group(3) else ""
            
            if direction:
                return f"{term} {numbers}{direction}"
            else:
                return f"{term} {numbers}"
        
        return self._concatenated_terms_pattern.sub(split_term, text)
    
    def _convert_compound_numbers(self, text: str) -> str:
        """
        Convierte números compuestos en inglés (>10) a dígitos individuales.
        
        Ejemplos:
        - "eight thirty five" → "eight three five"
        - "twenty one" → "two one"
        - "one hundred twenty three" → "one two three"
        
        También maneja números en formato de dígitos:
        - "eight 35" → "eight three five"
        """
        # Patrones de números compuestos comunes en aviación
        # Mantener las palabras de dígitos (one, two, etc.) no los dígitos numéricos
        compound_patterns = [
            # teens: ten one → one one (11), ten two → one two (12), etc.
            (r'\bten\s+(one|two|three|four|five|six|seven|eight|nine)\b', 
             lambda m: f"one {m.group(1)}"),
            # twenty X → two X
            (r'\btwenty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"two {m.group(1)}"),
            # thirty X → three X
            (r'\bthirty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"three {m.group(1)}"),
            # forty X → four X
            (r'\bforty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"four {m.group(1)}"),
            # fifty X → five X
            (r'\bfifty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"five {m.group(1)}"),
            # sixty X → six X
            (r'\bsixty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"six {m.group(1)}"),
            # seventy X → seven X
            (r'\bseventy\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"seven {m.group(1)}"),
            # eighty X → eight X
            (r'\beighty\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"eight {m.group(1)}"),
            # ninety X → nine X
            (r'\bninety\s+(one|two|three|four|five|six|seven|eight|nine)\b',
             lambda m: f"nine {m.group(1)}"),
            # Números de decenas solos: thirty → three zero, twenty → two zero, etc.
            (r'\b(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
             lambda m: self._tens_word_to_digits(m.group(1))),
        ]
        
        for pattern, replacement in compound_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _tens_word_to_digits(self, word: str) -> str:
        """Convierte palabras de decenas a dígitos."""
        tens_map = {
            'ten': 'one zero',
            'twenty': 'two zero',
            'thirty': 'three zero',
            'forty': 'four zero',
            'fifty': 'five zero',
            'sixty': 'six zero',
            'seventy': 'seven zero',
            'eighty': 'eight zero',
            'ninety': 'nine zero',
        }
        return tens_map.get(word.lower(), word)
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normaliza una lista de textos.
        
        Args:
            texts: Lista de strings
            
        Returns:
            Lista de textos normalizados
        """
        return [self.normalize(text) for text in texts]
    
    def normalize_dict(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Normaliza los valores de un diccionario.
        
        Args:
            data: Diccionario {key: text}
            
        Returns:
            Diccionario con valores normalizados
        """
        return {k: self.normalize(v) for k, v in data.items()}


def quick_normalize(text: str, **kwargs) -> str:
    """
    Función de conveniencia para normalización rápida.
    
    Args:
        text: Texto a normalizar
        **kwargs: Opciones para ATCTextNormalizer
        
    Returns:
        Texto normalizado
    """
    normalizer = ATCTextNormalizer(**kwargs)
    return normalizer.normalize(text)
