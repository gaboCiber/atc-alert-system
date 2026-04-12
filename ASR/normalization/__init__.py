"""
Módulo de normalización de textos ASR para ATC.
"""

from .text_normalizer import ATCTextNormalizer, quick_normalize
from .terminology import (
    expand_callsign,
    expand_number,
    expand_icao_spelling,
    number_to_word,
    airlines_icao,
    iata_to_icao,
    atc_terminology,
    nato_alphabet,
)

__all__ = [
    'ATCTextNormalizer',
    'quick_normalize',
    'expand_callsign',
    'expand_number',
    'expand_icao_spelling',
    'number_to_word',
    'airlines_icao',
    'iata_to_icao',
    'atc_terminology',
    'nato_alphabet',
]
