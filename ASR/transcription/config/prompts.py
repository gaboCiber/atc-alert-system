"""
Prompts configurables para modelos ASR.
"""

# Prompt básico de ATC (como en el notebook del usuario)
ATC_TERMINOLOGY = """
Air Traffic Control communications
"""

# Alfabeto fonético NATO
NATO_ALPHABET = """
alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliett, 
kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, 
uniform, victor, whiskey, xray, yankee, zulu
"""

# Terminología ATC común
ATC_TERMS = """
climb, climbing, descend, descending, passing, feet, knots, degrees, direct, 
maintain, identified, ILS, VFR, IFR, contact, frequency, turn, right, left, 
heading, altitude, flight, level, cleared, squawk, approach, runway, 
established, report, affirm, negative, wilco, roger, radio, radar
"""

# Términos específicos de Cuba (según el notebook)
CUBA_TERMS = """
one, two, three, four, five, six, seven, eight, niner, 
Habana, jetblue, spirit, Bemol, Bordo, Zeuss
"""

# Aerolíneas comunes en el espacio aéreo cubano
AIRLINES = """
American, Copa, JetBlue, Delta, Avianca, Spirit, United, Southwest, 
WestJet, Air Transat, Sunwing, Air Canada, Tampa, LAN Ecuador, Bimini, 
LAN Peru, Giant, Amerijet, TACA, Western, UPS, Aerolane, Aeromexico, 
Caribbean Airlines, Sharjah, Aires, LAN, Cayman, Air Canada Rouge, Dae, 
Línea Carga, Iberia, Viva, GXA, FedEx, Tom, Chasqui, GLG, TSU, Lacsa, 
Envoy, FAR, Aerolineas Argentinas, Frontier, Cargojet, Air Europa, 
Speedbird, Abex, I-Way, TAM, Condor, Airfrans, Flexjet, Turkish, Azul, 
Volaris, Republic
"""

# Prompt por defecto para Whisper (basado en el notebook)
DEFAULT_ATC_PROMPT = f"""{ATC_TERMINOLOGY.strip()} {NATO_ALPHABET.replace(chr(10), ' ').strip()} {ATC_TERMS.strip()} {CUBA_TERMS.strip()} {AIRLINES.strip()}"""

# Prompt minimalista (solo esencial)
MINIMAL_ATC_PROMPT = """
ATC communications, NATO alphabet, flight levels, frequencies, 
callsigns, JetBlue, Spirit, American, Habana, radar, climb, descend, 
maintain, heading, direct, cleared
""".strip()

# Prompt extendido con más terminología
EXTENDED_ATC_PROMPT = f"""{DEFAULT_ATC_PROMPT}

Additional ATC terminology:
request, cleared to, hold short, line up, take off, departure, arrival, 
ground, tower, center, approach, final, intercept, vectors, 
traffic, caution, confirm, say again, standby, check, read back, 
QNH, QFE, transition level, flight plan, waypoint, fix, VOR, NDB, 
DME, GPS, RNAV, RNP, holding pattern, standard instrument departure, 
standard terminal arrival route, missed approach, go around, 
emergency, priority, mayday, pan pan, squawk, ident
""".strip()

# Diccionario de prompts disponibles
AVAILABLE_PROMPTS = {
    "default": DEFAULT_ATC_PROMPT,
    "minimal": MINIMAL_ATC_PROMPT,
    "extended": EXTENDED_ATC_PROMPT,
    "none": None,
}


def get_prompt(name: str = "default") -> str:
    """
    Obtiene un prompt por nombre.
    
    Args:
        name: Nombre del prompt ("default", "minimal", "extended", "none")
        
    Returns:
        El prompt o None
    """
    return AVAILABLE_PROMPTS.get(name, DEFAULT_ATC_PROMPT)


def create_custom_prompt(
    include_nato: bool = True,
    include_terminology: bool = True,
    include_airlines: bool = True,
    include_cuba_terms: bool = True,
    extra_terms: str = ""
) -> str:
    """
    Crea un prompt personalizado seleccionando componentes.
    
    Args:
        include_nato: Incluir alfabeto NATO
        include_terminology: Incluir terminología ATC
        include_airlines: Incluir aerolíneas
        include_cuba_terms: Incluir términos cubanos
        extra_terms: Términos adicionales personalizados
        
    Returns:
        Prompt personalizado
    """
    parts = [ATC_TERMINOLOGY.strip()]
    parts.append("\n")
    
    if include_nato:
        parts.append("NATO ALPHABET: ")
        parts.append(NATO_ALPHABET.replace(chr(10), ' ').strip())
        parts.append("\n")
    if include_terminology:
        parts.append("ATC TERMINOLOGY: ")
        parts.append(ATC_TERMS.replace("\n", " ").strip())
        parts.append("\n")
    if include_airlines:
        parts.append("AIRLINES: ")
        parts.append(AIRLINES.replace("\n", " ").strip())
        parts.append("\n")
    if include_cuba_terms:
        parts.append("CUBA TERMS: ")
        parts.append(CUBA_TERMS.replace("\n", "").strip())
        parts.append("\n")
    if extra_terms:
        parts.append("EXTRA TERMS: ")
        parts.append(extra_terms)
    
    return " ".join(parts)
