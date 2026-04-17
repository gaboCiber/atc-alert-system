"""
Terminología aeronáutica para normalización de textos ASR.
Incluye: alfabeto NATO, números, códigos de aerolíneas y utilidades de expansión.
"""

import re
from typing import Dict, List, Optional

# ==========================================
# ALFABETO NATO
# ==========================================

nato_alphabet = {
    'A': 'alpha', 'B': 'bravo', 'C': 'charlie', 'D': 'delta',
    'E': 'echo', 'F': 'foxtrot', 'G': 'golf', 'H': 'hotel',
    'I': 'india', 'J': 'juliett', 'K': 'kilo', 'L': 'lima',
    'M': 'mike', 'N': 'november', 'O': 'oscar', 'P': 'papa',
    'Q': 'quebec', 'R': 'romeo', 'S': 'sierra', 'T': 'tango',
    'U': 'uniform', 'V': 'victor', 'W': 'whiskey', 'X': 'xray',
    'Y': 'yankee', 'Z': 'zulu',
}

# Invertido para búsqueda
nato_to_letter = {v: k for k, v in nato_alphabet.items()}

# ==========================================
# NÚMEROS
# ==========================================

number_to_word = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten'
}

word_to_number = {v: k for k, v in number_to_word.items()}

# ==========================================
# CÓDIGOS DE AEROLÍNEAS (ICAO / IATA)
# ==========================================

# Códigos ICAO de 3 letras → nombre de aerolínea
airlines_icao = {
    "AAB": "air belgium",
    "AAL": "american",
    "AAR": "asiana",
    "ABP": "bair",
    "ABR": "contract",
    "ABW": "airbridge cargo",
    "ACA": "air canada",
    "ACI": "aircalin",
    "ADB": "antonov bureau",
    "ADN": "aerodienst",
    "AEA": "europa",
    "AEE": "aegean",
    "AFL": "aeroflot",
    "AFR": "air france",
    "AHO": "air hamburg",
    "AIB": "airbus industrie",
    "AIC": "air india",
    "AIJ": "interjet",
    "AJU": "airjetsul",
    "ALK": "srilankan",
    "ALT": "aerlineas centrales",
    "AMC": "air malta",
    "AMQ": "amex",
    "ANA": "all nippon",
    "ANE": "air nostrum",
    "ANG": "niugini",
    "ANZ": "air new zealand",
    "AOJ": "asterix",
    "ARG": "aerolineas argentinas",
    "ASH": "mesa",
    "ASL": "air serbia",
    "ASM": "awesome",
    "ASY": "aussie",
    "ATL": "centralmex",
    "ATN": "amazon air",
    "AUA": "austrian",
    "AUI": "ukraine international",
    "AVA": "avianca",
    "AWC": "air charter",
    "AWI": "air wisconsin",
    "AWK": "airwork",
    "AXE": "galileo",
    "AXY": "legend",
    "AZA": "alitalia",
    "AZE": "arcus air",
    "AZG": "silk west",
    "BAO": "avangard",
    "BAW": "british",
    "BCI": "blue islands",
    "BCS": "eurotrans",
    "BEL": "brussels",
    "BLA": "all charter",
    "BMS": "bmi",
    "BOH": "bohemia",
    "BOV": "boliviana",
    "BPO": "pirol",
    "BRO": "broadsword",
    "BRU": "belarus avia",
    "BTI": "air baltic",
    "CAI": "corendon",
    "CAL": "continental",
    "CAZ": "eurocat",
    "CCA": "air china",
    "CEF": "czech air force",
    "CES": "china eastern",
    "CFG": "condor",
    "CFH": "care flight",
    "CGF": "clever",
    "CHH": "hainan",
    "CKS": "connie",
    "CLF": "clifton",
    "CLX": "cargolux",
    "CMB": "camber",
    "CMP": "copa",
    "COL": "colair",
    "CPA": "cathay pacific",
    "CRK": "bauhinia",
    "CSA": "czech",
    "CSN": "china southern",
    "CTN": "croatia",
    "CUB": "cubana",
    "CXA": "xiamen air",
    "DAL": "delta",
    "DCS": "twin star",
    "DFC": "dark blue",
    "DGX": "dasna",
    "DHK": "world express",
    "DLA": "dolomiti",
    "DLH": "lufthansa",
    "DSO": "dassault",
    "EAL": "eastern",
    "EAT": "trans europe",
    "EAV": "mayflower",
    "ECA": "eurocypria",
    "ECC": "eclair",
    "EDC": "saltire",
    "EDV": "endeavor",
    "EDW": "edelweiss",
    "EFD": "ever flight",
    "EIN": "aer lingus",
    "EJU": "alpine",
    "ELJ": "elite jet",
    "ELY": "el al",
    "ENT": "enter",
    "ENY": "envoy",
    "ETD": "etihad",
    "ETH": "ethiopian",
    "EVA": "eva",
    "EWG": "eurowings",
    "EXS": "channex",
    "EZS": "topswiss",
    "EZY": "easyjet",
    "FBU": "french bee",
    "FCA": "jetset",
    "FCK": "nav checker",
    "FDX": "fedex",
    "FHM": "eurobird",
    "FIN": "finnair",
    "FJI": "pacific",
    "FLG": "flagship",
    "FNK": "aurika",
    "FRF": "fairfleet",
    "FTC": "affaires tchad",
    "FTY": "fly tyrol",
    "FYL": "flyinglux",
    "GAC": "dream team",
    "GAF": "german air force",
    "GDK": "goldeck",
    "GIA": "indonesia",
    "GIZ": "afrilens",
    "GJS": "gojet",
    "GLP": "globeair",
    "GSW": "arrow jet",
    "HBA": "harbour air",
    "HFM": "moonraker",
    "HHN": "rooster",
    "HMJ": "harmony",
    "HRN": "heronair",
    "HVN": "vietnam",
    "HYP": "hyperion",
    "IBE": "iberia",
    "ICE": "icelandair",
    "ICV": "cargo med",
    "IFA": "red angel",
    "IJM": "jet management",
    "IRA": "iranair",
    "JAL": "japan airlines",
    "JBU": "jetblue",
    "JFA": "mosquito",
    "JMP": "jump run",
    "JST": "jetstar",
    "JTE": "jetex",
    "JZA": "jazz",
    "KAL": "korean air",
    "KLM": "klm",
    "KQA": "kenya",
    "KRP": "carpatair",
    "LAN": "latam",
    "LGL": "luxair",
    "LMJ": "masterjet",
    "LNX": "lonex",
    "LOF": "trans states",
    "LOT": "lot polish",
    "LRQ": "lux rescue",
    "LWG": "luxwing",
    "LXA": "red lion",
    "LXG": "luxor",
    "LXJ": "flexjet",
    "LZB": "flying bulgaria",
    "MAH": "malév",
    "MAL": "malev",
    "MAS": "malaysia",
    "MEA": "cedar jet",
    "MLD": "air moldova",
    "MMD": "mermaid",
    "MNB": "black sea",
    "MSR": "egyptair",
    "MTL": "mitavia",
    "MXA": "mexicana",
    "MXY": "breeze",
    "NAX": "norwegian",
    "NJE": "fraction",
    "NKS": "spirit",
    "NOS": "moonflower",
    "NWA": "northwest",
    "NWS": "northwind",
    "OMS": "mazoon",
    "ORF": "oman",
    "PAA": "pan am",
    "PAL": "philippine",
    "PCH": "pilatus",
    "PDT": "piedmont",
    "PFY": "pelflight",
    "PGT": "sunturk",
    "PNC": "panairsa",
    "PSA": "psa",
    "PTA": "ptarmigan",
    "PVL": "pal",
    "PWF": "private wings",
    "QAJ": "dagobert",
    "QFA": "qantas",
    "QGA": "quadriga",
    "QJE": "qjet",
    "QLK": "qlink",
    "QNR": "queen air",
    "QTR": "qatar",
    "QXE": "horizon",
    "RAM": "royal air maroc",
    "RCH": "reach",
    "RGA": "royal ghana",
    "ROF": "romavia",
    "ROT": "tarom",
    "ROU": "air canada rouge",
    "RPA": "republic",
    "RRR": "ascot/kittyhawk",
    "RXA": "rex",
    "RYR": "ryanair",
    "SAA": "south african",
    "SAS": "scandinavian",
    "SAZ": "swiss ambulance",
    "SBI": "siberian airlines",
    "SCR": "silver cloud",
    "SCX": "sun country",
    "SDM": "russia",
    "SDR": "sundair",
    "SFS": "southern frontier",
    "SIA": "singapore",
    "SKW": "skywest",
    "SLA": "sky lease",
    "SLG": "lifeguard",
    "SND": "arsam",
    "SON": "sunshine tours",
    "SPG": "spring air",
    "SRN": "sprintair",
    "SSG": "slovak government",
    "SUA": "air silesia",
    "SUI": "swiss air force",
    "SVA": "saudi arabian",
    "SVW": "silver arrows",
    "SWA": "southwest",
    "SWR": "swiss",
    "SWT": "swift",
    "SXS": "sunexpress",
    "TAM": "tam",
    "TAP": "tap",
    "TAR": "tunair",
    "TAY": "quality",
    "TGO": "transport canada",
    "THA": "thai",
    "THY": "turkish",
    "TIE": "time air",
    "TJS": "tyroljet",
    "TOM": "tui",
    "TOY": "toyota",
    "TRA": "transavia",
    "TUI": "tuifly",
    "TVF": "france soleil",
    "TVP": "jet travel",
    "TVS": "sky travel",
    "TWA": "trans world",
    "TWG": "twingoose",
    "UAE": "emirates",
    "UAL": "united",
    "UPS": "ups",
    "UZB": "uzbek",
    "VDA": "volga",
    "VIP": "sovereign",
    "VIR": "virgin",
    "VIV": "volaris",
    "VJT": "vista",
    "VKA": "vulkan air",
    "VLG": "vueling",
    "VMP": "vampire",
    "VNA": "vietnam",
    "VOZ": "velocity",
    "VQT": "vqtgg",
    "WJA": "westjet",
    "WUK": "wizz go",
    "WZZ": "wizz air",
    "XAX": "xanadu",
    "XGO": "pastis"
}


# Códigos IATA de 2 letras → ICAO (para conversión)
iata_to_icao = {
    'B6': 'JBU', 'NK': 'NKS', 'BA': 'BAW', 'AA': 'AAL', 'UA': 'UAL',
    'DL': 'DAL', 'WN': 'SWA', 'FX': 'FDX', '5X': 'UPS', 'IB': 'IBE',
    'AV': 'AVA', 'LA': 'LAN', 'CM': 'CMP', 'CU': 'CUB', 'VS': 'VIR',
    'BY': 'TOM', 'U2': 'EZY', 'FR': 'RYR', 'DE': 'CFG', 'VY': 'VLG',
    'TP': 'TAP', 'AF': 'AFR', 'LH': 'DLH', 'KL': 'KLM', 'LX': 'SWR',
    'JJ': 'TAM', 'AR': 'ARG', 'MX': 'MXA', 'Y4': 'VIV', 'AI': 'AIC',
    'AC': 'ACA', 'QK': 'JZA', 'WS': 'WJA', 'KE': 'KAL', 'JL': 'JAL',
    'NZ': 'ANZ', 'CX': 'CPA', 'EK': 'UAE', 'EY': 'ETD', 'QR': 'QTR',
    'SV': 'SVA', 'TK': 'THY', 'LY': 'ELY', 'SQ': 'SIA', 'MH': 'MAS',
    'PR': 'PAL', 'VN': 'VNA', 'NH': 'ANA', 'MU': 'CES', 'CZ': 'CSN',
    'CA': 'CCA', 'KQ': 'KQA', 'SA': 'SAA', 'ET': 'ETH', 'MS': 'MSR',
    'AT': 'RAM', 'OS': 'AUA', 'AY': 'FIN', 'SK': 'SAS', 'LO': 'LOT',
    'OK': 'CSA', 'EI': 'EIN', 'W6': 'WZZ', 'DY': 'NAX', 'FI': 'ICE',
    'EN': 'DLH', 'YX': 'RPA', 'MQ': 'ENY', 'OH': 'RPA', 'PT': 'PDT',
    'ZW': 'AWI', 'QX': 'QXE', 'MX': 'MXY', 'SY': 'SCX',
}

# ==========================================
# TERMINOLOGÍA ATC COMÚN
# ==========================================

atc_terminology = {
    'FL': 'flight level',
    'FLT': 'flight',
    'DEP': 'departure',
    'ARR': 'arrival',
    'APP': 'approach',
    'TWR': 'tower',
    'GND': 'ground',
    'RMP': 'ramp',
    'DEL': 'delivery',
    'CTR': 'center',
    'HAB': 'habana',
    'HAV': 'havana',
    'CAB': 'cuba',
    'ATC': 'atc',
    'PAX': 'passengers',
    'WX': 'weather',
    'WXR': 'weather radar',
    'WXCOND': 'weather conditions',
    'TFC': 'traffic',
    'TURB': 'turbulence',
    'TURBC': 'turbulence',
    'CB': 'cumulonimbus',
    'TS': 'thunderstorm',
    'RAD': 'radar',
    'NAV': 'navigation',
    'HDG': 'heading',
    'ALT': 'altitude',
    'SPD': 'speed',
    'CRS': 'course',
    'POS': 'position',
    'LOC': 'localizer',
    'GS': 'glide slope',
    'VOR': 'vor',
    'NDB': 'ndb',
    'DME': 'dme',
    'ILS': 'ils',
    'MLS': 'mls',
    'GPS': 'gps',
    'RNAV': 'rnav',
    'RNP': 'rnp',
    'FMS': 'fms',
    'AFCS': 'afcs',
    'AP': 'autopilot',
    'FD': 'flight director',
    'YD': 'yaw damper',
    'TAS': 'true airspeed',
    'IAS': 'indicated airspeed',
    'GS': 'ground speed',
    'VS': 'vertical speed',
    'MACH': 'mach',
    'FT': 'feet',
    'NM': 'nautical miles',
    'KM': 'kilometers',
    'SM': 'statute miles',
    'KG': 'kilograms',
    'LBS': 'pounds',
    'GAL': 'gallons',
    'L': 'liters',
    'QNH': 'qnh',
    'QFE': 'qfe',
    'QNE': 'qne',
    'HPa': 'hectopascals',
    'INHG': 'inches of mercury',
    'MMHG': 'millimeters of mercury',
    'RVR': 'runway visual range',
    'VIS': 'visibility',
    'CLD': 'cloud',
    'SCT': 'scattered',
    'BKN': 'broken',
    'OVC': 'overcast',
    'FEW': 'few',
    'CLR': 'clear',
    'SKC': 'sky clear',
    'CAVOK': 'cavok',
    'VMC': 'visual meteorological conditions',
    'IMC': 'instrument meteorological conditions',
    'IFR': 'instrument flight rules',
    'VFR': 'visual flight rules',
    'SVFR': 'special vfr',
    'PIC': 'pilot in command',
    'SIC': 'second in command',
    'FO': 'first officer',
    'CA': 'captain',
    'FA': 'flight attendant',
    'PF': 'pilot flying',
    'PM': 'pilot monitoring',
    'PNF': 'pilot non flying',
    'NOTAM': 'notam',
    'METAR': 'metar',
    'TAF': 'taf',
    'SIGMET': 'sigmet',
    'AIRMET': 'airmet',
    'PIREP': 'pirep',
    'ASHTAM': 'ashtam',
    'SNOWTAM': 'snowtam',
    'BIRDTAM': 'birdtam',
    'VOR': 'vor',
    'NDB': 'ndb',
    'DME': 'dme',
    'ILS': 'ils',
    'MLS': 'mls',
    'GPS': 'gps',
    'RNAV': 'rnav',
    'RNP': 'rnp',
    'FMS': 'fms',
    'AFCS': 'afcs',
    'AP': 'autopilot',
    'FD': 'flight director',
    'YD': 'yaw damper',
    'TAS': 'true airspeed',
    'IAS': 'indicated airspeed',
    'GS': 'ground speed',
    'VS': 'vertical speed',
    'MACH': 'mach',
    'FT': 'feet',
    'NM': 'nautical miles',
    'KM': 'kilometers',
    'SM': 'statute miles',
    'KG': 'kilograms',
    'LBS': 'pounds',
    'GAL': 'gallons',
    'L': 'liters',
    'QNH': 'qnh',
    'QFE': 'qfe',
    'QNE': 'qne',
    'HPa': 'hectopascals',
    'INHG': 'inches of mercury',
    'MMHG': 'millimeters of mercury',
    'RVR': 'runway visual range',
    'VIS': 'visibility',
    'CLD': 'cloud',
    'SCT': 'scattered',
    'BKN': 'broken',
    'OVC': 'overcast',
    'FEW': 'few',
    'CLR': 'clear',
    'SKC': 'sky clear',
    'CAVOK': 'cavok',
    'VMC': 'visual meteorological conditions',
    'IMC': 'instrument meteorological conditions',
    'IFR': 'instrument flight rules',
    'VFR': 'visual flight rules',
    'SVFR': 'special vfr',
    'PIC': 'pilot in command',
    'SIC': 'second in command',
    'FO': 'first officer',
    'CA': 'captain',
    'FA': 'flight attendant',
    'PF': 'pilot flying',
    'PM': 'pilot monitoring',
    'PNF': 'pilot non flying',
    'NOTAM': 'notam',
    'METAR': 'metar',
    'TAF': 'taf',
    'SIGMET': 'sigmet',
    'AIRMET': 'airmet',
    'PIREP': 'pirep',
    'ASHTAM': 'ashtam',
    'SNOWTAM': 'snowtam',
    'BIRDTAM': 'birdtam',
}

# ==========================================
# FUNCIONES DE EXPANSIÓN
# ==========================================

def expand_digit(digit: str) -> str:
    """Expande un dígito individual a palabra."""
    return number_to_word.get(digit, digit)


def expand_number(number_str: str) -> str:
    """
    Expande un número a palabras dígito por dígito.
    Ej: "1676" → "one six seven six"
    """
    words = []
    for char in number_str:
        if char.isdigit():
            words.append(number_to_word.get(char, char))
        else:
            words.append(char)
    return ' '.join(words)


def expand_callsign(callsign: str) -> Optional[str]:
    """
    Expande un callsign de aeronave.
    Ej: "JBU1676" → "jetblue one six seven six"
    Ej: "NKS236" → "spirit two three six"
    
    Args:
        callsign: String como "JBU1676" o "B6"
        
    Returns:
        String expandido o None si no se reconoce
    """
    if not callsign:
        return None
    
    callsign = callsign.strip().upper()
    
    # Patrón: Código ICAO (3 letras) + números
    icao_match = re.match(r'^([A-Z]{3})(\d+)$', callsign)
    if icao_match:
        code = icao_match.group(1)
        numbers = icao_match.group(2)
        airline_name = airlines_icao.get(code)
        if airline_name:
            expanded_nums = expand_number(numbers)
            return f"{airline_name} {expanded_nums}"
    
    # Patrón: Código IATA (2 letras) + números
    iata_match = re.match(r'^([A-Z]{2})(\d+)$', callsign)
    if iata_match:
        iata_code = iata_match.group(1)
        numbers = iata_match.group(2)
        icao_code = iata_to_icao.get(iata_code)
        if icao_code:
            airline_name = airlines_icao.get(icao_code)
            if airline_name:
                expanded_nums = expand_number(numbers)
                return f"{airline_name} {expanded_nums}"
    
    return None


def expand_icao_spelling(text: str) -> str:
    """
    Expande letras individuales a alfabeto NATO.
    Ej: "BEMOL" → "bravo echo mike oscar lima"
    """
    words = []
    for char in text.upper():
        if char in nato_alphabet:
            words.append(nato_alphabet[char])
        elif char.isalnum():
            words.append(char.lower())
    return ' '.join(words)


def get_airline_name(code: str) -> Optional[str]:
    """Obtiene el nombre de aerolínea a partir de código ICAO o IATA."""
    code = code.upper()
    
    # Intentar ICAO primero
    if code in airlines_icao:
        return airlines_icao[code]
    
    # Intentar IATA
    if code in iata_to_icao:
        icao = iata_to_icao[code]
        return airlines_icao.get(icao)
    
    return None


def extract_callsigns(text: str) -> List[str]:
    """
    Extrae posibles callsigns de un texto.
    Busca patrones como XXX123 o XX1234
    """
    # Patrón ICAO: 3 letras + 1-4 dígitos
    icao_pattern = r'\b([A-Z]{3}\d{1,4})\b'
    # Patrón IATA: 2 letras + 1-4 dígitos
    iata_pattern = r'\b([A-Z]{2}\d{1,4})\b'
    
    found = []
    found.extend(re.findall(icao_pattern, text.upper()))
    found.extend(re.findall(iata_pattern, text.upper()))
    
    return list(set(found))


# Backwards compatibility
airlines_codes = {k: v for k, v in airlines_icao.items()}
airlines_terminology = atc_terminology
numbers = number_to_word
