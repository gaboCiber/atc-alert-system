"""
Parser ATC usando BERT NER (Jzuluaga/bert-base-ner-atc-en-atco2-1h).

Pipeline: raw_text -> ATCTextNormalizer (expand) -> BERT NER -> entities
Cada entidad se post-procesa segun su tipo (callsign, command, value).

Uso:
    parser = BertATCParser()
    result = parser.parse("DAL451 climb FL330")
    result["callsign"]  # "DAL451"
    result["command_entity"]  # "climb"
    result["value_entity"]  # "flight level three three zero"
"""

import re
from typing import Dict, List, Optional, Any

from ASR.normalization.text_normalizer import ATCTextNormalizer
from Alert_System.integration.atc_compact_normalizer import ATCCompactNormalizer


# Mapa de palabras numericas a digitos (inverso de number_to_word)
_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "niner": "9",
}


class BertATCParser:
    """
    Parser ATC usando BERT NER de HuggingFace.

    El modelo fue entrenado en transcripciones ATC del corpus ATCO2,
    por lo que funciona mejor con texto en formato expandido
    ("delta four five one") que compacto ("DAL451").

    Pipeline:
        1. Expandir texto con ATCTextNormalizer (para match training data)
        2. BERT NER pipeline -> entities (callsign, command, value)
        3. Post-procesar cada entidad segun su tipo
    """

    MODEL_NAME = "Jzuluaga/bert-base-ner-atc-en-atco2-1h"

    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            model_name: Nombre del modelo en HuggingFace Hub
            confidence_threshold: Umbral minimo de confianza para aceptar una entidad
        """
        self.model_name = model_name or self.MODEL_NAME
        self.confidence_threshold = confidence_threshold
        self._pipeline = None
        self._load_error: Optional[str] = None

        self._normalizer = ATCTextNormalizer(
            expand_callsigns=True,
            expand_numbers=True,
            expand_icao=False,
            normalize_terminology=True,
            remove_punctuation=True,
            lowercase=True,
            convert_compound_numbers=True,
            split_concatenated_terms=True,
        )
        self._compact_normalizer = ATCCompactNormalizer()

    def _initialize(self) -> bool:
        """Carga el modelo BERT de forma lazy (solo la primera vez)."""
        if self._pipeline is not None:
            return True
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name
            )
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            return False

    @property
    def is_available(self) -> bool:
        """Verifica si el modelo esta disponible (carga si es necesario)."""
        return self._initialize()

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parsea una instruccion ATC usando BERT NER.

        Args:
            text: Texto crudo de la instruccion (formato ASR)

        Returns:
            Dict con:
                callsign: str o None (formato compacto ICAO, ej: "DAL451")
                callsign_entity: str o None (formato expandido)
                callsign_score: float
                command_entity: str o None (texto del comando, ej: "climb")
                command_score: float
                value_entity: str o None (texto del valor, ej: "flight level three three zero")
                value_score: float
                value_compact: str o None (valor en formato compacto, ej: "FL330")
                raw_entities: list de entidades crudas del NER
                confidence: float (confianza minima entre entidades encontradas)
                success: bool (True si se extrajo al menos callsign + un comando o valor)
        """
        result = {
            "callsign": None,
            "callsign_entity": None,
            "callsign_score": 0.0,
            "command_entity": None,
            "command_score": 0.0,
            "value_entity": None,
            "value_score": 0.0,
            "value_compact": None,
            "raw_entities": [],
            "confidence": 0.0,
            "success": False,
        }

        if not text or not self._initialize():
            return result

        # 1. Expandir el texto para que coincida con el dominio de entrenamiento del BERT
        expanded = self._normalizer.normalize(text)

        # 2. Ejecutar BERT NER
        try:
            entities = self._pipeline(expanded)
        except Exception:
            return result

        result["raw_entities"] = entities

        # 3. Agrupar entidades por tipo (tomar la de mayor confianza por tipo)
        callsign_span = None
        command_span = None
        value_span = None

        for entity in entities:
            group = entity.get("entity_group", "")
            score = entity.get("score", 0.0)
            if group == "callsign" and (
                callsign_span is None or score > callsign_span["score"]
            ):
                callsign_span = entity
            elif group == "command" and (
                command_span is None or score > command_span["score"]
            ):
                command_span = entity
            elif group == "value" and (
                value_span is None or score > value_span["score"]
            ):
                value_span = entity

        # 4. Post-procesar entidad callsign (expandido -> compacto ICAO)
        if callsign_span and callsign_span["score"] >= self.confidence_threshold:
            result["callsign_entity"] = callsign_span["word"]
            result["callsign_score"] = callsign_span["score"]

            compact_callsign = self._compact_normalizer.normalize(
                callsign_span["word"]
            )
            callsign = self._extract_callsign_compact(compact_callsign)
            if callsign:
                result["callsign"] = callsign

        # 5. Post-procesar entidad command
        if command_span and command_span["score"] >= self.confidence_threshold:
            result["command_entity"] = command_span["word"]
            result["command_score"] = command_span["score"]

        # 6. Post-procesar entidad value
        if value_span and value_span["score"] >= self.confidence_threshold:
            result["value_entity"] = value_span["word"]
            result["value_score"] = value_span["score"]

            # Compactar el value para consumo por regex de parametros
            value_compact = self._compact_normalizer.normalize(
                value_span["word"]
            ).strip()
            if value_compact:
                result["value_compact"] = value_compact

        # 7. Determinar confianza general y exito
        scores = [
            s
            for s in [
                result["callsign_score"],
                result["command_score"],
                result["value_score"],
            ]
            if s > 0
        ]
        result["confidence"] = min(scores) if scores else 0.0

        # Exito: al menos callsign + (command o value)
        has_callsign = result["callsign"] is not None
        has_command = result["command_entity"] is not None
        has_value = result["value_entity"] is not None
        result["success"] = has_callsign and (has_command or has_value)

        return result

    def _extract_callsign_compact(self, text: str) -> Optional[str]:
        """
        Extrae solo el callsign del resultado del compact normalizer.

        El compact normalizer puede devolver "DAL451 resto del texto".
        Este metodo extrae solo "DAL451".
        """
        # El formato esperado es: 2-4 letras mayusculas + 1-4 digitos
        match = re.search(r"\b([A-Z]{2,4}\d{1,4}[A-Z]?)\b", text.strip())
        if match:
            return match.group(1).upper()
        return None

    def __repr__(self) -> str:
        status = "available" if self._pipeline is not None else "unloaded"
        return f"BertATCParser(model={self.model_name}, status={status})"
