"""Sistema de prefiltrado de reglas genéricas con 3 capas: keywords, embeddings, LLM batch."""

import os
import hashlib
import pickle
import time
from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

from Alert_System.integration.schemas import ExecutableRule


# Keywords ATC para filtrado rápido
ATC_KEYWORDS = {
    "climb": {"climb", "ascend", "altitude", "flight level", "fl"},
    "descend": {"descend", "descent", "lower", "altitude", "flight level", "fl"},
    "altitude": {"altitude", "flight level", "fl", "msa", "minimum safe altitude", "above", "below"},
    "speed": {"speed", "knots", "slow", "reduce speed", "increase speed", "mach"},
    "heading": {"heading", "turn", "left", "right", "course", "track", "vector"},
    "runway": {"runway", "rwy", "taxi", "landing", "takeoff", "departure", "approach", "hold short", "cross"},
    "separation": {"separation", "distance", "nm", "miles", "conflict", "proximity", "maintain"},
    "emergency": {"emergency", "mayday", "pan", "squawk", "7700", "7600", "7500", "distress"},
    "clearance": {"clearance", "cleared", "permission", "authorize"},
    "weather": {"weather", "wind", "visibility", "ceiling", "cloud", "turbulence", "icing"},
    "communication": {"readback", "read back", "confirm", "acknowledge", "repeat", "say again"},
    "wake": {"wake", "turbulence", "heavy", "super", "spacing"},
    "holding": {"hold", "holding", "pattern", "orbit", "delay"},
    "route": {"route", "direct", "waypoint", "fix", "intersection", "proceed"},
    "phrasing": {"plain language", "phraseology", "standard", "non-standard"},
    "frequency": {"frequency", "freq", "radio", "contact", "monitor"},
    "lights": {"lights", "beacon", "strobe", "landing lights", "taxi lights"},
}


@dataclass
class FilterConfig:
    """Configuración para el filtrado de reglas."""
    use_keywords: bool = True
    use_embeddings: bool = True
    use_llm_batch: bool = True
    top_k: int = 30
    timeout_seconds: Optional[float] = None  # None = sin timeout
    embedding_cache_dir: str = ""
    verbose: bool = False


class RuleFilter:
    """Filtro de reglas genéricas con 3 capas: keywords, embeddings, LLM batch."""

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self._embedding_model: Any = None
        self._rule_embeddings: Dict[str, np.ndarray] = {}
        self._rules_hash: str = ""

    def _get_embedding_model(self) -> Any:
        """Carga modelo de embeddings (lazy)."""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self._embedding_model = None
        return self._embedding_model

    def _compute_rules_hash(self, rules: List[ExecutableRule]) -> str:
        """Hash del contenido de las reglas para invalidar cache."""
        content = "".join(
            f"{r.source_rule_id}:{r.raw_trigger}:{r.raw_constraint}:{r.condition_description}"
            for r in rules
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self) -> str:
        """Ruta al archivo de cache de embeddings."""
        cache_dir = self.config.embedding_cache_dir or os.path.join(
            os.path.dirname(__file__), "cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "rule_embeddings.pkl")

    def load_or_compute_embeddings(self, rules: List[ExecutableRule]) -> Dict[str, np.ndarray]:
        """Carga embeddings desde cache o los computa y guarda."""
        current_hash = self._compute_rules_hash(rules)
        cache_path = self._cache_path()

        # Intentar cargar desde cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("hash") == current_hash:
                    self._rule_embeddings = cached["embeddings"]
                    self._rules_hash = current_hash
                    return self._rule_embeddings
            except Exception:
                pass

        # Computar embeddings
        model = self._get_embedding_model()
        if model is None:
            return {}

        texts = []
        for rule in rules:
            text = " ".join(filter(None, [
                rule.raw_trigger or "",
                rule.raw_constraint or "",
                rule.condition_description or "",
                rule.rule_category or "",
            ]))
            texts.append(text.strip() or rule.source_rule_id)

        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self._rule_embeddings = {
            rule.source_rule_id: emb for rule, emb in zip(rules, embeddings)
        }
        self._rules_hash = current_hash

        # Guardar cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"hash": current_hash, "embeddings": self._rule_embeddings}, f)
        except Exception:
            pass

        return self._rule_embeddings

    def _extract_keywords(self, instruction_text: str) -> Set[str]:
        """Extrae keywords ATC del texto de la instrucción."""
        text = instruction_text.lower()
        found = set()
        for category, keywords in ATC_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    found.add(category)
                    break
        return found

    def _log(self, msg: str):
        if self.config.verbose:
            print(f"  [FILTER] {msg}", flush=True)

    def _keyword_filter(
        self,
        rules: List[ExecutableRule],
        instruction_keywords: Set[str],
    ) -> List[ExecutableRule]:
        """Capa 1a: filtra reglas por keywords."""
        self._log(f"Capa 1a - Keywords: {len(rules)} reglas, keywords={instruction_keywords}")
        if not instruction_keywords:
            self._log("  Sin keywords ATC conocidas, pasando todas a embeddings")
            return rules

        candidates = []
        for rule in rules:
            rule_text = " ".join(filter(None, [
                rule.raw_trigger or "",
                rule.raw_constraint or "",
                rule.condition_description or "",
                rule.rule_category or "",
            ])).lower()

            # Si no hay texto, incluirla (no podemos descartarla)
            if not rule_text.strip():
                candidates.append(rule)
                continue

            # Contar coincidencias de keywords
            matches = 0
            for category in instruction_keywords:
                for kw in ATC_KEYWORDS.get(category, set()):
                    if kw in rule_text:
                        matches += 1

            # Si hay al menos una coincidencia, incluir
            if matches > 0:
                candidates.append(rule)

        result = candidates if candidates else rules
        self._log(f"  Post-keywords: {len(result)} reglas (coincidieron {len(candidates)})")
        return result

    def _embedding_rank(
        self,
        candidates: List[ExecutableRule],
        instruction_text: str,
        top_k: int,
    ) -> List[ExecutableRule]:
        """Capa 1b: re-rank por similitud de embeddings."""
        self._log(f"Capa 1b - Embeddings: {len(candidates)} candidatas, top_k={top_k}")
        model = self._get_embedding_model()
        if model is None or not self._rule_embeddings:
            self._log(f"  Modelo embeddings no disponible, truncando a top_k={top_k}")
            return candidates[:top_k]

        instruction_embedding = model.encode(instruction_text, convert_to_numpy=True)

        scored = []
        for rule in candidates:
            emb = self._rule_embeddings.get(rule.source_rule_id)
            if emb is None:
                # Si no hay embedding, poner en cola baja
                scored.append((-1.0, rule))
                continue
            similarity = float(np.dot(instruction_embedding, emb) / (
                np.linalg.norm(instruction_embedding) * np.linalg.norm(emb) + 1e-8
            ))
            scored.append((similarity, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [rule for _, rule in scored[:top_k]]
        self._log(f"  Post-embeddings: {len(result)} reglas (scores: {[round(s,3) for s,_ in scored[:5]]})")
        return result

    def _llm_batch_filter(
        self,
        candidates: List[ExecutableRule],
        instruction_text: str,
        llm_config: Any,
    ) -> List[ExecutableRule]:
        """Capa 2: LLM batch relevance filter."""
        if not candidates:
            self._log("Capa 2 - LLM Batch: 0 candidatas, saltando")
            return []

        self._log(f"Capa 2 - LLM Batch: {len(candidates)} candidatas")

        # Verificar que Ollama/LLM está disponible antes de intentar la llamada
        try:
            import urllib.request
            base_url = getattr(llm_config, "base_url", "http://localhost:11434")
            test_url = base_url.rstrip("/v1") + "/api/tags"
            req = urllib.request.Request(test_url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status != 200:
                    self._log("  LLM no responde, saltando Capa 2")
                    return candidates
        except Exception:
            self._log("  LLM no disponible (¿Ollama corriendo?), saltando Capa 2")
            return candidates

        self._log("  LLM disponible, enviando batch...")
        try:
            from common.llm_client_factory import create_instructor_client
            from Alert_System.integration.schemas import RelevanceFilterResult

            client, mode = create_instructor_client(llm_config)

            # Construir prompt con reglas numeradas
            rules_text = "\n".join(
                f"{i}. [{r.source_rule_id}] {r.condition_description or r.raw_trigger or 'N/A'}"
                for i, r in enumerate(candidates)
            )

            system_prompt = (
                "Eres un experto en control de tráfico aéreo (ATC). "
                "Tu tarea es determinar qué reglas de seguridad son RELEVANTES para evaluar "
                "una instrucción ATC específica."
            )
            user_prompt = (
                f"Instrucción ATC: '{instruction_text}'\n\n"
                f"Reglas de seguridad disponibles (índices 0-based):\n{rules_text}\n\n"
                f"Indica cuáles reglas son RELEVANTES para detectar posibles violaciones "
                f"causadas por esta instrucción. Devuelve el resultado estructurado con relevances, "
                f"summary y relevant_count."
            )

            response = client.chat.completions.create(
                model=llm_config.name,
                response_model=RelevanceFilterResult,
                max_retries=llm_config.max_retries,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            relevant_indices = {
                rel.rule_index for rel in response.relevances if rel.is_relevant
            }
            filtered = [candidates[i] for i in relevant_indices if 0 <= i < len(candidates)]
            self._log(f"  Post-LLM batch: {len(filtered)} reglas relevantes de {len(candidates)}")
            return filtered if filtered else candidates

        except Exception as e:
            self._log(f"  Error en LLM batch: {e}, devolviendo todas las candidatas")
            return candidates

    def filter_rules(
        self,
        rules: List[ExecutableRule],
        instruction_text: str,
        llm_config: Any,
        timeout_seconds: float = 0.0,
        top_k: int = 30,
    ) -> List[ExecutableRule]:
        """
        Aplica las 3 capas de filtrado y devuelve las reglas a evaluar.

        Args:
            rules: Lista completa de reglas genéricas.
            instruction_text: Texto de la instrucción ATC.
            llm_config: Configuración del LLM.
            timeout_seconds: 0 = ilimitado; >0 = abortar si se agota.
            top_k: Cuántas reglas mantener tras el re-rank de embeddings.

        Returns:
            Lista filtrada de reglas a evaluar con LLM real.
        """
        start_time = time.monotonic()

        def _elapsed() -> float:
            if timeout_seconds is None or timeout_seconds <= 0:
                return 0.0
            return time.monotonic() - start_time

        def _timeout_reached() -> bool:
            if timeout_seconds is None or timeout_seconds <= 0:
                return False
            return _elapsed() >= timeout_seconds

        candidates = rules[:]
        self._log(f"=== Prefiltrado iniciado: {len(candidates)} reglas ===")

        # Capa 1a: Keywords
        if self.config.use_keywords:
            keywords = self._extract_keywords(instruction_text)
            candidates = self._keyword_filter(candidates, keywords)

        if _timeout_reached():
            return candidates

        # Capa 1b: Embeddings
        if self.config.use_embeddings and candidates:
            # Precalcular embeddings si no están cargados
            if not self._rule_embeddings:
                self.load_or_compute_embeddings(rules)
            candidates = self._embedding_rank(candidates, instruction_text, top_k)

        if _timeout_reached():
            self._log(f"=== Timeout alcanzado. Devolviendo {len(candidates)} reglas ===")
            return candidates

        # Capa 2: LLM Batch
        if self.config.use_llm_batch and candidates:
            candidates = self._llm_batch_filter(candidates, instruction_text, llm_config)

        self._log(f"=== Prefiltrado finalizado: {len(candidates)} reglas a evaluar ===")
        return candidates
