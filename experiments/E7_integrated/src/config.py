"""Configuración parametrizable para E7_integrated pipeline benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.llm_client_factory import ModelConfig


@dataclass
class FilterConfig:
    """Configuración del RuleFilter (3 capas: keywords, embeddings, LLM batch)."""
    use_keywords: bool = True
    use_embeddings: bool = True
    use_llm_batch: bool = True
    top_k: int = 10
    timeout_seconds: float = 30.0
    embedding_cache_dir: str = "cache/"
    verbose: bool = True


@dataclass
class LLMConfig:
    """Configuración del LLM para evaluación genérica y judge."""
    model_name: str = "gemma4:31b-cloud"
    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    timeout: int = 120
    max_retries: int = 1
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass
class BERTConfig:
    """Configuración del BERT NER parser."""
    model_name: str = "Jzuluaga/bert-base-ner-atc-en-atco2-1h"
    confidence_threshold: float = 0.5


@dataclass
class E7Config:
    """Configuración principal del experimento E7_integrated."""
    
    # Modelos
    bert: BERTConfig = field(default_factory=BERTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge_model: str = "gemma4:31b-cloud"
    
    # RuleFilter
    filter: FilterConfig = field(default_factory=FilterConfig)
    
    # Genéricas
    generic_eval_timeout_s: float = 300.0
    generic_max_retries: int = 1
    
    # Pipeline
    projection_minutes: int = 10
    verbose: bool = True
    skip_bert: bool = False
    skip_generic: bool = False
    
    # Paths
    compiled_rules_dir: Path = field(
        default_factory=lambda: Path("experiments/E4_compiled_rules/models/gpt-oss:120b-cloud")
    )
    rules_json_path: Path = field(
        default_factory=lambda: Path("rules.json")
    )
    ground_truth_dir: Path = field(
        default_factory=lambda: Path("experiments/E7_integrated/ground_truth")
    )
    results_dir: Path = field(
        default_factory=lambda: Path("experiments/E7_integrated/results")
    )
    figures_dir: Path = field(
        default_factory=lambda: Path("experiments/E7_integrated/results/figures")
    )
    
    # Resume
    partial_file: str = "partial_results.json"

    def resolve_paths(self, project_root: Path) -> None:
        """Resuelve rutas relativas respecto a la raíz del proyecto."""
        for attr in (
            "compiled_rules_dir",
            "rules_json_path",
            "ground_truth_dir",
            "results_dir",
            "figures_dir",
        ):
            p = getattr(self, attr)
            if not p.is_absolute():
                setattr(self, attr, project_root / p)
        cache = Path(self.filter.embedding_cache_dir)
        if not cache.is_absolute():
            self.filter.embedding_cache_dir = str(project_root / cache)
    
    def __post_init__(self):
        """Crear directorios si no existen."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        Path(self.filter.embedding_cache_dir).mkdir(parents=True, exist_ok=True)


def to_model_config(cfg: "E7Config | LLMConfig", timeout_s: float | None = None) -> "ModelConfig":
    """Convierte LLMConfig de E7 a ModelConfig del Alert_System."""
    from common.llm_client_factory import ModelConfig

    llm = cfg.llm if hasattr(cfg, "llm") else cfg
    generic_timeout = timeout_s if timeout_s is not None else getattr(cfg, "generic_eval_timeout_s", 300.0)
    return ModelConfig(
        name=llm.model_name,
        provider=llm.provider,
        base_url=llm.base_url,
        api_key=llm.api_key,
        max_retries=llm.max_retries,
        timeout=int(generic_timeout),
    )