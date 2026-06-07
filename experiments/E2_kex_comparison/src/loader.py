import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

from .config import E2Config

KEX_TYPES = ["entities", "relationships", "events", "rules", "procedures"]


@dataclass
class KexPageResult:
    page_number: int
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    procedures: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    original_text: str = ""
    granularity: str = ""
    sentence_results: List[Dict[str, Any]] = field(default_factory=list)

    def get_by_type(self, kex_type: str) -> List[Dict[str, Any]]:
        return getattr(self, kex_type, [])

    def get_chunk(self, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by index (0-based)"""
        if 0 <= chunk_index < len(self.sentence_results):
            return self.sentence_results[chunk_index]
        return None

    def get_chunk_by_type(self, chunk_index: int, kex_type: str) -> List[Dict[str, Any]]:
        """Get items of a specific type from a specific chunk"""
        chunk = self.get_chunk(chunk_index)
        if chunk and "ner" in chunk:
            ner = chunk["ner"]
            if isinstance(ner, dict):
                return ner.get(kex_type, [])
        return []

    def chunk_count(self) -> int:
        """Get the number of chunks in this page"""
        return len(self.sentence_results)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "entities": self.entities,
            "relationships": self.relationships,
            "events": self.events,
            "rules": self.rules,
            "procedures": self.procedures,
            "errors": self.errors,
        }


@dataclass
class ModelResult:
    model_name: str
    pages: Dict[int, KexPageResult] = field(default_factory=dict)
    available_pages: List[int] = field(default_factory=list)


def load_kex_page(filepath: Path) -> Optional[KexPageResult]:
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        sentence_results = data.get("sentence_results", [])

        result = KexPageResult(
            page_number=data.get("page_number", 1),
            entities=[],
            relationships=[],
            events=[],
            rules=[],
            procedures=[],
            errors=[],
            original_text=data.get("original_text", ""),
            granularity=data.get("granularity", ""),
            sentence_results=sentence_results,
        )

        if sentence_results:
            all_entities = []
            all_relationships = []
            all_events = []
            all_rules = []
            all_procedures = []
            for sr in sentence_results:
                ner = sr.get("ner", {})
                if isinstance(ner, dict):
                    all_entities.extend(ner.get("entities", []))
                    all_relationships.extend(ner.get("relationships", []))
                    all_events.extend(ner.get("events", []))
                    all_rules.extend(ner.get("rules", []))
                    all_procedures.extend(ner.get("procedures", []))
            result.entities = all_entities
            result.relationships = all_relationships
            result.events = all_events
            result.rules = all_rules
            result.procedures = all_procedures

        return result
    except (json.JSONDecodeError, IOError, IndexError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_error_page(filepath: Path) -> List[Dict[str, Any]]:
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("errors", [])
        return []
    except (json.JSONDecodeError, IOError):
        return []


def load_model_pages(model_dir: Path) -> ModelResult:
    model_name = model_dir.name
    pages: Dict[int, KexPageResult] = {}
    available: List[int] = []

    pattern = re.compile(r"pagina_(\d+)\.json")
    for f in model_dir.glob("pagina_*.json"):
        if "_errors" in f.stem:
            continue
        m = pattern.match(f.name)
        if not m:
            continue
        page_num = int(m.group(1))
        result = load_kex_page(f)
        if result:
            pages[page_num] = result
            available.append(page_num)

    for f in model_dir.glob("pagina_*_errors.json"):
        m = re.match(r"pagina_(\d+)_errors\.json", f.name)
        if m:
            page_num = int(m.group(1))
            errors = load_error_page(f)
            if page_num in pages:
                pages[page_num].errors = errors
            else:
                pr = KexPageResult(page_number=page_num, errors=errors)
                pages[page_num] = pr

    available.sort()
    return ModelResult(model_name=model_name, pages=pages, available_pages=available)


def load_ground_truth(gt_dir: Path) -> Dict[int, KexPageResult]:
    pages: Dict[int, KexPageResult] = {}
    pattern = re.compile(r"pagina_(\d+)\.json")
    for f in gt_dir.glob("pagina_*.json"):
        if "_errors" in f.stem:
            continue
        m = pattern.match(f.name)
        if not m:
            continue
        page_num = int(m.group(1))
        result = load_kex_page(f)
        if result:
            pages[page_num] = result

    for f in gt_dir.glob("pagina_*_errors.json"):
        m = re.match(r"pagina_(\d+)_errors\.json", f.name)
        if m:
            page_num = int(m.group(1))
            errors = load_error_page(f)
            if page_num in pages:
                pages[page_num].errors = errors

    return pages


def discover_model_dirs(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    return sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda x: x.name)


@dataclass
class ExperimentData:
    model_names: List[str]
    model_results: Dict[str, ModelResult]
    ground_truth: Dict[int, KexPageResult]
    pages: List[int]

    @classmethod
    def from_config(cls, cfg: E2Config) -> "ExperimentData":
        model_dirs = discover_model_dirs(cfg.models_dir)
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {cfg.models_dir}")

        model_results: Dict[str, ModelResult] = {}
        all_pages_set = set()

        for mdir in model_dirs:
            result = load_model_pages(mdir)
            model_results[mdir.name] = result
            all_pages_set.update(result.available_pages)

        gt_pages = load_ground_truth(cfg.ground_truth_dir)
        if not gt_pages:
            raise FileNotFoundError(f"No ground truth files found in {cfg.ground_truth_dir}")

        gt_page_set = set(gt_pages.keys())
        common_pages = sorted(all_pages_set & gt_page_set)
        if not common_pages:
            raise ValueError("No common pages between models and ground truth")

        return cls(
            model_names=sorted(model_results.keys()),
            model_results=model_results,
            ground_truth=gt_pages,
            pages=common_pages,
        )

    def get_model_page(self, model_name: str, page: int) -> Optional[KexPageResult]:
        return self.model_results.get(model_name, ModelResult("", {}, [])).pages.get(page)
