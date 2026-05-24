import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from config import E1Config


@dataclass
class Chunk:
    index: int
    text: str
    char_count: int

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            index=d.get("chunk_index", d.get("index", 0)),
            text=d.get("text", ""),
            char_count=d.get("char_count", len(d.get("text", ""))),
        )


@dataclass
class PageChunks:
    page_number: int
    total_chunks: int
    chunks: List[Chunk] = field(default_factory=list)
    granularity: str = "chunk"
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict, source: str = "") -> "PageChunks":
        return cls(
            page_number=d.get("page_number", 1),
            total_chunks=d.get("total_chunks", 0),
            chunks=[Chunk.from_dict(c) for c in d.get("chunks", [])],
            granularity=d.get("granularity", "chunk"),
            source=source,
        )

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "total_chunks": self.total_chunks,
            "granularity": self.granularity,
            "source": self.source,
            "chunks": [
                {"chunk_index": c.index, "text": c.text, "char_count": c.char_count}
                for c in self.chunks
            ],
        }


@dataclass
class ModelResult:
    model_name: str
    pages: Dict[int, PageChunks] = field(default_factory=dict)
    available_pages: List[int] = field(default_factory=list)


def load_page_chunks(filepath: Path, source: str = "") -> Optional[PageChunks]:
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PageChunks.from_dict(data, source=source)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_model_chunks(model_dir: Path) -> ModelResult:
    model_name = model_dir.name
    pages: Dict[int, PageChunks] = {}
    available: List[int] = []

    pattern = re.compile(r"pagina_(\d+)_chunks\.json")
    for f in model_dir.glob("pagina_*_chunks.json"):
        m = pattern.match(f.name)
        if not m:
            continue
        page_num = int(m.group(1))
        result = load_page_chunks(f, source=model_name)
        if result:
            pages[page_num] = result
            available.append(page_num)

    available.sort()
    return ModelResult(model_name=model_name, pages=pages, available_pages=available)


def load_ground_truth(gt_dir: Path) -> Dict[int, PageChunks]:
    pages: Dict[int, PageChunks] = {}
    pattern = re.compile(r"pagina_(\d+)_chunks\.json")
    for f in gt_dir.glob("pagina_*_chunks.json"):
        m = pattern.match(f.name)
        if not m:
            continue
        page_num = int(m.group(1))
        result = load_page_chunks(f, source="ground_truth")
        if result:
            pages[page_num] = result
    return pages


def discover_model_dirs(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    return sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda x: x.name)


@dataclass
class ExperimentData:
    model_names: List[str]
    model_results: Dict[str, ModelResult]
    ground_truth: Dict[int, PageChunks]
    pages: List[int]

    @classmethod
    def from_config(cls, cfg: E1Config) -> "ExperimentData":
        model_dirs = discover_model_dirs(cfg.models_dir)
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {cfg.models_dir}")

        model_results: Dict[str, ModelResult] = {}
        model_pages: Dict[str, List[int]] = {}

        for mdir in model_dirs:
            result = load_model_chunks(mdir)
            model_results[mdir.name] = result
            model_pages[mdir.name] = result.available_pages

        gt_pages = load_ground_truth(cfg.ground_truth_dir)
        if not gt_pages:
            raise FileNotFoundError(f"No ground truth files found in {cfg.ground_truth_dir}")

        gt_page_list = sorted(gt_pages.keys())
        all_model_pages = set()
        for mp in model_pages.values():
            all_model_pages.update(mp)

        common_pages = sorted(all_model_pages & set(gt_page_list))
        if not common_pages:
            raise ValueError("No common pages between models and ground truth")

        return cls(
            model_names=sorted(model_results.keys()),
            model_results=model_results,
            ground_truth=gt_pages,
            pages=common_pages,
        )

    def get_model_page_chunks(self, model_name: str, page: int) -> Optional[PageChunks]:
        return self.model_results.get(model_name, ModelResult("", {}, [])).pages.get(page)