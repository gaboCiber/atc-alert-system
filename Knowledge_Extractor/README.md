# Knowledge Extractor

Extract aeronautical knowledge (entities, relationships, rules, procedures) from ATC documentation using LLMs with structured output.

## Features

- **PDF Text Extraction** with custom margins (PyMuPDF)
- **Three Granularity Levels**: page, sentence, chunk (NLTK + LLM segmentation)
- **Context-aware Extraction** with semantic embeddings (sentence-transformers)
- **Resume Capability**: Start from specific page with previous entities as context
- **Sequential ID Management**: E001, R001, RULE001, EV001, P001, D001
- **Structured Output** via Pydantic/Instructor with automatic validation
- **Memory Management**: Automatic Ollama model unloading after processing
- **Granular Processing**: Process by page, sentence, or logical chunks

## Architecture

```
Knowledge_Extractor/
├── config/           # Configuration (PipelineConfig, ResumeConfig, prompts)
├── core/             # DocumentProcessor, TextSegmenter, SentenceExtractor, ContextManager
├── extractors/       # KEXExtractor (Instructor-based), JSONParser (fallback)
├── schemas/          # Pydantic models (AeronauticalExtraction, Entity, Relationship, etc.)
├── pipeline/         # KnowledgeExtractionPipeline with state management
├── utils/            # IDManager, FileUtils
├── cli.py            # Command-line interface
├── main.py           # Simple entry point
└── __main__.py       # python -m Knowledge_Extractor
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Ollama running locally (default: http://localhost:11434)
- Python 3.10+

## Usage

### CLI

#### Basic Extraction
```bash
# Sentence granularity (default)
python -m Knowledge_Extractor document.pdf -m llama3.2 -o output/

# Page granularity (fastest)
python -m Knowledge_Extractor document.pdf -g page

# Logical chunking with LLM (best for complex docs)
python -m Knowledge_Extractor document.pdf -g chunk
```

#### Resume from Specific Page
```bash
# Start from page 5, loading entities from pages 1-4
python -m Knowledge_Extractor document.pdf --start-page 5 --resume

# Resume using different directory for previous results
python -m Knowledge_Extractor document.pdf --start-page 10 --previous-dir output/old_run/
```

#### Custom Margins
```bash
# Extract only from specific PDF area (left, bottom, right, top in points)
python -m Knowledge_Extractor document.pdf --margins 34 76 1 33
```

### Granularity Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `page` | Process each page as single unit | Fast processing, simple docs |
| `sentence` | Split into individual sentences | Balanced speed/accuracy |
| `chunk` | Group sentences into logical blocks with LLM | Complex documents with rules/procedures |

### Python API

```python
from Knowledge_Extractor import extract_knowledge

# Simple extraction
results = extract_knowledge("document.pdf", output_dir="output")
```

### Advanced Pipeline Usage

```python
from Knowledge_Extractor.config.settings import PipelineConfig, ResumeConfig
from Knowledge_Extractor.pipeline.orchestrator import KnowledgeExtractionPipeline

# Resume from page 5
resume_config = ResumeConfig(
    start_page=5,
    load_previous_entities=True,
    previous_output_dir="output/previous_run"
)

config = PipelineConfig(
    model_name="llama3.2",
    granularity="sentence",
    resume=resume_config
)

pipeline = KnowledgeExtractionPipeline(config)
results = pipeline.process("document.pdf")

print(f"Processed {pipeline.state.processed_pages} pages")
print(f"Accumulated {pipeline.context_manager.get_entity_count()} entities")
```

## Output Structure

Each page generates a JSON file (`pagina_N.json`):

```json
{
  "texto_original": "...",
  "granularity": "sentence",
  "sentence_results": [
    {
      "chunk_text": "...",
      "ner": {
        "entities": [...],
        "relationships": [...],
        "rules": [...]
      },
      "context": {
        "contexto_entidades_usadas": 10,
        "entidades_acumuladas_total": 45,
        "last_ids": {"entities": "E012", "rules": "RULE003"}
      }
    }
  ],
  "last_ids_summary": {...}
}
```

## Configuration

### Model Settings
- `name`: Ollama model (default: llama3.2)
- `base_url`: Ollama API endpoint
- `max_retries`: Retry attempts for failed extractions

### Embedding Settings
- `model_name`: sentence-transformers model
- `top_k`: Number of context entities to select
- `threshold`: Similarity threshold for context selection

### Resume Settings
- `start_page`: Page to start from (1-indexed)
- `load_previous_entities`: Load entities from previous runs
- `previous_output_dir`: Custom directory for loading state

## Extracted Schema

The system extracts:
- **Entities**: Aircraft, Runways, Procedures, Phases, etc.
- **Relationships**: Structural, spatial, procedural, operational
- **Events**: Runway crossing, clearance issued, etc.
- **Rules**: Obligations, prohibitions, permissions with formal logic
- **Procedures**: Step-by-step workflows
- **Definitions**: Glossary terms

All with unique IDs and cross-referencing.

## License

Academic use only
