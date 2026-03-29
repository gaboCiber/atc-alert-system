# Knowledge Extractor

Extract aeronautical knowledge (entities, relationships, rules, procedures) from ATC documentation using LLMs with structured output.

## Architecture

```
Knowledge_Extractor/
├── config/           # Configuration and prompts
├── core/             # Document processing, segmentation, embeddings
├── extractors/       # LLM extractors (KEX with Instructor)
├── schemas/          # Pydantic models
├── pipeline/         # Orchestration
├── utils/            # Utilities (ID management, file ops)
├── cli.py            # Command-line interface
└── main.py           # Simple entry point
```

## Usage

### CLI
```bash
# Extract from PDF (default: sentence granularity)
python -m Knowledge_Extractor.cli document.pdf -m llama3.2 -o output/

# With logical chunking (NLTK + LLM segmentation)
python -m Knowledge_Extractor.cli document.pdf -g chunk

# Full page processing
python -m Knowledge_Extractor.cli document.pdf -g page

# With custom margins
python -m Knowledge_Extractor.cli document.pdf --margins 34 76 1 33
```

### Granularity Options

- **page**: Process each page as a single unit (fastest, less context)
- **sentence**: Split into individual sentences (balanced)
- **chunk**: Group sentences into logical chunks using LLM (best for complex documents)

### Python API
```python
from Knowledge_Extractor import extract_knowledge

results = extract_knowledge("document.pdf", output_dir="output")
```

### Pipeline
```python
from Knowledge_Extractor.config.settings import PipelineConfig
from Knowledge_Extractor.pipeline.orchestrator import KnowledgeExtractionPipeline

config = PipelineConfig(model_name="llama3.2")
pipeline = KnowledgeExtractionPipeline(config)
results = pipeline.process("document.pdf")
```

## Features

- **PDF Text Extraction** with custom margins
- **Semantic Segmentation** using sentence embeddings
- **Context-aware Extraction** with accumulated entities
- **Sequential ID Management** (E001, R001, RULE001...)
- **Structured Output** via Pydantic/Instructor
- **Granular Processing** (page or sentence level)

## Requirements

- Ollama running locally (default: http://localhost:11434)
- Python 3.10+
- See requirements.txt for package dependencies

## License

Academic use only
