# Knowledge Extractor

Extract aeronautical knowledge (entities, relationships, rules, procedures) from ATC documentation using LLMs with structured output.

## Features

- **PDF Text Extraction** with custom margins (PyMuPDF)
- **Three Granularity Levels**: page, sentence, chunk (NLTK + LLM segmentation)
- **Context-aware Extraction** with semantic embeddings (sentence-transformers)
- **Expanded Context Types**: Entities, Definitions, Rules, and Relationships for richer context
- **External Chunks Source**: Load pre-generated chunks from another folder
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

#### LLM Provider Configuration

Use different LLM providers (Ollama, OpenAI, Gemini native, Anthropic):

```bash
# Ollama (default - openai provider)
python -m Knowledge_Extractor document.pdf -m llama3.2

# OpenAI
python -m Knowledge_Extractor document.pdf -m gpt-4o --provider openai --api-key $OPENAI_API_KEY

# Gemini (native API - uses instructor.from_gemini)
python -m Knowledge_Extractor document.pdf -m gemini-1.5-flash --provider gemini --api-key $GEMINI_API_KEY

# Gemini (OpenAI-compatible endpoint - still uses openai provider)
python -m Knowledge_Extractor document.pdf -m gemini-1.5-flash --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key $GEMINI_API_KEY

# Anthropic/Claude
python -m Knowledge_Extractor document.pdf -m claude-3-sonnet-20240229 --provider anthropic --api-key $ANTHROPIC_API_KEY
```

**Note**: Install optional dependencies for native providers:
```bash
# For Gemini native support
pip install google-generativeai

# For Anthropic support
pip install anthropic
```

#### Context Control

Control which context types are included in the extraction prompt:

```bash
# Include all context types (default)
python -m Knowledge_Extractor document.pdf

# Exclude specific types
python -m Knowledge_Extractor document.pdf --no-definitions --no-rules

# Adjust context limits per type
python -m Knowledge_Extractor document.pdf --definition-limit 20 --rule-limit 10 --relationship-limit 15
```

#### External Chunks Source

Load pre-generated chunks from another folder:

```bash
# Use chunks from external source
python -m Knowledge_Extractor document.pdf --chunks-source "path/to/chunks/"

# Combine with output directory
python -m Knowledge_Extractor document.pdf --chunks-source "output/old_run/" -o "output/new_run/"

# Partial reprocessing: skip PDF extraction for pages with external chunks
python -m Knowledge_Extractor document.pdf --chunks-source "chunks_dir/" --start-page 5
```

**Note**: When `--chunks-source` is provided, pages with existing chunk files (`pagina_N_chunks.json`) will skip PDF text extraction entirely and use the external chunks. All chunks (external or generated) are saved to the output directory.

#### Chunk-Only Mode

Extract and save chunks without running KEX extraction:

```bash
# Extract only chunks, skip KEX
python -m Knowledge_Extractor document.pdf --chunk-only -o chunks_output/

# Combine with chunk granularity
python -m Knowledge_Extractor document.pdf --chunk-only -g chunk -o chunks_output/

# Reorganize chunks from external source
python -m Knowledge_Extractor document.pdf --chunk-only --chunks-source "existing_chunks/" -o "new_chunks/"
```

**Note**: In chunk-only mode, only `pagina_N_chunks.json` files are created. The full extraction results (`pagina_N.json`) can be generated later using `--chunks-source` flag.

#### Resume from Specific Page / Page Range

Process a specific page range instead of the entire document:

```bash
# Start from page 5, loading entities from pages 1-4
python -m Knowledge_Extractor document.pdf --start-page 5 --resume

# Process only pages 5-10
python -m Knowledge_Extractor document.pdf --start-page 5 --final-page 10

# Process up to page 3 only
python -m Knowledge_Extractor document.pdf --final-page 3

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
from Knowledge_Extractor.config.settings import PipelineConfig, ResumeConfig, EmbeddingConfig
from Knowledge_Extractor.pipeline.orchestrator import KnowledgeExtractionPipeline

# Configure expanded context
embedding_config = EmbeddingConfig(
    definition_top_k=10,
    rule_top_k=5,
    relationship_top_k=10,
    include_definitions=True,
    include_rules=True,
    include_relationships=True,
)

# Resume from page 5 with external chunks
resume_config = ResumeConfig(
    start_page=5,
    load_previous_entities=True,
    previous_output_dir="output/previous_run"
)

config = PipelineConfig(
    model_name="llama3.2",
    granularity="sentence",
    embedding=embedding_config,
    resume=resume_config,
    chunks_source_dir="path/to/external/chunks",  # Optional: load pre-generated chunks
)

pipeline = KnowledgeExtractionPipeline(config)
results = pipeline.process("document.pdf")

print(f"Processed {pipeline.state.processed_pages} pages")
print(f"Accumulated {pipeline.context_manager.get_entity_count()} entities")
print(f"Accumulated {pipeline.context_manager.get_definition_count()} definitions")
print(f"Accumulated {pipeline.context_manager.get_rule_count()} rules")
print(f"Accumulated {pipeline.context_manager.get_relationship_count()} relationships")
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
- `name`: LLM model name (default: llama3.2)
- `provider`: LLM provider type - "openai", "gemini", or "anthropic" (default: openai)
- `base_url`: API base URL for OpenAI-compatible providers (default: http://localhost:11434/v1 for Ollama)
- `api_key`: API key for authentication (default: ollama)
- `max_retries`: Retry attempts for failed extractions

### Embedding Settings
- `model_name`: sentence-transformers model
- `top_k`: Number of context entities to select
- `threshold`: Similarity threshold for context selection
- `definition_top_k`: Max definitions in context (default: 10)
- `rule_top_k`: Max rules in context (default: 5)
- `relationship_top_k`: Max relationships in context (default: 10)
- `include_definitions`: Include definitions in context (default: True)
- `include_rules`: Include rules in context (default: True)
- `include_relationships`: Include relationships in context (default: True)

### Resume Settings
- `start_page`: Page to start from (1-indexed)
- `load_previous_entities`: Load entities from previous runs
- `previous_output_dir`: Custom directory for loading state

### Pipeline Settings
- `chunks_source_dir`: Directory with pre-generated chunks (optional)
- `granularity`: Processing level (page, sentence, chunk)

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
