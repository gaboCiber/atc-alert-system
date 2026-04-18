# ASR (Automatic Speech Recognition) Module for ATC

Complete module for transcription, normalization, and evaluation of ATC (Air Traffic Control) audio.

## Dependencies & Installation

### Requirements

```txt
# Transcription
openai-whisper          # For WhisperModel
faster-whisper          # For FasterWhisperModel
transformers            # For HuggingFaceModel
torch                   # PyTorch backend
tqdm                    # Progress bars

# Evaluation
jiwer                   # WER metrics (optional but recommended)
python-docx             # DOCX ground truth reading

# Normalization
# (uses only Python standard libraries)
```

### Installation

```bash
# Install from requirements.txt
pip install -r ASR/requirements.txt

# Or install manually as needed
pip install openai-whisper faster-whisper transformers torch tqdm
pip install jiwer python-docx
```

---

## Module Structure

```
ASR/
├── transcription/     # Transcription pipeline with multiple models
│   ├── models/        # ASR model implementations
│   ├── output/        # Output format handlers (CSV/JSON)
│   └── config/        # Prompts and configuration
├── normalization/     # Text normalization for evaluation
├── evaluation/        # WER evaluation and ASR metrics
├── requirements.txt   # Module dependencies
└── README.md          # This file
```

---

## 1. Transcription (`ASR.transcription`)

ASR transcription pipeline with support for multiple models and output formats.

### Supported Models

| Model | Description | Requires Prompt |
|-------|-------------|-----------------|
| `WhisperModel` | OpenAI Whisper (tiny, base, small, medium, turbo, large-v1/2/3) | Yes |
| `WhisperPromptedModel` | Whisper with built-in ATC prompt | No (internal) |
| `WhisperATCModel` | HuggingFace WhisperATC (v2/v3) optimized for ATC | No |
| `HuggingFaceModel` | Any HuggingFace ASR model | Optional |
| `FasterWhisperModel` | Whisper optimized with CTranslate2 | Yes |

### Basic Usage

```python
from ASR.transcription import WhisperModel, TranscriptionPipeline, WhisperATCModel

# Whisper with prompt
whisper = WhisperModel(
    model_name="large-v3",
    device="auto",           # auto | cpu | cuda
    prompt="default",        # default | minimal | extended | none
    fp16=False
)

pipeline = TranscriptionPipeline(
    model=whisper,
    output_format="csv",     # csv | json
    show_progress=True
)

# Transcribe entire directory
results = pipeline.run_directory(
    directory="./recordings",
    output_path="./output/transcriptions.csv",
    extensions=(".mp3", ".wav", ".flac"),
    recursive=True
)

# Or transcribe list of files
results = pipeline.run(
    audio_files=["audio1.mp3", "audio2.wav"],
    output_path="./output/results.csv",
    model_column="model"
)
```

### WhisperATC (ATC Specialized Model)

```python
from ASR.transcription import WhisperATCModel

# No prompt required - already optimized for ATC
atc_model = WhisperATCModel(
    model_version="v3",      # v2 | v3
    device="auto",
    return_timestamps=True
)

pipeline = TranscriptionPipeline(atc_model, output_format="json")
pipeline.run_directory("./audio", "./output.json")
```

### FasterWhisper

```python
from ASR.transcription import FasterWhisperModel

faster = FasterWhisperModel(
    model_name="large-v3",
    device="cuda",
    compute_type="int8",     # int8 | int8_float16 | float16 | float32
    beam_size=5
)
```

### Available Prompts

```python
from ASR.transcription import get_prompt, create_custom_prompt, AVAILABLE_PROMPTS

# Predefined prompts
prompt = get_prompt("default")   # Full with NATO, terminology, airlines
prompt = get_prompt("minimal")   # Essential for ATC
prompt = get_prompt("extended")  # Additional terminology
prompt = get_prompt("none")      # No prompt

# Custom prompt
prompt = create_custom_prompt(
    include_nato=True,
    include_terminology=True,
    include_airlines=True,
    include_cuba_terms=True,
    extra_terms="custom terms"
)
```

### Dynamic Prompts (Per-File Context)

The `transcribe()` method and `TranscriptionPipeline` support dynamic prompts that can be different for each audio file. This is useful for providing context-specific information like nearby waypoints and callsigns.

```python
from ASR.transcription import WhisperModel, TranscriptionPipeline, create_custom_prompt
from ASR.evaluation.data_loaders import Atco2DataLoader
from pathlib import Path

# Create model once
model = WhisperModel(model_name="large-v3", device="auto")

# Define dynamic prompt provider
loader = Atco2DataLoader()

def dynamic_prompt_provider(audio_path: str) -> str:
    """Generate context-specific prompt for each audio file"""
    info_path = str(Path(audio_path).with_suffix(".info"))
    
    try:
        waypoints = loader.extract_waypoints_from_info(info_path)
        callsigns = loader.extract_callsigns_from_info(info_path)
    except FileNotFoundError:
        return create_custom_prompt(extra_terms="")
    
    # Build context string
    parts = []
    if waypoints:
        parts.append(f"Nearby waypoints: {', '.join(waypoints)}")
    if callsigns:
        cs_str = ", ".join([f"{code} ({phonetic})" for code, phonetic in callsigns.items()])
        parts.append(f"Common callsigns: {cs_str}")
    
    return create_custom_prompt(extra_terms=". ".join(parts))

# Create pipeline with dynamic prompt provider
pipeline = TranscriptionPipeline(
    model=model,
    output_format="csv",
    show_progress=True,
    prompt_provider=dynamic_prompt_provider  # Callback for each file
)

# Process directory - each file gets its own prompt based on its .info file
results = pipeline.run_directory(
    directory="./ASR/Recordings/ATCO2-ASRdataset-v1_beta/DATA",
    output_path="./results.csv",
    extensions=(".wav",)
)
```

**Key Features:**
- `transcribe(audio_path, prompt=...)`: Override the model's default prompt for a single transcription
- `prompt_provider`: Callback function that receives the audio path and returns a custom prompt
- The model is loaded only once, but each file can have a different prompt

### CLI (Command Line Interface)

```bash
# Whisper with default prompt
python -m ASR.transcription.cli \
    --model whisper \
    --model-size large-v3 \
    --input ./audio \
    --output results.csv \
    --prompt default

# WhisperATC (best for ATC)
python -m ASR.transcription.cli \
    --model whisperatc \
    --version v3 \
    --input ./audio \
    --output results.json \
    --timestamps

# HuggingFace with custom model
python -m ASR.transcription.cli \
    --model huggingface \
    --hf-model "jlvdoorn/whisper-large-v3-atco2-asr" \
    --input ./audio \
    --output results.csv

# FasterWhisper
python -m ASR.transcription.cli \
    --model faster-whisper \
    --model-size large-v3 \
    --device cuda \
    --input ./audio \
    --output results.csv

# Append mode (add results to existing CSV)
python -m ASR.transcription.cli \
    --model whisperatc \
    --input ./new_audio \
    --output results.csv \
    --append

# Checkpoint mode (resume if interrupted)
python -m ASR.transcription.cli \
    --model whisper \
    --model-size large-v3 \
    --input ./audio \
    --output results.json \
    --checkpoint
```

### CLI Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--model` | whisper, whisperatc, huggingface, faster-whisper | Model type |
| `--model-size` | tiny, base, small, medium, large-v1/2/3 | Whisper size |
| `--hf-model` | string | HuggingFace model ID |
| `--chunk-length-s` | int | Chunk length in seconds for HuggingFace (None disables chunking, default: None) |
| `--version` | v2, v3 | WhisperATC version |
| `--device` | auto, cpu, cuda | Device |
| `--prompt` | default, minimal, extended, none | Prompt type |
| `--format` | csv, json | Output format |
| `--timestamps` | flag | Include timestamps |
| `--append` | flag | Append results to existing file (CSV only) |
| `--checkpoint` | flag | Enable checkpoint for resuming interrupted transcriptions |
| `--no-progress` | flag | Hide progress bar |
| `--extensions` | string | Extensions (default: .mp3,.wav,.flac,.m4a,.ogg,.mkv) |

### Output Format

**CSV**: One row per model, columns are audio files
```csv
model,audio1.mp3,audio2.mp3
whisper-base,text1,text2
whisper-large,text1_alt,text2_alt
```

**JSON**: Full structure with timestamps and metadata
```json
{
  "metadata": {
    "export_date": "2024-01-01T12:00:00",
    "format": "json",
    "num_results": 10
  },
  "results": [
    {
      "file_path": "audio.mp3",
      "model_name": "whisper-large-v3",
      "text": "transcription...",
      "timestamps": [
        {"start": 0.0, "end": 3.5, "text": "segment 1"}
      ],
      "metadata": {"language": "en"}
    }
  ]
}
```

---

## 2. Normalization (`ASR.normalization`)

ATC text normalization for comparison and WER evaluation.

### Features

- **Callsign expansion**: `JBU1676` → `jetblue one six seven six`
- **Number expansion** (all digits): `FL340` → `flight level three four zero`, `877561` → `eight seven seven five six one`
- **Runway number expansion**: `16R` → `one six right`, `16L` → `one six left`
- **Frequency expansion**: `118.5` → `one one eight point five`
- **ATC terminology normalization**: `climb` → `climb`, `descend` → `descend`, etc.
- **Optional ICAO expansion**: `BEMOL` → `bravo echo mike oscar lima`
- **Punctuation removal** and **lowercase conversion**

### Usage

```python
from ASR.normalization import ATCTextNormalizer, quick_normalize

# Full normalization
normalizer = ATCTextNormalizer(
    expand_callsigns=True,       # Expand JBU1676 → jetblue one six seven six
    expand_numbers=True,         # Expand 340 → three four zero
    expand_icao=False,           # Expand BEMOL → bravo echo...
    normalize_terminology=True,  # Replace ATC abbreviations
    remove_punctuation=True,     # Remove punctuation
    lowercase=True               # Convert to lowercase
)

text = "JBU1676 climb FL340 maintain heading 090, runway 16R"
normalized = normalizer.normalize(text)
# Result: "jetblue one six seven six climb flight level three four zero maintain heading zero niner zero runway one six right"

# Full number expansion works with any digit sequence
text = "China 877561, established on the localizer"
normalizer.normalize(text)
# Result: "china eight seven seven five six one established on the localizer"

# Quick function with default values
normalized = quick_normalize(text)
```

### Available Terminology

```python
from ASR.normalization import (
    expand_callsign,      # Expand callsign with ICAO
    expand_number,        # Expand number to words
    expand_icao_spelling, # Expand word to NATO alphabet
    number_to_word,       # Dictionary number→word
    airlines_icao,        # Dictionary airlines ICAO
    iata_to_icao,         # Dictionary IATA→ICAO
    atc_terminology,      # Dictionary ATC terms
    nato_alphabet,        # Dictionary NATO alphabet
)
```

---

## 3. Evaluation (`ASR.evaluation`)

ASR transcription evaluation against ground truth using WER and other metrics.

### Data Loaders (OOP-Based)

The evaluation module now uses an object-oriented approach with data loaders for different datasets:

| Loader | Description | Use Case |
|--------|-------------|----------|
| `EcnaDataLoader` | Loader for ECNA dataset format | ECNA ground truth (DOCX + audio) |
| `Atco2DataLoader` | Loader for ATCO2 dataset format | ATCO2 structured data |
| `BaseDataLoader` | Abstract base for custom loaders | Extend for new datasets |

### Basic Usage (New API with Loaders)

```python
from ASR.evaluation import ASREvaluator, EcnaDataLoader, Atco2DataLoader

# Create data loader for your dataset
loader = EcnaDataLoader()  # or Atco2DataLoader()

# Initialize evaluator with automatic ATC normalization
evaluator = ASREvaluator(
    use_jiwer=True,           # Use jiwer for WER calculation
    use_atc_normalizer=True  # Automatically normalize with ATCTextNormalizer
)

# Evaluate all models in transcription CSV
results = evaluator.evaluate_all_models_with_loader(
    data_loader=loader,
    ground_truth_path="./ground_truth.docx",
    transcriptions_path="./transcriptions.csv",
    detailed=True
)

# View results per model
for model_name, model_results in results.items():
    agg = evaluator.aggregate_metrics(model_results)
    print(f"{model_name}: WER={agg['average_wer']:.2%}")
```

### Legacy API (Still Supported)

```python
from ASR.evaluation import ASREvaluator, load_ground_truth, load_transcriptions_by_timestamp
from ASR.normalization import ATCTextNormalizer

# Manual loading and normalization
gt_raw = load_ground_truth("./ground_truth.docx")
trans_raw = load_transcriptions_by_timestamp("./transcriptions.csv")

normalizer = ATCTextNormalizer()
gt_norm = {ts: normalizer.normalize(txt) for ts, txt in gt_raw.items()}
trans_norm = {m: {ts: normalizer.normalize(txt) for ts, txt in d.items()} 
              for m, d in trans_raw.items()}

evaluator = ASREvaluator(use_jiwer=True)
results = evaluator.evaluate_all_models(gt_norm, trans_norm, detailed=True)
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `WER` | Word Error Rate |
| `MER` | Match Error Rate |
| `WIL` | Word Information Lost |
| `WIP` | Word Information Preserved |
| `CER` | Character Error Rate |

### ASREvaluationResult

```python
from ASR.evaluation import ASREvaluationResult

result = ASREvaluationResult(
    model_name="whisper-large-v3",
    timestamp="22.31.28",
    reference="normalized ground truth",
    hypothesis="normalized prediction",
    wer=0.15,                    # 15% WER
    mer=0.12,
    wil=0.18,
    wip=0.82,
    cer=0.08,
    num_ref_words=45,
    num_hyp_words=43,
    errors={                     # Error details
        "substitutions": 5,
        "insertions": 2,
        "deletions": 3
    }
)
```

### Reports and Comparison

```python
from ASR.evaluation import ASREvaluator

evaluator = ASREvaluator()

# Print formatted report with normalized text samples
evaluator.print_evaluation_report(results, show_samples=True)

# Compare models - returns sorted list by WER (lower is better)
comparison = evaluator.compare_models(results)
# [("whisper-large-v3", 0.12), ("whisper-base", 0.18), ...]
```

The `ASREvaluator` automatically applies `ATCTextNormalizer` when `use_atc_normalizer=True` (default), ensuring consistent preprocessing for both reference and hypothesis text before WER calculation.

---

## 4. Complete Evaluation Script (`run_evaluation.py`)

Command-line script for complete evaluation.

```bash
python ASR/run_evaluation.py \
    --ground-truth ./ground_truth.docx \
    --transcriptions ./transcriptions.csv \
    --normalize \
    --show-samples 5
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--ground-truth` | Path to DOCX file with ground truth |
| `--transcriptions` | Path to CSV with transcriptions |
| `--normalize` | Apply normalization before evaluation |
| `--no-normalize` | Do not normalize (raw text) |
| `--show-samples` | Number of normalization samples to show |

---

## Complete Example: Transcription + Evaluation Pipeline

```python
from ASR.transcription import WhisperATCModel, TranscriptionPipeline
from ASR.evaluation import ASREvaluator, load_ground_truth, load_transcriptions_by_timestamp
from ASR.normalization import ATCTextNormalizer

# 1. Transcribe audio
model = WhisperATCModel(model_version="v3")
pipeline = TranscriptionPipeline(model, output_format="csv")
pipeline.run_directory("./recordings", "./output/transcriptions.csv")

# 2. Evaluate against ground truth
gt_raw = load_ground_truth("./ground_truth.docx")
trans_raw = load_transcriptions_by_timestamp("./output/transcriptions.csv")

normalizer = ATCTextNormalizer()
gt_norm = {ts: normalizer.normalize(txt) for ts, txt in gt_raw.items()}
trans_norm = {
    model: {ts: normalizer.normalize(txt) for ts, txt in data.items()}
    for model, data in trans_raw.items()
}

evaluator = ASREvaluator()
results = evaluator.evaluate_all_models(gt_norm, trans_norm)

# 3. View results
for model, res in results.items():
    print(f"{model}: WER = {res.wer:.2%}")
```

---

## Recommended Workflow

1. **Record ATC audio** → Save in supported format (.mp3, .wav, .flac)
2. **Transcribe** → Use `TranscriptionPipeline` with appropriate model
3. **Normalize** → Apply `ATCTextNormalizer` to ground truth and transcriptions
4. **Evaluate** → Use `ASREvaluator` to calculate WER
5. **Analyze** → Compare models and adjust prompts based on results
