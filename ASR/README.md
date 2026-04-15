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
```

### CLI Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--model` | whisper, whisperatc, huggingface, faster-whisper | Model type |
| `--model-size` | tiny, base, small, medium, large-v1/2/3 | Whisper size |
| `--hf-model` | string | HuggingFace model ID |
| `--version` | v2, v3 | WhisperATC version |
| `--device` | auto, cpu, cuda | Device |
| `--prompt` | default, minimal, extended, none | Prompt type |
| `--format` | csv, json | Output format |
| `--timestamps` | flag | Include timestamps |
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

- Callsign expansion: `JBU1676` → `jetblue one six seven six`
- Number expansion: `FL340` → `flight level three four zero`
- Frequency expansion: `118.5` → `one one eight point five`
- ATC terminology normalization
- Optional ICAO expansion: `BEMOL` → `bravo echo mike oscar lima`
- Punctuation removal
- Lowercase conversion

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

text = "JBU1676 climb FL340 maintain heading 090"
normalized = normalizer.normalize(text)
# Result: "jetblue one six seven six climb flight level three four zero maintain heading zero niner zero"

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

### Basic Usage

```python
from ASR.evaluation import ASREvaluator, load_ground_truth, load_transcriptions_by_timestamp, align_data
from ASR.normalization import ATCTextNormalizer

# 1. Load ground truth from DOCX
ground_truth_raw = load_ground_truth("./ground_truth.docx")
# {timestamp: full_text, ...}

# 2. Load transcriptions from CSV (Whisper format)
transcriptions_raw = load_transcriptions_by_timestamp("./transcriptions.csv")
# {model: {timestamp: text, ...}, ...}

# 3. Normalize
normalizer = ATCTextNormalizer()
ground_truth = {ts: normalizer.normalize(txt) for ts, txt in ground_truth_raw.items()}
transcriptions = {
    model: {ts: normalizer.normalize(txt) for ts, txt in model_data.items()}
    for model, model_data in transcriptions_raw.items()
}

# 4. Evaluate
evaluator = ASREvaluator(use_jiwer=True)
results = evaluator.evaluate_all_models(
    ground_truth=ground_truth,
    transcriptions=transcriptions,
    detailed=True
)

# 5. View results
for model_name, result in results.items():
    print(f"{model_name}: WER={result.wer:.2%}")
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
from ASR.evaluation import print_evaluation_report, compare_models

# Print formatted report
print_evaluation_report(results)

# Compare models
comparison = compare_models(results)
# Returns model ranking by WER
```

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
