# Módulo ASR (Automatic Speech Recognition) para ATC

Módulo completo para transcripción, normalización y evaluación de audio ATC (Air Traffic Control).

## Estructura

```
ASR/
├── transcription/     # Pipeline de transcripción con múltiples modelos
├── normalization/     # Normalización de textos para evaluación
├── evaluation/        # Evaluación WER y métricas ASR
└── run_evaluation.py  # Script principal de evaluación
```

---

## 1. Transcripción (`ASR.transcription`)

Pipeline de transcripción ASR con soporte para múltiples modelos y formatos de salida.

### Modelos Soportados

| Modelo | Descripción | Requiere Prompt |
|--------|-------------|-----------------|
| `WhisperModel` | OpenAI Whisper (tiny, base, small, medium, large-v1/2/3) | Sí |
| `WhisperPromptedModel` | Whisper con prompt ATC predefinido | No (interno) |
| `WhisperATCModel` | HuggingFace WhisperATC (v2/v3) optimizado para ATC | No |
| `HuggingFaceModel` | Cualquier modelo HuggingFace ASR | Opcional |
| `FasterWhisperModel` | Whisper optimizado con CTranslate2 | Sí |

### Uso Básico

```python
from ASR.transcription import WhisperModel, TranscriptionPipeline, WhisperATCModel

# Whisper con prompt
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

# Transcribir directorio completo
results = pipeline.run_directory(
    directory="./recordings",
    output_path="./output/transcriptions.csv",
    extensions=(".mp3", ".wav", ".flac"),
    recursive=True
)

# O transcribir lista de archivos
results = pipeline.run(
    audio_files=["audio1.mp3", "audio2.wav"],
    output_path="./output/resultados.csv",
    model_column="model"
)
```

### WhisperATC (Modelo Especializado ATC)

```python
from ASR.transcription import WhisperATCModel

# No requiere prompt - ya está optimizado para ATC
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

### Prompts Disponibles

```python
from ASR.transcription import get_prompt, create_custom_prompt, AVAILABLE_PROMPTS

# Prompts predefinidos
prompt = get_prompt("default")   # Completo con NATO, terminología, aerolíneas
prompt = get_prompt("minimal")   # Esencial para ATC
prompt = get_prompt("extended")  # Más terminología adicional
prompt = get_prompt("none")      # Sin prompt

# Prompt personalizado
prompt = create_custom_prompt(
    include_nato=True,
    include_terminology=True,
    include_airlines=True,
    include_cuba_terms=True,
    extra_terms="términos personalizados"
)
```

### CLI (Línea de Comandos)

```bash
# Whisper con prompt por defecto
python -m ASR.transcription.cli \
    --model whisper \
    --model-size large-v3 \
    --input ./audio \
    --output resultados.csv \
    --prompt default

# WhisperATC (mejor para ATC)
python -m ASR.transcription.cli \
    --model whisperatc \
    --version v3 \
    --input ./audio \
    --output resultados.json \
    --timestamps

# HuggingFace con modelo personalizado
python -m ASR.transcription.cli \
    --model huggingface \
    --hf-model "jlvdoorn/whisper-large-v3-atco2-asr" \
    --input ./audio \
    --output resultados.csv

# FasterWhisper
python -m ASR.transcription.cli \
    --model faster-whisper \
    --model-size large-v3 \
    --device cuda \
    --input ./audio \
    --output resultados.csv
```

### Parámetros del CLI

| Parámetro | Opciones | Descripción |
|-----------|----------|-------------|
| `--model` | whisper, whisperatc, huggingface, faster-whisper | Tipo de modelo |
| `--model-size` | tiny, base, small, medium, large-v1/2/3 | Tamaño Whisper |
| `--hf-model` | string | ID HuggingFace |
| `--version` | v2, v3 | Versión WhisperATC |
| `--device` | auto, cpu, cuda | Dispositivo |
| `--prompt` | default, minimal, extended, none | Tipo de prompt |
| `--format` | csv, json | Formato salida |
| `--timestamps` | flag | Incluir timestamps |
| `--no-progress` | flag | Ocultar barra progreso |
| `--extensions` | string | Extensiones (default: .mp3,.wav,.flac,.m4a,.ogg,.mkv) |

### Formato de Salida

**CSV**: Una fila por modelo, columnas son archivos de audio
```csv
model,audio1.mp3,audio2.mp3
whisper-base,texto1,texto2
whisper-large,texto1_alt,texto2_alt
```

**JSON**: Estructura completa con timestamps y metadata
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
      "text": "transcripción...",
      "timestamps": [
        {"start": 0.0, "end": 3.5, "text": "fragmento 1"}
      ],
      "metadata": {"language": "en"}
    }
  ]
}
```

---

## 2. Normalización (`ASR.normalization`)

Normalización de textos ATC para comparación y evaluación WER.

### Funcionalidades

- Expansión de callsigns: `JBU1676` → `jetblue one six seven six`
- Expansión de números: `FL340` → `flight level three four zero`
- Expansión de frecuencias: `118.5` → `one one eight point five`
- Normalización de terminología ATC
- Expansión opcional ICAO: `BEMOL` → `bravo echo mike oscar lima`
- Eliminación de puntuación
- Conversión a minúsculas

### Uso

```python
from ASR.normalization import ATCTextNormalizer, quick_normalize

# Normalización completa
normalizer = ATCTextNormalizer(
    expand_callsigns=True,      # Expande JBU1676 → jetblue one six seven six
    expand_numbers=True,        # Expande 340 → three four zero
    expand_icao=False,          # Expande BEMOL → bravo echo...
    normalize_terminology=True, # Reemplaza abreviaturas ATC
    remove_punctuation=True,    # Elimina puntuación
    lowercase=True              # Convierte a minúsculas
)

text = "JBU1676 climb FL340 maintain heading 090"
normalized = normalizer.normalize(text)
# Resultado: "jetblue one six seven six climb flight level three four zero maintain heading zero niner zero"

# Función rápida con valores por defecto
normalized = quick_normalize(text)
```

### Terminología Disponible

```python
from ASR.normalization import (
    expand_callsign,      # Expande callsign con ICAO
    expand_number,        # Expande número a palabras
    expand_icao_spelling, # Expande palabra a alfabeto NATO
    number_to_word,       # Diccionario número→palabra
    airlines_icao,        # Diccionario aerolíneas ICAO
    iata_to_icao,         # Diccionario IATA→ICAO
    atc_terminology,      # Diccionario términos ATC
    nato_alphabet,        # Diccionario alfabeto NATO
)
```

---

## 3. Evaluación (`ASR.evaluation`)

Evaluación de transcripciones ASR contra ground truth usando WER y otras métricas.

### Uso Básico

```python
from ASR.evaluation import ASREvaluator, load_ground_truth, load_transcriptions_by_timestamp, align_data
from ASR.normalization import ATCTextNormalizer

# 1. Cargar ground truth desde DOCX
ground_truth_raw = load_ground_truth("./ground_truth.docx")
# {timestamp: texto_completo, ...}

# 2. Cargar transcripciones desde CSV (formato Whisper)
transcriptions_raw = load_transcriptions_by_timestamp("./transcriptions.csv")
# {modelo: {timestamp: texto, ...}, ...}

# 3. Normalizar
normalizer = ATCTextNormalizer()
ground_truth = {ts: normalizer.normalize(txt) for ts, txt in ground_truth_raw.items()}
transcriptions = {
    model: {ts: normalizer.normalize(txt) for ts, txt in model_data.items()}
    for model, model_data in transcriptions_raw.items()
}

# 4. Evaluar
evaluator = ASREvaluator(use_jiwer=True)
results = evaluator.evaluate_all_models(
    ground_truth=ground_truth,
    transcriptions=transcriptions,
    detailed=True
)

# 5. Ver resultados
for model_name, result in results.items():
    print(f"{model_name}: WER={result.wer:.2%}")
```

### Métricas Disponibles

| Métrica | Descripción |
|---------|-------------|
| `WER` | Word Error Rate (tasa de error de palabras) |
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
    reference="ground truth normalizado",
    hypothesis="predicción normalizada",
    wer=0.15,                    # 15% WER
    mer=0.12,
    wil=0.18,
    wip=0.82,
    cer=0.08,
    num_ref_words=45,
    num_hyp_words=43,
    errors={                     # Detalle de errores
        "substitutions": 5,
        "insertions": 2,
        "deletions": 3
    }
)
```

### Reportes y Comparación

```python
from ASR.evaluation import print_evaluation_report, compare_models

# Imprimir reporte formateado
print_evaluation_report(results)

# Comparar modelos
comparison = compare_models(results)
# Retorna ranking de modelos por WER
```

---

## 4. Script de Evaluación Completo (`run_evaluation.py`)

Script de línea de comandos para evaluación completa.

```bash
python ASR/run_evaluation.py \
    --ground-truth ./ground_truth.docx \
    --transcriptions ./transcriptions.csv \
    --normalize \
    --show-samples 5
```

### Parámetros

| Parámetro | Descripción |
|-----------|-------------|
| `--ground-truth` | Ruta al archivo DOCX con ground truth |
| `--transcriptions` | Ruta al CSV con transcripciones |
| `--normalize` | Aplicar normalización antes de evaluar |
| `--no-normalize` | No normalizar (texto crudo) |
| `--show-samples` | Número de muestras de normalización a mostrar |

---

## Ejemplo Completo: Pipeline de Transcripción + Evaluación

```python
from ASR.transcription import WhisperATCModel, TranscriptionPipeline
from ASR.evaluation import ASREvaluator, load_ground_truth, load_transcriptions_by_timestamp
from ASR.normalization import ATCTextNormalizer

# 1. Transcribir audio
model = WhisperATCModel(model_version="v3")
pipeline = TranscriptionPipeline(model, output_format="csv")
pipeline.run_directory("./recordings", "./output/transcriptions.csv")

# 2. Evaluar contra ground truth
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

# 3. Ver resultados
for model, res in results.items():
    print(f"{model}: WER = {res.wer:.2%}")
```

---

## Dependencias

```txt
# Transcripción
openai-whisper          # Para WhisperModel
faster-whisper          # Para FasterWhisperModel
transformers            # Para HuggingFaceModel
torch                   # Backend PyTorch
tqdm                    # Barras de progreso

# Normalización
# (solo dependencias estándar de Python)

# Evaluación
jiwer                   # Métricas WER (opcional pero recomendado)
python-docx             # Lectura de ground truth DOCX
```

---

## Instalación

```bash
# Instalar dependencias básicas
pip install -r requirements.txt

# O instalar manualmente según necesidad
pip install openai-whisper faster-whisper transformers torch tqdm
pip install jiwer python-docx
```

---

## Flujo de Trabajo Recomendado

1. **Grabar audio ATC** → Guardar en formato soportado (.mp3, .wav, .flac)
2. **Transcribir** → Usar `TranscriptionPipeline` con modelo apropiado
3. **Normalizar** → Aplicar `ATCTextNormalizer` a ground truth y transcripciones
4. **Evaluar** → Usar `ASREvaluator` para calcular WER
5. **Analizar** → Comparar modelos y ajustar prompts según resultados
