# ATC Alert System

Sistema integral de alertas de seguridad para controladores de tráfico aéreo (ATC) que combina reconocimiento de voz automático (ASR), extracción de conocimiento de documentación aeronáutica (KEX), y detección de violaciones de seguridad en tiempo real usando state projection.

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ATC Alert System                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │     ASR      │───▶│ Knowledge    │───▶│ Alert System │               │
│  │  Transcribe  │    │ Extractor    │    │  Pipeline    │               │
│  │  Audio → Text│    │  PDF → Rules │    │  Rules →     │               │
│  └──────────────┘    └──────────────┘    │  Alerts      │               │
│                                            └──────────────┘               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Flujo de Datos                                 │   │
│  │                                                                     │   │
│  │  Audio ATC → ASR → Transcripción → Normalización →                │   │
│  │  Instrucción Parseada → State Projection → Rule Evaluation →     │   │
│  │  Alert Generation → ATCO Decision → Commit/Rollback               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Fuente de Reglas                                │   │
│  │                                                                     │   │
│  │  Documentación PDF → Knowledge Extractor → Reglas Estructuradas →  │   │
│  │  KEX Adapter → Condition Evaluators → Rule Engine                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Componentes del Sistema

### 1. ASR (Automatic Speech Recognition)

Módulo de transcripción de audio ATC con múltiples modelos y normalización especializada.

**Características principales:**
- **Modelos soportados**: OpenAI Whisper, WhisperATC (optimizado para ATC), HuggingFace, FasterWhisper
- **Normalización ATC**: Expansión de callsigns, números, pistas, frecuencias, terminología
- **Evaluación**: Métricas WER, MER, WIL, WIP, CER con normalización automática
- **Reducción de ruido**: DeepFilterNet (requiere Python 3.9 en venv separado)
- **Prompts dinámicos**: Contexto específico por archivo (waypoints, callsigns)
- **Multi-model pipeline**: Comparación de modelos con checkpointing

**Uso básico:**
```python
from ASR.transcription import WhisperATCModel, TranscriptionPipeline

model = WhisperATCModel(model_version="v3")
pipeline = TranscriptionPipeline(model, output_format="csv")
pipeline.run_directory("./recordings", "./output/transcriptions.csv")
```

**Documentación completa**: Ver [ASR/README.md](ASR/README.md)

### 2. Knowledge Extractor

Sistema de extracción de conocimiento aeronáutico de documentación PDF usando LLMs con salida estructurada.

**Características principales:**
- **Extracción estructurada**: Entidades, relaciones, reglas, eventos, procedimientos
- **Contexto semántico**: Embeddings para selección relevante de contexto acumulado
- **Dos modos de extracción**: Joint (rápido) vs Sequential (más preciso)
- **Granularidad flexible**: Page, sentence, o chunking con LLM
- **Validación robusta**: Pydantic + Instructor con validación estricta
- **Resume capability**: Reanuda desde página específica con estado previo
- **Multi-provider LLM**: Ollama, OpenAI, Gemini, Anthropic

**Uso básico:**
```python
from Knowledge_Extractor import extract_knowledge

results = extract_knowledge("document.pdf", output_dir="output")
```

**Documentación completa**: Ver [Knowledge_Extractor/README.md](Knowledge_Extractor/README.md)

### 3. Alert System

Sistema de detección de violaciones de seguridad en tiempo real usando state projection.

**Características principales:**
- **State Projection**: Simulación "what-if" antes de modificar estado real
- **Gestión transaccional**: Commit/rollback con override del ATCO
- **Arquitectura híbrida**: Reglas conocidas + reglas genéricas del KEX
- **Evaluadores de condiciones**: Altitude, Separation, Runway, Generic
- **Pipeline de 8 pasos**: Input → Normalization → Projection → Evaluation → Alert → Decision
- **Integración ASR**: Adaptador para transcripciones → instrucciones
- **Integración KEX**: Adaptador para reglas → evaluadores

**Uso básico:**
```python
from Alert_System.pipeline.alert_pipeline import AlertPipeline
from Alert_System.core.state_manager import StateManager
from Alert_System.rule_engine.engine import RuleEngine

state_manager = StateManager(initial_state=traffic_state)
rule_engine = RuleEngine()
pipeline = AlertPipeline(state_manager, rule_engine)

result = pipeline.process_instruction("AAL123 descend to 4000")
```

**Documentación completa**: Ver [Alert_System/README.md](Alert_System/README.md)

## Instalación

### Requisitos Previos

- Python 3.10+
- Ollama (para modelos locales) o API keys para OpenAI/Gemini/Anthropic
- Python 3.9 en venv separado (opcional, para DeepFilterNet)

### Instalación del Proyecto

```bash
# Clonar el repositorio
git clone <repository-url>
cd atc-alert-system

# Instalar dependencias principales
pip install -r requirements.txt

# Instalar dependencias de ASR
pip install -r ASR/requirements.txt

# Instalar dependencias de Knowledge Extractor
pip install -r Knowledge_Extractor/requirements.txt
```

### Configuración de Ollama (Opcional)

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelo Whisper
ollama pull whisper-large-v3

# Descargar modelo para KEX (ej: llama3.2)
ollama pull llama3.2
```

### Configuración de DeepFilterNet (Opcional)

```bash
cd ASR/noise_reduction/scripts
chmod +x setup_venv.sh
./setup_venv.sh
```

## Flujo de Trabajo End-to-End

### Escenario 1: Transcripción y Evaluación ASR

```python
from ASR.transcription import WhisperATCModel, TranscriptionPipeline
from ASR.evaluation import ASREvaluator, EcnaDataLoader

# 1. Transcribir audio
model = WhisperATCModel(model_version="v3")
pipeline = TranscriptionPipeline(model, output_format="csv")
pipeline.run_directory("./recordings", "./output/transcriptions.csv")

# 2. Evaluar contra ground truth
loader = EcnaDataLoader()
evaluator = ASREvaluator(use_jiwer=True, use_atc_normalizer=True)
results = evaluator.evaluate_all_models_with_loader(
    data_loader=loader,
    ground_truth_path="./ground_truth.docx",
    transcriptions_path="./output/transcriptions.csv",
    detailed=True
)
```

### Escenario 2: Extracción de Conocimiento y Alertas

```python
from Knowledge_Extractor import extract_knowledge
from Alert_System.pipeline.alert_pipeline import AlertPipeline
from Alert_System.core.state_manager import StateManager
from Alert_System.rule_engine.engine import RuleEngine

# 1. Extraer reglas de documentación PDF
kex_results = extract_knowledge("aicp_document.pdf", output_dir="kex_output")
rules = kex_results.get("rules", [])

# 2. Inicializar sistema de alertas
state_manager = StateManager(initial_state=traffic_state)
rule_engine = RuleEngine()
pipeline = AlertPipeline(state_manager, rule_engine)

# 3. Cargar reglas del KEX
pipeline.load_kex_rules(rules)

# 4. Procesar instrucción ATC
result = pipeline.process_instruction("AAL123 descend to 4000")

# 5. Revisar alertas
if result.alerts_generated:
    for alert in result.alerts_generated:
        print(f"⚠️ {alert.severity}: {alert.title}")
        print(f"   {alert.explanation}")
```

### Escenario 3: Pipeline Completo Integrado

```python
from ASR.transcription import WhisperATCModel, TranscriptionPipeline
from Alert_System.integration.asr_adapter import ASRAdapter
from Alert_System.pipeline.alert_pipeline import AlertPipeline

# 1. Transcribir audio ATC
model = WhisperATCModel(model_version="v3")
asr_pipeline = TranscriptionPipeline(model, output_format="json")
transcription = asr_pipeline.run_single("atc_audio.wav")

# 2. Adaptar a instrucción parseada
asr_adapter = ASRAdapter()
instruction = asr_adapter.adapt(transcription)

# 3. Procesar en pipeline de alertas
result = alert_pipeline.process_instruction(pre_parsed=instruction)

# 4. Decisión del ATCO
if result.alerts_generated:
    # Presentar alertas al controlador
    # El controlador decide COMMIT o ROLLBACK
    if result.alerts_generated[0].is_critical():
        result.final_decision = "ROLLBACK"
    else:
        result.final_decision = "COMMIT"
```

## Estructura del Proyecto

```
atc-alert-system/
├── ASR/                          # Módulo de reconocimiento de voz
│   ├── transcription/            # Modelos ASR y pipeline
│   ├── normalization/            # Normalización de texto ATC
│   ├── evaluation/               # Evaluación WER y métricas
│   ├── noise_reduction/          # DeepFilterNet para reducción de ruido
│   └── requirements.txt
│
├── Knowledge_Extractor/          # Módulo de extracción de conocimiento
│   ├── core/                     # Procesamiento de documentos y contexto
│   ├── extractors/               # Extracción LLM con Instructor
│   ├── schemas/                  # Modelos Pydantic
│   ├── pipeline/                 # Orquestador de extracción
│   ├── integration/              # Adaptadores KEX y ASR
│   └── requirements.txt
│
├── Alert_System/                 # Sistema de alertas de seguridad
│   ├── core/                     # State projection y state management
│   ├── models/                   # Modelos de instrucciones, estado, alertas
│   ├── rule_engine/              # Motor de reglas y evaluadores
│   ├── pipeline/                 # Pipeline de 8 pasos
│   ├── integration/              # Adaptadores KEX y ASR
│   └── config/                   # Patrones de reglas JSON
│
└── README.md                     # Este archivo
```

## Tecnologías Utilizadas

### ASR
- **openai-whisper**: Modelo base Whisper
- **faster-whisper**: Optimización CTranslate2
- **transformers**: HuggingFace Transformers
- **torch/torchaudio**: Backend PyTorch
- **jiwer**: Métricas WER
- **sentence-transformers**: Embeddings semánticos

### Knowledge Extractor
- **PyMuPDF**: Extracción de texto PDF
- **nltk**: Tokenización y segmentación
- **instructor**: Salida estructurada LLM
- **openai**: Cliente OpenAI (compatible Ollama)
- **pydantic**: Validación de datos

### Alert System
- **pydantic**: Modelos de datos
- **Python estándar**: Sin dependencias externas pesadas

## Tests

### Ejecutar Tests de ASR

```bash
# Tests de transcripción
pytest tests/asr/test_transcription.py -v

# Tests de normalización
pytest tests/asr/test_normalization.py -v

# Tests de evaluación
pytest tests/asr/test_evaluation.py -v
```

### Ejecutar Tests de Alert System

```bash
# Tests del pipeline
pytest tests/alert_system/test_pipeline.py -v
```

## Configuración

### Variables de Entorno

No requeridas actualmente. El sistema usa configuración en código y archivos JSON.

### Archivos de Configuración

- `Alert_System/config/rule_patterns.json`: Patrones de mapeo de reglas KEX
- `ASR/transcription/config/prompts.py`: Prompts para modelos Whisper

## Contribución

Este proyecto es de uso académico. Para contribuir:

1. Fork el repositorio
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## Licencia

Uso académico únicamente.

## Contacto

Para preguntas o contribuciones, contactar al equipo de desarrollo.
