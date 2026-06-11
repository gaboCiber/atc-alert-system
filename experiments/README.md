# Experimentos — ATC Alert System

Serie de 6 experimentos que evalúan componentes del sistema de alertas ATC, desde la segmentación de texto hasta el benchmark integrado del pipeline completo.

## Progresión E1 → E6

```
E1: Chunk Comparison         (segmentación de texto)
  ↓
E2: KEX Comparison           (extracción de conocimiento)
  ↓
E3: ASR Comparison           (transcripción de audio)
  ↓
E4: Compiled Rules           (generación de código Python)
  ↓
E5: Alert Pipeline           (compiled vs generic LLM)
  ↓
E6: System Benchmark         (latencia + precisión end-to-end)
```

## Resumen por experimento

| # | Nombre | ¿Qué mide? | Input | Output principal |
|---|--------|-----------|-------|-----------------|
| **E1** | Chunk Comparison | Calidad de segmentación de texto aeronáutico en chunks lógicos | Páginas del manual ICAO | Boundary P/R/F1, Char/Word/ROUGE-L F1 |
| **E2** | KEX Comparison | Calidad de extracción de conocimiento (entidades, relaciones, eventos, reglas, procedimientos) | Páginas del manual ICAO | Structural F1, Content, CrossRef, Semantic Score |
| **E3** | ASR Comparison | Precisión de transcripción de audio ATC (WER/MER/WIL/WIP) | Audios ATC | WER, MER, WIL, WIP, Subs/Ins/Dels |
| **E4** | Compiled Rules | Capacidad de generar código Python ejecutable desde reglas KEX | Reglas KEX + TrafficStates | Classification, Validation, Execution, Semantic Score |
| **E5** | Alert Pipeline | Comparación estrategia compiled vs generic LLM para evaluar reglas | Instrucciones ATC + TrafficStates | Precision, Recall, F1, Severity Accuracy, Latency |
| **E6** | System Benchmark | Latencia y precisión del sistema completo (BERT + native + compiled + generic + pipeline) | Instrucciones ATC + TrafficStates | Latency stats, P/R/F1, Judge scores |

## Hallazgos clave

| Hallazgo | Experimentos | Detalle |
|----------|-------------|---------|
| Compiled ≈ native en velocidad | E4, E5, E6 | Ambos ~0.01ms, ~200,000× más rápido que generic |
| Generic LLM es el cuello de botella | E5, E6 | ~1.8s por evaluación vs ~0.01ms de compiled |
| Precisión similar entre estrategias | E5 | Compiled 0.813 vs Generic 0.808 (diferencia marginal) |
| BERT añade ~49ms de latencia fija | E6 | Independiente de la instrucción, es el parseo NLP |
| Severidad es el punto débil | E4, E5, E6 | Los modelos generan severidades incorrectas frecuentemente |
| RULE004 (runway) tiene severity_accuracy=0.0 | E6 | La regla compilada no asigna la severidad esperada |
| WER excelente con whisper-large | E3 | Solo 2.94% WER, todas las diferencias son eliminaciones |
| 5 tipos de KEX tienen dificultad desigual | E2 | Reglas y procedimientos son más difíciles que entidades |

## Dependencias entre experimentos

```
E1 ──→ E2 (mismos documentos ICAO como input)
E2 ──→ E4 (reglas KEX → compilación a código)
E4 ──→ E5 (código compilado → estrategia compiled)
E4 ──→ E6 (código compilado → componente compiled del benchmark)
E2 ──→ E6 (descripciones de reglas → componente generic del benchmark)
E5 ──→ E6 (metodología de comparación → benchmark integrado)
```

E3 es independiente: evalúa ASR, no produce inputs para los demás experimentos.

## Estructura de cada experimento

```
experiments/E{N}_{nombre}/
├── explain.md                # Documentación detallada del experimento
├── README.md                 # Instrucciones de uso (inglés)
├── config.json               # Pesos y parámetros del scoring
├── ground_truth/             # Datos de referencia
│   ├── test_cases.json       # (E4, E5, E6)
│   └── ...                   # Archivos específicos
├── src/                      # Código fuente del evaluador
│   ├── run.py                # Entry point CLI
│   ├── evaluator.py          # Lógica de evaluación
│   ├── report.py             # Generación de figuras + summary.json
│   ├── semantic_judge.py     # (E2, E4, E5, E6) LLM-as-a-Judge
│   └── ...                   # Otros módulos específicos
├── results/                  # Outputs generados
│   ├── summary.json          # Métricas agregadas
│   ├── detailed_results.json # Resultados por ítem
│   └── figures/              # 6–8 visualizaciones PNG
└── models/                   # (E1, E2, E3) Outputs de modelos evaluados
```

## Scoring: evolución a través de los experimentos

| Exp | Fórmula | Pesos |
|-----|---------|-------|
| E1 | 0.15×ChunkCount + 0.20×BoundaryF1 + 0.25×CharF1 + 0.20×WordF1 + 0.20×ROUGE-L | Pesos en `config.json` |
| E2 | 0.15×StructuralF1 + 0.10×Content + 0.15×CrossRef + 0.60×Semantic | Dominancia del juez semántico |
| E3 | WER (principal), MER, WIL, WIP (secundarias) | Sin fórmula combinada |
| E4 | 0.15×Cls + 0.15×Val + 0.30×Exec + 0.40×Sem | Ejecución y semántica pesan más |
| E5 | 0.30×Precision + 0.30×Recall + 0.20×Severity + 0.20×Semantic | Balance entre detección y calidad |
| E6 | Latencia (ranking) + Precisión (P/R/F1) + Judge (semantic) | Tres dimensiones separadas, sin pesos combinados |

## Visualizaciones comunes

Cada experimento genera 6–8 figuras PNG en `results/figures/`:

| Tipo de figura | E1 | E2 | E3 | E4 | E5 | E6 |
|----------------|----|----|----|----|----|----|
| Barras de ranking (score) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Latencia | — | — | — | — | ✓ | ✓ |
| Matriz de confusión | ✓ | — | — | — | ✓ | — |
| Precisión/Recall/F1 | — | — | — | — | ✓ | ✓ |
| Desglose por ítem | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| Juez semántico | — | ✓ | — | ✓ | ✓ | ✓ |
| Radar/comparativa | — | ✓ | — | — | — | — |
| Pipeline steps | — | — | — | — | — | ✓ |
| Latency breakdown | — | — | — | — | — | ✓ |
| Latency flame | — | — | — | — | — | ✓ |

## Navegación rápida

- [E1 — Chunk Comparison](E1_chunk_comparison/explain.md)
- [E2 — KEX Comparison](E2_kex_comparison/explain.md)
- [E3 — ASR Comparison](E3_asr_comparison/explain.md)
- [E4 — Compiled Rules](E4_compiled_rules/explain.md)
- [E5 — Alert Pipeline](E5_alert_pipeline/explain.md)
- [E6 — System Benchmark](E6_system_benchmark/explain.md)
