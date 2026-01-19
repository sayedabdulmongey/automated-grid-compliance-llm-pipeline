# Implementation Plan - Automated Grid Compliance LLM Pipeline

## Goal Description

Build an end-to-end MLOps pipeline to fine-tune a Small Language Model (SLM) for the energy sector, specifically as a **Technical Compliance QA Assistant** for Grid Connection Standards. This aligns with the "Interview task" requirements and the "summary.pdf" blueprint.

## Technical Decisions

> [!IMPORTANT]
> **Data Sources**: We will focus on these 3 documents:
>
> 1. **ENA Engineering Recommendation G99 (Issue 2)**: [Link](<https://dcode.org.uk/assets/250307ena-erec-g99-issue-2-(2025).pdf>) (The "Bible")
> 2. **UK Power Networks: Electric Vehicle Connections (EDS 08-5050)**: [Link](https://media.umbraco.io/uk-power-networks/baik5mop/eds-08-5050-electric-vehicle-connections.pdf) (The "Specifics")
> 3. **SP Energy Networks (SPEN): Connecting your EV fleet**: [Link](https://www.spenergynetworks.co.uk/userfiles/file/Connecting%20your%20EV%20fleet%20V3.pdf) (The "Commercial/Fleet")

> **Tools**:
>
> - **Model**: **Qwen 2.5 (3B or 7B)** - Chosen for its superior logic and technical reasoning capabilities.
> - **PDF Extraction**: `docling` (Layout-aware, handles tables).
> - **Orchestration**: A simple **Python-based runner** or **Makefile** for end-to-end automation without over-engineering.
> - **Fine-Tuning**: `unsloth` for fast, memory-efficient LoRA/QLoRA.
> - **Serving**: `FastAPI` + `vLLM` or `HuggingFace Pipeline`.

## Proposed Changes

### Structure

```
automated-grid-compliance-llm-pipeline/
├── config/
│   └── config.yaml          # Central configuration
├── src/
│   ├── data_collection/     # Scrapers/Downloaders
│   ├── processing/          # PDF extraction (Docling), cleaning, chunking
│   ├── generation/          # Synthetic QA generation (Three-Tier Strategy)
│   ├── training/            # Fine-tuning scripts (Unsloth)
│   ├── evaluation/          # ROUGE/BLEU & Benchmarking
│   └── serving/             # FastAPI app + inference engine
├── scripts/                 # Orchestration (runner.py or Makefile)
├── requirements.txt
└── .env.example
```

## [Stage 1] Data Collection & Processing

#### `src/data_collection/scraper.py`

- Automate the download of the 3 key PDFs.

#### `src/processing/pdf_processor.py`

- Use `docling` to extract text + tables as Markdown.
- **Selective Slicing (G99 focus)**: Only process pages critical for EV/V2G and domestic compliance.
- **Documentation**: Document the "Rationale" for each selected range in a `data_processing_notes.md`.

## G99 Slicing Strategy (Priority Data)

To handle the 500-page G99 document, we will filter for:

1. **Pages 26–44**: Definitions (V2G, Storage) & Thresholds for Type A.
2. **Pages 61–65**: Connect & Notify logic (SGI-1, 2, 3).
3. **Pages 75–80**: Phase balance (16A limit) & Shared systems (32A limit).
4. **Pages 89–98**: LV Connection Earthing (Safety Diagrams).
5. **Pages 125–126**: Protection Settings (The "Holy Grail" Table 10.1).
6. **Pages 137–144**: Type A Technical Requirements & Frequency Response.
7. **Page 199**: V2G Compliance responsibilities (Section 15.5).
8. **Pages 312–317**: Storage Exceptions (Annex A.4.2).
9. **Pages 318–319**: Phase Imbalance Maths (Annex A.5).

## UKPN EDS 08-5050 Slicing Strategy

Focus on physical connection and charging safety:

1. **Pages 8–9**: Process Logic & Supply Matrix (Table 5-1).
2. **Page 11**: Unmetered Supply (UMS) Technical Limits (Table 6-1).
3. **Pages 11–13**: Earthing & Separation Rules (Table 7-1, 2.5m rule).
4. **Page 14**: Mandatory Notification routing.
5. **Page 17**: Approved Open PEN Devices (Table B-1).

## SPEN EV Fleet Guide Slicing Strategy

Focus on commercial planning and fleet logistics:

1. **Pages 5–6**: Capacity Assessment (Authorsied Capacity vs Demand) & The **"30% Rule"**.
2. **Page 10**: Operational Mitigation (Timed Profile Connections, Storage).
3. **Pages 11–13**: Cost & Timeline Case Studies (Small/Flexible/Large benchmarks).
4. **Pages 14–15**: DNO-Specific Definitions (Rapid vs Ultra-Rapid speeds).

### [Stage 2] Dataset Creation (The "Three-Tier" Strategy)

#### `src/generation/qa_generator.py`

Generate specialized Q&A pairs using a "Senior Engineer" persona via **Gemini 2.0 Flash**:

- **Schema**: JSON structure with `question` and `answer` keys.
- **Storage**: All data (chunks + generated Q&A) stored in a **SQLite database** (`data/pipeline.db`).
- **Tier 1 (Fact)**: Hard numerical limits from G99 tables.
- **Tier 2 (Design)**: Compliance rules from UKPN EDS.
- **Tier 3 (Scenario)**: Fleet logistics and project estimation from SPEN Guide.
- **Metadata**: Source file and page attribution included for every entry.

### [Stage 3] Fine-Tuning

#### `src/training/trainer.py`

- Implement LoRA fine-tuning for Qwen 2.5 using `unsloth`.
- Track metrics via `wandb` or `mlflow`.

### [Stage 4] Evaluation & Serving

#### `src/evaluation/benchmark.py`

- Benchmark fine-tuned performance vs baseline on a generated "Gold Standard" test set.

#### `src/serving/api.py`

- FastAPI endpoint exposing the fine-tuned model for QA.

## Verification Plan

### Automated Tests

- **Unit Tests**: Verify PDF extraction text quality and chunking logic.
- **Pipeline Test**: A dry-run of the `runner.py` with minimal data to ensure all stages connect correctly.

### Manual Verification

- **Quality Check**: Inspect generated Q&A pairs for engineering accuracy.
- **Inference Check**: Test the served API with a specific edge-case question (e.g., "PME earthing for outdoor chargers").
