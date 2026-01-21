# Automated Grid Compliance LLM Pipeline

An end-to-end MLOps pipeline for fine-tuning a Small Language Model (SLM) to serve as a **Technical Compliance QA Assistant** for UK EV charging and grid connection standards.

---

## 📚 Project Overview

This pipeline automates:

1. **PDF Processing**: Selective extraction of critical pages from technical standards using `pypdf` and `docling`.
2. **Synthetic Q&A Generation**: Creating training data using Groq's `openai/gpt-oss-120b` model with tier-specific prompts.
3. **Dataset Storage**: Dual storage in SQLite (local) and HuggingFace Hub (cloud).
4. **Future Stages**: Fine-tuning with Qwen 2.5, evaluation, and serving via FastAPI.

---

## 📖 Knowledge Base (Data Sources)

The model is trained on three UK energy sector documents:

| Document                                            | Role                                               | Link                                                                                                         |
| --------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **ENA Engineering Recommendation G99 (Issue 2)**    | Technical "Bible" for generation and export limits | [Download](<https://dcode.org.uk/assets/250307ena-erec-g99-issue-2-(2025).pdf>)                              |
| **UK Power Networks: EV Connections (EDS 08-5050)** | DNO rules for connection, earthing, and safety     | [Download](https://media.umbraco.io/uk-power-networks/baik5mop/eds-08-5050-electric-vehicle-connections.pdf) |
| **SP Energy Networks: Connecting your EV fleet**    | Commercial workflow and project estimation         | [Download](https://www.spenergynetworks.co.uk/userfiles/file/Connecting%20your%20EV%20fleet%20V3.pdf)        |

---

## 🚀 Setup Instructions

### 1. System Dependencies

**Important**: Before installing Python packages, install these system dependencies required by `docling` for OCR and image processing:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libgl1
```

**Why these are needed:**

- **`tesseract-ocr`**: Open-source OCR engine used by `docling` to extract text from scanned PDFs and images.
- **`libgl1`**: OpenGL library required by `opencv-python` (a dependency of `docling`) for image processing.

Without these, you'll encounter errors like:

- `TesseractNotFoundError` when processing PDFs with images
- `ImportError: libGL.so.1` when docling tries to load image processing libraries

### 2. Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Data Preparation

Download the three PDFs (links above) and place them in `data/raw/` with these **exact names**:

```
data/raw/
├── G99_Issue_2.pdf
├── UKPN_EDS_08_5050.pdf
└── SPEN_EV_Fleet_Guide.pdf
```

### 4. Environment Variables

Create a `.env` file in the project root:

```bash
# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# HuggingFace Token (for pushing datasets)
HF_TOKEN=your_huggingface_token_here

# HuggingFace Repository ID
HF_REPO_ID=your-username/grid-compliance-qa
```

---

## 🔧 Pipeline Stages

### Stage 1: PDF Processing

Extract layout-aware content from PDFs:

```bash
python src/processing/pdf_processor.py
```

**What it does:**

1. **Slices PDFs**: Extracts only relevant pages (defined in `SLICING_CONFIG`) using `pypdf`.
2. **Docling Conversion**: Processes sliced PDFs to extract:
   - Text content
   - Tables and diagrams
   - Metadata with original page numbers
3. **Stores in SQLite**: Saves chunks to `data/pipeline.db` → `document_chunks` table.

**Schema: `document_chunks`**

- `id`: Auto-increment primary key
- `source_file`: PDF filename (e.g., `G99_Issue_2.pdf`)
- `page_number`: Original page number
- `content`: Extracted markdown text
- `extracted_at`: Timestamp

---

### Stage 2: Q&A Generation

Generate synthetic training data using Groq:

```bash
# Generate Q&A pairs
python src/generation/qa_generator.py --generate

# View statistics
python src/generation/qa_generator.py --stats

# Export to JSONL
python src/generation/qa_generator.py --export

# Push to HuggingFace
python src/generation/qa_generator.py --push-hf
```

**How it works:**

1. **Three-Tier Strategy**: Each PDF uses a specialized prompt:
   - **Tier 1 (G99)**: Factual questions about numerical thresholds, classifications, protection settings.
   - **Tier 2 (UKPN)**: Design/procedural questions about earthing, separation rules, and notifications.
   - **Tier 3 (SPEN)**: Scenario-based questions for fleet planning and cost estimation.

2. **Pydantic Schema Validation**: Uses structured output with:

   ```python
   {
     "items": [
       {"question": "...", "answer": "..."},
       ...
     ]
   }
   ```

3. **Rate Limiting**: Sleeps every 20 requests to avoid API limits.

4. **Dual Storage**:
   - **Local**: SQLite `training_dataset` table
   - **Cloud**: HuggingFace Hub dataset

**Schema: `training_dataset`**

- `id`: Auto-increment primary key
- `chunk_id`: Foreign key to `document_chunks`
- `source_file`: PDF filename
- `page_number`: Original page number
- `question`: Generated question
- `answer`: Generated answer
- `generated_at`: Timestamp

**Output Format (JSONL & HuggingFace)**

```json
{
  "input": "What is the frequency response requirement for Type A modules?",
  "output": "Type A modules must comply with Section 11.2.3...",
  "metadata": {
    "source": "G99_Issue_2.pdf",
    "page": 142
  }
}
```

---

## 📊 Project Structure

```
automated-grid-compliance-llm-pipeline/
├── data/
│   ├── raw/                         # Manual PDF downloads
│   ├── interim/                     # Sliced PDFs (temporary)
│   ├── output/                      # JSONL exports
│   └── pipeline.db                  # SQLite database
├── notebooks/
│   └── exploration.ipynb            # Data exploration notebook
├── src/
│   ├── processing/
│   │   ├── pdf_processor.py         # PDF slicing + docling extraction
│   │   └── data_processing_notes.md # Rationale for page selection
│   ├── generation/
│   │   └── qa_generator.py          # Groq-based Q&A generation
│   ├── training/
│   │   └── qwen_qlora_finetuning.ipynb  # QLoRA fine-tuning notebook
│   └── serving/
│       └── app.py                   # FastAPI inference server
├── Dockerfile                       # Container configuration
├── docker-compose.yml               # Multi-service orchestration
├── requirements.txt                 # All project dependencies
├── .env.example                     # Environment template
└── README.md
```

---

## 🛠️ Dependencies

Key libraries:

- **`pypdf`**: PDF slicing
- **`docling`**: Layout-aware PDF extraction (requires `tesseract-ocr` + `libgl1`)
- **`groq`**: API client for `openai/gpt-oss-120b`
- **`pydantic`**: Schema validation for LLM outputs
- **`datasets`** + **`huggingface_hub`**: Cloud dataset storage

---

## 📝 Dataset Statistics

After running `--generate`:

```
Total QA pairs: 416
By Source:
  G99_Issue_2.pdf: 294
  SPEN_EV_Fleet_Guide.pdf: 77
  UKPN_EDS_08_5050.pdf: 45
```

🔗 **Dataset on HuggingFace**: [sayedsalem/grid-compliance-qa](https://huggingface.co/datasets/sayedsalem/grid-compliance-qa)

---

## 🧠 Stage 3: Fine-Tuning with QLoRA

Fine-tune Qwen 2.5-7B using QLoRA (4-bit quantization) on the generated dataset.

### Training Configuration

| Parameter           | Value                                                                       |
| ------------------- | --------------------------------------------------------------------------- |
| Base Model          | `Qwen/Qwen2.5-7B-Instruct`                                                  |
| Quantization        | 4-bit NF4 with double quantization                                          |
| LoRA Rank (r)       | 16                                                                          |
| LoRA Alpha          | 32                                                                          |
| Target Modules      | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Batch Size          | 2 (with 4 gradient accumulation steps)                                      |
| Learning Rate       | 2e-4                                                                        |
| Epochs              | 3                                                                           |
| Max Sequence Length | 512                                                                         |

### Data Split

- **Training**: 396 samples
- **Testing**: 20 samples (balanced: 10 G99, 6 SPEN, 4 UKPN)

### Run Training

Open the notebook in Google Colab (requires T4 GPU):

```bash
src/training/qwen_qlora_finetuning.ipynb
```

---

## 📊 Evaluation Results

### Metrics Comparison: Baseline vs Fine-Tuned

| Metric          | Baseline        | Fine-Tuned      | Improvement  |
| --------------- | --------------- | --------------- | ------------ |
| **ROUGE-1**     | 0.1764          | 0.3827          | **+116.96%** |
| **ROUGE-2**     | 0.0517          | 0.1646          | **+218.52%** |
| **ROUGE-L**     | 0.1141          | 0.2993          | **+162.36%** |
| **BLEU**        | 0.0182          | 0.1005          | **+452.08%** |
| **Avg Latency** | 19.91s          | 10.81s          | **-45.7%**   |
| **Throughput**  | 0.050 samples/s | 0.093 samples/s | **+84.3%**   |

### Sample Predictions

**Question**: _What is the aggregate power rating for a Power Generating Facility comprised of three 400 kW Type A Synchronous Power Generating Modules?_

| Model            | Response                                                                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ground Truth** | The total power... is 1.2 MW.                                                                                                                             |
| **Baseline**     | The aggregate power rating would be the sum of the individual power ratings. Given: Each module...                                                        |
| **Fine-Tuned**   | A Power Generating Facility made up of three 400 kW Type A Synchronous Power Generating Modules has an aggregate power rating of **1.2 MW** (3 x 400 kW). |

🔗 **Model on HuggingFace**: [sayedsalem/qwen2.5-7b-grid-compliance](https://huggingface.co/sayedsalem/qwen2.5-7b-grid-compliance)

---

## 📈 Experiment Tracking

Training runs are tracked with **Weights & Biases**:

- Project: `grid-compliance-qwen-qlora`
- Metrics logged: training loss, eval loss, ROUGE, BLEU, latency, throughput

🔗 **WandB Report**: [View Full Training Report](https://api.wandb.ai/links/sayedsalem767-ml-eng-/j9uuzgsb)

---

## � Stage 4: API Serving with FastAPI

Deploy the fine-tuned model as a REST API.

### Running Locally

```bash
cd src/serving
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Endpoint  | Method | Description                |
| --------- | ------ | -------------------------- |
| `/`       | GET    | API info and links         |
| `/health` | GET    | Health check status        |
| `/ask`    | POST   | Ask a single question      |
| `/batch`  | POST   | Process multiple questions |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the 30% rule for EV fleet load management?"}'
```

### API Documentation

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>

---

## 🐳 Stage 5: Docker Containerization

Deploy the API using Docker with GPU support.

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA 12.1+

### Build and Run

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop the service
docker-compose down
```

### Environment Variables

| Variable         | Default                                 | Description                  |
| ---------------- | --------------------------------------- | ---------------------------- |
| `MODEL_ID`       | `sayedsalem/qwen2.5-7b-grid-compliance` | HuggingFace model ID         |
| `MAX_NEW_TOKENS` | `256`                                   | Max tokens to generate       |
| `HF_TOKEN`       | -                                       | HuggingFace token (optional) |

### Production Deployment

```bash
# Run in production mode
docker-compose -f docker-compose.yml up -d

# Scale if needed (requires load balancer)
docker-compose up -d --scale api=2
```

---

## ✅ Project Status

| Stage                   | Status  | Description                          |
| ----------------------- | ------- | ------------------------------------ |
| **1. PDF Processing**   | ✅ Done | Docling extraction with page slicing |
| **2. Q&A Generation**   | ✅ Done | Groq-powered synthetic data          |
| **3. Fine-Tuning**      | ✅ Done | QLoRA on Qwen 2.5-7B                 |
| **4. Evaluation**       | ✅ Done | ROUGE/BLEU metrics                   |
| **5. API Serving**      | ✅ Done | FastAPI inference server             |
| **6. Containerization** | ✅ Done | Docker + docker-compose              |

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- **ENA**, **UK Power Networks**, **SP Energy Networks** for public technical documentation.
- **Groq** for API access.
- **HuggingFace** for dataset hosting.
