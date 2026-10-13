# AI-Auto-Test-Evaluator (AI-Tutor)
### Intelligent Academic Assignment Evaluation Platform

An offline, privacy-preserving, AI-powered academic evaluation platform for grading handwritten, scanned, and digital student assignments. Built with **FastAPI**, **Ollama**, **ChromaDB**, **OpenCV**, and multi-engine Vision OCR capabilities.

---

## 🌟 Key Features

### 1. Teacher Management & Knowledge Ingestion
- **Document Ingestion**: Teachers can upload Question Papers and Model Answers/Rubrics (PDF or text).
- **Automated Question Parsing**: Regex and heuristic decomposition into individual questions, sub-questions (e.g., `Q1`, `Q2(a)`), and max mark allocations.
- **Semantic Vector Storage**: Indexes exam guidelines and model answers into a persistent **ChromaDB** vector database using `qwen3-embedding:4b`.
- **Review & Override**: Teachers can inspect student answer sheets side-by-side with OCR extractions, adjust marks per question, and add personalized teacher remarks.

### 2. Student Submission & Evaluation Pipeline
- **Multi-Format Uploads**: Supports single/multi-page PDFs and images (`PNG`, `JPG`, `WEBP`).
- **Answer Segmentation (Phase 3)**: Automatically segments continuous student handwriting into discrete answers mapped to question IDs.
- **Rubric-Grounded Evaluation (Phase 4)**: Evaluates student answers on a per-question basis against model answers and rubrics using local LLMs.
- **Transparent Feedback**: Provides marks awarded, rubric reasoning, identified missing concepts, and encouraging tutor feedback.
- **Edge-Case Detection**: Automatically flags skipped questions, ambiguous mappings, or unmapped content for teacher review (`review_required`).

### 3. Diagnostic OCR Lab
- **Interactive Image Preprocessing**: Real-time calibration of OpenCV filters (Bilateral Denoising, Grayscale, Adaptive Gaussian Thresholding, and Deskewing).
- **Multi-Engine Telemetry**: Live performance measurement of execution time (ms), skew angles, and character counts.

---

## 🔍 OCR Engines & Hardware Compatibility Guide

The platform integrates four distinct OCR engines. Depending on your GPU VRAM and CPU capacity, choose the engine best suited for your hardware:

| OCR Engine | Architecture / Backend | Minimum VRAM | Recommended Hardware | Status / Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| **Qwen2.5-VL** | `qwen2.5vl:3b` via **Ollama** (GGUF Q4_K_M) | **~2.2 GB** | Consumer Laptops (RTX 3050, 4GB+ VRAM) or Apple Silicon | 🌟 **Recommended Default**: Fast (2–5s/page), accurate handwriting recognition, zero OOM errors. |
| **Standard OCR** | OpenCV Preprocessing + Gemma Vision / PaddleOCR | **< 500 MB** | Any PC / CPU only | ⚡ **Ultra-Lightweight**: Instant, ideal for low-end machines and prototyping. |
| **Surya OCR** | PyTorch SegFormer (`det3`) + Transformer (`rec2`) | **≥ 6 GB** | Mid/High-Tier GPUs (RTX 3060 12GB, RTX 4070+) | ⚠️ **Heavy on 4GB VRAM**: Full pages can cause CUDA OOM on 4GB GPUs. Runs on CPU, but takes 45–90s per page. |
| **Chandra OCR** | `datalab-to/chandra-ocr-2` (~8B Vision-Language model) | **≥ 12 GB** | Workstations / Servers (TITAN Xp, RTX 3090, A100) | 🚫 **Not for 4GB–8GB GPUs**: Requires >10GB VRAM on CUDA; loading on CPU consumes >16GB system RAM. |

> **Hardware Advice for 4GB VRAM Laptops (e.g. RTX 3050)**:  
> Use **Qwen2.5-VL (`qwen2.5vl:3b`)** or **Standard OCR**. They fit comfortably in available VRAM (~3.2 GB usable) alongside Windows display overhead without memory spikes.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, Pydantic v2, AnyIO
- **Vector Database**: ChromaDB (`chromadb`)
- **Text & Embedding Pipeline**: LangChain (`langchain-core`, `langchain-experimental`), `qwen3-embedding:4b`
- **Vision & Document Processing**: OpenCV (`opencv-python-headless`), PyMuPDF (`fitz`), Pillow, pdfplumber, PyPDF2
- **Local LLM Engine**: Ollama (`qwen2.5vl:3b`, `gemma3:270m`, `gemma3:latest`, `phi3:latest`)
- **Frontend**: Vanilla HTML5, Modern CSS3 (Variables, Glassmorphism, Responsive Grid), ES6+ JavaScript
- **Test Suite**: Pytest (23 automated unit and integration tests)

---

## 📁 Repository Structure

```text
AI-Tutor/
├── AI_Tutor_Phase4_Synthetic_Dataset/  # Phase 4 test PDFs, rubric, and expected results
│   ├── student_A_strong_answers.pdf
│   ├── student_B_partial_and_skipped.pdf
│   ├── student_C_mapping_edge_cases.pdf
│   ├── teacher_question_paper.pdf
│   ├── teacher_model_answers_and_rubric.pdf
│   ├── expected_results.json
│   └── expected_results.csv
├── Dataset/                            # Real-world computer network question/answer sheets
├── app/
│   ├── data/
│   │   ├── chroma_db/                  # Persistent ChromaDB vector index
│   │   ├── documents/                  # Ingested teacher documents per test ID
│   │   └── store.json                  # Persistent JSON storage for tests & submissions
│   ├── services/
│   │   ├── chandra_ocr.py              # Isolated Chandra OCR service (12GB+ VRAM)
│   │   ├── chunker.py                  # Semantic document chunker
│   │   ├── embedding.py                # Ollama embedding integration (qwen3-embedding:4b)
│   │   ├── evaluator.py                # Semantic evaluator & rubric scoring service
│   │   ├── gpu_memory.py               # GPU memory cleanup & cache management
│   │   ├── knowledge_store.py          # Persistent ChromaDB teacher knowledge index
│   │   ├── ocr.py                      # Standard OCR service (OpenCV + Gemma)
│   │   ├── preprocess.py               # OpenCV image preprocessing pipeline
│   │   ├── question_parser.py          # Question decomposition & structure parser
│   │   ├── qwen_ocr.py                 # Qwen2.5-VL Vision OCR service via Ollama
│   │   ├── student_segmenter.py        # Student answer segmentation & mapping
│   │   ├── surya_service.py            # Surya OCR service with CPU/GPU memory routing
│   │   └── teacher_ingestion.py        # End-to-end teacher material ingestion service
│   ├── static/                         # Frontend styling, scripts, and media assets
│   ├── templates/                      # Jinja2 HTML templates
│   └── main.py                         # FastAPI application & API endpoints
├── tests/
│   ├── test_extraction.py              # Tests for image extraction & filters
│   ├── test_phase4.py                  # Tests for segmented evaluation & rubrics
│   ├── test_qwen_ocr.py                # Tests for Qwen2.5-VL OCR isolation & routing
│   └── test_test_management.py         # Tests for teacher test lifecycle & API endpoints
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Platform documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12
- **Ollama**: Installed and running locally ([ollama.ai](https://ollama.ai))

### 2. Download Recommended Ollama Models
Open your terminal and run:
```bash
# Vision OCR Engine (Fast and fits in 4GB VRAM)
ollama pull qwen2.5vl:3b

# Evaluator LLM (Ultra-lightweight, 290MB)
ollama pull gemma3:270m

# Knowledge Base Embeddings
ollama pull qwen3-embedding:4b

# Optional: Higher-capacity evaluator models
ollama pull phi3:latest
ollama pull gemma3:latest
```

### 3. Installation
Clone the repository and set up a virtual environment:
```bash
# Clone the repository
git clone https://github.com/ohm-shahh/AI-Auto-Test-Evaluator.git
cd AI-Auto-Test-Evaluator

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(Optional for GPU acceleration with PyTorch)*:  
If you have an NVIDIA GPU, install the PyTorch build matching your CUDA version (e.g., CUDA 12.8):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128
```

### 4. Run the Application
Start the FastAPI server:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 🧪 Running the Test Suite

The test suite validates OCR engine isolation, document parsing, question segmentation, and rubric evaluation:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/
```

All **23 unit and integration tests** should pass:
- `tests/test_extraction.py` (5 tests)
- `tests/test_phase4.py` (3 tests)
- `tests/test_qwen_ocr.py` (12 tests)
- `tests/test_test_management.py` (3 tests)

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Application UI (Teacher, Student, Lab) |
| `GET` | `/api/tests` | List all tests with submission counts and average scores |
| `GET` | `/api/tests/active` | List published, active tests for students |
| `POST` | `/api/tests` | Create a new test |
| `GET` | `/api/tests/{test_id}` | Fetch test metadata and question breakdown |
| `DELETE`| `/api/tests/{test_id}` | Delete test and associated ChromaDB vectors |
| `POST` | `/api/tests/{test_id}/ingest` | Ingest Question Paper & Model Answer PDFs into vector DB |
| `GET` | `/api/tests/{test_id}/knowledge-status` | Inspect ingestion status and chunk statistics |
| `POST` | `/api/tests/{test_id}/submit` | Submit student assignment sheet (PDF or image) with selected OCR engine |
| `GET` | `/api/tests/{test_id}/submissions` | View all student submissions for a test |
| `POST` | `/api/submissions/{sub_id}/evaluate` | Trigger Phase 4 per-question semantic rubric grading |
| `PATCH`| `/api/submissions/{sub_id}` | Teacher manual score override and remarks |
| `GET` | `/api/ocr/status` | Real-time health check of all 4 OCR engines |
| `POST` | `/api/extract` | Diagnostic extraction endpoint with OpenCV filters |
| `POST` | `/api/surya/extract` | Dedicated diagnostic endpoint for Surya OCR |

---

## 📊 Evaluation Schema Example

The LLM evaluation produces strict, rubric-grounded JSON for every segmented question:

```json
{
  "marks_awarded": 4.5,
  "max_marks": 5.0,
  "reasoning": "Student correctly defined the primary key constraint and identified non-null requirement. Minor omission on composite keys.",
  "missing_concepts": [
    "Composite key definition"
  ],
  "feedback": "Strong answer! To get full marks, also mention that primary keys can be composed of multiple attributes (composite keys).",
  "evaluation_status": "completed",
  "needs_teacher_review": false
}
```

---

## 📄 License & Attribution

Developed for academic assignment evaluation research. 100% offline and local inference ensuring student data privacy.