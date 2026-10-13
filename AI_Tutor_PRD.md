# Product Requirements Document (PRD) & System Architecture Specification
## Project: AI Tutor — Intelligent Academic Assignment Evaluation Platform

---

## 1. Executive Summary & Vision

The **AI Tutor Platform** is an offline, privacy-preserving, local AI-powered academic evaluation system based on the research specification (*AI_Tutor_Research_Document.pdf*). 

The platform automates the end-to-end evaluation of handwritten, scanned, and digital student assignments:
1. **Teachers** can publish tests by entering questions manually or by **directly uploading Question Paper and Model Answer PDFs** to auto-extract problem statements, rubrics, and answer keys. Teachers can view real-time student submissions, inspect actual uploaded answer sheets side-by-side with OCR extractions, and override marks.
2. **Students** can browse active assignments, upload handwritten photo solutions or multi-page PDFs, and receive **instant, transparent AI grading** with concept coverage checklists, missing elements, and constructive tutoring feedback.
3. **Diagnostic OCR Lab** allows real-time calibration of OpenCV filters (deskewing, smoothing, binarization) and extraction telemetry.

**Key Design Principle**: 100% local execution using local open-weight LLMs via **Ollama** and local computer vision pipelines (OpenCV / PyMuPDF / PaddleOCR) with **zero external cloud API dependencies, zero API costs, and full data privacy**.

---

## 2. Technology Stack & Environment

| Component | Technology | Role / Specification |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** (Python 3.11+) | Async REST API, lifespan engine management, static asset delivery. |
| **Frontend Architecture** | **Vanilla HTML5, CSS3, ES6+ JS** | Single-page multi-portal app (Student, Teacher, OCR Lab). Minimalist white theme with modern gradients. |
| **Local LLM Engine** | **Ollama** (`http://127.0.0.1:11434`) | Offline inference runner for vision and evaluation models. |
| **Vision OCR Model** | **`gemma3:latest`** (Gemma 3 Vision) | Multimodal vision transcription of complex cursive handwriting, layouts, and math. |
| **Grading / Evaluator LLM** | **`gemma3:270m` / `phi3:latest`** | Semantic evaluation against model answer and rubric with structured JSON output. |
| **Image Preprocessing** | **OpenCV (`cv2`) & NumPy** | Auto-deskewing (`minAreaRect`), bilateral noise filtering, adaptive Gaussian binarization. |
| **PDF Extraction Engine** | **PyMuPDF (`fitz`), pdfplumber, PyPDF2** | Dual-tier extraction: digital text streams + high-DPI rasterization. |
| **Fallback OCR Engine** | **PaddleOCR (PP-OCRv6)** | Pre-warmed CPU engine for sub-second non-LLM fallback. |
| **Persistence Layer** | **Local JSON Store (`app/data/store.json`)** | File-backed persistent storage for assignments and student submission records. |

---

## 3. System Architecture & Workflows

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND USER INTERFACE                         │
│  [Student Portal]      [Teacher Dashboard]      [OCR Diagnostic Lab]   │
│  - Assignment Browser  - Test Builder (QP Upload)- OpenCV Filter Toggles│
│  - Solution Submissions- Submission Table       - Live Telemetry (ms)  │
│  - AI Report & Badges  - Sheet Inspection Modal - Side-by-Side Preview │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP REST
┌────────────────────────────────────▼───────────────────────────────────┐
│                      FASTAPI APPLICATION BACKEND                       │
│                        (app/main.py — Port 8000)                       │
├────────────────────────────────────┬───────────────────────────────────┤
│          ROUTING & ORCHESTRATION   │        PERSISTENCE LAYER          │
│  - /api/tests                      │  - app/data/store.json            │
│  - /api/tests/{id}/submit          │  - In-memory cache                │
│  - /api/submissions/{id} (PATCH)   │                                   │
│  - /api/extract                    │                                   │
└──────────────────┬─────────────────┴───────────────────┬───────────────┘
                   │                                     │
┌──────────────────▼──────────────────┐┌─────────────────▼───────────────┐
│  MODULE 1 & 2: INGESTION & OCR     ││  MODULE 3: LLM EVALUATION     │
│  (preprocess.py & ocr.py)           ││  (evaluator.py)               │
├─────────────────────────────────────┤├───────────────────────────────┤
│ 1. PDF Digital Stream Extraction    ││ 1. Semantic Rubric Grading    │
│    (pdfplumber / PyPDF2)            ││ 2. Concept Checklist Matching │
│ 2. OpenCV Preprocessing Pipeline    ││ 3. Actionable Tutor Feedback  │
│    (Bilateral filter, deskew)       ││ 4. Fallback Heuristic Grader  │
│ 3. Ollama Vision (gemma3:latest)    ││ Model: gemma3:270m / phi3     │
│ 4. Pre-warmed PaddleOCR Fallback    ││                               │
└─────────────────────────────────────┘└───────────────────────────────┘
```

---

## 4. Detailed Module Specifications

### Module 1: Document Ingestion & OpenCV Preprocessing
* **File**: `app/services/preprocess.py`
* **Features**:
  * **Safe Scaling**: Automatically downscales smartphone 12MP+ camera photos to a max dimension of 1600px, reducing RAM spikes by 90% while maintaining crisp character strokes.
  * **Thumbnail-Based Auto-Deskewing**: Calculates rotation angle on a small 400px thumbnail using `minAreaRect` on Otsu text pixels, then applies the rotation matrix `M` to the full page. Completely eliminates `MemoryError: Unable to allocate MiB` errors.
  * **Bilateral Smoothing**: Removes shadows and page grain while preserving high-contrast ink boundaries.
  * **Adaptive Gaussian Binarization**: Produces high-contrast thresholded output for diagnostic inspection.

### Module 2: Multi-Tier OCR & Handwriting Recognition
* **File**: `app/services/ocr.py`
* **Features**:
  * **Primary Multimodal Engine**: Transcribes handwriting via **Ollama `gemma3:latest`**. Avoids the token repetition bugs and CUDA host memory crashes encountered in older Qwen models on Windows.
  * **Optimized Image Transmission**: Compresses images into high-quality JPEG streams (`quality: 92`, max dimension 1024px) before sending to Ollama.
  * **Pre-Warmed PaddleOCR Fallback**: Background lifespan task pre-loads PaddleOCR in memory during server startup for sub-second failover.

### Module 3: Semantic LLM Evaluation & Pedagogical Grading
* **File**: `app/services/evaluator.py`
* **Features**:
  * Uses local Ollama (`gemma3:270m` or `phi3:latest`) with strict JSON schema prompts.
  * **Output Schema**:
    ```json
    {
      "marks_obtained": 8.5,
      "max_marks": 10.0,
      "status": "Evaluated",
      "reasoning": "The student correctly explained gradient descent and distinguished Batch vs SGD, but omitted the mathematical learning rate formula.",
      "concepts_covered": [
        "Cost function minimization via negative gradient",
        "Distinction between full-batch and single-sample updates"
      ],
      "concepts_missing": [
        "Mathematical update formula: theta = theta - alpha * grad(J)"
      ],
      "constructive_feedback": "Review the gradient descent parameter update equation to ensure full marks on mathematical definitions."
    }
    ```
  * **Heuristic Fallback Engine**: If Ollama is temporarily offline, a TF-IDF / keyword semantic matcher grades submissions safely without throwing 500 errors.

### Teacher Portal & Inspection Experience
* **Files**: `app/templates/index.html`, `app/static/js/app.js`, `app/static/css/style.css`
* **Features**:
  * **Direct Question Paper PDF Upload**: Teachers drag & drop question paper PDFs to auto-extract problem statements into the form.
  * **Direct Model Solution PDF Upload**: Auto-extracts ground truth answer keys.
  * **1-Click Sample Assignment Loader**: Populates complete test templates for instant testing.
  * **Side-by-Side Inspection Modal (`#inspect-modal`)**: Allows teachers to view the student's actual uploaded sheet image alongside the extracted OCR text and AI marks.
  * **Score Override**: Teachers can adjust marks and save custom feedback remarks with instant table updates.

### Responsive UI & Zoom System
* Calibrated at **100% standard browser zoom** with a spacious `1440px` max-width container, compact modern cards, and responsive side-by-side inspection viewports.

---

## 5. API Endpoint Specifications

| Method | Endpoint | Request Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/tests` | None | Lists all active assignments and submission counts. |
| `POST` | `/api/tests` | `JSON: { id, title, subject, max_marks, due_date, question, model_answer, rubric }` | Creates a new assignment. |
| `GET` | `/api/tests/{id}/submissions` | None | Retrieves all student submissions for a test. |
| `POST` | `/api/tests/{id}/submit` | `FormData: file, student_name, student_roll, apply_grayscale, etc.` | Uploads student solution, runs OCR, evaluates via LLM, and stores result. |
| `PATCH` | `/api/submissions/{id}` | `JSON: { marks, remarks }` | Allows teacher to manually override marks and save remarks. |
| `POST` | `/api/extract` | `FormData: file, apply_grayscale, apply_denoise, apply_deskew, apply_threshold` | Diagnostic endpoint returning extracted text, execution time, and preprocessed image URL. |
| `GET` | `/api/samples/{type}` | None (`type = image \| pdf`) | Returns base64 sample files for quick testing. |

---

## 6. Directory Structure

```
AI-Tutor/
├── app/
│   ├── data/
│   │   └── store.json               # Persistent JSON database (Tests & Submissions)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── evaluator.py             # LLM Semantic Grading Engine (Ollama / Heuristic)
│   │   ├── ocr.py                   # Multi-tier Vision OCR (Gemma 3 Vision + PaddleOCR)
│   │   └── preprocess.py            # OpenCV pipeline (Bilateral, auto-deskew, binarize)
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css            # Minimalist White & Gradient Responsive CSS
│   │   └── js/
│   │       └── app.js               # Frontend Controller (Portals, Dropzones, Modals)
│   ├── templates/
│   │   └── index.html               # Single-Page Multi-Portal Interface
│   ├── __init__.py
│   └── main.py                      # FastAPI App, Lifespan Pre-warming, REST Endpoints
├── tests/
│   └── test_extraction.py           # Unit tests
├── AI_Tutor_PRD.md                  # Complete PRD & System Spec Document
├── AI_Tutor_Progress_Report.html    # Standalone printable progress report
├── requirements.txt                 # Python dependencies
└── D25DCS172 DEV SHAH - Assignment-1.pdf # Sample student test document
```

---

## 7. How to Run the Project

```bash
# 1. Activate Python virtual environment
..\myenv\Scripts\activate

# 2. Ensure Ollama is running with required models
ollama pull gemma3:latest
ollama pull gemma3:270m

# 3. Launch FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 4. Open in browser
http://127.0.0.1:8000
```

---

## 8. Remaining Roadmap (Future Modules)

1. **Module 4: Student Analytics & Weak-Area Radar Charts**:
   - Aggregate submission scores over time.
   - Generate student-specific and class-wide concept mastery graphs.
2. **Module 5: Multi-User Authentication & Database Migration**:
   - Role-based login (Teacher vs. Student accounts).
   - Migration from local `store.json` to MongoDB / PostgreSQL for multi-tenant scalability.
