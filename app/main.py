import base64
import time
import cv2
import logging
import os
import io
import json
import uuid
from datetime import datetime
from typing import Optional, List
import pdfplumber
import PyPDF2
import fitz
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel

from app.services.preprocess import preprocess_pipeline
from app.services.ocr import ocr_service
from app.services.evaluator import evaluator_service
from app.services.chandra_ocr import chandra_service
from app.services.qwen_ocr import qwen_ocr_service
from app.services.student_segmenter import student_segmenter_service
from app.services.teacher_ingestion import teacher_ingestion_service
from app.services.knowledge_store import knowledge_store
try:
    from app.services.surya_service import surya_ocr_service
except Exception as e:
    surya_ocr_service = None

import asyncio
import anyio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-Tutor Academic Evaluation Platform (Lazy loading OCR models on demand)...")
    yield
    logger.info("Shutting down AI-Tutor platform...")

app = FastAPI(
    title="AI-Tutor Academic Evaluation Platform",
    description="A complete intelligent assignment evaluation platform featuring OCR, OpenCV, and Ollama LLM grading.",
    lifespan=lifespan
)

# Determine paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
data_dir = os.path.join(BASE_DIR, "data")

os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# ─────────────────────────────────────────────────────────────
# IN-MEMORY & PERSISTENT DATA STORAGE (TESTS & SUBMISSIONS)
# ─────────────────────────────────────────────────────────────
DB_FILE = os.path.join(data_dir, "store.json")

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    
    # Default seed data
    return {
        "tests": [
            {
                "id": "test_1_ML",
                "title": "Machine Learning Test 1: Supervised Learning & Optimization",
                "subject": "Machine Learning & AI",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "due_date": "2026-10-15",
                "max_marks": 10.0,
                "question": "1. Explain the fundamental concept of Gradient Descent in Supervised Learning.\n2. Differentiate between Batch Gradient Descent and Stochastic Gradient Descent (SGD).\n3. Why is the learning rate hyperparameter crucial?",
                "model_answer": "1. Gradient Descent is an iterative optimization algorithm used to minimize the cost/loss function by moving in the direction of steepest descent (negative gradient of the loss with respect to parameters).\n2. Batch Gradient Descent computes gradients over the entire dataset before updating parameters (stable but slow on large datasets). SGD updates weights for each individual training example (fast, introduces noise that helps escape local minima).\n3. The learning rate controls the step size taken during each iteration. A very high learning rate can overshoot the global minimum and diverge; a very small learning rate leads to extremely slow convergence.",
                "rubric": "Cost function minimization: 3 marks; Batch vs SGD distinction: 4 marks; Learning rate explanation & tradeoffs: 3 marks."
            }
        ],
        "submissions": []
    }

def save_db(data):
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save database: {e}")

db = load_db()

# ─────────────────────────────────────────────────────────────
# DIGITAL PDF EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────

def extract_via_pdfplumber_stream(pdf_bytes: bytes, page_index: int) -> Optional[str]:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_index >= len(pdf.pages):
                return None
            page = pdf.pages[page_index]
            text = page.extract_text()
            if text and text.strip():
                return text
            return None
    except Exception as e:
        logger.error(f"[pdfplumber] Error on page {page_index + 1}: {str(e)}")
        return None

def extract_via_pypdf2_stream(pdf_bytes: bytes, page_index: int) -> Optional[str]:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        if page_index >= len(reader.pages):
            return None
        page = reader.pages[page_index]
        text = page.extract_text()
        if text and text.strip():
            return text
        return None
    except Exception as e:
        logger.error(f"[PyPDF2] Error on page {page_index + 1}: {str(e)}")
        return None


def process_document_extraction(
    content: bytes,
    filename: str,
    apply_grayscale: bool = True,
    apply_denoise: bool = True,
    apply_threshold: bool = True,
    apply_deskew: bool = True
) -> dict:
    """
    Central extraction pipeline for PDF and images.
    """
    start_time = time.time()
    is_pdf = filename.lower().endswith('.pdf')
    ocr_results = []
    preprocessed_data_url = None
    first_page_metadata = {}
    
    if is_pdf:
        pdf_doc = fitz.open(stream=content, filetype="pdf")
        num_pages = len(pdf_doc)
        if num_pages == 0:
            raise ValueError("The PDF file contains no pages.")
            
        processed_first_page_img = None
        for page_idx in range(num_pages):
            page_text = extract_via_pdfplumber_stream(content, page_idx)
            if not page_text:
                page_text = extract_via_pypdf2_stream(content, page_idx)
                
            if page_text:
                ocr_results.append({
                    "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "text": page_text.strip(),
                    "confidence": 1.0,
                    "page": page_idx + 1,
                    "source": "digital"
                })
            else:
                page = pdf_doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                img_data = pix.tobytes("png")
                
                page_processed_img, page_metadata = preprocess_pipeline(
                    image_bytes=img_data,
                    apply_grayscale=apply_grayscale,
                    apply_denoise=apply_denoise,
                    apply_threshold=apply_threshold,
                    apply_deskew=apply_deskew
                )
                
                page_ocr = ocr_service.extract_text(page_processed_img)
                for line in page_ocr:
                    line["page"] = page_idx + 1
                    line["source"] = "ocr"
                    ocr_results.append(line)
                    
                if page_idx == 0:
                    processed_first_page_img = page_processed_img
                    first_page_metadata = page_metadata
                    
        if processed_first_page_img is not None:
            _, buffer = cv2.imencode('.jpg', processed_first_page_img)
            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            preprocessed_data_url = f"data:image/jpeg;base64,{base64_encoded}"
            
        metadata = {
            "is_pdf": True,
            "num_pages": num_pages,
            "original_size": first_page_metadata.get("original_size", {"width": 0, "height": 0}),
            "processed_size": first_page_metadata.get("processed_size", {"width": 0, "height": 0}),
            "skew_angle": first_page_metadata.get("skew_angle", 0.0)
        }
    else:
        processed_img, metadata = preprocess_pipeline(
            image_bytes=content,
            apply_grayscale=apply_grayscale,
            apply_denoise=apply_denoise,
            apply_threshold=apply_threshold,
            apply_deskew=apply_deskew
        )
        metadata["is_pdf"] = False
        ocr_results = ocr_service.extract_text(processed_img)
        for line in ocr_results:
            line["source"] = "ocr"
            
        try:
            _, buffer = cv2.imencode('.jpg', processed_img)
            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            preprocessed_data_url = f"data:image/jpeg;base64,{base64_encoded}"
        except Exception:
            preprocessed_data_url = None

    elapsed_time = time.time() - start_time
    
    if is_pdf:
        pages_text = {}
        for line in ocr_results:
            p = line.get("page", 1)
            if p not in pages_text:
                pages_text[p] = []
            pages_text[p].append(line["text"])
        
        full_text_parts = []
        for p in sorted(pages_text.keys()):
            full_text_parts.extend(pages_text[p])
        full_text = "\n".join(full_text_parts)
    else:
        full_text = "\n".join([line["text"] for line in ocr_results])
        
    metadata["processing_time_seconds"] = round(elapsed_time, 3)
    metadata["confidence_average"] = round(
        sum([line["confidence"] for line in ocr_results]) / len(ocr_results), 3
    ) if ocr_results else 0.0

    return {
        "text": full_text,
        "lines": ocr_results,
        "metadata": metadata,
        "preprocessed_image": preprocessed_data_url
    }


# ─────────────────────────────────────────────────────────────
# APPLICATION ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

# --- Test Management Endpoints ---

@app.get("/api/tests")
async def get_all_tests(active_only: bool = False):
    """List all teacher posted tests with submission counts and average scores."""
    tests_summary = []
    for t in db.get("tests", []):
        is_published = t.get("is_published", True)
        status = t.get("status", "active")

        if active_only and (not is_published or status != "active"):
            continue

        subs = [s for s in db.get("submissions", []) if s.get("test_id") == t.get("id")]
        sub_count = len(subs)
        avg_score = 0
        evaluated_subs = [
            s for s in subs
            if s.get("evaluation") and isinstance(s.get("evaluation"), dict) and "marks_obtained" in s["evaluation"]
        ]
        if evaluated_subs:
            total_marks = sum(s["evaluation"].get("marks_obtained", 0) for s in evaluated_subs)
            avg_score = round(total_marks / len(evaluated_subs), 1)

        tests_summary.append({
            **t,
            "status": status,
            "is_published": is_published,
            "submissions_count": sub_count,
            "average_score": avg_score
        })
    return {"success": True, "tests": tests_summary, "total": len(tests_summary)}

@app.get("/api/tests/active")
async def get_active_tests():
    """List active, published tests for the student portal."""
    return await get_all_tests(active_only=True)

class CreateTestSchema(BaseModel):
    id: str
    title: str
    subject: str
    max_marks: float
    due_date: str
    question: str
    model_answer: str
    rubric: Optional[str] = ""
    status: Optional[str] = "active"
    is_published: Optional[bool] = True

@app.post("/api/tests")
async def create_new_test(payload: CreateTestSchema):
    """Teacher creates a new test."""
    clean_id = payload.id.strip().replace(" ", "_")
    if not clean_id:
        clean_id = f"test_{uuid.uuid4().hex[:6]}"
        
    # Check if ID already exists
    if any(t["id"] == clean_id for t in db.get("tests", [])):
        raise HTTPException(status_code=400, detail=f"Test ID '{clean_id}' already exists. Please choose another ID.")
        
    new_test = {
        "id": clean_id,
        "title": payload.title,
        "subject": payload.subject,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "due_date": payload.due_date,
        "max_marks": payload.max_marks,
        "question": payload.question,
        "model_answer": payload.model_answer,
        "rubric": payload.rubric,
        "status": payload.status or "active",
        "is_published": True if payload.is_published is not False else False,
        "questions": [],
        "source_documents": [],
        "knowledge_status": "processing",
        "knowledge_error": None,
        "knowledge_stats": None
    }
    db["tests"].insert(0, new_test)
    save_db(db)

    # Immediately trigger ingestion for the newly created test
    try:
        ingest_res = teacher_ingestion_service.ingest_test_materials(
            test_id=clean_id,
            db_ref=db,
            save_db_fn=save_db,
            manual_qp_text=payload.question,
            manual_model_text=payload.model_answer,
            manual_rubric=payload.rubric
        )
        new_test["knowledge_status"] = "ready"
        new_test["knowledge_stats"] = ingest_res
    except Exception as e:
        logger.warning(f"[INGEST] Automatic ingestion notice for test '{clean_id}': {e}")
        new_test["knowledge_status"] = "failed"
        new_test["knowledge_error"] = str(e)
        save_db(db)

    return {"success": True, "test": new_test}

@app.post("/api/tests/{test_id}/ingest")
async def ingest_test_materials_endpoint(
    test_id: str,
    question_paper: Optional[UploadFile] = File(None),
    model_answer: Optional[UploadFile] = File(None),
    question_text: Optional[str] = Form(None),
    model_answer_text: Optional[str] = Form(None),
    rubric_text: Optional[str] = Form(None)
):
    """
    Ingest question paper and model answer/rubric for an existing test into
    persistent ChromaDB and structured DB.
    """
    test = next((t for t in db["tests"] if t["id"] == test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found.")

    qp_content = await question_paper.read() if question_paper else None
    qp_filename = question_paper.filename if question_paper else None
    model_content = await model_answer.read() if model_answer else None
    model_filename = model_answer.filename if model_answer else None

    q_text = question_text or test.get("question")
    m_text = model_answer_text or test.get("model_answer")
    r_text = rubric_text or test.get("rubric")

    try:
        result = teacher_ingestion_service.ingest_test_materials(
            test_id=test_id,
            db_ref=db,
            save_db_fn=save_db,
            qp_file_content=qp_content,
            qp_filename=qp_filename,
            model_file_content=model_content,
            model_filename=model_filename,
            manual_qp_text=q_text,
            manual_model_text=m_text,
            manual_rubric=r_text
        )
        return result
    except Exception as e:
        logger.error(f"[INGEST] Ingestion endpoint failed for test '{test_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/api/tests/{test_id}/knowledge-status")
async def get_test_knowledge_status(test_id: str):
    """
    Check the current teacher knowledge ingestion status of a test.
    """
    test = next((t for t in db["tests"] if t["id"] == test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found.")
        
    return {
        "success": True,
        "test_id": test_id,
        "knowledge_status": test.get("knowledge_status", "uploaded"),
        "knowledge_error": test.get("knowledge_error"),
        "knowledge_updated_at": test.get("knowledge_updated_at"),
        "knowledge_stats": test.get("knowledge_stats"),
        "questions_count": len(test.get("questions", [])),
        "source_documents": test.get("source_documents", [])
    }

@app.get("/api/debug/chroma/inspect")
async def inspect_chroma_knowledge_base():
    """
    Development debug inspection of the persistent ChromaDB collection.
    """
    try:
        return {
            "success": True,
            "inspection": knowledge_store.inspect_collection()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tests/{test_id}")
async def get_test_details(test_id: str):
    """Get single test information."""
    test = next((t for t in db.get("tests", []) if t["id"] == test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found.")
    normalized_test = {
        **test,
        "status": test.get("status", "active"),
        "is_published": test.get("is_published", True)
    }
    return {"success": True, "test": normalized_test}

@app.delete("/api/tests/{test_id}")
async def delete_test(test_id: str):
    """Delete a test, its submissions, files, and knowledge base."""
    global db
    import shutil
    
    test_exists = False
    new_tests = []
    for test in db.get("tests", []):
        if test["id"] == test_id:
            test_exists = True
        else:
            new_tests.append(test)
            
    if not test_exists:
        raise HTTPException(status_code=404, detail="Test not found.")
        
    db["tests"] = new_tests
    db["submissions"] = [sub for sub in db.get("submissions", []) if sub.get("test_id") != test_id]
    save_db(db)
    
    test_docs_dir = os.path.join(data_dir, "documents", test_id)
    if os.path.exists(test_docs_dir):
        try:
            shutil.rmtree(test_docs_dir)
        except Exception as e:
            logger.error(f"Failed to delete test documents dir {test_docs_dir}: {e}")
            
    try:
        from app.services.knowledge_store import knowledge_store
        knowledge_store.delete_test_chunks(test_id)
    except Exception as e:
        logger.error(f"Failed to delete test chunks from knowledge store: {e}")
        
    return {"success": True, "message": "Test deleted successfully"}

# --- Student Submission & Evaluation Endpoints ---

@app.post("/api/tests/{test_id}/submit")
async def submit_student_solution(
    test_id: str,
    file: UploadFile = File(...),
    student_name: str = Form("Anonymous Student"),
    student_roll: str = Form("STU-001"),
    ocr_engine: str = Form("standard"),
    apply_grayscale: bool = Form(True),
    apply_denoise: bool = Form(True),
    apply_threshold: bool = Form(True),
    apply_deskew: bool = Form(True)
):
    """
    Student submits solution sheet (image or PDF).
    Backend extracts text using the chosen canonical OCR engine, evaluates with LLM, and stores result.
    """
    test = next((t for t in db["tests"] if t["id"] == test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found.")

    canonical_engine = (ocr_engine or "standard").strip().lower()
    if canonical_engine not in ("standard", "chandra", "qwen2.5vl", "surya"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid OCR engine '{canonical_engine}'. Supported engines: standard, chandra, qwen2.5vl, surya."
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read file: {e}")

    # 1. Strict OCR Routing (Zero Silent Fallback)
    logger.info(f"[OCR] Student submission for test '{test_id}'. Requested engine: {canonical_engine}")
    extracted_text = ""
    extraction_metadata = {}
    preprocessed_data_url = None

    if canonical_engine == "chandra":
        logger.info("[OCR] Routing student submission to Chandra OCR")
        logger.info("[OCR] Standard/Gemma OCR: NOT USED")
        logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
        res = await anyio.to_thread.run_sync(chandra_service.run_chandra_ocr, content, file.filename)
        if not res.get("success", False):
            error_detail = res.get("error", "Chandra OCR extraction failed.")
            logger.error(f"[OCR] Chandra OCR failed on student submission: {error_detail}")
            raise HTTPException(status_code=400, detail=f"Chandra OCR extraction failed: {error_detail}")
        extracted_text = res["text"]
        extraction_metadata = res.get("metadata", {})
        preprocessed_data_url = res.get("preprocessed_image")

    elif canonical_engine == "surya":
        logger.info("[OCR] Routing student submission to Surya OCR")
        logger.info("[OCR] Standard/Gemma OCR: NOT USED")
        logger.info("[OCR] Chandra OCR: NOT USED")
        logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
        if not surya_ocr_service:
            raise HTTPException(status_code=500, detail="Surya OCR service is not available.")
        res = await asyncio.to_thread(surya_ocr_service.extract_from_bytes, content, file.filename)
        if not res.get("success", False):
            error_detail = res.get("error", "Surya OCR extraction failed.")
            logger.error(f"[OCR] Surya OCR failed on student submission: {error_detail}")
            raise HTTPException(status_code=400, detail=f"Surya OCR extraction failed: {error_detail}")
        extracted_text = res.get("text", "")
        extraction_metadata = res.get("metadata", {})
        preprocessed_data_url = res.get("preprocessed_image")

    elif canonical_engine == "qwen2.5vl":
        logger.info("[OCR] Routing student submission to Qwen2.5-VL OCR (qwen2.5vl:3b)")
        logger.info("[OCR] Standard/Gemma OCR: NOT USED")
        logger.info("[OCR] Chandra OCR: NOT USED")
        res = await anyio.to_thread.run_sync(qwen_ocr_service.run_qwen_ocr, content, file.filename)
        if not res.get("success", False):
            error_detail = res.get("error", "Qwen2.5-VL OCR extraction failed.")
            logger.error(f"[OCR] Qwen2.5-VL OCR failed on student submission: {error_detail}")
            raise HTTPException(status_code=400, detail=f"Qwen2.5-VL OCR extraction failed: {error_detail}")
        extracted_text = res["text"]
        extraction_metadata = res.get("metadata", {})
        preprocessed_data_url = res.get("preprocessed_image")

    else:
        logger.info("[OCR] Routing student submission to Standard OCR")
        logger.info("[OCR] Standard OCR using Gemma/Ollama pipeline")
        logger.info("[OCR] Chandra OCR: NOT USED")
        logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
        try:
            extraction = process_document_extraction(
                content=content,
                filename=file.filename,
                apply_grayscale=apply_grayscale,
                apply_denoise=apply_denoise,
                apply_threshold=apply_threshold,
                apply_deskew=apply_deskew
            )
            extracted_text = extraction["text"]
            extraction_metadata = extraction.get("metadata", {})
            preprocessed_data_url = extraction.get("preprocessed_image")
        except Exception as e:
            logger.error(f"[OCR] Standard extraction failed: {e}")
            raise HTTPException(status_code=400, detail=f"Standard OCR extraction failed: {str(e)}")

    extraction_metadata["ocr_engine"] = canonical_engine
    student_text = extracted_text

    # 2. Semantic Evaluation via Evaluator LLM (Moved to explicit evaluation endpoint for Phase 4)
    eval_result = None

    submission_id = f"sub_{uuid.uuid4().hex[:8]}"
    
    # 3. Segment and Map Student Answers (Phase 3)
    teacher_questions = test.get("questions", [])
    try:
        segmented_answers = student_segmenter_service.segment_answers(student_text, teacher_questions)
        for seg in segmented_answers:
            seg["submission_id"] = submission_id
            seg["test_id"] = test_id
            seg["ocr_engine"] = canonical_engine
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        segmented_answers = []

    # 4. Store submission record
    original_data_url = None
    if not file.filename.lower().endswith('.pdf'):
        try:
            mime = "image/png" if file.filename.lower().endswith(".png") else "image/jpeg"
            b64_orig = base64.b64encode(content).decode('utf-8')
            original_data_url = f"data:{mime};base64,{b64_orig}"
        except Exception:
            original_data_url = None

    record = {
        "id": submission_id,
        "test_id": test_id,
        "test_title": test["title"],
        "student_name": student_name,
        "student_roll": student_roll,
        "filename": file.filename,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ocr_engine": canonical_engine,
        "extracted_text": student_text,
        "evaluation": eval_result,
        "teacher_override": None,
        "teacher_remarks": None,
        "metadata": extraction_metadata,
        "preprocessed_image": preprocessed_data_url,
        "original_image": original_data_url,
        "segmented_answers": segmented_answers
    }

    db["submissions"].insert(0, record)
    save_db(db)

    return {
        "success": True,
        "submission_id": submission_id,
        "ocr_engine": canonical_engine,
        "extracted_text": student_text,
        "evaluation": eval_result,
        "metadata": extraction_metadata,
        "preprocessed_image": preprocessed_data_url,
        "segmented_answers": segmented_answers
    }

@app.get("/api/tests/{test_id}/submissions")
async def get_test_submissions(test_id: str):
    """Teacher views all submissions for a specific test."""
    subs = [s for s in db["submissions"] if s.get("test_id") == test_id]
    return {"success": True, "submissions": subs}

class OverrideSchema(BaseModel):
    marks: float
    remarks: Optional[str] = ""

@app.patch("/api/submissions/{submission_id}")
async def teacher_override_submission(submission_id: str, payload: OverrideSchema):
    """Teacher manually updates marks or adds remarks for a student submission."""
    sub = next((s for s in db["submissions"] if s["id"] == submission_id), None)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found.")
        
    sub["teacher_override"] = payload.marks
    sub["teacher_remarks"] = payload.remarks
    save_db(db)
    return {"success": True, "submission": sub}

@app.post("/api/submissions/{submission_id}/evaluate")
async def evaluate_student_submission(submission_id: str):
    """
    Phase 4: Automatically evaluates a student submission on a per-question basis.
    """
    sub = next((s for s in db["submissions"] if s["id"] == submission_id), None)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found.")
        
    test = next((t for t in db["tests"] if t["id"] == sub["test_id"]), None)
    if not test:
        raise HTTPException(status_code=404, detail="Test for submission not found.")

    teacher_questions = {q["id"]: q for q in test.get("questions", [])}
    
    total_awarded = 0.0
    total_max = 0.0
    review_count = 0
    evaluated_count = 0

    segmented = sub.get("segmented_answers", [])
    if not segmented:
        return {"success": False, "detail": "No segmented answers to evaluate."}
        
    for seg in segmented:
        status = seg.get("mapping_status")
        if status in ("review_required", "unmapped"):
            seg["evaluation_result"] = {
                "needs_teacher_review": True,
                "reasoning": "Mapping uncertain; teacher review required.",
                "marks_awarded": 0.0,
                "max_marks": 0.0
            }
            review_count += 1
            continue
            
        tq_id = seg.get("teacher_question_id")
        tq = teacher_questions.get(tq_id)
        if not tq:
            seg["evaluation_result"] = {
                "needs_teacher_review": True,
                "reasoning": "Teacher question missing from structured store.",
                "marks_awarded": 0.0,
                "max_marks": 0.0
            }
            review_count += 1
            continue
            
        raw_max = tq.get("max_marks")
        try:
            max_marks = float(raw_max) if raw_max is not None else 0.0
        except (ValueError, TypeError):
            max_marks = 0.0

        if max_marks <= 0.0 or tq.get("is_parent"):
            seg["evaluation_result"] = {
                "needs_teacher_review": True,
                "reasoning": "Parent or unweighted question; teacher review required.",
                "marks_awarded": 0.0,
                "max_marks": 0.0
            }
            review_count += 1
            continue
        student_ans = seg.get("extracted_answer", "").strip()
        
        if not student_ans:
            seg["evaluation_result"] = {
                "marks_awarded": 0.0,
                "max_marks": max_marks,
                "reasoning": "Student left the answer blank.",
                "missing_concepts": ["All concepts missing."],
                "feedback": "Please attempt this question next time.",
                "needs_teacher_review": False,
                "evaluation_status": "completed"
            }
            total_max += max_marks
            evaluated_count += 1
            continue
            
        # Optional Context Retrieval
        retrieved_context = ""
        try:
            from app.services.embedding import embedding_service
            q_emb = embedding_service.get_embedding(student_ans)
            if q_emb:
                matches = knowledge_store.query_test_knowledge(test["id"], q_emb, top_k=2)
                if matches:
                    retrieved_context = "\\n".join([m["document"] for m in matches if m["distance"] < 0.8])
        except Exception as e:
            logger.warning(f"Retrieval failed during evaluation: {e}")

        # Call Evaluator
        try:
            eval_res = evaluator_service.evaluate_segmented_answer(
                student_answer=student_ans,
                question=tq.get("question_text", ""),
                model_answer=tq.get("model_answer", ""),
                max_marks=max_marks,
                rubric=tq.get("rubric", ""),
                retrieved_context=retrieved_context
            )
            seg["evaluation_result"] = eval_res
            
            if eval_res.get("needs_teacher_review"):
                review_count += 1
            else:
                total_awarded += eval_res.get("marks_awarded", 0.0)
                total_max += max_marks
                evaluated_count += 1
                
        except Exception as e:
            logger.error(f"Failed to evaluate segment: {e}")
            seg["evaluation_result"] = {
                "needs_teacher_review": True,
                "reasoning": f"Evaluator error: {str(e)}",
                "marks_awarded": 0.0,
                "max_marks": max_marks
            }
            review_count += 1
            
    # Update submission object
    sub["evaluation"] = {
        "marks_obtained": total_awarded,
        "max_marks": total_max,
        "evaluated_count": evaluated_count,
        "review_count": review_count,
        "status": "evaluated"
    }
    save_db(db)
    
    return {
        "success": True,
        "submission_id": submission_id,
        "evaluation": sub["evaluation"],
        "segmented_answers": segmented
    }

# --- Diagnostic & Quick Test Endpoints ---

@app.get("/api/samples/{sample_name}")
async def get_sample_file(sample_name: str):
    """Fetches sample files for quick developer testing."""
    root_dir = os.path.dirname(BASE_DIR)
    file_map = {
        "image": os.path.join(root_dir, "handwritten_sample.png"),
        "pdf": os.path.join(root_dir, "AI_Tutor_Research_Document.pdf")
    }
    target_path = file_map.get(sample_name)
    if not target_path or not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Sample not found.")
        
    with open(target_path, "rb") as f:
        data = f.read()
        
    b64 = base64.b64encode(data).decode('utf-8')
    mime = "application/pdf" if target_path.endswith(".pdf") else "image/png"
    return {
        "filename": os.path.basename(target_path),
        "data_url": f"data:{mime};base64,{b64}",
        "mime_type": mime
    }

@app.get("/api/ocr/status")
async def get_ocr_engine_status():
    """Returns availability and status of OCR engines."""
    chandra_ready, chandra_msg = chandra_service.is_available()
    qwen_ready, qwen_msg = qwen_ocr_service.is_available()
    surya_ready = surya_ocr_service is not None
    surya_msg = "Ready (PyTorch with GPU/CPU auto memory routing)" if surya_ready else "Not available"
    return {
        "success": True,
        "standard_ocr": "Online (Gemma 3 Vision / PaddleOCR)",
        "chandra_ocr": {
            "available": chandra_ready,
            "status": chandra_msg
        },
        "qwen2.5vl_ocr": {
            "available": qwen_ready,
            "status": qwen_msg,
            "model": qwen_ocr_service.MODEL_NAME
        },
        "surya_ocr": {
            "available": surya_ready,
            "status": surya_msg
        }
    }

@app.post("/api/extract")
async def extract_handwritten_document(
    file: UploadFile = File(...),
    ocr_engine: str = Form("standard"),
    apply_grayscale: bool = Form(True),
    apply_denoise: bool = Form(True),
    apply_threshold: bool = Form(True),
    apply_deskew: bool = Form(True)
):
    """Standalone OCR diagnostic endpoint supporting Standard, Chandra, and Qwen2.5-VL OCR."""
    selected_engine = (ocr_engine or "standard").strip().lower()
    if selected_engine not in ("standard", "chandra", "qwen2.5vl", "surya"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "filename": file.filename,
                "ocr_engine": selected_engine,
                "error": f"Invalid OCR engine '{selected_engine}'. Supported engines: standard, chandra, qwen2.5vl, surya.",
                "detail": f"Engine '{selected_engine}' is not supported.",
                "text": "",
                "lines": [],
                "metadata": {}
            }
        )

    logger.info(f"[OCR] Requested engine: {selected_engine}")
    try:
        content = await file.read()

        if selected_engine == "chandra":
            logger.info("[OCR] Routing to Chandra OCR")
            logger.info("[OCR] Standard/Gemma OCR: NOT USED")
            logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
            res = await anyio.to_thread.run_sync(chandra_service.run_chandra_ocr, content, file.filename)
            if not res.get("success", False):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "filename": file.filename,
                        "ocr_engine": "chandra",
                        "error": res.get("error", "Chandra OCR processing failed."),
                        "detail": res.get("detail", res.get("error")),
                        "text": "",
                        "lines": [],
                        "metadata": res.get("metadata", {}),
                        "preprocessed_image": res.get("preprocessed_image")
                    }
                )
            logger.info(f"[OCR] Completed using chandra ({len(res.get('text', ''))} characters)")
            return {
                "success": True,
                "filename": file.filename,
                "text": res["text"],
                "lines": res.get("lines", []),
                "metadata": res.get("metadata", {}),
                "preprocessed_image": res.get("preprocessed_image"),
                "ocr_engine": "chandra"
            }

        elif selected_engine == "surya":
            logger.info("[OCR] Routing to Surya OCR")
            logger.info("[OCR] Standard/Gemma OCR: NOT USED")
            logger.info("[OCR] Chandra OCR: NOT USED")
            logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
            if not surya_ocr_service:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "filename": file.filename,
                        "ocr_engine": "surya",
                        "error": "Surya OCR service is not available.",
                        "text": "",
                        "lines": [],
                        "metadata": {}
                    }
                )
            res = await asyncio.to_thread(surya_ocr_service.extract_from_bytes, content, file.filename)
            if not res.get("success", False):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "filename": file.filename,
                        "ocr_engine": "surya",
                        "error": res.get("error", "Surya OCR processing failed."),
                        "detail": res.get("detail", res.get("error")),
                        "text": "",
                        "lines": [],
                        "metadata": res.get("metadata", {}),
                        "preprocessed_image": res.get("preprocessed_image")
                    }
                )
            logger.info(f"[OCR] Completed using surya ({len(res.get('text', ''))} characters)")
            return {
                "success": True,
                "filename": file.filename,
                "text": res["text"],
                "lines": res.get("lines", []),
                "metadata": res.get("metadata", {}),
                "preprocessed_image": res.get("preprocessed_image"),
                "ocr_engine": "surya"
            }

        elif selected_engine == "qwen2.5vl":
            logger.info("[OCR] Routing to Qwen2.5-VL OCR")
            logger.info("[OCR] Standard/Gemma OCR: NOT USED")
            logger.info("[OCR] Chandra OCR: NOT USED")
            res = await anyio.to_thread.run_sync(qwen_ocr_service.run_qwen_ocr, content, file.filename)
            if not res.get("success", False):
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "filename": file.filename,
                        "ocr_engine": "qwen2.5vl",
                        "error": res.get("error", "Qwen2.5-VL OCR processing failed."),
                        "detail": res.get("detail", res.get("error")),
                        "text": "",
                        "lines": [],
                        "metadata": res.get("metadata", {}),
                        "preprocessed_image": res.get("preprocessed_image")
                    }
                )
            logger.info(f"[OCR] Completed using qwen2.5vl ({len(res.get('text', ''))} characters)")
            return {
                "success": True,
                "filename": file.filename,
                "text": res["text"],
                "lines": res.get("lines", []),
                "metadata": res.get("metadata", {}),
                "preprocessed_image": res.get("preprocessed_image"),
                "ocr_engine": "qwen2.5vl"
            }

        else: # standard
            logger.info("[OCR] Routing to Standard OCR")
            logger.info("[OCR] Standard OCR using Gemma/Ollama and OpenCV pipeline")
            logger.info("[OCR] Chandra OCR: NOT USED")
            logger.info("[OCR] Qwen2.5-VL OCR: NOT USED")
            res = process_document_extraction(
                content=content,
                filename=file.filename,
                apply_grayscale=apply_grayscale,
                apply_denoise=apply_denoise,
                apply_threshold=apply_threshold,
                apply_deskew=apply_deskew
            )
            logger.info(f"[OCR] Completed using standard ({len(res.get('text', ''))} characters)")
            return {
                "success": True,
                "filename": file.filename,
                "text": res["text"],
                "lines": res["lines"],
                "metadata": {**res["metadata"], "ocr_engine": "standard"},
                "preprocessed_image": res["preprocessed_image"],
                "ocr_engine": "standard"
            }
    except Exception as e:
        logger.error(f"Error in extract endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "ocr_engine": selected_engine,
                "detail": f"Extraction failed: {str(e)}",
                "error": str(e),
                "text": "",
                "lines": [],
                "metadata": {}
            }
        )

@app.post("/api/surya/extract")
async def extract_via_surya(
    file: UploadFile = File(...),
    device_mode: str = Form("auto")
):
    """Dedicated endpoint to run Surya OCR with PyTorch + CUDA/CPU on uploaded PDF or Image."""
    try:
        content = await file.read()
        if not surya_ocr_service:
            raise HTTPException(status_code=500, detail="Surya OCR service is not available.")
        res = await asyncio.to_thread(surya_ocr_service.extract_from_bytes, content, file.filename, device_mode)
        return res
    except Exception as e:
        logger.error(f"Error in Surya extract endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Surya OCR extraction failed: {str(e)}",
                "engine": "Surya OCR",
                "text": ""
            }
        )
