import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app, db
from app.services.qwen_ocr import QwenVLOCRService, qwen_ocr_service, SYSTEM_OCR_PROMPT
from app.services.student_segmenter import StudentAnswerSegmenter

client = TestClient(app)

def create_dummy_png_bytes(width=100, height=100, color="white") -> bytes:
    """Helper to generate dummy PNG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_multipage_pdf_bytes(page_texts: list[str]) -> bytes:
    """Helper to generate a multi-page PDF with PyMuPDF."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=300, height=300)
        page.insert_text((30, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def get_or_create_test_fixture():
    """Ensure a valid test fixture exists in db['tests'] for submission testing."""
    existing = next((t for t in db.get("tests", []) if t["id"] == "CN_001"), None)
    if existing:
        return "CN_001"
    test_id = "test_qwen_unit"
    existing = next((t for t in db.get("tests", []) if t["id"] == test_id), None)
    if not existing:
        fixture = {
            "id": test_id,
            "title": "Qwen Unit Test Paper",
            "subject": "CS",
            "max_marks": 20.0,
            "due_date": "2026-12-31",
            "status": "active",
            "is_published": True,
            "questions": [
                {
                    "id": "q1",
                    "question_number": "1",
                    "sub_question_number": "",
                    "question_text": "Define latency and throughput.",
                    "max_marks": 10.0,
                    "model_answer": "Latency is delay, throughput is data rate."
                },
                {
                    "id": "q2",
                    "question_number": "2",
                    "sub_question_number": "",
                    "question_text": "Explain TCP handshake.",
                    "max_marks": 10.0,
                    "model_answer": "SYN, SYN-ACK, ACK."
                }
            ]
        }
        db.setdefault("tests", []).append(fixture)
    return test_id


# ==============================================================================
# TEST 1: Valid qwen2.5vl engine selection
# ==============================================================================
def test_valid_qwen_engine_selection_on_extract():
    """Verify that qwen2.5vl engine is accepted and invoked on /api/extract."""
    img_bytes = create_dummy_png_bytes()
    mock_chat_resp = {"message": {"content": "1. Latency is delay."}}

    with patch.object(qwen_ocr_service, "is_available", return_value=(True, "Ready")), \
         patch.object(qwen_ocr_service._client, "chat", return_value=mock_chat_resp) as mock_chat:

        files = {"file": ("student_sheet.png", img_bytes, "image/png")}
        data = {"ocr_engine": "qwen2.5vl"}

        resp = client.post("/api/extract", files=files, data=data)
        assert resp.status_code == 200, resp.text
        res_json = resp.json()
        assert res_json["success"] is True
        assert res_json["ocr_engine"] == "qwen2.5vl"
        assert "1. Latency is delay." in res_json["text"]
        assert mock_chat.called


# ==============================================================================
# TEST 2: Invalid engine selection
# ==============================================================================
def test_invalid_engine_selection():
    """Verify that invalid engine choices are strictly rejected on both endpoints."""
    img_bytes = create_dummy_png_bytes()
    test_id = get_or_create_test_fixture()

    # Test /api/extract with invalid engine
    files = {"file": ("student_sheet.png", img_bytes, "image/png")}
    data = {"ocr_engine": "unsupported_vlm"}
    resp = client.post("/api/extract", files=files, data=data)
    assert resp.status_code == 400
    res_json = resp.json()
    assert res_json["success"] is False
    assert "Invalid OCR engine 'unsupported_vlm'" in res_json["error"]

    # Test /api/tests/{test_id}/submit with invalid engine
    files_submit = {"file": ("submission.png", img_bytes, "image/png")}
    data_submit = {"student_name": "Test Student", "student_id": "S101", "ocr_engine": "fake_ocr"}
    resp_submit = client.post(f"/api/tests/{test_id}/submit", files=files_submit, data=data_submit)
    assert resp_submit.status_code == 400
    assert "Invalid OCR engine 'fake_ocr'" in resp_submit.json()["detail"]


# ==============================================================================
# TEST 3: Qwen calls the correct model name
# ==============================================================================
def test_qwen_calls_correct_model_name():
    """Verify that Ollama chat is invoked with exact model 'qwen2.5vl:3b'."""
    service = QwenVLOCRService()
    img_bytes = create_dummy_png_bytes()

    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"model": "qwen2.5vl:3b"}]}
    mock_client.chat.return_value = {"message": {"content": "Sample OCR line"}}
    service._client = mock_client

    result = service.run_qwen_ocr(img_bytes, "sample.png")
    assert result["success"] is True

    # Assert model name passed to Ollama chat
    assert mock_client.chat.call_count == 1
    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "qwen2.5vl:3b"
    assert result["metadata"]["model_name"] == "qwen2.5vl:3b"
    assert result["metadata"]["ocr_engine"] == "qwen2.5vl"


# ==============================================================================
# TEST 4: Correct image payload construction
# ==============================================================================
def test_correct_image_payload_construction():
    """Verify system OCR prompt, temperature, and non-empty image payload bytes."""
    service = QwenVLOCRService()
    img_bytes = create_dummy_png_bytes(width=200, height=200)

    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"model": "qwen2.5vl:3b"}]}
    mock_client.chat.return_value = {"message": {"content": "Q1: 192.168.1.1 0x7F"}}
    service._client = mock_client

    result = service.run_qwen_ocr(img_bytes, "page.png")
    assert result["success"] is True

    call_kwargs = mock_client.chat.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == SYSTEM_OCR_PROMPT
    assert "Do not evaluate, grade, solve, or assign marks" in messages[0]["content"]
    assert len(messages[0]["images"]) == 1
    image_bytes_sent = messages[0]["images"][0]
    assert isinstance(image_bytes_sent, bytes)
    assert len(image_bytes_sent) > 0

    options = call_kwargs["options"]
    assert options["temperature"] == 0.1


# ==============================================================================
# TEST 5: Multi-page PDF processing in correct order
# ==============================================================================
def test_multipage_pdf_processing_order():
    """Verify that multi-page PDFs are processed sequentially and labeled."""
    service = QwenVLOCRService()
    pdf_bytes = create_multipage_pdf_bytes(["Page 1 Question 1", "Page 2 Question 2", "Page 3 Question 3"])

    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"model": "qwen2.5vl:3b"}]}

    # Return distinct transcription for each page call
    mock_client.chat.side_effect = [
        {"message": {"content": "1. Answer to question 1 from page 1"}},
        {"message": {"content": "2. Answer to question 2 from page 2"}},
        {"message": {"content": "3. Answer to question 3 from page 3"}},
    ]
    service._client = mock_client

    result = service.run_qwen_ocr(pdf_bytes, "answers.pdf")

    assert result["success"] is True
    assert mock_client.chat.call_count == 3
    assert result["metadata"]["is_pdf"] is True
    assert result["metadata"]["num_pages"] == 3

    # Check order and boundary markers
    full_text = result["text"]
    assert "--- Page 1 ---\n1. Answer to question 1 from page 1" in full_text
    assert "--- Page 2 ---\n2. Answer to question 2 from page 2" in full_text
    assert "--- Page 3 ---\n3. Answer to question 3 from page 3" in full_text
    p1_pos = full_text.find("--- Page 1 ---")
    p2_pos = full_text.find("--- Page 2 ---")
    p3_pos = full_text.find("--- Page 3 ---")
    assert p1_pos < p2_pos < p3_pos


# ==============================================================================
# TEST 6: Ollama model unavailable
# ==============================================================================
def test_ollama_model_unavailable():
    """Verify actionable failure when Ollama server or model is unavailable."""
    service = QwenVLOCRService()
    mock_client = MagicMock()
    # List does not include qwen2.5vl:3b
    mock_client.list.return_value = {"models": [{"model": "other_model:latest"}]}
    service._client = mock_client

    avail, msg = service.is_available()
    assert avail is False
    assert "not found" in msg

    img_bytes = create_dummy_png_bytes()
    result = service.run_qwen_ocr(img_bytes, "test.png")
    assert result["success"] is False
    assert result["ocr_engine"] == "qwen2.5vl"
    assert "unavailable" in result["error"].lower()
    # Ensure chat inference was never triggered
    assert mock_client.chat.call_count == 0


# ==============================================================================
# TEST 7: OCR response missing or malformed
# ==============================================================================
def test_ocr_response_missing_or_malformed():
    """Verify safe error handling when Ollama returns an exception or malformed payload."""
    service = QwenVLOCRService()
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"model": "qwen2.5vl:3b"}]}
    # Ollama raises an internal exception during inference
    mock_client.chat.side_effect = RuntimeError("Ollama context length exceeded")
    service._client = mock_client

    img_bytes = create_dummy_png_bytes()
    result = service.run_qwen_ocr(img_bytes, "test.png")
    assert result["success"] is False
    assert result["ocr_engine"] == "qwen2.5vl"
    assert "Ollama context length exceeded" in result["error"]
    assert result["text"] == ""


# ==============================================================================
# TEST 8: No silent fallback to Standard/Gemma or Chandra
# ==============================================================================
def test_no_silent_fallback_when_qwen_fails():
    """Verify that when qwen2.5vl fails on student submission, it does NOT fall back."""
    test_id = get_or_create_test_fixture()
    img_bytes = create_dummy_png_bytes()

    with patch.object(qwen_ocr_service, "run_qwen_ocr", return_value={"success": False, "error": "Model crashed"}), \
         patch("app.main.chandra_service.run_chandra_ocr") as mock_chandra, \
         patch("app.main.process_document_extraction") as mock_standard:

        files = {"file": ("submission.png", img_bytes, "image/png")}
        data = {"student_name": "Test Student", "student_id": "S101", "ocr_engine": "qwen2.5vl"}
        resp = client.post(f"/api/tests/{test_id}/submit", files=files, data=data)

        # Must fail with 400 and clear error detail
        assert resp.status_code == 400
        assert "Qwen2.5-VL OCR extraction failed" in resp.json()["detail"]

        # Neither Standard nor Chandra OCR must be touched!
        assert not mock_chandra.called, "Silent fallback to Chandra occurred!"
        assert not mock_standard.called, "Silent fallback to Standard OCR occurred!"


# ==============================================================================
# TEST 9: Qwen output passed to existing StudentAnswerSegmenter
# ==============================================================================
def test_qwen_output_passed_to_segmenter():
    """Verify that Qwen's transcribed text passes cleanly into StudentAnswerSegmenter."""
    segmenter = StudentAnswerSegmenter()
    teacher_questions = [
        {"id": "tq1", "question_number": "1", "sub_question_number": "", "question_text": "Define latency."},
        {"id": "tq2", "question_number": "2", "sub_question_number": "", "question_text": "Explain handshake."}
    ]

    qwen_ocr_text = """
    1. Latency is the total transit time taken for data packets from source to destination.
    It is measured in milliseconds.

    2. Three-way handshake involves SYN, SYN-ACK, and ACK packets.
    """

    segments = segmenter.segment_answers(qwen_ocr_text, teacher_questions)
    assert len(segments) == 2

    seg1 = next(s for s in segments if s["detected_question_number"] == "1")
    assert seg1["mapping_status"] == "mapped"
    assert seg1["teacher_question_id"] == "tq1"
    assert "transit time" in seg1["extracted_answer"]

    seg2 = next(s for s in segments if s["detected_question_number"] == "2")
    assert seg2["mapping_status"] == "mapped"
    assert seg2["teacher_question_id"] == "tq2"
    assert "Three-way handshake" in seg2["extracted_answer"]


# ==============================================================================
# TEST 10: Existing Standard and Chandra routes remain unchanged
# ==============================================================================
def test_existing_standard_and_chandra_routes_unchanged():
    """Verify that standard and chandra engine routes are completely preserved."""
    img_bytes = create_dummy_png_bytes()

    # Chandra route
    with patch("app.main.chandra_service.run_chandra_ocr", return_value={
        "success": True,
        "text": "Chandra recognized text",
        "lines": [],
        "metadata": {"engine": "chandra"},
        "preprocessed_image": None
    }) as mock_chandra:
        files = {"file": ("test.png", img_bytes, "image/png")}
        resp = client.post("/api/extract", files=files, data={"ocr_engine": "chandra"})
        assert resp.status_code == 200
        assert resp.json()["ocr_engine"] == "chandra"
        assert resp.json()["text"] == "Chandra recognized text"
        assert mock_chandra.called

    # Standard route
    with patch("app.main.process_document_extraction", return_value={
        "text": "Standard recognized text",
        "lines": [],
        "metadata": {"engine": "standard"},
        "preprocessed_image": None
    }) as mock_standard:
        files = {"file": ("test.png", img_bytes, "image/png")}
        resp = client.post("/api/extract", files=files, data={"ocr_engine": "standard"})
        assert resp.status_code == 200
        assert resp.json()["ocr_engine"] == "standard"
        assert resp.json()["text"] == "Standard recognized text"
        assert mock_standard.called


# ==============================================================================
# TEST 11: Submission and OCR output persistence
# ==============================================================================
def test_submission_and_ocr_output_persistence():
    """Verify that a student submission via qwen2.5vl persists properly in store without mutating teacher records."""
    test_id = get_or_create_test_fixture()
    img_bytes = create_dummy_png_bytes()

    simulated_ocr = "1. Latency is network delay.\n\n2. TCP sends SYN packet."
    mock_run_result = {
        "success": True,
        "ocr_engine": "qwen2.5vl",
        "text": simulated_ocr,
        "lines": [{"box": [[0, 0]], "text": "line", "confidence": 0.95}],
        "metadata": {"ocr_engine": "qwen2.5vl", "model_name": "qwen2.5vl:3b", "is_pdf": False, "num_pages": 1},
        "preprocessed_image": None
    }

    # Store initial state of teacher questions
    test_record = next(t for t in db["tests"] if t["id"] == test_id)
    initial_questions_copy = json.dumps(test_record["questions"])

    with patch.object(qwen_ocr_service, "run_qwen_ocr", return_value=mock_run_result):
        files = {"file": ("submission.png", img_bytes, "image/png")}
        data = {
            "student_name": "Persistent Student",
            "student_roll": "STU_999",
            "ocr_engine": "qwen2.5vl"
        }
        resp = client.post(f"/api/tests/{test_id}/submit", files=files, data=data)
        assert resp.status_code == 200, resp.text
        res_data = resp.json()
        sub_id = res_data["submission_id"]
        assert res_data["ocr_engine"] == "qwen2.5vl"

        # Verify stored submission in db["submissions"]
        stored_sub = next((s for s in db["submissions"] if s["id"] == sub_id), None)
        assert stored_sub is not None
        assert stored_sub["ocr_engine"] == "qwen2.5vl"
        assert stored_sub["student_name"] == "Persistent Student"
        assert stored_sub["student_roll"] == "STU_999"
        assert stored_sub["extracted_text"] == simulated_ocr
        assert len(stored_sub["segmented_answers"]) == 2

        # Verify teacher record was not overwritten or mutated
        after_questions = json.dumps(test_record["questions"])
        assert initial_questions_copy == after_questions

        # Clean up test submission so store is not polluted
        db["submissions"] = [s for s in db["submissions"] if s["id"] != sub_id]
        from app.main import save_db
        save_db(db)


# ==============================================================================
# TEST 12: Live local model inference with qwen2.5vl:3b (if available)
# ==============================================================================
def test_live_local_qwen_model_inference():
    """Real live inference test using local Ollama model qwen2.5vl:3b."""
    avail, msg = qwen_ocr_service.is_available()
    if not avail:
        pytest.skip(f"Local Ollama qwen2.5vl:3b model not available: {msg}")

    # Use a small test image with printed/simple text
    test_img = Image.new("RGB", (200, 100), color="white")
    # Draw simple content or pass as is
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = qwen_ocr_service.run_qwen_ocr(img_bytes, "live_test.png")
    assert result["success"] is True
    assert result["ocr_engine"] == "qwen2.5vl"
    assert result["metadata"]["model_name"] == "qwen2.5vl:3b"
    assert "processing_time_seconds" in result["metadata"]
    assert result["metadata"]["processing_time_seconds"] > 0
