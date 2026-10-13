import os
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from app.main import (
    app,
    extract_via_pdfplumber_stream,
    extract_via_pypdf2_stream
)
from app.services.preprocess import preprocess_pipeline
from app.services.ocr import ocr_service

# Paths to sample files in the workspace root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "handwritten_sample.png")
PDF_PATH = os.path.join(BASE_DIR, "AI_Tutor_Research_Document.pdf")

client = TestClient(app)

def test_preprocess_pipeline():
    """
    Test the preprocessing pipeline with the sample image.
    """
    assert os.path.exists(IMAGE_PATH), f"Sample image not found at {IMAGE_PATH}"
    
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        
    # Run the preprocessor
    processed_img, metadata = preprocess_pipeline(
        image_bytes=image_bytes,
        apply_grayscale=True,
        apply_denoise=True,
        apply_threshold=True,
        apply_deskew=True
    )
    
    # Assert processed image is a single-channel grayscale image (because apply_threshold=True)
    assert isinstance(processed_img, np.ndarray)
    assert len(processed_img.shape) == 2
    
    # Check metadata structure
    assert "original_size" in metadata
    assert "processed_size" in metadata
    assert "skew_angle" in metadata
    assert metadata["original_size"]["width"] > 0
    assert metadata["original_size"]["height"] > 0
    assert metadata["processed_size"]["width"] > 0
    assert metadata["processed_size"]["height"] > 0

def test_ocr_service_basic():
    """
    Test the OCR service with a dummy small numpy array to verify it initializes and runs.
    """
    # Create a simple white image containing a black square
    dummy_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    # Let's see if we can run OCR (even if no text is found, it shouldn't raise exception)
    results = ocr_service.extract_text(dummy_img)
    assert isinstance(results, list)

def test_pdf_digital_extraction():
    """
    Test the digital PDF text extraction helper functions on the research document PDF.
    """
    assert os.path.exists(PDF_PATH), f"Sample PDF not found at {PDF_PATH}"
    
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
        
    # extract via pdfplumber
    text_plumber = extract_via_pdfplumber_stream(pdf_bytes, 0)
    # extract via pypdf2
    text_pypdf = extract_via_pypdf2_stream(pdf_bytes, 0)
    
    # At least one should successfully extract some text if it's a digital PDF
    assert text_plumber is not None or text_pypdf is not None
    extracted = text_plumber or text_pypdf
    assert len(extracted.strip()) > 0
    print(f"\nExtracted digital text (first 100 chars): {extracted[:100]}")

def test_api_extract_image():
    """
    Test the FastAPI /api/extract endpoint with the sample handwritten image.
    """
    assert os.path.exists(IMAGE_PATH), f"Sample image not found at {IMAGE_PATH}"
    
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("handwritten_sample.png", f, "image/png")}
        data = {
            "apply_grayscale": "true",
            "apply_denoise": "true",
            "apply_threshold": "true",
            "apply_deskew": "true"
        }
        response = client.post("/api/extract", files=files, data=data)
        
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert "text" in res_json
    assert "lines" in res_json
    assert "metadata" in res_json
    assert "preprocessed_image" in res_json
    assert res_json["metadata"]["is_pdf"] is False
    assert res_json["metadata"]["confidence_average"] >= 0.0

def test_api_extract_pdf():
    """
    Test the FastAPI /api/extract endpoint with the sample research document PDF.
    """
    assert os.path.exists(PDF_PATH), f"Sample PDF not found at {PDF_PATH}"
    
    with open(PDF_PATH, "rb") as f:
        files = {"file": ("AI_Tutor_Research_Document.pdf", f, "application/pdf")}
        data = {
            "apply_grayscale": "true",
            "apply_denoise": "true",
            "apply_threshold": "true",
            "apply_deskew": "true"
        }
        response = client.post("/api/extract", files=files, data=data)
        
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert "text" in res_json
    assert "lines" in res_json
    assert "metadata" in res_json
    assert res_json["metadata"]["is_pdf"] is True
    assert res_json["metadata"]["num_pages"] > 0
