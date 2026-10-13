import io
import time
import base64
import logging
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from PIL import Image
import ollama
from ollama import Client

logger = logging.getLogger(__name__)

SYSTEM_OCR_PROMPT = (
    "You are an expert OCR transcription engine.\n"
    "Transcribe all handwritten and printed text from this image exactly as written.\n"
    "Preserve all question numbers, sub-question identifiers (e.g., Q1, 1., 2(a), (b)), mathematical equations, formulas, calculations, technical terms, IP addresses, and hexadecimal strings verbatim.\n"
    "Maintain the original line breaks, paragraph structure, and answer sequence.\n"
    "Do not evaluate, grade, solve, or assign marks to the answers.\n"
    "Do not summarize, paraphrase, or invent text. If a word or symbol is ambiguous, provide your best faithful reading.\n"
    "Output only the transcribed text with no preamble, explanations, conversational filler, or markdown code fence wrappers."
)

class QwenVLOCRService:
    """
    Isolated service for Qwen2.5-VL OCR integration using Ollama model 'qwen2.5vl:3b'.
    Strictly isolated with zero silent fallbacks to Gemma or Chandra.
    Extracts raw OCR text from image or PDF page-by-page.
    """
    MODEL_NAME = "qwen2.5vl:3b"
    ENGINE_ID = "qwen2.5vl"

    def __init__(self, client: Optional[Client] = None, timeout: float = 120.0):
        self.timeout = timeout
        self._client = client or Client(timeout=timeout)

    def is_available(self) -> tuple[bool, str]:
        """
        Check if Ollama server is responsive and qwen2.5vl:3b model is available.
        """
        try:
            models_info = self._client.list()
            models_list = getattr(models_info, "models", None) or (
                models_info.get("models", []) if isinstance(models_info, dict) else []
            )
            available_names = []
            for m in models_list:
                name = (
                    getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or (m.get("model") if isinstance(m, dict) else m.get("name") if isinstance(m, dict) else None)
                )
                if name:
                    available_names.append(str(name).lower())

            if any(self.MODEL_NAME in n for n in available_names):
                return True, f"Qwen2.5-VL OCR Ready ({self.MODEL_NAME})"
            return False, f"Model '{self.MODEL_NAME}' not found in Ollama. Available: {available_names}"
        except Exception as e:
            return False, f"Ollama connection error: {str(e)}"

    def _clean_transcription(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def run_qwen_ocr(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Executes Qwen2.5-VL OCR on an uploaded PDF or image file page-by-page.
        Strictly preserves raw text with zero silent fallback.
        """
        start_time = time.time()
        is_pdf = filename.lower().endswith(".pdf")
        preview_data_url = None
        pil_images: List[Image.Image] = []

        # 1. Convert input PDF or image into PIL Images preserving page order
        try:
            if is_pdf:
                doc = fitz.open(stream=content, filetype="pdf")
                if len(doc) == 0:
                    raise ValueError("The uploaded PDF file contains no pages.")

                for page_idx in range(len(doc)):
                    page = doc.load_page(page_idx)
                    # Render page at 2.0x scale (144 dpi) for clear text
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    img_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_images.append(pil_img)
            else:
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
                pil_images.append(pil_img)

            # Generate preview image data URL from page 1
            if pil_images:
                preview_img = pil_images[0]
                buf = io.BytesIO()
                preview_copy = preview_img.copy()
                if max(preview_copy.size) > 1024:
                    preview_copy.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                preview_copy.save(buf, format="JPEG", quality=88)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                preview_data_url = f"data:image/jpeg;base64,{b64_str}"

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Qwen2.5-VL OCR] Document decoding failed: {e}", exc_info=True)
            return {
                "success": False,
                "ocr_engine": self.ENGINE_ID,
                "error": f"Failed to load document: {str(e)}",
                "detail": str(e),
                "text": "",
                "lines": [],
                "metadata": {
                    "ocr_engine": self.ENGINE_ID,
                    "model_name": self.MODEL_NAME,
                    "processing_time_seconds": round(elapsed, 3),
                    "is_pdf": is_pdf,
                    "num_pages": 0
                },
                "preprocessed_image": None
            }

        num_pages = len(pil_images)

        # 2. Verify model availability before initiating inference
        avail, avail_msg = self.is_available()
        if not avail:
            elapsed = time.time() - start_time
            logger.error(f"[Qwen2.5-VL OCR] Model availability check failed: {avail_msg}")
            return {
                "success": False,
                "ocr_engine": self.ENGINE_ID,
                "error": f"Qwen2.5-VL OCR unavailable: {avail_msg}",
                "detail": avail_msg,
                "text": "",
                "lines": [],
                "metadata": {
                    "ocr_engine": self.ENGINE_ID,
                    "model_name": self.MODEL_NAME,
                    "is_pdf": is_pdf,
                    "num_pages": num_pages,
                    "processing_time_seconds": round(elapsed, 3),
                },
                "preprocessed_image": preview_data_url
            }

        # 3. Page-by-page inference with Qwen2.5-VL vision model
        page_texts: List[str] = []
        try:
            logger.info(
                f"[Qwen2.5-VL OCR] Processing {num_pages} page(s) sequentially with model '{self.MODEL_NAME}'..."
            )

            for page_idx, img in enumerate(pil_images):
                page_start = time.time()
                logger.info(f"[Qwen2.5-VL OCR] Page {page_idx + 1}/{num_pages} inference starting...")

                img_to_send = img
                if max(img.size) > 1536:
                    img_to_send = img.copy()
                    img_to_send.thumbnail((1536, 1536), Image.Resampling.LANCZOS)

                buf = io.BytesIO()
                img_to_send.save(buf, format="JPEG", quality=95)
                img_bytes = buf.getvalue()

                response = self._client.chat(
                    model=self.MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": SYSTEM_OCR_PROMPT,
                        "images": [img_bytes]
                    }],
                    options={
                        "temperature": 0.1,
                    }
                )

                raw_content = ""
                if response and hasattr(response, "message") and response.message:
                    raw_content = response.message.content or ""
                elif isinstance(response, dict):
                    raw_content = response.get("message", {}).get("content", "")

                clean_text = self._clean_transcription(raw_content)
                page_dur = time.time() - page_start
                logger.info(
                    f"[Qwen2.5-VL OCR] Page {page_idx + 1}/{num_pages} completed in {page_dur:.2f}s "
                    f"({len(clean_text)} characters)."
                )

                if is_pdf:
                    page_texts.append(f"--- Page {page_idx + 1} ---\n{clean_text}")
                else:
                    page_texts.append(clean_text)

            extracted_text = "\n\n".join(page_texts) if is_pdf else "\n".join(page_texts)
            elapsed = time.time() - start_time
            logger.info(
                f"[Qwen2.5-VL OCR] Completed all {num_pages} page(s) in {elapsed:.2f}s "
                f"({len(extracted_text)} total characters)."
            )

            lines = []
            for line in extracted_text.split("\n"):
                c_line = line.strip()
                if c_line:
                    lines.append({
                        "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        "text": c_line,
                        "confidence": 0.95
                    })

            return {
                "success": True,
                "ocr_engine": self.ENGINE_ID,
                "text": extracted_text,
                "lines": lines,
                "metadata": {
                    "ocr_engine": self.ENGINE_ID,
                    "model_name": self.MODEL_NAME,
                    "is_pdf": is_pdf,
                    "num_pages": num_pages,
                    "processing_time_seconds": round(elapsed, 3),
                    "confidence_average": 0.95,
                    "skew_angle": 0.0
                },
                "preprocessed_image": preview_data_url
            }

        except Exception as e:
            elapsed = time.time() - start_time
            err_msg = str(e)
            logger.error(f"[Qwen2.5-VL OCR] Inference error: {err_msg}", exc_info=True)
            return {
                "success": False,
                "ocr_engine": self.ENGINE_ID,
                "error": f"Qwen2.5-VL OCR extraction failed: {err_msg}",
                "detail": err_msg,
                "text": "",
                "lines": [],
                "metadata": {
                    "ocr_engine": self.ENGINE_ID,
                    "model_name": self.MODEL_NAME,
                    "is_pdf": is_pdf,
                    "num_pages": num_pages,
                    "processing_time_seconds": round(elapsed, 3),
                    "confidence_average": None,
                    "skew_angle": 0.0
                },
                "preprocessed_image": preview_data_url
            }

# Singleton instance
qwen_ocr_service = QwenVLOCRService()
