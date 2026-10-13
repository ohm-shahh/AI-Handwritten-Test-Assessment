import io
import os
import time
import base64
import logging
import tempfile
from typing import Dict, Any, List
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class ChandraOCRService:
    """
    Isolated service for Chandra OCR integration (Phase 1).
    Does NOT modify or touch the existing Standard OCR or Surya OCR pipelines.
    Extracts raw OCR text only without any LLM corrections, RAG, or post-processing.
    """
    def __init__(self):
        self._manager = None
        self._init_error = None

    def is_available(self) -> tuple[bool, str]:
        """
        Check if the chandra package and inference modules are importable.
        """
        try:
            import chandra
            from chandra.model import InferenceManager
            return True, "Chandra OCR Ready"
        except Exception as e:
            return False, f"Chandra OCR Unavailable: {str(e)}"

    def _get_manager(self):
        """
        Lazily initializes the Chandra InferenceManager with memory-safe allocation
        specifically tuned for 12GB VRAM GPUs (TITAN Xp).
        Reserves ~3GB of GPU VRAM headroom for vision token generation and activations.
        """
        if self._manager is None:
            try:
                import torch
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                from chandra.model import InferenceManager
                from transformers import AutoModelForImageTextToText, AutoProcessor
                from chandra.settings import settings
                from app.services.gpu_memory import release_ollama_vram, release_cuda_cache

                # Explicitly release Ollama background VRAM before Chandra runs (lifecycle only, no OCR)
                release_ollama_vram()
                release_cuda_cache()

                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    # Reserve 3.5 GiB GPU VRAM headroom for attention and vision tokens
                    safe_b = max(int(4 * (1024**3)), min(int(7.5 * (1024**3)), free_b - int(3.5 * (1024**3))))
                    safe_gib = f"{safe_b / (1024**3):.1f}GiB"
                    max_memory = {0: safe_gib, "cpu": "24GiB"}
                else:
                    max_memory = {"cpu": "24GiB"}

                logger.info(f"Loading Chandra model checkpoint with max_memory={max_memory}...")
                model = AutoModelForImageTextToText.from_pretrained(
                    settings.MODEL_CHECKPOINT,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory=max_memory
                )
                model = model.eval()
                processor = AutoProcessor.from_pretrained(settings.MODEL_CHECKPOINT)
                processor.tokenizer.padding_side = "left"
                model.processor = processor

                manager = InferenceManager.__new__(InferenceManager)
                manager.method = "hf"
                manager.model = model
                self._manager = manager
                logger.info("Chandra InferenceManager initialized successfully with safe VRAM headroom.")
            except Exception as e:
                logger.error(f"Failed to initialize Chandra InferenceManager: {e}", exc_info=True)
                raise

        return self._manager

    def run_chandra_ocr(self, content: bytes, filename: str) -> Dict[str, Any]:
        """
        Executes Chandra OCR on an uploaded PDF or image page-by-page.
        Returns the raw extracted text as-is.
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
                    raise ValueError("Uploaded PDF has 0 pages.")

                for page_idx in range(len(doc)):
                    page = doc.load_page(page_idx)
                    # Render page at 1.5x scale for sharp text while preserving VRAM
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                    img_bytes = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_images.append(pil_img)

                    if page_idx == 0:
                        b64_str = base64.b64encode(img_bytes).decode("utf-8")
                        preview_data_url = f"data:image/png;base64,{b64_str}"
            else:
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
                pil_images.append(pil_img)

                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=92)
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                preview_data_url = f"data:image/jpeg;base64,{b64_str}"

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Chandra OCR] Document loading failed: {e}", exc_info=True)
            return {
                "success": False,
                "ocr_engine": "chandra",
                "error": f"Failed to load document: {str(e)}",
                "detail": str(e),
                "text": "",
                "metadata": {
                    "ocr_engine": "chandra",
                    "processing_time_seconds": round(elapsed, 3),
                    "is_pdf": is_pdf,
                },
                "preprocessed_image": preview_data_url
            }

        # 2. Run Chandra OCR inference page-by-page
        try:
            import torch
            from chandra.model.schema import BatchInputItem

            manager = self._get_manager()
            logger.info(f"[Chandra OCR] Processing {len(pil_images)} page(s) sequentially...")

            results = []
            for page_idx, img in enumerate(pil_images):
                logger.info(f"[Chandra OCR] Running inference on page {page_idx + 1}/{len(pil_images)}...")
                # Resize image to optimal dimension for fast vision model inference
                if max(img.size) > 1536:
                    img = img.copy()
                    img.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
                page_batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]
                page_results = manager.generate(page_batch, max_output_tokens=2048)
                results.extend(page_results)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 3. Extract text verbatim (NO LLM alteration, NO post-processing)
            page_texts = []
            for page_idx, res in enumerate(results):
                page_text = getattr(res, "markdown", None) or getattr(res, "html", "") or str(res)
                page_text = page_text.strip()
                if is_pdf:
                    page_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
                else:
                    page_texts.append(page_text)

            extracted_text = "\n\n".join(page_texts) if is_pdf else "\n".join(page_texts)
            elapsed = time.time() - start_time
            logger.info(f"[Chandra OCR] Finished in {elapsed:.2f}s. Extracted {len(extracted_text)} characters.")

            return {
                "success": True,
                "ocr_engine": "chandra",
                "text": extracted_text,
                "lines": [],
                "metadata": {
                    "ocr_engine": "chandra",
                    "is_pdf": is_pdf,
                    "num_pages": len(pil_images),
                    "processing_time_seconds": round(elapsed, 3),
                    "confidence_average": None,
                    "skew_angle": 0.0
                },
                "preprocessed_image": preview_data_url
            }

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            logger.error(f"[Chandra OCR] Inference error: {error_msg}", exc_info=True)

            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                user_msg = "Chandra OCR ran out of GPU memory (CUDA OOM). TITAN Xp 12GB VRAM may be insufficient for full 10.6GB model parameters."
            elif "model.safetensors" in error_msg.lower() or "incomplete" in error_msg.lower():
                user_msg = "Chandra model weights download is in progress or incomplete."
            else:
                user_msg = f"Chandra OCR error: {error_msg}"

            return {
                "success": False,
                "ocr_engine": "chandra",
                "error": user_msg,
                "detail": error_msg,
                "text": "",
                "lines": [],
                "metadata": {
                    "ocr_engine": "chandra",
                    "is_pdf": is_pdf,
                    "num_pages": len(pil_images),
                    "processing_time_seconds": round(elapsed, 3),
                    "confidence_average": None,
                    "skew_angle": 0.0
                },
                "preprocessed_image": preview_data_url
            }

# Singleton instance
chandra_service = ChandraOCRService()
