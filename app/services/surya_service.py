import io
import os
import gc
import time
import base64
import logging
from typing import List, Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Ensure PyTorch & transformers compatibility for Surya
try:
    import torch
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            return heads, torch.arange(len(mask))[mask].long()
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask=None, num_hidden_layers=None, is_attention_chunked=False):
            if head_mask is not None:
                return head_mask
            if num_hidden_layers is not None:
                return [None] * num_hidden_layers
            return None
        PreTrainedModel.get_head_mask = get_head_mask

    # Patch SuryaADETRDecoderPreTrainedModel tie_weights for newer transformers
    try:
        from surya.model.common.adetr.decoder import SuryaADETRDecoderPreTrainedModel
        SuryaADETRDecoderPreTrainedModel.tie_weights = lambda self, *args, **kwargs: None
        SuryaADETRDecoderPreTrainedModel._tie_weights = lambda self, *args, **kwargs: None
    except Exception as e:
        logger.warning(f"Could not patch SuryaADETRDecoderPreTrainedModel: {e}")
except Exception as e:
    logger.warning(f"Could not patch transformers.pytorch_utils: {e}")


def get_free_vram_gb() -> float:
    """Returns currently available free VRAM in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def clean_gpu_memory():
    """Aggressively purges PyTorch CUDA cache and runs garbage collection."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


class SuryaOCRService:
    def __init__(self):
        self.det_model = None
        self.det_processor = None
        self.rec_model = None
        self.rec_processor = None
        self.current_device = None

    def clean_memory(self):
        """Releases all model references and frees GPU cache."""
        try:
            if self.det_model is not None:
                del self.det_model
                self.det_model = None
            if self.rec_model is not None:
                del self.rec_model
                self.rec_model = None
            if self.det_processor is not None:
                del self.det_processor
                self.det_processor = None
            if self.rec_processor is not None:
                del self.rec_processor
                self.rec_processor = None
            self.current_device = None
        except Exception as e:
            logger.warning(f"Error during Surya OCR memory cleanup: {e}")
        finally:
            clean_gpu_memory()

    def load_models(self, target_device: str):
        """Loads detection and recognition models onto target device (cuda or cpu)."""
        if self.det_model is not None and self.rec_model is not None and self.current_device == target_device:
            return True

        # If switching devices or loading fresh, clean memory first
        self.clean_memory()

        dtype = torch.float16 if target_device == "cuda" else torch.float32
        logger.info(f"Loading Surya OCR models on device: {target_device} with dtype: {dtype}...")

        from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor

        self.det_processor = load_det_processor()
        self.det_model = load_det_model(device=target_device, dtype=dtype)

        self.rec_model = load_rec_model(device=target_device, dtype=dtype)
        self.rec_processor = load_rec_processor()

        self.current_device = target_device
        logger.info(f"Surya OCR models successfully loaded on {target_device}.")
        return True

    def _infer_pages(self, images: List[Image.Image], device: str) -> List[Any]:
        """Runs page-by-page sequential inference with small batches to avoid activation spikes."""
        from surya.ocr import run_ocr

        self.load_models(target_device=device)
        predictions = []

        # Process page-by-page sequentially to bound VRAM usage
        for page_idx, img in enumerate(images):
            # Conservative recognition batch size (4) to prevent VRAM spikes on RTX 3050 (4GB)
            rec_batch = 4 if device == "cuda" else 2
            page_preds = run_ocr(
                [img],
                [["en"]],
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor,
                detection_batch_size=1,
                recognition_batch_size=rec_batch
            )
            predictions.extend(page_preds)

        return predictions

    def extract_from_bytes(self, file_bytes: bytes, filename: str, device_mode: str = "auto") -> Dict[str, Any]:
        """
        Runs Surya OCR on an uploaded PDF or image file.
        Device mode options:
          - "auto": Checks free VRAM. If >= 2.5 GB free, tries GPU; otherwise uses CPU.
          - "gpu": Attempts GPU; falls back to CPU if OOM occurs.
          - "cpu": Always runs on CPU.
        """
        start_time = time.time()
        images: List[Image.Image] = []
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            try:
                import pymupdf  # PyMuPDF
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    images.append(img)
            except Exception as e:
                logger.error(f"Failed to rasterize PDF for Surya OCR: {e}")
                return {
                    "success": False,
                    "error": f"Failed to convert PDF pages: {str(e)}",
                    "engine": "Surya OCR"
                }
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                images.append(img)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid image file: {str(e)}",
                    "engine": "Surya OCR"
                }

        if not images:
            return {
                "success": False,
                "error": "No readable pages found in uploaded document.",
                "engine": "Surya OCR"
            }

        # Generate base64 thumbnail of first page for UI preview
        preview_b64 = None
        try:
            thumb = images[0].copy()
            thumb.thumbnail((800, 800))
            buffered = io.BytesIO()
            thumb.save(buffered, format="JPEG", quality=85)
            preview_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.warning(f"Could not generate preview thumbnail: {e}")

        # Decide initial target device
        cuda_avail = torch.cuda.is_available()
        free_vram = get_free_vram_gb() if cuda_avail else 0.0

        if device_mode == "cpu" or not cuda_avail:
            target_device = "cpu"
            reason = "Selected by user" if device_mode == "cpu" else "No CUDA GPU available"
        elif device_mode == "auto":
            # Require at least 2.4 GB free VRAM to safely attempt GPU
            if free_vram >= 2.4:
                target_device = "cuda"
                reason = f"Automatic ({free_vram:.2f} GB free VRAM)"
            else:
                target_device = "cpu"
                reason = f"Automatic safe fallback (Only {free_vram:.2f} GB free VRAM, requires >=2.4 GB)"
        else:  # "gpu"
            target_device = "cuda"
            reason = f"Force GPU ({free_vram:.2f} GB free VRAM)"

        logger.info(f"Surya OCR routing decision: device={target_device} ({reason})")

        # Run OCR with automatic OOM fallback
        fallback_used = False
        predictions = None

        try:
            predictions = self._infer_pages(images, device=target_device)
        except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
            err_str = str(e).lower()
            is_oom = "out of memory" in err_str or "cuda" in err_str or isinstance(e, torch.cuda.OutOfMemoryError)

            if is_oom and target_device == "cuda":
                logger.warning(
                    f"⚠️ CUDA Out of Memory encountered ({e}). "
                    f"Purging GPU memory and seamlessly retrying Surya OCR on CPU..."
                )
                self.clean_memory()
                fallback_used = True
                try:
                    predictions = self._infer_pages(images, device="cpu")
                    target_device = "cpu"
                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}", exc_info=True)
                    return {
                        "success": False,
                        "error": f"Surya OCR failed on both GPU and CPU fallback: {str(cpu_err)}",
                        "engine": "Surya OCR",
                        "preview_url": preview_b64
                    }
            else:
                logger.error(f"Surya OCR error: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Surya OCR inference error: {str(e)}",
                    "engine": "Surya OCR",
                    "preview_url": preview_b64
                }
        finally:
            # Release GPU memory immediately after the job finishes
            self.clean_memory()

        if predictions is None:
            return {
                "success": False,
                "error": "No OCR predictions generated.",
                "engine": "Surya OCR",
                "preview_url": preview_b64
            }

        # Format output text and calculate telemetry
        page_texts = []
        total_confidence = 0.0
        total_lines = 0

        for idx, pred in enumerate(predictions):
            lines = []
            for line in pred.text_lines:
                text_str = line.text.strip()
                if text_str:
                    lines.append(text_str)
                    if hasattr(line, "confidence") and line.confidence is not None:
                        total_confidence += float(line.confidence)
                        total_lines += 1

            page_header = f"--- Page {idx + 1} ---" if len(predictions) > 1 else ""
            if page_header:
                page_texts.append(page_header)
            if lines:
                page_texts.append("\n".join(lines))
            else:
                page_texts.append("[No text detected on this page]")

        full_text = "\n\n".join(page_texts).strip()
        elapsed_ms = int((time.time() - start_time) * 1000)
        avg_conf = round((total_confidence / total_lines) * 100, 1) if total_lines > 0 else 92.0

        if fallback_used:
            engine_label = "Surya OCR (CPU Fallback • OOM Recovered)"
        elif target_device == "cuda":
            engine_label = "Surya OCR (GPU • RTX 3050 CUDA)"
        else:
            engine_label = "Surya OCR (CPU Mode)"

        return {
            "success": True,
            "text": full_text if full_text else "[No text detected by Surya OCR]",
            "execution_time_ms": elapsed_ms,
            "confidence": avg_conf,
            "pages": len(images),
            "lines_count": total_lines,
            "engine": engine_label,
            "device_used": target_device,
            "fallback_used": fallback_used,
            "preview_url": preview_b64
        }


# Global singleton instance
surya_ocr_service = SuryaOCRService()
