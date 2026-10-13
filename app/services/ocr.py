import logging
import cv2
import numpy as np
import ollama
from ollama import Client

logger = logging.getLogger(__name__)

# Initialize client with a high timeout (120 seconds)
ollama_client = Client(timeout=120.0)

class OCRService:
    def __init__(self, model_name: str = "gemma3:latest"):
        self.model_name = model_name
        self._paddle_engine = None

    def _get_paddle_engine(self):
        if self._paddle_engine is None:
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR engine...")
                self._paddle_engine = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self._paddle_engine = None
            return self._paddle_engine

    def warmup(self):
        """Pre-warms the OCR engine at server startup."""
        try:
            logger.info("Pre-warming OCR engines at startup...")
            engine = self._get_paddle_engine()
            if engine is not None:
                dummy = np.full((64, 256, 3), 255, dtype=np.uint8)
                cv2.putText(dummy, "AI Tutor Ready", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                engine.ocr(dummy)
                logger.info("PaddleOCR engine pre-warmed successfully!")
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    def _extract_via_paddle(self, img_array: np.ndarray):
        """
        Runs OCR on a numpy image array using PaddleOCR.
        """
        try:
            logger.info("Running PaddleOCR extraction...")
            if not isinstance(img_array, np.ndarray) or img_array.size == 0:
                return []

            # Ensure contiguous uint8 array
            img = np.ascontiguousarray(img_array, dtype=np.uint8)
            if len(img.shape) == 2:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_img = img

            # Scale to optimal OCR dimension (1200 max dim) for sub-second CPU execution
            h, w = bgr_img.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                bgr_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            engine = self._get_paddle_engine()
            if engine is None:
                return []

            results = engine.ocr(bgr_img)
            parsed_results = []
            
            if not results:
                return parsed_results

            for item in results:
                # PaddleX 3.x dictionary format
                if isinstance(item, dict):
                    rec_texts = item.get('rec_texts', [])
                    rec_scores = item.get('rec_scores', [])
                    rec_polys = item.get('rec_polys', [])
                    for i, text in enumerate(rec_texts):
                        clean_text = str(text).strip()
                        if clean_text:
                            score = float(rec_scores[i]) if i < len(rec_scores) else 0.95
                            poly = rec_polys[i].tolist() if i < len(rec_polys) and hasattr(rec_polys[i], 'tolist') else [[0,0],[0,0],[0,0],[0,0]]
                            parsed_results.append({
                                "box": poly,
                                "text": clean_text,
                                "confidence": round(score, 3)
                            })
                # Classic PaddleOCR nested list format
                elif isinstance(item, list):
                    for line in item:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            box = line[0]
                            text_conf = line[1]
                            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                                clean_text = str(text_conf[0]).strip()
                                if clean_text:
                                    parsed_results.append({
                                        "box": box,
                                        "text": clean_text,
                                        "confidence": round(float(text_conf[1]), 3)
                                    })

            logger.info(f"PaddleOCR successfully extracted {len(parsed_results)} lines.")
            return parsed_results
        except Exception as e:
            logger.error(f"Error executing PaddleOCR: {e}")
            return []

    def extract_text(self, img_array: np.ndarray):
        """
        Runs OCR on a numpy image array using Google Gemma 3 Vision LLM via Ollama
        for state-of-the-art handwriting accuracy, with automatic PaddleOCR fallback.
        Returns:
            list of dicts containing text, bounding box coordinates, and confidence.
        """
        # 1. Primary engine: Ollama Vision Model (Gemma 3 Vision) for superior handwriting recognition
        try:
            logger.info(f"Transcribing handwriting with Vision LLM '{self.model_name}' via Ollama...")
            
            # Scale image proportionally (max dim 1024) to maintain crisp handwriting while conserving memory
            max_dim = 1024
            h, w = img_array.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_img = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                resized_img = img_array

            # Encode as high-quality JPEG for compact transmission
            success, encoded_img = cv2.imencode('.jpg', resized_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if success:
                img_bytes = encoded_img.tobytes()
                prompt = (
                    "You are an OCR engine. Transcribe all handwritten and printed text from this image exactly as written. "
                    "Maintain original line breaks and layout structure. "
                    "Do not add intro text, commentary, or markdown code blocks. "
                    "Output only the exact transcribed text."
                )

                response = ollama_client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [img_bytes]
                    }],
                    options={
                        "temperature": 0.1,
                        "num_ctx": 2048,
                    }
                )

                transcribed_text = response.message.content
                if transcribed_text and transcribed_text.strip():
                    logger.info(f"Successfully transcribed text via {self.model_name}!")
                    parsed_results = []
                    for line in transcribed_text.split('\n'):
                        clean_line = line.strip()
                        if clean_line:
                            parsed_results.append({
                                "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                                "text": clean_line,
                                "confidence": 0.98
                            })
                    if parsed_results:
                        return parsed_results
        except Exception as e:
            logger.warning(f"Ollama Vision LLM ({self.model_name}) unavailable/error: {str(e)}")
            logger.info("Falling back to local PaddleOCR engine...")

        # 2. Seamless Fallback: Local PaddleOCR
        return self._extract_via_paddle(img_array)

# Instantiate singleton
ocr_service = OCRService()


