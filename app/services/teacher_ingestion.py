import os
import io
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import pdfplumber
import PyPDF2
import docx

from app.services.embedding import embedding_service
from app.services.knowledge_store import knowledge_store
from app.services.question_parser import (
    QuestionPaperParser,
    ModelAnswerParser,
    map_questions_and_answers
)
from app.services.chunker import teacher_chunker
from app.services.preprocess import preprocess_pipeline
from app.services.ocr import ocr_service

logger = logging.getLogger(__name__)

class TeacherTestIngestionService:
    """
    Teacher Knowledge Ingestion Service (Phase 2).
    Orchestrates file storage, extraction, deterministic parsing, structured DB sync,
    semantic chunking, Ollama qwen3-embedding:4b embedding, and persistent ChromaDB index.
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.doc_storage_root = os.path.join(base_dir, "data", "documents")
        os.makedirs(self.doc_storage_root, exist_ok=True)
        self.qp_parser = QuestionPaperParser()
        self.model_parser = ModelAnswerParser()

    def extract_text_from_file(self, content: bytes, filename: str) -> str:
        """
        Extracts clean text from PDF, DOCX, TXT, or Image files.
        Digital PDF text is extracted directly; scanned PDFs/images route to OCR service.
        """
        lower_name = filename.lower()
        logger.info(f"[INGEST] Extracting text from document '{filename}' ({len(content)} bytes)...")

        # 1. Plain text / Markdown
        if lower_name.endswith(('.txt', '.md', '.json', '.csv')):
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                return content.decode('latin-1', errors='ignore')

        # 2. Word Document (DOCX)
        if lower_name.endswith('.docx'):
            try:
                doc = docx.Document(io.BytesIO(content))
                extracted_lines = [p.text for p in doc.paragraphs if p.text.strip()]
                # Also include table text
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text:
                            extracted_lines.append(row_text)
                return "\n".join(extracted_lines)
            except Exception as e:
                logger.error(f"[INGEST] Failed extracting DOCX '{filename}': {e}")
                raise RuntimeError(f"Error parsing Word document ({filename}): {str(e)}")

        # 3. PDF (Digital text first, OCR for scanned pages)
        if lower_name.endswith('.pdf'):
            try:
                pdf_doc = fitz.open(stream=content, filetype="pdf")
                num_pages = len(pdf_doc)
                if num_pages == 0:
                    raise ValueError(f"PDF '{filename}' contains no pages.")

                all_pages_text = []
                for p_idx in range(num_pages):
                    # Attempt selectable digital text extraction
                    page_text = None
                    try:
                        with pdfplumber.open(io.BytesIO(content)) as pdf:
                            if p_idx < len(pdf.pages):
                                page_text = pdf.pages[p_idx].extract_text()
                    except Exception:
                        pass

                    if not page_text or not page_text.strip():
                        # Fallback to PyMuPDF digital extract
                        page_text = pdf_doc[p_idx].get_text()

                    if page_text and page_text.strip():
                        all_pages_text.append(page_text.strip())
                    else:
                        # Scanned page: render image and route to OCR abstraction
                        logger.info(f"[INGEST] Page {p_idx + 1} of '{filename}' is scanned/image; routing to OCR service...")
                        page = pdf_doc.load_page(p_idx)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                        img_bytes = pix.tobytes("png")
                        
                        proc_img, _ = preprocess_pipeline(
                            image_bytes=img_bytes,
                            apply_grayscale=True,
                            apply_denoise=True,
                            apply_threshold=False,
                            apply_deskew=True
                        )
                        ocr_lines = ocr_service.extract_text(proc_img)
                        ocr_text = "\n".join([line["text"] for line in ocr_lines if line.get("text")])
                        if ocr_text.strip():
                            all_pages_text.append(ocr_text.strip())

                return "\n\n".join(all_pages_text)
            except Exception as e:
                logger.error(f"[INGEST] PDF extraction error for '{filename}': {e}")
                raise RuntimeError(f"Error extracting PDF ({filename}): {str(e)}")

        # 4. Image file (PNG, JPG, JPEG, WEBP, BMP, TIFF)
        if lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            try:
                proc_img, _ = preprocess_pipeline(
                    image_bytes=content,
                    apply_grayscale=True,
                    apply_denoise=True,
                    apply_threshold=False,
                    apply_deskew=True
                )
                ocr_lines = ocr_service.extract_text(proc_img)
                return "\n".join([line["text"] for line in ocr_lines if line.get("text")])
            except Exception as e:
                logger.error(f"[INGEST] Image OCR extraction error for '{filename}': {e}")
                raise RuntimeError(f"Error extracting image ({filename}): {str(e)}")

        raise ValueError(f"Unsupported document format: {filename}")

    def store_original_file(
        self,
        test_id: str,
        file_content: bytes,
        original_filename: str,
        doc_type: str
    ) -> Dict[str, Any]:
        """
        Stores an uploaded file unchanged in app/data/documents/{test_id}/
        without overwriting existing files, returning source document metadata.
        """
        test_dir = os.path.join(self.doc_storage_root, test_id)
        os.makedirs(test_dir, exist_ok=True)

        safe_filename = os.path.basename(original_filename).replace(" ", "_")
        dest_path = os.path.join(test_dir, safe_filename)

        # Write original bytes unchanged
        with open(dest_path, "wb") as f:
            f.write(file_content)

        doc_record = {
            "id": f"doc_{uuid_or_hash(test_id, safe_filename)}",
            "test_id": test_id,
            "document_type": doc_type,
            "original_filename": original_filename,
            "storage_path": dest_path,
            "file_size_bytes": len(file_content),
            "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.info(f"[INGEST] Stored original file: {dest_path} ({len(file_content)} bytes)")
        return doc_record

    def ingest_test_materials(
        self,
        test_id: str,
        db_ref: Dict[str, Any],
        save_db_fn: Any,
        qp_file_content: Optional[bytes] = None,
        qp_filename: Optional[str] = None,
        model_file_content: Optional[bytes] = None,
        model_filename: Optional[str] = None,
        manual_qp_text: Optional[str] = None,
        manual_model_text: Optional[str] = None,
        manual_rubric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full Phase 2 Ingestion Pipeline for a test:
        1. Verifies qwen3-embedding:4b availability in Ollama.
        2. Stores original documents unchanged.
        3. Extracts text from documents / manual inputs.
        4. Deterministically parses structured questions, marks, and sub-questions.
        5. Parses model answers and rubrics, and maps Question ↔ Model Answer.
        6. Updates structured DB records (authoritative exam storage).
        7. Semantically chunks questions with LangChain SemanticChunker.
        8. Computes embeddings via qwen3-embedding:4b.
        9. Persists chunks and embeddings in ChromaDB (idempotent upsert).
        """
        logger.info(f"[INGEST] Test {test_id} ingestion started.")

        # Step 1: Verify Ollama embedding model availability
        embed_ok, embed_msg = embedding_service.is_available()
        if not embed_ok:
            logger.error(f"[INGEST] Embedding verification failed: {embed_msg}")
            # Mark test knowledge_status as failed in DB
            self._update_db_status(db_ref, save_db_fn, test_id, "failed", error=embed_msg)
            raise RuntimeError(f"Embedding configuration error: {embed_msg}")

        # Update status to processing
        self._update_db_status(db_ref, save_db_fn, test_id, "processing")

        try:
            source_documents: List[Dict[str, Any]] = []

            # Step 2: Store original files and extract text
            qp_text = ""
            if qp_file_content and qp_filename:
                doc_rec = self.store_original_file(test_id, qp_file_content, qp_filename, "question_paper")
                source_documents.append(doc_rec)
                qp_text = self.extract_text_from_file(qp_file_content, qp_filename)
            elif manual_qp_text and manual_qp_text.strip():
                qp_text = manual_qp_text.strip()
                # Also save text version to documents folder for completeness
                text_bytes = qp_text.encode("utf-8")
                doc_rec = self.store_original_file(test_id, text_bytes, "question_paper.txt", "question_paper")
                source_documents.append(doc_rec)

            model_text = ""
            if model_file_content and model_filename:
                doc_rec = self.store_original_file(test_id, model_file_content, model_filename, "model_answer")
                source_documents.append(doc_rec)
                model_text = self.extract_text_from_file(model_file_content, model_filename)
            elif manual_model_text and manual_model_text.strip():
                model_text = manual_model_text.strip()
                text_bytes = model_text.encode("utf-8")
                doc_rec = self.store_original_file(test_id, text_bytes, "model_answer.txt", "model_answer")
                source_documents.append(doc_rec)

            logger.info(f"[INGEST] Question paper extracted ({len(qp_text)} chars).")
            logger.info(f"[INGEST] Model answer extracted ({len(model_text)} chars).")

            # Step 3: Parse structured questions & model answers
            parsed_questions = self.qp_parser.parse(qp_text)
            parsed_answers_map = self.model_parser.parse(model_text)

            logger.info(f"[INGEST] Parsed {len(parsed_questions)} questions.")
            logger.info(f"[INGEST] Parsed {len(parsed_answers_map)} model answers.")

            # Step 4: Map Question ↔ Model Answer
            mapped_questions = map_questions_and_answers(
                questions=parsed_questions,
                answers_map=parsed_answers_map,
                general_rubric=manual_rubric
            )
            logger.info("[INGEST] Question-answer mapping complete.")

            # Step 5: Update structured DB (Authoritative source of truth)
            test_obj = next((t for t in db_ref.get("tests", []) if t["id"] == test_id), None)
            if not test_obj:
                raise ValueError(f"Test with ID '{test_id}' not found in structured DB.")

            test_obj["questions"] = mapped_questions
            test_obj["source_documents"] = source_documents
            test_obj["knowledge_status"] = "processing"
            save_db_fn(db_ref)

            # Step 6: Semantic chunking
            source_file_label = qp_filename or (source_documents[0]["original_filename"] if source_documents else "manual_entry")
            semantic_chunks = teacher_chunker.chunk_questions(
                test_id=test_id,
                questions=mapped_questions,
                source_file=source_file_label
            )
            logger.info(f"[INGEST] Semantic chunks created: {len(semantic_chunks)}.")

            if not semantic_chunks:
                raise ValueError("No semantic chunks could be created from the provided materials.")

            chunk_texts = [c[0] for c in semantic_chunks]
            chunk_metas = [c[1] for c in semantic_chunks]

            # Step 7: Generate embeddings with qwen3-embedding:4b
            logger.info(f"[EMBED] Model: {embedding_service.model_name}")
            embeddings = embedding_service.get_embeddings(chunk_texts)
            logger.info(f"[EMBED] Generated {len(embeddings)} embeddings.")

            # Step 8: Persistent ChromaDB upsert (idempotent)
            logger.info(f"[CHROMA] Collection: {knowledge_store.collection_name}")
            # Delete old chunks to prevent stale embeddings
            knowledge_store.delete_test_chunks(test_id)
            stored_count = knowledge_store.upsert_chunks(
                test_id=test_id,
                chunk_documents=chunk_texts,
                embeddings=embeddings,
                metadatas=chunk_metas
            )
            logger.info(f"[CHROMA] Stored {stored_count} documents.")

            # Step 9: Finalize structured DB status
            test_obj["knowledge_status"] = "ready"
            test_obj["knowledge_error"] = None
            test_obj["knowledge_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            test_obj["knowledge_stats"] = {
                "questions_extracted": len(mapped_questions),
                "answers_processed": len(parsed_answers_map),
                "chunks_created": len(semantic_chunks),
                "embeddings_created": len(embeddings),
                "chroma_collection": knowledge_store.collection_name
            }
            save_db_fn(db_ref)

            logger.info(f"[INGEST] Test {test_id} knowledge base READY.")

            return {
                "success": True,
                "test_id": test_id,
                "questions_extracted": len(mapped_questions),
                "answers_processed": len(parsed_answers_map),
                "chunks_created": len(semantic_chunks),
                "embeddings_created": len(embeddings),
                "chroma_collection": knowledge_store.collection_name
            }

        except Exception as e:
            logger.error(f"[INGEST] Pipeline failed for test {test_id}: {e}", exc_info=True)
            self._update_db_status(db_ref, save_db_fn, test_id, "failed", error=str(e))
            raise

    def _update_db_status(
        self,
        db_ref: Dict[str, Any],
        save_db_fn: Any,
        test_id: str,
        status: str,
        error: Optional[str] = None
    ):
        test_obj = next((t for t in db_ref.get("tests", []) if t["id"] == test_id), None)
        if test_obj:
            test_obj["knowledge_status"] = status
            test_obj["knowledge_error"] = error
            save_db_fn(db_ref)

def uuid_or_hash(test_id: str, filename: str) -> str:
    import hashlib
    raw = f"{test_id}_{filename}_{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]

# Singleton instance
teacher_ingestion_service = TeacherTestIngestionService()
