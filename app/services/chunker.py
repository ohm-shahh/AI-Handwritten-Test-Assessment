import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_experimental.text_splitter import SemanticChunker
from app.services.embedding import embedding_service, OllamaEmbeddingService

logger = logging.getLogger(__name__)

class TeacherSemanticChunker:
    """
    Question-level semantic chunker using LangChain SemanticChunker backed by
    Ollama qwen3-embedding:4b.
    Ensures unrelated questions are never grouped together.
    """

    def __init__(self, embed_service: Optional[OllamaEmbeddingService] = None):
        self.embedding_service = embed_service or embedding_service
        try:
            self.splitter = SemanticChunker(
                embeddings=self.embedding_service,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )
            logger.info("[CHUNKER] Initialized LangChain SemanticChunker with qwen3-embedding:4b.")
        except Exception as e:
            logger.error(f"[CHUNKER] Failed initializing SemanticChunker: {e}")
            raise

    def build_question_text_unit(self, q: Dict[str, Any]) -> str:
        """
        Builds a coherent semantic unit string containing question, model answer, and rubric.
        """
        parts = []
        q_num = q.get("question_number", "")
        sec = q.get("section")
        sec_prefix = f"Section {sec} - " if sec else ""
        marks = q.get("max_marks")
        marks_str = f" [{marks} Marks]" if marks is not None else ""
        
        parts.append(f"{sec_prefix}Question {q_num}{marks_str}:\n{q.get('question_text', '').strip()}")

        if q.get("model_answer"):
            parts.append(f"Model Answer:\n{q['model_answer'].strip()}")

        if q.get("rubric"):
            parts.append(f"Marking Scheme & Rubric:\n{q['rubric'].strip()}")

        if q.get("marking_points"):
            pts_str = "\n".join([f"- {pt}" for pt in q["marking_points"]])
            parts.append(f"Key Points:\n{pts_str}")

        return "\n\n".join(parts)

    def chunk_questions(
        self,
        test_id: str,
        questions: List[Dict[str, Any]],
        source_file: str = "teacher_upload"
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Chunks questions individually while preserving question-level semantic boundaries.
        Returns a list of (chunk_text, chunk_metadata).
        """
        all_chunks: List[Tuple[str, Dict[str, Any]]] = []

        if not questions:
            logger.warning(f"[CHUNKER] No questions provided for test_id='{test_id}'.")
            return []

        logger.info(f"[CHUNKER] Creating semantic chunks for {len(questions)} questions in test '{test_id}'...")

        for q_idx, q in enumerate(questions):
            unit_text = self.build_question_text_unit(q)
            if not unit_text.strip():
                continue

            # Split question unit semantically
            try:
                # If text is very short, SemanticChunker might return 1 chunk directly
                sub_chunks = self.splitter.split_text(unit_text)
                if not sub_chunks:
                    sub_chunks = [unit_text]
            except Exception as e:
                logger.warning(f"[CHUNKER] SemanticChunker fallback for Q{q.get('question_number')}: {e}")
                sub_chunks = [unit_text]

            q_id = q.get("id") or f"q_{q_idx + 1}"
            q_num = q.get("question_number", str(q_idx + 1))
            sec = q.get("section") or ""
            marks = q.get("max_marks")

            for c_idx, chunk_content in enumerate(sub_chunks):
                if not chunk_content.strip():
                    continue

                metadata = {
                    "test_id": test_id,
                    "question_id": q_id,
                    "question_number": str(q_num),
                    "section": str(sec),
                    "max_marks": float(marks) if marks is not None else -1.0,
                    "document_type": "model_answer",
                    "chunk_type": "question_answer_rubric",
                    "source_file": source_file,
                    "chunk_index": c_idx,
                    "total_chunks_for_question": len(sub_chunks)
                }
                all_chunks.append((chunk_content.strip(), metadata))

        logger.info(f"[CHUNKER] Created {len(all_chunks)} semantic chunks for test '{test_id}'.")
        return all_chunks

# Singleton instance
teacher_chunker = TeacherSemanticChunker()
