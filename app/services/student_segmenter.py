import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StudentAnswerSegmenter:
    """
    Segments raw OCR text from a student submission into individual answers
    and maps them to the teacher's structured questions.
    """
    def __init__(self):
        # Matches patterns like "Q1", "Q.1", "Question 1", "1)", "1.", "1(a)", "Q1(a)"
        self.question_pattern = re.compile(
            r'^\s*(?:Q\.?|Question\s*)?(\d+)(?:\s*[\.\)]|\s*\(([a-z])\))?\s*',
            re.IGNORECASE | re.MULTILINE
        )

    def segment_answers(self, ocr_text: str, teacher_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits OCR text by question headers and maps to teacher question IDs.
        """
        segmented = []
        
        # Split by page if present
        pages = re.split(r'--- Page (\d+) ---', ocr_text)
        
        # If no page markers (e.g. standard OCR on single image), wrap it
        if len(pages) == 1:
            pages = ["", "1", ocr_text]

        current_q_num = None
        current_sub_q = None
        current_text = []
        current_page = 1
        
        def save_segment():
            nonlocal current_text, current_q_num, current_sub_q, current_page
            if current_text or current_q_num:
                text_content = "\n".join(current_text).strip()
                if text_content or current_q_num:
                    segmented.append({
                        "detected_question_number": str(current_q_num) if current_q_num else None,
                        "sub_question_identifier": str(current_sub_q).lower() if current_sub_q else None,
                        "extracted_answer": text_content,
                        "page_reference": current_page,
                        "teacher_question_id": None,
                        "mapping_status": "unmapped"
                    })
            current_text = []
        
        for i in range(1, len(pages), 2):
            try:
                page_num = int(pages[i])
            except ValueError:
                page_num = current_page
            
            page_content = pages[i+1]
            lines = page_content.split('\n')
            
            for line in lines:
                match = self.question_pattern.match(line)
                if match:
                    save_segment()
                    current_page = page_num
                    current_q_num = match.group(1)
                    current_sub_q = match.group(2) if match.group(2) else None
                    remainder = line[match.end():].strip()
                    if remainder:
                        current_text.append(remainder)
                else:
                    if line.strip():
                        current_text.append(line)
        
        save_segment()
        
        # Mapping logic
        for seg in segmented:
            q_num = seg["detected_question_number"]
            sub_q = seg["sub_question_identifier"]
            
            if q_num:
                # Find matching teacher question
                matched = False
                for tq in teacher_questions:
                    tq_num = str(tq.get("question_number", ""))
                    tq_sub = tq.get("sub_question_number")
                    
                    # Direct match
                    if tq_num == q_num and (not tq_sub or str(tq_sub).lower() == sub_q):
                        seg["teacher_question_id"] = tq["id"]
                        seg["mapping_status"] = "mapped"
                        matched = True
                        break
                    
                    # Sometimes teacher Q is "1(a)" and parsed as question_number="1(a)"
                    expected_combined = f"{q_num}({sub_q})" if sub_q else q_num
                    if tq_num == expected_combined:
                        seg["teacher_question_id"] = tq["id"]
                        seg["mapping_status"] = "mapped"
                        matched = True
                        break
                        
                if not matched:
                    seg["mapping_status"] = "review_required"

        return segmented

student_segmenter_service = StudentAnswerSegmenter()
