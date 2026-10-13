import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Common section headers: Section A, Part 1, Group B, Section - I
SECTION_PATTERN = re.compile(
    r'^(?:SECTION|PART|GROUP)\s*[-:–]?\s*([A-Za-z0-9IVXLCDM]+)',
    re.IGNORECASE
)

# Question headers: Q1, Q.1, Q 1, Question 1, 1., 1), Question 1:
QUESTION_HEADER_PATTERN = re.compile(
    r'^(?:(?:Q(?:uestion)?\.?\s*(\d+[a-zA-Z]?)\.?)|(\d+)\.|\((\d+)\)|(\d+)\))\s*[-:–]?\s*',
    re.IGNORECASE
)

# Sub-question headers within or after question: 1(a), (a), a), a., (i), i)
SUBQUESTION_PATTERN = re.compile(
    r'^(?:\(?([a-z]|[ivxlcdm]+)\)|\b([a-z])\.)\s*[-:–]?\s*',
    re.IGNORECASE
)

# Combined question+subquestion like "1(a)" or "Q1.b" (explicit parenthesis or single letter after dot)
COMBINED_Q_SUB_PATTERN = re.compile(
    r'^(?:Q(?:uestion)?\.?\s*)?(?:(\d+)\s*\(([a-zA-Z]|[ivxlcdm]+)\)|(\d+)\.([a-zA-Z])\b)\s*[-:–]?\s*',
    re.IGNORECASE
)

# Marks patterns: [5], (5 Marks), 5 marks, 5M, (10)
MARKS_PATTERNS = [
    re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?|m)?\s*\]', re.IGNORECASE),
    re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?|m)\s*\)', re.IGNORECASE),
    re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)\b', re.IGNORECASE),
    re.compile(r'\b(\d+(?:\.\d+)?)\s*[mM]\b'),
    re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*\)\s*$'),  # (5) at end of string
]

def extract_marks_from_text(text: str) -> Tuple[Optional[float], str]:
    """
    Extracts numerical max_marks from text and returns (marks, cleaned_text).
    If no marks detected, returns (None, original_text).
    """
    cleaned = text
    detected_marks = None
    
    for pat in MARKS_PATTERNS:
        match = pat.search(cleaned)
        if match:
            try:
                detected_marks = float(match.group(1))
                if detected_marks.is_integer():
                    detected_marks = int(detected_marks)
                # Remove the marks token from cleaned text
                cleaned = pat.sub('', cleaned).strip()
                break
            except (ValueError, TypeError):
                continue
                
    return detected_marks, cleaned

def normalize_q_num(raw_num: str) -> str:
    """Normalizes question identifiers (e.g., 'Q. 1(a)' -> '1(a)', '01' -> '1')."""
    if not raw_num:
        return ""
    clean = raw_num.strip().lower()
    clean = re.sub(r'^(?:q|question)\.?\s*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[\s\.\:\–\-]+', '', clean)
    clean = clean.lstrip('0') or '0'
    return clean

class QuestionPaperParser:
    """
    Deterministic rule-based parser for Question Papers.
    Preserves hierarchical structure (parent questions & sub-questions),
    sections, and marks. Never silently discards content.
    """

    def parse(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            logger.warning("[PARSER] Empty question paper text provided.")
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        questions: List[Dict[str, Any]] = []
        current_section: Optional[str] = None
        current_parent_q: Optional[Dict[str, Any]] = None
        current_item: Optional[Dict[str, Any]] = None

        for line in lines:
            # Check for section header
            sec_match = SECTION_PATTERN.match(line)
            if sec_match:
                current_section = sec_match.group(1).upper()
                logger.debug(f"[PARSER] Detected section: {current_section}")
                continue

            # Check for combined pattern like "1(a)" or "Q1(a)"
            comb_match = COMBINED_Q_SUB_PATTERN.match(line)
            if comb_match:
                parent_num = comb_match.group(1) or comb_match.group(3)
                sub_num = (comb_match.group(2) or comb_match.group(4)).lower()
                q_num = f"{parent_num}({sub_num})"
                content = line[comb_match.end():].strip().lstrip('.:-– ')
                marks, cleaned_content = extract_marks_from_text(content)

                # Ensure parent question exists or create placeholder
                parent_id = f"q_{parent_num}"
                if not any(q["id"] == parent_id for q in questions):
                    parent_q = {
                        "id": parent_id,
                        "question_number": parent_num,
                        "parent_question_id": None,
                        "sub_question_number": None,
                        "section": current_section,
                        "question_text": f"Question {parent_num}",
                        "max_marks": None,
                        "is_parent": True
                    }
                    questions.append(parent_q)

                sub_id = f"q_{parent_num}_{sub_num}"
                current_item = {
                    "id": sub_id,
                    "question_number": q_num,
                    "parent_question_id": parent_id,
                    "sub_question_number": sub_num,
                    "section": current_section,
                    "question_text": cleaned_content or content,
                    "max_marks": marks,
                    "is_parent": False
                }
                questions.append(current_item)
                continue

            # Check for top-level question header: Q1, 1., Question 1
            q_match = QUESTION_HEADER_PATTERN.match(line)
            if q_match:
                q_num = (q_match.group(1) or q_match.group(2) or q_match.group(3) or q_match.group(4))
                content = line[q_match.end():].strip().lstrip('.:-– ')
                marks, cleaned_content = extract_marks_from_text(content)

                q_id = f"q_{q_num}"
                current_item = {
                    "id": q_id,
                    "question_number": q_num,
                    "parent_question_id": None,
                    "sub_question_number": None,
                    "section": current_section,
                    "question_text": cleaned_content or content,
                    "max_marks": marks,
                    "is_parent": False
                }
                current_parent_q = current_item
                questions.append(current_item)
                continue

            # Check for sub-question header: a), (a), a., i)
            sub_match = SUBQUESTION_PATTERN.match(line)
            if sub_match and current_parent_q:
                sub_num = (sub_match.group(1) or sub_match.group(2)).lower()
                parent_num = current_parent_q["question_number"]
                q_num = f"{parent_num}({sub_num})"
                content = line[sub_match.end():].strip().lstrip('.:-– ')
                marks, cleaned_content = extract_marks_from_text(content)

                current_parent_q["is_parent"] = True
                sub_id = f"q_{parent_num}_{sub_num}"
                current_item = {
                    "id": sub_id,
                    "question_number": q_num,
                    "parent_question_id": current_parent_q["id"],
                    "sub_question_number": sub_num,
                    "section": current_section,
                    "question_text": cleaned_content or content,
                    "max_marks": marks,
                    "is_parent": False
                }
                questions.append(current_item)
                continue

            # Continuation line for the current item
            if current_item:
                marks, cleaned_line = extract_marks_from_text(line)
                if marks is not None and current_item["max_marks"] is None:
                    current_item["max_marks"] = marks
                    current_item["question_text"] += " " + cleaned_line
                else:
                    current_item["question_text"] += " " + line
            else:
                # Content before any question header (e.g. instructions/title)
                logger.info(f"[PARSER] Preserved leading preamble: '{line[:60]}...'")

        # Fallback: if no questions were identified deterministically, preserve full raw text
        if not questions:
            logger.warning("[PARSER] Could not deterministically segment questions. Preserving raw text as single question.")
            marks, cleaned = extract_marks_from_text(text)
            questions.append({
                "id": "q_1",
                "question_number": "1",
                "parent_question_id": None,
                "sub_question_number": None,
                "section": current_section,
                "question_text": cleaned.strip() or text.strip(),
                "max_marks": marks,
                "is_parent": False,
                "parsing_flag": "unsegmented_raw_text"
            })

        logger.info(f"[PARSER] Parsed {len(questions)} questions from question paper.")
        return questions


class ModelAnswerParser:
    """
    Deterministic parser for Model Answers, Rubrics, and Marking Schemes.
    Segments entries by question identifiers and extracts marking points,
    keywords, and rubrics.
    """

    def parse(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary mapping normalized question_number -> {
            "model_answer": str,
            "rubric": str,
            "marking_points": List[str],
            "step_marks": List[Dict[str, Any]]
        }
        """
        if not text or not text.strip():
            logger.warning("[PARSER] Empty model answer text provided.")
            return {}

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result: Dict[str, Dict[str, Any]] = {}
        current_key: Optional[str] = None
        current_answer_lines: List[str] = []
        current_rubric_lines: List[str] = []
        current_points: List[str] = []

        def commit_current():
            nonlocal current_key, current_answer_lines, current_rubric_lines, current_points
            if current_key:
                ans_text = "\n".join(current_answer_lines).strip()
                rubric_text = "\n".join(current_rubric_lines).strip()
                result[current_key] = {
                    "model_answer": ans_text,
                    "rubric": rubric_text,
                    "marking_points": list(current_points),
                    "step_marks": []
                }
            current_answer_lines = []
            current_rubric_lines = []
            current_points = []

        for line in lines:
            # Check for question key like "1(a)", "Q1", "Answer 1", "Question 1:"
            # or "Q1:" / "1."
            comb_m = COMBINED_Q_SUB_PATTERN.match(line)
            if comb_m:
                commit_current()
                parent_num = comb_m.group(1) or comb_m.group(3)
                sub_num = (comb_m.group(2) or comb_m.group(4)).lower()
                current_key = normalize_q_num(f"{parent_num}({sub_num})")
                rest = line[comb_m.end():].strip()
                if rest:
                    current_answer_lines.append(rest)
                continue

            ans_sub_m = re.match(r'^(?:Answer|Ans|Sol|Solution)\s*[-:–]?\s*(?:(\d+)\s*\(([a-zA-Z]|[ivxlcdm]+)\)|(\d+)\.([a-zA-Z])\b)\s*[-:–]?\s*', line, re.IGNORECASE)
            if ans_sub_m:
                commit_current()
                parent_num = ans_sub_m.group(1) or ans_sub_m.group(3)
                sub_num = (ans_sub_m.group(2) or ans_sub_m.group(4)).lower()
                current_key = normalize_q_num(f"{parent_num}({sub_num})")
                rest = line[ans_sub_m.end():].strip()
                if rest:
                    current_answer_lines.append(rest)
                continue

            q_m = QUESTION_HEADER_PATTERN.match(line)
            ans_m = re.match(r'^(?:Answer|Ans|Sol|Solution)\s*(\d+[a-zA-Z]?|[a-zA-Z])\s*[-:–]?\s*', line, re.IGNORECASE)
            
            if q_m:
                commit_current()
                q_num = q_m.group(1) or q_m.group(2) or q_m.group(3) or q_m.group(4)
                current_key = normalize_q_num(q_num)
                rest = line[q_m.end():].strip()
                if rest:
                    current_answer_lines.append(rest)
                continue
            elif ans_m:
                commit_current()
                q_num = ans_m.group(1)
                current_key = normalize_q_num(q_num)
                rest = line[ans_m.end():].strip()
                if rest:
                    current_answer_lines.append(rest)
                continue

            # Rubric or marking scheme indicators: "Rubric:", "Marking Scheme:", "Marks:", "Keywords:"
            rubric_m = re.match(r'^(?:Rubric|Marking Scheme|Key Points?|Breakdown)\s*[-:–]?\s*(.*)', line, re.IGNORECASE)
            if rubric_m:
                rubric_content = rubric_m.group(1).strip()
                if rubric_content:
                    current_rubric_lines.append(rubric_content)
                continue

            # Marking point indicator: "- Point 1 (2 marks)" or "• Step 1: ..."
            point_m = re.match(r'^[-•*]\s*(.+)', line)
            if point_m:
                pt = point_m.group(1).strip()
                current_points.append(pt)
                current_answer_lines.append(line)
                continue

            # Regular continuation line
            if current_key:
                current_answer_lines.append(line)

        commit_current()

        # If no key patterns matched, associate full text with "all" or "1"
        if not result and lines:
            logger.info("[PARSER] Model answer without explicit question headers; storing as single answer unit.")
            result["1"] = {
                "model_answer": "\n".join(lines).strip(),
                "rubric": "",
                "marking_points": [],
                "step_marks": []
            }

        logger.info(f"[PARSER] Parsed {len(result)} model answer sections.")
        return result


def map_questions_and_answers(
    questions: List[Dict[str, Any]],
    answers_map: Dict[str, Dict[str, Any]],
    general_rubric: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Deterministically associates questions with model answers and rubrics based on normalized identifiers.
    Flags ambiguous or unmatched items cleanly without silently discarding data.
    """
    mapped_questions = []

    for q in questions:
        q_num = q.get("question_number", "")
        norm_key = normalize_q_num(q_num)

        # Try exact normalized key match
        matched_ans = answers_map.get(norm_key)

        # Try without sub-parent if subquestion
        if not matched_ans and "(" in norm_key:
            sub_key = norm_key.replace("(", "").replace(")", "")
            matched_ans = answers_map.get(sub_key)

        # Fallback to single general answer if only one exists
        if not matched_ans and len(answers_map) == 1 and "1" in answers_map and len(questions) == 1:
            matched_ans = answers_map["1"]

        if matched_ans:
            q["model_answer"] = matched_ans.get("model_answer", "")
            q["rubric"] = matched_ans.get("rubric") or general_rubric or ""
            q["marking_points"] = matched_ans.get("marking_points", [])
            q["step_marks"] = matched_ans.get("step_marks", [])
            q["mapping_status"] = "matched"
        else:
            q["model_answer"] = ""
            q["rubric"] = general_rubric or ""
            q["marking_points"] = []
            q["step_marks"] = []
            q["mapping_status"] = "unmatched"
            logger.warning(f"[PARSER] Question '{q_num}' has no matching model answer (flagged as unmatched).")

        mapped_questions.append(q)

    logger.info(f"[PARSER] Question-answer mapping complete. {sum(1 for q in mapped_questions if q['mapping_status'] == 'matched')}/{len(mapped_questions)} matched.")
    return mapped_questions
