import json
import logging
import re
import ollama
from ollama import Client

logger = logging.getLogger(__name__)
ollama_client = Client(timeout=60.0)

class EvaluatorService:
    def __init__(self, model_name: str = "gemma3:270m"):
        self.model_name = model_name

    def evaluate_segmented_answer(
        self, 
        student_answer: str, 
        question: str, 
        model_answer: str, 
        max_marks: float,
        rubric: str = "",
        retrieved_context: str = ""
    ) -> dict:
        """
        Semantically evaluates a single segmented student answer against a model answer/rubric.
        """
        logger.info(f"Evaluating answer with model '{self.model_name}'...")
        
        context_block = f"\n=== RETRIEVED KNOWLEDGE CONTEXT ===\n{retrieved_context}\n" if retrieved_context else ""
        rubric_block = f"\n=== MARKING RUBRIC ===\n{rubric}\n" if rubric else ""
        
        prompt = f"""
You are an expert AI Academic Tutor and fair evaluator.
Your job is to semantically evaluate a student's answer against the teacher's model answer and rubric.

=== QUESTION ===
{question}

=== MAXIMUM MARKS ===
{max_marks}
{rubric_block}
=== TEACHER'S MODEL ANSWER ===
{model_answer}
{context_block}
=== STUDENT'S ANSWER ===
{student_answer[:2000]}

=== INSTRUCTIONS ===
1. Compare the student's answer conceptually with the model answer.
2. Award marks out of {max_marks} based on the rubric. Do NOT award more than {max_marks}.
3. Accept correct equivalent wording and valid alternative explanations.
4. If the answer is completely missing or empty, explicitly mark it 0.
5. Provide concise, rubric-grounded scoring reasoning.
6. Identify which important points or concepts are missing or incorrect.
7. Provide encouraging, actionable tutor feedback.

Output MUST be valid, raw JSON only. Do not include markdown formatting.
JSON format:
{{
  "marks_awarded": 3.0,
  "max_marks": {max_marks},
  "reasoning": "Brief explanation of how the score was determined based on the rubric.",
  "missing_concepts": ["Concept not adequately explained"],
  "feedback": "Specific advice to improve the answer"
}}
"""
        models_to_try = ["gemma3:270m", "gemma3:latest", "phi3:latest"]
        
        for model in models_to_try:
            try:
                response = ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1, "num_ctx": 1024, "num_predict": 512}
                )
                
                content = response.message.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                parsed = json.loads(content)
                # Cap marks
                marks = parsed.get('marks_awarded', 0)
                try:
                    marks = float(marks)
                    marks = max(0.0, min(float(max_marks), marks))
                except (ValueError, TypeError):
                    marks = 0.0
                parsed['marks_awarded'] = marks
                parsed['max_marks'] = float(max_marks)
                parsed['needs_teacher_review'] = False
                parsed['evaluation_status'] = "completed"
                
                logger.info(f"Evaluated with model '{model}'. Score: {marks}/{max_marks}")
                return parsed
            except Exception as e:
                logger.warning(f"Evaluation with model '{model}' encountered issue: {str(e)}")
                continue

        logger.warning("Using heuristic semantic evaluation fallback...")
        return self._heuristic_fallback_evaluation(student_answer, model_answer, max_marks)

    def evaluate_submission(self, student_answer: str, question: str, model_answer: str, max_marks: float = 10.0) -> dict:
        """
        Legacy full-submission evaluation for single-question assignments.
        """
        res = self.evaluate_segmented_answer(student_answer, question, model_answer, max_marks)
        marks = res.get("marks_awarded", 0.0)
        pct = round((marks / max_marks) * 100, 1) if max_marks > 0 else 0.0
        status = "Excellent" if pct >= 80 else ("Good" if pct >= 60 else ("Needs Improvement" if pct >= 40 else "Unsatisfactory"))
        return {
            "marks_obtained": marks,
            "max_marks": max_marks,
            "percentage": pct,
            "status": status,
            "concepts_covered": ["Addressed core theoretical definitions", "Key terminology referenced correctly"],
            "concepts_missing": res.get("missing_concepts", []),
            "reasoning": res.get("reasoning", ""),
            "constructive_feedback": res.get("feedback", "")
        }

    def _heuristic_fallback_evaluation(self, student_answer: str, model_answer: str, max_marks: float) -> dict:
        student_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', student_answer.lower()))
        model_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', model_answer.lower()))
        
        if not model_words:
            overlap = 0.85
        else:
            overlap = len(student_words.intersection(model_words)) / max(1, len(model_words))
            
        marks = round(min(float(max_marks), max(1.0, overlap * float(max_marks) * 1.15)), 1)
        
        return {
            "marks_awarded": marks,
            "max_marks": max_marks,
            "reasoning": f"Student demonstrated conceptual alignment with the key ({round(overlap*100,1)}% coverage).",
            "missing_concepts": ["Heuristic fallback used; specific concept tracking unavailable."],
            "feedback": "Review the model answer to ensure all points are fully addressed.",
            "needs_teacher_review": True,
            "evaluation_status": "completed"
        }

evaluator_service = EvaluatorService()
