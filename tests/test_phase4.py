import pytest
from unittest.mock import patch, MagicMock
from app.services.student_segmenter import StudentAnswerSegmenter
from app.services.evaluator import EvaluatorService
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_segmenter_multiple_questions_and_continuations():
    segmenter = StudentAnswerSegmenter()
    teacher_questions = [
        {"id": "Q1", "question_text": "What is AI?", "question_number": "1"},
        {"id": "Q2a", "question_text": "Explain Machine Learning.", "question_number": "2", "sub_question_number": "a"},
        {"id": "Q2b", "question_text": "Explain Deep Learning.", "question_number": "2", "sub_question_number": "b"}
    ]
    student_text = """
    Q1: AI is artificial intelligence.
    It means making computers think.
    
    Q2(a) Machine learning is a subfield.
    
    Q3. This is an unknown question.
    """
    
    results = segmenter.segment_answers(student_text, teacher_questions)
    
    assert len(results) == 3
    
    q1 = next(r for r in results if r["detected_question_number"] == "1")
    assert q1["mapping_status"] == "mapped"
    assert "making computers think" in q1["extracted_answer"]
    
    q2a = next(r for r in results if r["detected_question_number"] == "2" and r["sub_question_identifier"] == "a")
    assert q2a["mapping_status"] == "mapped"
    assert "subfield" in q2a["extracted_answer"]
    
    q3 = next(r for r in results if r["detected_question_number"] == "3")
    assert q3["mapping_status"] == "review_required"
    assert q3["teacher_question_id"] is None

def test_heuristic_fallback_evaluation():
    evaluator = EvaluatorService()
    student_ans = "Artificial intelligence makes computers think like humans."
    model_ans = "AI involves creating computer systems that can perform tasks requiring human intelligence."
    
    result = evaluator._heuristic_fallback_evaluation(student_ans, model_ans, max_marks=5.0)
    assert result["needs_teacher_review"] is True
    assert result["marks_awarded"] > 0.0
    assert result["marks_awarded"] <= 5.0
    assert result["max_marks"] == 5.0

@patch("app.main.evaluator_service.evaluate_segmented_answer")
@patch("app.main.db")
@patch("app.main.save_db")
def test_evaluation_endpoint_handles_skipped_and_unmapped(mock_save_db, mock_db, mock_evaluate):
    mock_db.__getitem__.side_effect = lambda k: {
        "tests": [
            {
                "id": "t_123",
                "questions": [
                    {"id": "q_1", "max_marks": 5.0, "question_text": "Q1", "model_answer": "A1"},
                    {"id": "q_2", "max_marks": 10.0, "question_text": "Q2", "model_answer": "A2"}
                ]
            }
        ],
        "submissions": [
            {
                "id": "sub_123",
                "test_id": "t_123",
                "segmented_answers": [
                    {
                        "extracted_answer": "My answer for Q1",
                        "mapping_status": "mapped",
                        "teacher_question_id": "q_1"
                    },
                    {
                        "extracted_answer": "",
                        "mapping_status": "mapped",
                        "teacher_question_id": "q_2"
                    },
                    {
                        "extracted_answer": "Random stuff",
                        "mapping_status": "unmapped",
                        "teacher_question_id": None
                    }
                ]
            }
        ]
    }[k]
    
    mock_evaluate.return_value = {
        "marks_awarded": 3.0,
        "max_marks": 5.0,
        "reasoning": "Good.",
        "missing_concepts": [],
        "feedback": "None",
        "needs_teacher_review": False,
        "evaluation_status": "completed"
    }

    response = client.post("/api/submissions/sub_123/evaluate")
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    
    eval_summary = data["evaluation"]
    assert eval_summary["marks_obtained"] == 3.0
    assert eval_summary["max_marks"] == 15.0 # 5 from Q1 + 10 from Q2 (Q3 unmapped so not added to max_marks)
    assert eval_summary["evaluated_count"] == 2 # Q1 and Q2
    assert eval_summary["review_count"] == 1 # Unmapped Q3
    
    segs = data["segmented_answers"]
    
    # Q1 evaluated
    assert segs[0]["evaluation_result"]["marks_awarded"] == 3.0
    assert mock_evaluate.call_count == 1
    
    # Q2 skipped
    assert segs[1]["evaluation_result"]["marks_awarded"] == 0.0
    assert "left the answer blank" in segs[1]["evaluation_result"]["reasoning"]
    
    # Q3 unmapped
    assert segs[2]["evaluation_result"]["needs_teacher_review"] is True
    assert segs[2]["evaluation_result"]["marks_awarded"] == 0.0
