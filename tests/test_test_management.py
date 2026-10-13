import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app, db

client = TestClient(app)

def test_get_all_tests_with_none_evaluation():
    """Verify get_all_tests does not crash when submission evaluation is None."""
    test_db = {
        "tests": [
            {
                "id": "t_active_1",
                "title": "Active Test 1",
                "subject": "CS",
                "max_marks": 20.0,
                "due_date": "2026-11-01",
                "status": "active",
                "is_published": True
            },
            {
                "id": "t_archived_1",
                "title": "Archived Test 1",
                "subject": "CS",
                "max_marks": 10.0,
                "due_date": "2026-10-01",
                "status": "archived",
                "is_published": True
            }
        ],
        "submissions": [
            {
                "id": "sub_none_eval",
                "test_id": "t_active_1",
                "evaluation": None
            },
            {
                "id": "sub_valid_eval",
                "test_id": "t_active_1",
                "evaluation": {
                    "marks_obtained": 18.0,
                    "max_marks": 20.0,
                    "status": "Evaluated"
                }
            }
        ]
    }
    with patch("app.main.db", test_db):
        res = client.get("/api/tests")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["tests"]) == 2
        
        t1 = next(t for t in data["tests"] if t["id"] == "t_active_1")
        assert t1["submissions_count"] == 2
        assert t1["average_score"] == 18.0  # calculated only on evaluated submission

def test_get_active_tests_filtering():
    """Verify /api/tests/active only returns active, published tests."""
    test_db = {
        "tests": [
            {
                "id": "t_active",
                "title": "Active Test",
                "subject": "Math",
                "max_marks": 10.0,
                "due_date": "2026-12-01",
                "status": "active",
                "is_published": True
            },
            {
                "id": "t_archived",
                "title": "Archived Test",
                "subject": "Math",
                "max_marks": 10.0,
                "due_date": "2026-10-01",
                "status": "archived",
                "is_published": True
            },
            {
                "id": "t_unpublished",
                "title": "Draft Test",
                "subject": "Math",
                "max_marks": 10.0,
                "due_date": "2026-12-01",
                "status": "active",
                "is_published": False
            }
        ],
        "submissions": []
    }
    with patch("app.main.db", test_db):
        res = client.get("/api/tests/active")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["tests"]) == 1
        assert data["tests"][0]["id"] == "t_active"

def test_create_new_test_success_and_duplicate():
    """Verify test creation succeeds and duplicate test ID is rejected with 400."""
    test_db = {
        "tests": [
            {"id": "existing_test", "title": "Existing", "subject": "CS"}
        ],
        "submissions": []
    }
    with patch("app.main.db", test_db), \
         patch("app.main.save_db") as mock_save, \
         patch("app.main.teacher_ingestion_service.ingest_test_materials") as mock_ingest:
        
        mock_ingest.return_value = {"chunks_created": 3}

        # Attempt duplicate ID
        dup_payload = {
            "id": "existing_test",
            "title": "New Title",
            "subject": "CS",
            "max_marks": 10.0,
            "due_date": "2026-11-01",
            "question": "Q1",
            "model_answer": "A1",
            "rubric": ""
        }
        res_dup = client.post("/api/tests", json=dup_payload)
        assert res_dup.status_code == 400
        assert "already exists" in res_dup.json()["detail"]

        # Attempt new unique ID
        new_payload = {
            "id": "unique_new_test",
            "title": "Unique New Test",
            "subject": "Physics",
            "max_marks": 25.0,
            "due_date": "2026-11-20",
            "question": "1. What is Newton's second law?",
            "model_answer": "1. F = ma.",
            "rubric": "Formula: 25 marks."
        }
        res_new = client.post("/api/tests", json=new_payload)
        assert res_new.status_code == 200
        new_data = res_new.json()
        assert new_data["success"] is True
        created = new_data["test"]
        assert created["id"] == "unique_new_test"
        assert created["status"] == "active"
        assert created["is_published"] is True
        assert created["knowledge_status"] == "ready"
        assert mock_save.called
