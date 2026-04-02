import unittest

from fastapi.testclient import TestClient

from server.app import app


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_tasks_endpoint(self):
        response = self.client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("tasks", payload)
        self.assertIn("action_schema", payload)
        self.assertGreaterEqual(len(payload["tasks"]), 3)

    def test_grader_endpoint(self):
        response = self.client.post(
            "/grader",
            json={
                "task_id": "task_diagnose",
                "drift_events": ["quantization_applied"],
                "remediations": ["revert_quantization"],
                "budget_used": 20,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("score", payload)
        self.assertGreaterEqual(payload["score"], 0.0)
        self.assertLessEqual(payload["score"], 1.0)

    def test_baseline_endpoint(self):
        response = self.client.post("/baseline")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("scores", payload)
        self.assertIn("mean_score", payload)
        self.assertEqual(set(payload["scores"].keys()), {
            "task_detect_localize",
            "task_diagnose",
            "task_multi_drift",
        })


if __name__ == "__main__":
    unittest.main()
