import unittest

from server.graders import grade_submission
from server.tasks import TASK_REGISTRY, list_tasks


class TestGradersAndTasks(unittest.TestCase):
    def test_task_registry_has_three_tasks(self):
        self.assertGreaterEqual(len(TASK_REGISTRY), 3)
        ids = {task["id"] for task in TASK_REGISTRY.values()}
        self.assertEqual(
            ids,
            {"task_detect_localize", "task_diagnose", "task_multi_drift"},
        )

    def test_list_tasks_contains_required_fields(self):
        tasks = list_tasks()
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("difficulty", task)
            self.assertIn("budget", task)
            self.assertIn("max_steps", task)

    def test_grader_returns_normalized_score(self):
        cfg = TASK_REGISTRY["task_multi_drift"]
        score = grade_submission(
            task_id="task_multi_drift",
            submitted_drifts=["prompt_template_change", "safety_filter_misconfig"],
            submitted_remediations=[
                "rollback_prompt_template",
                "recalibrate_safety_filter",
            ],
            active_drifts=cfg["active_drifts"],
            budget_used=15,
            budget_total=cfg["budget"],
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.9)

    def test_grader_penalizes_false_positives(self):
        cfg = TASK_REGISTRY["task_diagnose"]
        good = grade_submission(
            task_id="task_diagnose",
            submitted_drifts=["quantization_applied"],
            submitted_remediations=["revert_quantization"],
            active_drifts=cfg["active_drifts"],
            budget_used=20,
            budget_total=cfg["budget"],
        )
        noisy = grade_submission(
            task_id="task_diagnose",
            submitted_drifts=["quantization_applied", "prompt_template_change"],
            submitted_remediations=[
                "revert_quantization",
                "rollback_prompt_template",
            ],
            active_drifts=cfg["active_drifts"],
            budget_used=20,
            budget_total=cfg["budget"],
        )
        self.assertGreater(good, noisy)


if __name__ == "__main__":
    unittest.main()
