import unittest

from server.graders import grade_submission
from server.simulation import DriftType


class TestGraders(unittest.TestCase):
    def test_perfect_submission_scores_high(self):
        score = grade_submission(
            task_id="task_detect_localize",
            submitted_drifts=["data_contamination"],
            submitted_remediations=["rollback_finetune_checkpoint"],
            active_drifts=[DriftType.DATA_CONTAMINATION],
            budget_used=0,
            budget_total=999,
        )
        self.assertGreaterEqual(score, 0.95)

    def test_wrong_diagnosis_scores_low(self):
        score = grade_submission(
            task_id="task_detect_localize",
            submitted_drifts=["quantization_applied"],
            submitted_remediations=["revert_quantization"],
            active_drifts=[DriftType.DATA_CONTAMINATION],
            budget_used=0,
            budget_total=999,
        )
        self.assertLess(score, 0.15)

    def test_scores_always_between_0_and_1(self):
        for task_id in ["task_detect_localize", "task_diagnose", "task_multi_drift"]:
            score = grade_submission(
                task_id=task_id,
                submitted_drifts=["garbage_drift_xyz", "another_fake"],
                submitted_remediations=["fake_remediation"],
                active_drifts=[DriftType.QUANTIZATION_APPLIED],
                budget_used=100,
                budget_total=60,
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_false_positives_penalized(self):
        score_clean = grade_submission(
            task_id="task_detect_localize",
            submitted_drifts=["data_contamination"],
            submitted_remediations=["rollback_finetune_checkpoint"],
            active_drifts=[DriftType.DATA_CONTAMINATION],
            budget_used=0,
            budget_total=999,
        )
        score_fp = grade_submission(
            task_id="task_detect_localize",
            submitted_drifts=["data_contamination", "quantization_applied"],
            submitted_remediations=["rollback_finetune_checkpoint"],
            active_drifts=[DriftType.DATA_CONTAMINATION],
            budget_used=0,
            budget_total=999,
        )
        self.assertGreater(score_clean, score_fp)

    def test_task3_requires_both_drifts(self):
        score_both = grade_submission(
            task_id="task_multi_drift",
            submitted_drifts=["prompt_template_change", "safety_filter_misconfig"],
            submitted_remediations=["rollback_prompt_template", "recalibrate_safety_filter"],
            active_drifts=[DriftType.PROMPT_TEMPLATE_CHANGE, DriftType.SAFETY_FILTER_MISCONFIG],
            budget_used=30,
            budget_total=50,
        )
        score_one = grade_submission(
            task_id="task_multi_drift",
            submitted_drifts=["prompt_template_change"],
            submitted_remediations=["rollback_prompt_template"],
            active_drifts=[DriftType.PROMPT_TEMPLATE_CHANGE, DriftType.SAFETY_FILTER_MISCONFIG],
            budget_used=30,
            budget_total=50,
        )
        self.assertGreater(score_both, score_one)
        self.assertGreater(score_one, 0.0)

    def test_spurious_remediations_penalized(self):
        score_clean = grade_submission(
            task_id="task_diagnose",
            submitted_drifts=["quantization_applied"],
            submitted_remediations=["revert_quantization"],
            active_drifts=[DriftType.QUANTIZATION_APPLIED],
            budget_used=20,
            budget_total=60,
        )
        score_noisy = grade_submission(
            task_id="task_diagnose",
            submitted_drifts=["quantization_applied"],
            submitted_remediations=["revert_quantization", "rollback_prompt_template", "scale_serving_infra"],
            active_drifts=[DriftType.QUANTIZATION_APPLIED],
            budget_used=20,
            budget_total=60,
        )
        self.assertGreater(score_clean, score_noisy)

    def test_empty_submission_scores_zero(self):
        score = grade_submission(
            task_id="task_detect_localize",
            submitted_drifts=[],
            submitted_remediations=[],
            active_drifts=[DriftType.DATA_CONTAMINATION],
            budget_used=0,
            budget_total=20,
        )
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()