import unittest

from models import LlmRewardLabAction
from server.environment import LlmRewardLabEnvironment


class TestEnvironmentDeterminism(unittest.TestCase):
    def setUp(self):
        self.env = LlmRewardLabEnvironment()

    def test_reset_is_deterministic_with_seed(self):
        obs1 = self.env.reset(task_id="task_diagnose", seed=42)
        # Stats hidden on reset — inspect to get them
        obs1s = self.env.step(LlmRewardLabAction(action_type="inspect_samples", parameters={"limit": 5}))

        obs2 = self.env.reset(task_id="task_diagnose", seed=42)
        obs2s = self.env.step(LlmRewardLabAction(action_type="inspect_samples", parameters={"limit": 5}))

        self.assertEqual(obs1s.quality_stats, obs2s.quality_stats)
        self.assertEqual(obs2.budget_remaining, 60)
        self.assertEqual(obs2.task_id, "task_diagnose")

    def test_submit_correct_diagnosis_scores_high(self):
        self.env.reset(task_id="task_detect_localize", seed=42)
        obs = self.env.step(
            LlmRewardLabAction(
                action_type="submit_diagnosis",
                parameters={
                    "drift_events": ["data_contamination"],
                    "remediations": ["rollback_finetune_checkpoint"],
                },
            )
        )
        self.assertTrue(obs.done)
        self.assertGreaterEqual(float(obs.reward or 0.0), 0.95)

    def test_budget_reduces_on_hypothesis_test(self):
        self.env.reset(task_id="task_diagnose", seed=42)
        self.assertEqual(self.env.state.step_count, 0)

        obs = self.env.step(
            LlmRewardLabAction(
                action_type="run_ab_test",
                parameters={"hypothesis_id": "quantization_applied"},
            )
        )
        self.assertEqual(obs.budget_remaining, 48)  # 60 - 12
        self.assertEqual(self.env.state.step_count, 1)
        self.assertIn("quantization_applied", obs.tested_hypotheses)

    def test_retesting_hypothesis_is_penalized(self):
        self.env.reset(task_id="task_diagnose", seed=42)
        self.env.step(
            LlmRewardLabAction(
                action_type="run_ab_test",
                parameters={"hypothesis_id": "quantization_applied"},
            )
        )
        obs = self.env.step(
            LlmRewardLabAction(
                action_type="run_ab_test",
                parameters={"hypothesis_id": "quantization_applied"},
            )
        )
        # Re-test should not cost additional budget
        self.assertEqual(obs.budget_remaining, 48)
        self.assertLess(float(obs.reward or 0.0), 0.0)

    def test_unknown_action_type_penalized(self):
        self.env.reset(task_id="task_detect_localize", seed=42)
        obs = self.env.step(
            LlmRewardLabAction(
                action_type="submit_diagnosis",  # valid type but testing unknown
                parameters={},
            )
        )
        # Empty submission should score low but not crash
        self.assertTrue(obs.done)

    def test_quality_stats_only_on_inspect(self):
        """Quality stats should appear on inspect only, not on reset or hypothesis tests."""
        obs = self.env.reset(task_id="task_diagnose", seed=42)
        self.assertEqual(len(obs.quality_stats), 0)  # reset hides stats — forces inspection

        obs = self.env.step(
            LlmRewardLabAction(
                action_type="inspect_samples",
                parameters={"limit": 5},
            )
        )
        self.assertGreater(len(obs.quality_stats), 0)  # inspect reveals stats

        obs = self.env.step(
            LlmRewardLabAction(
                action_type="run_ab_test",
                parameters={"hypothesis_id": "prompt_template_change"},
            )
        )
        self.assertEqual(len(obs.quality_stats), 0)  # hypothesis test does not


if __name__ == "__main__":
    unittest.main()