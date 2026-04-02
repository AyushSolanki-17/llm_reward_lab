import unittest

from models import LlmRewardLabAction
from server.environment import LlmRewardLabEnvironment


class TestEnvironmentDeterminism(unittest.TestCase):
    def setUp(self):
        self.env = LlmRewardLabEnvironment()

    def test_reset_is_deterministic_with_seed(self):
        obs1 = self.env.reset(task_id="task_diagnose", seed=42)
        stats1 = obs1.quality_stats

        obs2 = self.env.reset(task_id="task_diagnose", seed=42)
        stats2 = obs2.quality_stats

        self.assertEqual(stats1, stats2)
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
        self.assertGreaterEqual(float(obs.reward or 0.0), 0.99)

    def test_budget_reduces_on_hypothesis_test(self):
        self.env.reset(task_id="task_diagnose", seed=42)
        initial_budget = self.env.state.step_count  # step_count check for progression too
        self.assertEqual(initial_budget, 0)

        obs = self.env.step(
            LlmRewardLabAction(
                action_type="test_hypothesis",
                parameters={"hypothesis_id": "quantization_applied"},
            )
        )
        self.assertEqual(obs.budget_remaining, 48)  # 60 - 12
        self.assertEqual(self.env.state.step_count, 1)
        self.assertIn("quantization_applied", obs.tested_hypotheses)


if __name__ == "__main__":
    unittest.main()
