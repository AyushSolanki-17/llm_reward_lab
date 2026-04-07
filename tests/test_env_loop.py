import unittest

from models import LlmRewardLabAction
from server.environment import LlmRewardLabEnvironment


class TestEnvLoop(unittest.TestCase):
    def test_full_episode(self):
        env = LlmRewardLabEnvironment()

        obs = env.reset(task_id="task_detect_localize", seed=42)
        self.assertEqual(obs.task_id, "task_detect_localize")
        self.assertGreater(obs.budget_remaining, 0)
        self.assertEqual(len(obs.quality_stats), 0)  # stats hidden until inspect
        self.assertFalse(obs.done)

        obs = env.step(
            LlmRewardLabAction(
                action_type="inspect_samples",
                parameters={"task_type": "summarization", "limit": 10},
            )
        )
        self.assertLessEqual(len(obs.samples), 10)
        self.assertTrue(all(s.task_type == "summarization" for s in obs.samples))

        obs = env.step(
            LlmRewardLabAction(
                action_type="submit_diagnosis",
                parameters={
                    "drift_events": ["data_contamination"],
                    "remediations": ["rollback_finetune_checkpoint"],
                },
            )
        )
        self.assertTrue(obs.done)
        self.assertGreaterEqual(float(obs.reward or 0.0), 0.0)
        self.assertLessEqual(float(obs.reward or 0.0), 1.0)

        obs = env.step(
            LlmRewardLabAction(action_type="inspect_samples", parameters={}),
        )
        self.assertTrue(obs.done)

    def test_all_three_tasks(self):
        env = LlmRewardLabEnvironment()
        for task_id in ["task_detect_localize", "task_diagnose", "task_multi_drift"]:
            obs = env.reset(task_id=task_id, seed=42)
            self.assertEqual(obs.task_id, task_id)
            obs = env.step(
                LlmRewardLabAction(
                    action_type="submit_diagnosis",
                    parameters={"drift_events": [], "remediations": []},
                )
            )
            self.assertTrue(obs.done)
            self.assertGreaterEqual(float(obs.reward or 0.0), 0.0)
            self.assertLessEqual(float(obs.reward or 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
