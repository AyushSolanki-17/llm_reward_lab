import unittest

from server.simulation import DriftSimulator, DriftType


class TestSimulation(unittest.TestCase):
    def test_deterministic(self):
        """Same seed must always produce same pool."""
        s1 = DriftSimulator(seed=42).generate_pool(100, [DriftType.DATA_CONTAMINATION])
        s2 = DriftSimulator(seed=42).generate_pool(100, [DriftType.DATA_CONTAMINATION])
        self.assertEqual([s.final_quality for s in s1], [s.final_quality for s in s2])

    def test_drift_actually_lowers_quality(self):
        """Drifted pool must have lower mean quality than clean pool."""
        clean = DriftSimulator(seed=42).generate_pool(200, active_drifts=[])
        drifted = DriftSimulator(seed=42).generate_pool(
            200,
            [DriftType.DATA_CONTAMINATION],
        )
        clean_mean = sum(s.final_quality for s in clean) / len(clean)
        drifted_mean = sum(s.final_quality for s in drifted) / len(drifted)
        self.assertLess(drifted_mean, clean_mean)

    def test_drift_affects_correct_task_type(self):
        """DATA_CONTAMINATION should only hurt summarization, not coding."""
        pool = DriftSimulator(seed=42).generate_pool(500, [DriftType.DATA_CONTAMINATION])
        summarization = [s for s in pool if s.task_type == "summarization"]
        coding = [s for s in pool if s.task_type == "coding"]
        summ_mean = sum(s.final_quality for s in summarization) / len(summarization)
        coding_mean = sum(s.final_quality for s in coding) / len(coding)
        self.assertLess(summ_mean, coding_mean)

    def test_quantization_affects_long_inputs_only(self):
        """QUANTIZATION_APPLIED should only hurt long coding/qa inputs."""
        pool = DriftSimulator(seed=42).generate_pool(500, [DriftType.QUANTIZATION_APPLIED])
        long_coding = [
            s
            for s in pool
            if s.task_type == "coding" and s.input_length == "long"
        ]
        short_coding = [
            s
            for s in pool
            if s.task_type == "coding" and s.input_length == "short"
        ]
        long_mean = sum(s.final_quality for s in long_coding) / len(long_coding)
        short_mean = sum(s.final_quality for s in short_coding) / len(short_coding)
        self.assertLess(long_mean, short_mean)


if __name__ == "__main__":
    unittest.main()
