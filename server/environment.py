from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    Hypothesis,
    LlmRewardLabAction,
    LlmRewardLabObservation,
    OutputSample,
)
from server.graders import grade_submission
from server.simulation import DriftSimulator, SimulatedOutput
from server.tasks import TASK_REGISTRY


class LlmRewardLabEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._seed = 42
        self._simulator = DriftSimulator(seed=self._seed)

        self._pool: list[SimulatedOutput] = []
        self._active_drifts = []
        self._budget_total = 0
        self._budget_remaining = 0
        self._task_cfg: dict | None = None
        self._tested_hypotheses: dict[str, str] = {}
        self._done = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> LlmRewardLabObservation:
        task_id = kwargs.get("task_id", "task_detect_localize")
        if task_id not in TASK_REGISTRY:
            task_id = "task_detect_localize"

        self._seed = seed if seed is not None else 42
        self._simulator = DriftSimulator(seed=self._seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._done = False
        self._tested_hypotheses = {}

        self._task_cfg = TASK_REGISTRY[task_id]
        self._active_drifts = self._task_cfg["active_drifts"]
        self._budget_total = self._task_cfg["budget"]
        self._budget_remaining = self._budget_total
        self._pool = self._simulator.generate_pool(
            n=self._task_cfg["pool_size"],
            active_drifts=self._active_drifts,
        )

        return self._build_observation(
            reward=0.0,
            last_action_result=f"Task '{task_id}' started. Inspect quality and diagnose drift.",
        )

    def step(
        self,
        action: LlmRewardLabAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> LlmRewardLabObservation:
        del timeout_s, kwargs

        if self._task_cfg is None:
            return self.reset()

        if self._done:
            return self._build_observation(
                reward=0.0,
                last_action_result="Episode already finished. Call reset().",
            )

        self._state.step_count += 1

        if action.action_type == "inspect_samples":
            return self._handle_inspect(action)
        if action.action_type == "test_hypothesis":
            return self._handle_test_hypothesis(action)
        if action.action_type == "request_probe":
            return self._handle_probe(action)
        return self._handle_submit(action)

    @property
    def state(self) -> State:
        return self._state

    def _handle_inspect(self, action: LlmRewardLabAction) -> LlmRewardLabObservation:
        params = action.parameters
        task_type = params.get("task_type")
        input_length = params.get("input_length")
        limit = int(params.get("limit", 20))
        limit = max(1, min(limit, 50))

        selected = self._filter_samples(task_type=task_type, input_length=input_length)[
            :limit
        ]
        reward = 0.01 if selected else -0.02
        return self._build_observation(
            reward=reward,
            samples=selected,
            last_action_result=f"Returned {len(selected)} samples.",
        )

    def _handle_test_hypothesis(
        self,
        action: LlmRewardLabAction,
    ) -> LlmRewardLabObservation:
        assert self._task_cfg is not None
        hypothesis_id = str(action.parameters.get("hypothesis_id", ""))
        hypotheses = self._task_cfg["hypotheses"]

        if hypothesis_id not in hypotheses:
            return self._build_observation(
                reward=-0.05,
                last_action_result=f"Unknown hypothesis '{hypothesis_id}'.",
            )

        cost = int(hypotheses[hypothesis_id]["cost"])
        if self._budget_remaining < cost:
            return self._build_observation(
                reward=-0.05,
                last_action_result="Insufficient budget.",
            )

        self._budget_remaining -= cost
        is_true = hypothesis_id in {drift.value for drift in self._active_drifts}
        result = "CONFIRMED active drift." if is_true else "Rejected."
        self._tested_hypotheses[hypothesis_id] = result
        reward = 0.05 if is_true else -0.02
        return self._build_observation(
            reward=reward,
            last_action_result=f"Hypothesis {hypothesis_id}: {result}",
        )

    def _handle_probe(self, action: LlmRewardLabAction) -> LlmRewardLabObservation:
        count = int(action.parameters.get("count", 6))
        count = max(1, min(count, 20))
        cost = count
        if self._budget_remaining < cost:
            return self._build_observation(
                reward=-0.05,
                last_action_result="Insufficient budget for probe.",
            )

        self._budget_remaining -= cost
        probe_pool = self._simulator.generate_pool(
            n=max(30, count * 3),
            active_drifts=self._active_drifts,
        )
        filtered = [
            sample
            for sample in probe_pool
            if sample.task_type
            == action.parameters.get("task_type", sample.task_type)
            and sample.input_length
            == action.parameters.get("input_length", sample.input_length)
        ][:count]

        return self._build_observation(
            reward=0.02 if filtered else -0.02,
            samples=filtered,
            last_action_result=f"Probe returned {len(filtered)} samples.",
        )

    def _handle_submit(self, action: LlmRewardLabAction) -> LlmRewardLabObservation:
        assert self._task_cfg is not None
        params = action.parameters
        score = grade_submission(
            task_id=self._task_cfg["id"],
            submitted_drifts=list(params.get("drift_events", [])),
            submitted_remediations=list(params.get("remediations", [])),
            active_drifts=self._active_drifts,
            budget_used=self._budget_total - self._budget_remaining,
            budget_total=self._budget_total,
        )
        self._done = True
        return self._build_observation(
            reward=score,
            last_action_result=f"Diagnosis submitted. Final score={score:.4f}",
        )

    def _build_observation(
        self,
        reward: float,
        last_action_result: str,
        samples: list[SimulatedOutput] | None = None,
    ) -> LlmRewardLabObservation:
        assert self._task_cfg is not None

        max_steps = int(self._task_cfg["max_steps"])
        if not self._done and self._state.step_count >= max_steps:
            self._done = True
            reward = min(reward, 0.0)
            last_action_result = "Max step limit reached. Submit earlier."

        stats: dict[str, dict] = defaultdict(
            lambda: {"scores": [], "sample_count": 0.0},
        )
        for sample in self._pool:
            bucket = stats[sample.task_type]
            bucket["scores"].append(sample.final_quality)
            bucket["sample_count"] += 1.0

        quality_stats = {}
        for task_type, bucket in stats.items():
            scores = bucket["scores"]
            mean = sum(scores) / len(scores)
            variance = sum((value - mean) ** 2 for value in scores) / len(scores)
            quality_stats[task_type] = {
                "mean": round(mean, 4),
                "std": round(variance**0.5, 4),
                "sample_count": bucket["sample_count"],
            }

        hypotheses = [
            Hypothesis(
                hypothesis_id=h_id,
                description=h_cfg["description"],
                cost=h_cfg["cost"],
            )
            for h_id, h_cfg in self._task_cfg["hypotheses"].items()
        ]

        return LlmRewardLabObservation(
            done=self._done,
            reward=round(float(reward), 4),
            metadata={"episode_id": self._state.episode_id},
            samples=[self._to_output_sample(sample) for sample in (samples or [])],
            budget_remaining=self._budget_remaining,
            budget_total=self._budget_total,
            available_hypotheses=hypotheses,
            tested_hypotheses=dict(self._tested_hypotheses),
            quality_stats=quality_stats,
            step_count=self._state.step_count,
            task_id=self._task_cfg["id"],
            last_action_result=last_action_result,
        )

    def _filter_samples(
        self,
        task_type: str | None = None,
        input_length: str | None = None,
    ) -> list[SimulatedOutput]:
        filtered = self._pool
        if task_type:
            filtered = [sample for sample in filtered if sample.task_type == task_type]
        if input_length:
            filtered = [
                sample for sample in filtered if sample.input_length == input_length
            ]
        return filtered

    def _to_output_sample(self, sample: SimulatedOutput) -> OutputSample:
        return OutputSample(
            sample_id=sample.sample_id,
            task_type=sample.task_type,
            input_length=sample.input_length,
            quality_score=sample.final_quality,
            timestamp=self._state.step_count,
            metadata={"drift_applied": sample.drift_applied},
        )
