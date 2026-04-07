from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import random

from models import (
    BusinessMetrics,
    Hypothesis,
    IncidentContext,
    LlmRewardLabAction,
    LlmRewardLabObservation,
    OutputSample,
)
from server.graders import grade_submission
from server.simulation import DriftSimulator, SimulatedOutput
from server.tasks import TASK_REGISTRY

# Incident context templates keyed by difficulty
_INCIDENT_TEMPLATES = {
    "easy": {
        "severity": "HIGH",
        "started_minutes_ago": 47,
        "affected_users_percent": 18.0,
        "reported_issue": "Users reporting noticeably worse summarization quality.",
    },
    "medium": {
        "severity": "MEDIUM",
        "started_minutes_ago": 126,
        "affected_users_percent": 9.0,
        "reported_issue": "Intermittent quality complaints on longer code-review tasks.",
    },
    "hard": {
        "severity": "CRITICAL",
        "started_minutes_ago": 203,
        "affected_users_percent": 34.0,
        "reported_issue": "Broad quality drop with inconsistent patterns — multiple teams affected.",
    },
}


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
        self._inspected = False
        self._done = False
        self._incident_context: IncidentContext | None = None
        self._business_metrics: BusinessMetrics | None = None
        self._obs_rng = random.Random(self._seed)

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
        self._inspected = False
        self._tested_hypotheses = {}

        self._task_cfg = TASK_REGISTRY[task_id]
        self._active_drifts = self._task_cfg["active_drifts"]
        self._budget_total = self._task_cfg["budget"]
        self._budget_remaining = self._budget_total
        self._pool = self._simulator.generate_pool(
            n=self._task_cfg["pool_size"],
            active_drifts=self._active_drifts,
        )

        # Observation noise RNG (deterministic per seed)
        self._obs_rng = random.Random(self._seed + 7)

        # Generate incident context based on task difficulty
        difficulty = self._task_cfg.get("difficulty", "easy")
        tpl = _INCIDENT_TEMPLATES.get(difficulty, _INCIDENT_TEMPLATES["easy"])
        self._incident_context = IncidentContext(**tpl)

        # Generate business metrics reflecting the severity of active drifts
        drift_severity = len(self._active_drifts) * 0.12
        self._business_metrics = BusinessMetrics(
            user_satisfaction=round(0.85 - drift_severity + self._obs_rng.gauss(0, 0.02), 3),
            error_rate=round(0.05 + drift_severity * 0.8 + self._obs_rng.gauss(0, 0.01), 3),
            cost_per_1k_requests=round(0.028 + self._obs_rng.gauss(0, 0.003), 4),
        )

        return self._build_observation(
            reward=0.0,
            last_action_result=f"Task '{task_id}' started. Inspect samples to see quality breakdown.",
            include_stats=False,
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
        if action.action_type == "run_ab_test":
            return self._handle_ab_test(action)
        if action.action_type == "run_targeted_eval":
            return self._handle_targeted_eval(action)
        if action.action_type == "submit_diagnosis":
            return self._handle_submit(action)

        return self._build_observation(
            reward=-0.05,
            last_action_result=f"Unknown action_type: {action.action_type}",
        )

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

        # First inspection is free; subsequent ones cost 1 budget
        if self._inspected and self._budget_remaining > 0:
            self._budget_remaining -= 1

        self._inspected = True
        # Reward: useful inspection (+0.05) vs empty result (-0.02)
        # Filtered inspections (agent is narrowing down) get a bonus
        has_filter = task_type is not None or input_length is not None
        if selected:
            reward = 0.05 if has_filter else 0.03
        else:
            reward = -0.02
        return self._build_observation(
            reward=reward,
            samples=selected,
            last_action_result=f"Returned {len(selected)} samples (filtered: task_type={task_type}, input_length={input_length}).",
            include_stats=True,
        )

    def _handle_ab_test(
        self,
        action: LlmRewardLabAction,
    ) -> LlmRewardLabObservation:
        assert self._task_cfg is not None
        hypothesis_id = str(action.parameters.get("hypothesis_id", ""))
        hypotheses = self._task_cfg["hypotheses"]

        if hypothesis_id not in hypotheses:
            return self._build_observation(
                reward=-0.05,
                last_action_result=f"Unknown hypothesis '{hypothesis_id}'. Available: {list(hypotheses.keys())}",
            )

        if hypothesis_id in self._tested_hypotheses:
            return self._build_observation(
                reward=-0.03,
                last_action_result=f"Hypothesis '{hypothesis_id}' already tested: {self._tested_hypotheses[hypothesis_id]}",
            )

        cost = int(hypotheses[hypothesis_id]["cost"])
        if self._budget_remaining < cost:
            self._degrade_business_metrics(satisfaction_delta=-0.03, error_delta=0.02)
            return self._build_observation(
                reward=-0.05,
                last_action_result=f"Insufficient budget. Need {cost}, have {self._budget_remaining}. System continues degrading.",
            )

        self._budget_remaining -= cost
        is_true = hypothesis_id in {drift.value for drift in self._active_drifts}
        result = "CONFIRMED active drift." if is_true else "Rejected — no evidence of this drift."
        self._tested_hypotheses[hypothesis_id] = result
        # Stronger signal: confirmed drift +0.12, wrong hypothesis -0.04
        reward = 0.12 if is_true else -0.04
        if not is_true:
            # Wrong investigation wastes time — system continues degrading
            self._degrade_business_metrics(satisfaction_delta=-0.02, error_delta=0.01)
        return self._build_observation(
            reward=reward,
            last_action_result=f"Hypothesis '{hypothesis_id}': {result} (cost={cost}, budget_remaining={self._budget_remaining})",
        )

    def _handle_targeted_eval(self, action: LlmRewardLabAction) -> LlmRewardLabObservation:
        count = int(action.parameters.get("count", 6))
        count = max(1, min(count, 20))
        cost = count
        if self._budget_remaining < cost:
            return self._build_observation(
                reward=-0.05,
                last_action_result=f"Insufficient budget for probe. Need {cost}, have {self._budget_remaining}.",
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
            reward=0.03 if filtered else -0.02,
            samples=filtered,
            last_action_result=f"Probe returned {len(filtered)} fresh samples (cost={cost}).",
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

        # Explanation quality bonus: reward agents that justify their reasoning
        explanation = str(params.get("explanation", ""))
        explanation_score = self._score_explanation(explanation)
        if explanation_score > 0:
            score = min(0.999, score + explanation_score)

        self._done = True
        result_msg = f"Diagnosis submitted. Final score={score:.4f}"
        if explanation_score > 0:
            result_msg += f" (explanation bonus: +{explanation_score:.3f})"
        return self._build_observation(
            reward=score,
            last_action_result=result_msg,
        )

    @staticmethod
    def _score_explanation(explanation: str) -> float:
        """Score explanation quality based on structural heuristics.
        Returns a bonus in [0.0, 0.05] — small but meaningful."""
        if not explanation or len(explanation) < 20:
            return 0.0

        bonus = 0.0
        lower = explanation.lower()

        # Mentions specific evidence (task types, patterns, stats)
        evidence_terms = ["summarization", "coding", "qa", "long", "short", "medium",
                          "mean", "std", "variance", "quality", "score", "drop"]
        evidence_hits = sum(1 for t in evidence_terms if t in lower)
        if evidence_hits >= 2:
            bonus += 0.015

        # Shows reasoning chain (cause -> effect language)
        reasoning_terms = ["because", "therefore", "since", "indicates", "suggests",
                           "consistent with", "rules out", "confirms", "evidence"]
        reasoning_hits = sum(1 for t in reasoning_terms if t in lower)
        if reasoning_hits >= 1:
            bonus += 0.015

        # Mentions specific drift types or remediations
        drift_terms = ["contamination", "quantization", "template", "safety", "filter",
                        "router", "latency", "context window"]
        drift_hits = sum(1 for t in drift_terms if t in lower)
        if drift_hits >= 1:
            bonus += 0.02

        return round(min(0.05, bonus), 4)

    def _degrade_business_metrics(self, satisfaction_delta: float, error_delta: float) -> None:
        """Wrong actions have real consequences — business metrics worsen."""
        if self._business_metrics is None:
            return
        self._business_metrics = BusinessMetrics(
            user_satisfaction=round(max(0.0, self._business_metrics.user_satisfaction + satisfaction_delta), 3),
            error_rate=round(min(1.0, self._business_metrics.error_rate + error_delta), 3),
            cost_per_1k_requests=self._business_metrics.cost_per_1k_requests,
        )

    def _build_observation(
        self,
        reward: float,
        last_action_result: str,
        samples: list[SimulatedOutput] | None = None,
        include_stats: bool = False,
    ) -> LlmRewardLabObservation:
        assert self._task_cfg is not None

        max_steps = int(self._task_cfg["max_steps"])
        if not self._done and self._state.step_count >= max_steps:
            self._done = True
            reward = min(reward, 0.0)
            last_action_result = f"Max step limit ({max_steps}) reached. Episode ended."

        # Only include quality stats when explicitly requested (inspect/reset)
        # Partial observability: stats have sampling noise to simulate real monitoring
        quality_stats = {}
        if include_stats:
            stats: dict[str, dict] = defaultdict(
                lambda: {"scores": [], "sample_count": 0.0},
            )
            for sample in self._pool:
                bucket = stats[sample.task_type]
                bucket["scores"].append(sample.final_quality)
                bucket["sample_count"] += 1.0

            # Misleading bias: some tasks may have an artificial stat skew
            # that tempts the agent toward a wrong diagnosis (red herring)
            misleading = self._task_cfg.get("misleading_bias", {})

            for task_type, bucket in stats.items():
                scores = bucket["scores"]
                mean = sum(scores) / len(scores)
                variance = sum((value - mean) ** 2 for value in scores) / len(scores)
                # Add observation noise — agent must reason under uncertainty
                noise_mean = self._obs_rng.gauss(0, 0.015)
                noise_std = self._obs_rng.gauss(0, 0.008)
                # Apply misleading bias if configured for this task type
                bias = misleading.get(task_type, 0.0)
                quality_stats[task_type] = {
                    "mean": round(max(0.0, min(1.0, mean + noise_mean + bias)), 4),
                    "std": round(max(0.0, (variance**0.5) + noise_std), 4),
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

        # Clamp final score strictly within (0, 1) for done episodes
        clamped_reward = round(float(reward), 4)
        if self._done and clamped_reward >= 1.0:
            clamped_reward = 0.999
        elif self._done and clamped_reward <= 0.0:
            clamped_reward = 0.001

        return LlmRewardLabObservation(
            done=self._done,
            reward=clamped_reward,
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
            incident_context=self._incident_context,
            business_metrics=self._business_metrics,
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
            metadata={"source": "prod_eval_stream"},
        )