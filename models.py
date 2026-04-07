"""Typed OpenEnv models for the LLM regression detection environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

ActionType = Literal[
    "inspect_samples",
    "run_ab_test",
    "run_targeted_eval",
    "submit_diagnosis",
]


class OutputSample(BaseModel):
    sample_id: str
    task_type: str
    input_length: str
    quality_score: float
    timestamp: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    hypothesis_id: str
    description: str
    cost: int


class LlmRewardLabAction(Action):
    action_type: ActionType = Field(
        ...,
        description="inspect_samples | run_ab_test | run_targeted_eval | submit_diagnosis",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "inspect_samples    -> {task_type?: str, input_length?: str, limit?: int}\n"
            "run_ab_test        -> {hypothesis_id: str}\n"
            "run_targeted_eval  -> {task_type?: str, input_length?: str, count?: int}\n"
            "submit_diagnosis   -> {drift_events: List[str], remediations: List[str], explanation?: str}"
        ),
    )


class IncidentContext(BaseModel):
    """Simulated production incident ticket — adds realism and urgency."""
    severity: str = "MEDIUM"
    started_minutes_ago: int = 0
    affected_users_percent: float = 0.0
    reported_issue: str = ""


class BusinessMetrics(BaseModel):
    """Simulated business impact metrics — creates trade-off decisions."""
    user_satisfaction: float = 0.0
    error_rate: float = 0.0
    cost_per_1k_requests: float = 0.0


class LlmRewardLabObservation(Observation):
    samples: List[OutputSample] = Field(default_factory=list)
    budget_remaining: int = 0
    budget_total: int = 0
    available_hypotheses: List[Hypothesis] = Field(default_factory=list)
    tested_hypotheses: Dict[str, str] = Field(default_factory=dict)
    quality_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    step_count: int = 0
    task_id: str = ""
    last_action_result: Optional[str] = None
    incident_context: Optional[IncidentContext] = None
    business_metrics: Optional[BusinessMetrics] = None
