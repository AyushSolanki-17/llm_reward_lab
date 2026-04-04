import random
from dataclasses import dataclass
from typing import List
from enum import Enum

TASK_TYPES = ["summarization", "qa", "coding", "translation", "classification"]
INPUT_LENGTHS = ["short", "medium", "long"]


class DriftType(str, Enum):
    PROMPT_TEMPLATE_CHANGE = "prompt_template_change"
    QUANTIZATION_APPLIED = "quantization_applied"
    SAFETY_FILTER_MISCONFIG = "safety_filter_misconfig"
    DATA_CONTAMINATION = "data_contamination"
    CONTEXT_WINDOW_BUG = "context_window_bug"
    INFRA_LATENCY = "infra_latency"
    ROUTER_BUG = "router_bug"


DRIFT_CATALOG = {
    DriftType.PROMPT_TEMPLATE_CHANGE: {
        "description": "Upstream prompt template was silently modified",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.18,
        "remediation": "rollback_prompt_template",
    },
    DriftType.QUANTIZATION_APPLIED: {
        "description": "4-bit quantization applied to reduce serving cost",
        "affected_task_types": ["coding", "qa"],
        "affected_input_lengths": ["long"],
        "quality_delta": -0.25,
        "remediation": "revert_quantization",
    },
    DriftType.SAFETY_FILTER_MISCONFIG: {
        "description": "Safety filter misconfigured — false positives on sensitive keywords",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.12,
        "affected_probability": 0.3,
        "remediation": "recalibrate_safety_filter",
    },
    DriftType.DATA_CONTAMINATION: {
        "description": "Fine-tuning batch contaminated with low-quality summarization data",
        "affected_task_types": ["summarization"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.30,
        "remediation": "rollback_finetune_checkpoint",
    },
    DriftType.CONTEXT_WINDOW_BUG: {
        "description": "Context window handling bug truncates long inputs incorrectly",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["long"],
        "quality_delta": -0.22,
        "remediation": "patch_context_window_handler",
    },
    DriftType.INFRA_LATENCY: {
        "description": "Serving infrastructure latency spike causing timeout-induced quality drops",
        "affected_task_types": ["coding", "summarization"],
        "affected_input_lengths": ["medium", "long"],
        "quality_delta": -0.15,
        "remediation": "scale_serving_infra",
    },
    DriftType.ROUTER_BUG: {
        "description": "Traffic router sending requests to a stale model checkpoint",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.20,
        "affected_probability": 0.4,
        "remediation": "fix_traffic_routing",
    },
}


@dataclass
class SimulatedOutput:
    sample_id: str
    task_type: str
    input_length: str
    base_quality: float
    final_quality: float
    drift_applied: List[str]


class DriftSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_pool(
        self,
        n: int = 200,
        active_drifts: List[DriftType] | None = None,
    ) -> List[SimulatedOutput]:
        pool = []
        active_drifts = active_drifts or []

        for i in range(n):
            task = self.rng.choice(TASK_TYPES)
            length = self.rng.choice(INPUT_LENGTHS)
            base_q = round(self.rng.gauss(0.78, 0.08), 3)
            base_q = max(0.0, min(1.0, base_q))

            final_q = base_q
            applied = []

            for drift in active_drifts:
                spec = DRIFT_CATALOG[drift]
                task_match = (
                    spec["affected_task_types"] == ["all"]
                    or task in spec["affected_task_types"]
                )
                len_match = (
                    spec.get("affected_input_lengths") == ["all"]
                    or length in spec.get("affected_input_lengths", ["all"])
                )
                prob_match = self.rng.random() < spec.get("affected_probability", 1.0)

                if task_match and len_match and prob_match:
                    final_q = max(0.0, final_q + spec["quality_delta"])
                    applied.append(drift.value)

            pool.append(
                SimulatedOutput(
                    sample_id=f"sample_{i:04d}",
                    task_type=task,
                    input_length=length,
                    base_quality=base_q,
                    final_quality=round(final_q, 3),
                    drift_applied=applied,
                )
            )

        return pool