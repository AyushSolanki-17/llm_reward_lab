import random
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

TASK_TYPES = ["summarization", "qa", "coding", "translation", "classification"]
INPUT_LENGTHS = ["short", "medium", "long"]

# ── DRIFT EVENT DEFINITIONS ──────────────────────────────────────────────────

class DriftType(str, Enum):
    # affects ALL task types uniformly
    PROMPT_TEMPLATE_CHANGE  = "prompt_template_change"
    # affects coding + qa on LONG inputs only
    QUANTIZATION_APPLIED    = "quantization_applied"
    # random false positives, task-agnostic
    SAFETY_FILTER_MISCONFIG = "safety_filter_misconfig"
    # affects summarization only
    DATA_CONTAMINATION      = "data_contamination"
    # affects LONG inputs across all tasks
    CONTEXT_WINDOW_BUG      = "context_window_bug"

DRIFT_CATALOG = {
    DriftType.PROMPT_TEMPLATE_CHANGE: {
        "description": "Upstream prompt template was silently modified",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["all"],
        # drops quality by 0.18 uniformly
        "quality_delta": -0.18,
        "remediation": "rollback_prompt_template"
    },
    DriftType.QUANTIZATION_APPLIED: {
        "description": "4-bit quantization applied to reduce serving cost",
        "affected_task_types": ["coding", "qa"],
        "affected_input_lengths": ["long"],
        "quality_delta": -0.25,
        "remediation": "revert_quantization"
    },
    DriftType.SAFETY_FILTER_MISCONFIG: {
        "description": "Safety filter misconfigured — false positives on sensitive keywords",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.12,
        "affected_probability": 0.3,    # only 30% of outputs affected, randomly
        "remediation": "recalibrate_safety_filter"
    },
    DriftType.DATA_CONTAMINATION: {
        "description": "Fine-tuning batch contaminated with low-quality summarization data",
        "affected_task_types": ["summarization"],
        "affected_input_lengths": ["all"],
        "quality_delta": -0.30,
        "remediation": "rollback_finetune_checkpoint"
    },
    DriftType.CONTEXT_WINDOW_BUG: {
        "description": "Context window handling bug truncates long inputs incorrectly",
        "affected_task_types": ["all"],
        "affected_input_lengths": ["long"],
        "quality_delta": -0.22,
        "remediation": "patch_context_window_handler"
    },
}

@dataclass
class SimulatedOutput:
    sample_id: str
    task_type: str
    input_length: str
    # pre-drift true quality
    base_quality: float
    # post-drift observed quality
    final_quality: float
    # which drift events affected this sample
    drift_applied: List[str]

class DriftSimulator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_pool(
        self,
        n: int = 200,
        active_drifts: List[DriftType] = None
    ) -> List[SimulatedOutput]:
        """
        Generate n synthetic LLM outputs with drift applied.
        Fully deterministic given seed + active_drifts.
        """
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
                task_match = (spec["affected_task_types"] == ["all"]
                              or task in spec["affected_task_types"])
                len_match  = (spec.get("affected_input_lengths") == ["all"]
                              or length in spec.get("affected_input_lengths", ["all"]))
                prob_match = self.rng.random() < spec.get("affected_probability", 1.0)

                if task_match and len_match and prob_match:
                    final_q = max(0.0, final_q + spec["quality_delta"])
                    applied.append(drift.value)

            pool.append(SimulatedOutput(
                sample_id=f"sample_{i:04d}",
                task_type=task,
                input_length=length,
                base_quality=base_q,
                final_quality=round(final_q, 3),
                drift_applied=applied,
            ))

        return pool