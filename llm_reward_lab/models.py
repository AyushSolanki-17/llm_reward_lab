# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Llm Reward Lab Environment.

The llm_reward_lab environment is a simple test environment that echoes back messages.
"""

from typing import Optional, List, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel

# OBSERVATION Models
class OutputSample(BaseModel):
    sample_id: str
    # summarization | qa | coding | translation | classification
    task_type: str
    # short | medium | long
    input_length: str
    # 0.0–1.0, observable by agent
    quality_score: float
    # episode step when produced
    timestamp: int
    metadata: Dict[str, Any] = {}

class Hypothesis(BaseModel):
    hypothesis_id: str
    description: str
    # budget units to test this hypothesis
    cost: int


class LlmRewardLabAction(Action):
    """
        Action for the Llm Reward Lab environment - just a message to echo.
        One step the agent takes inside the environment.
        action_type controls what happens:
              inspect_samples   → view a slice of outputs filtered by task_type / input_length
              test_hypothesis   → formally test one hypothesis, costs budget
              submit_diagnosis  → agent declares root causes + remediation plan (ends episode)
              request_probe     → request a targeted probe output on a specific input pattern
    """

    message: str = Field(..., description="Message to echo back")
    action_type: str = Field(
        ...,
        description="One of: inspect_samples | test_hypothesis | submit_diagnosis | request_probe"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "inspect_samples   → {task_type?: str, input_length?: 'short'|'long', limit?: int}\n"
            "test_hypothesis   → {hypothesis_id: str}\n"
            "submit_diagnosis  → {drift_events: List[str], remediations: List[str]}\n"
            "request_probe     → {task_type: str, input_length: str, count: int}"
        )
    )


class LlmRewardLabObservation(Observation):
    """
        Observation from the Llm Reward Lab environment - the echoed message.
        Describes what agent sees after each step.
    """

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
    # Current samples visible to agent
    samples: List[OutputSample] = Field(default_factory=list)

    # Budget state
    budget_remaining: int
    budget_total: int

    # Available hypotheses to test (Task 2 & 3)
    available_hypotheses: List[Hypothesis] = Field(default_factory=list)

    # Hypothesis test results revealed so far
    tested_hypotheses: Dict[str, str] = Field(
        default_factory=dict,
        description="hypothesis_id → result_summary string"
    )

    # Aggregate quality stats per task_type (always visible)
    quality_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="task_type → {mean: float, std: float, sample_count: int}"
    )

    # Episode state
    step_count: int
    task_id: str
    done: bool = False

    # Feedback from last action
    last_action_result: Optional[str] = None