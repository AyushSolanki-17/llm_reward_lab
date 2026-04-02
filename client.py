"""Client for llm-regression-detector."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import LlmRewardLabAction, LlmRewardLabObservation


class LlmRewardLabEnv(EnvClient[LlmRewardLabAction, LlmRewardLabObservation, State]):
    def _step_payload(self, action: LlmRewardLabAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[LlmRewardLabObservation]:
        obs_data = dict(payload.get("observation", {}))
        if "reward" not in obs_data:
            obs_data["reward"] = payload.get("reward")
        if "done" not in obs_data:
            obs_data["done"] = payload.get("done", False)

        observation = LlmRewardLabObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
