"""
Inference Script — LLM Reward Lab
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import LlmRewardLabEnv
from models import LlmRewardLabAction
from server.environment import LlmRewardLabEnvironment
from server.tasks import TASK_REGISTRY

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.2")
BENCHMARK = os.getenv("LLM_REWARD_LAB_BENCHMARK", "llm_reward_lab")
TASK_IDS = ["task_detect_localize", "task_diagnose", "task_multi_drift"]
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1

REMEDIATION_BY_DRIFT = {
    "data_contamination": "rollback_finetune_checkpoint",
    "quantization_applied": "revert_quantization",
    "prompt_template_change": "rollback_prompt_template",
    "safety_filter_misconfig": "recalibrate_safety_filter",
    "context_window_bug": "patch_context_window_handler",
    "infra_latency": "scale_serving_infra",
    "router_bug": "fix_traffic_routing",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an LLM regression diagnostician. You interact with an environment that
    monitors LLM output quality and must identify active drift events.

    Available actions (respond with exactly one JSON object per turn):
    1. {"action_type": "inspect_samples", "parameters": {"limit": 20}}
       - View output samples and quality stats. Optional filters: task_type, input_length.
    2. {"action_type": "run_ab_test", "parameters": {"hypothesis_id": "<id>"}}
       - Test whether a specific drift is active. Costs budget.
    3. {"action_type": "run_targeted_eval", "parameters": {"count": 6}}
       - Request fresh probe samples. Costs budget.
    4. {"action_type": "submit_diagnosis", "parameters": {"drift_events": [...], "remediations": [...]}}
       - Submit your final diagnosis. Use this when confident.

    Strategy:
    - First inspect samples to understand quality patterns.
    - Test hypotheses starting with cheapest or most likely ones.
    - Track your budget carefully — don't waste it on unlikely hypotheses.
    - Submit diagnosis before running out of steps.
    - Each confirmed drift needs a matching remediation.

    Respond with ONLY a valid JSON object. No explanation, no markdown.
""").strip()


# ── Logging helpers (hackathon stdout format) ──────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM-driven action selection ───────────────────────────────────────────────


def build_user_prompt(obs_summary: dict, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Task: {obs_summary['task_id']}
        Budget remaining: {obs_summary['budget_remaining']}/{obs_summary['budget_total']}
        Steps used: {obs_summary['step_count']}
        Last result: {obs_summary['last_action_result']}
        Quality stats: {json.dumps(obs_summary['quality_stats'], indent=2)}
        Available hypotheses: {json.dumps(obs_summary['available_hypotheses'], indent=2)}
        Tested hypotheses: {json.dumps(obs_summary['tested_hypotheses'], indent=2)}
        Previous actions:
        {history_block}

        Decide your next action. Respond with a single JSON object.
    """).strip()


def observation_to_summary(obs) -> dict:
    return {
        "task_id": obs.task_id,
        "budget_remaining": obs.budget_remaining,
        "budget_total": obs.budget_total,
        "quality_stats": obs.quality_stats,
        "available_hypotheses": [
            {"id": h.hypothesis_id, "description": h.description, "cost": h.cost}
            for h in obs.available_hypotheses
        ],
        "tested_hypotheses": obs.tested_hypotheses,
        "step_count": obs.step_count,
        "last_action_result": obs.last_action_result,
    }


def get_llm_action(client: OpenAI, obs_summary: dict, step: int, history: List[str]) -> dict:
    user_prompt = build_user_prompt(obs_summary, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return {"action_type": "inspect_samples", "parameters": {"limit": 20}}


# ── Deterministic fallback agent ───────────────────────────────────────────────


def fallback_action(obs_summary: dict, step: int, max_steps: int) -> dict:
    """Simple inspect -> test -> submit loop. No hardcoded answers."""
    tested = obs_summary["tested_hypotheses"]
    hypotheses = obs_summary["available_hypotheses"]
    budget = obs_summary["budget_remaining"]
    steps_left = max_steps - step

    # Always submit on last step
    if steps_left <= 1:
        return _build_submit_action(tested)

    # Step 1: inspect samples to get quality stats
    if step == 1:
        return {"action_type": "inspect_samples", "parameters": {"limit": 20}}

    # Steps 2+: test affordable hypotheses in order
    for h in sorted(hypotheses, key=lambda x: x["cost"]):
        if h["id"] not in tested and h["cost"] <= budget:
            return {"action_type": "run_ab_test", "parameters": {"hypothesis_id": h["id"]}}

    # All tested or budget exhausted — submit
    return _build_submit_action(tested)


def _build_submit_action(tested: dict) -> dict:
    confirmed = [h_id for h_id, result in tested.items() if "CONFIRMED" in result]
    remediations = [
        REMEDIATION_BY_DRIFT[d] for d in confirmed if d in REMEDIATION_BY_DRIFT
    ]
    # If nothing confirmed, make a conservative guess
    if not confirmed:
        confirmed = ["prompt_template_change"]
        remediations = ["rollback_prompt_template"]
    return {
        "action_type": "submit_diagnosis",
        "parameters": {"drift_events": confirmed, "remediations": remediations},
    }


# ── Sync baseline (called by server /baseline endpoint) ───────────────────────


def run_baseline() -> dict:
    """Sync baseline using the environment directly. Returns reproducible scores."""
    env = LlmRewardLabEnvironment()
    scores: dict[str, float] = {}

    for task_id in TASK_IDS:
        if task_id not in TASK_REGISTRY:
            continue

        obs = env.reset(task_id=task_id, seed=42)
        max_steps = int(TASK_REGISTRY[task_id]["max_steps"])
        obs_summary = observation_to_summary(obs)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            action_dict = fallback_action(obs_summary, step, max_steps)
            action = LlmRewardLabAction(
                action_type=action_dict["action_type"],
                parameters=action_dict.get("parameters", {}),
            )
            obs = env.step(action)
            obs_summary = observation_to_summary(obs)

            if obs.done:
                break

        scores[task_id] = float(obs.reward or 0.0)

    mean_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    return {"seed": 42, "scores": scores, "mean_score": mean_score}


# ── Async hackathon entry point (uses Docker image via client) ─────────────────


async def run_task(env, client: Optional[OpenAI], task_id: str, max_steps: int) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset(task_id=task_id, seed=42)
        obs_summary = observation_to_summary(obs)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            if client:
                action_dict = get_llm_action(client, obs_summary, step, history)
            else:
                action_dict = fallback_action(obs_summary, step, max_steps)

            action = LlmRewardLabAction(
                action_type=action_dict["action_type"],
                parameters=action_dict.get("parameters", {}),
            )

            obs = await env.step(action)
            obs_summary = observation_to_summary(obs)

            reward = obs.reward or 0.0
            done = obs.done
            error = obs.last_action_result if reward < 0 else None

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_dict['action_type']}({json.dumps(action_dict.get('parameters', {}))})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward={reward:.2f}, result={obs.last_action_result}")

            if done:
                break

        if rewards:
            score = max(rewards[-1], 0.0)
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def run_task_local(llm_client: Optional[OpenAI], task_id: str, max_steps: int) -> float:
    """Run a task using the local environment directly (no Docker)."""
    env = LlmRewardLabEnvironment()
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id, seed=42)
        obs_summary = observation_to_summary(obs)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            if llm_client:
                action_dict = get_llm_action(llm_client, obs_summary, step, history)
            else:
                action_dict = fallback_action(obs_summary, step, max_steps)

            action = LlmRewardLabAction(
                action_type=action_dict["action_type"],
                parameters=action_dict.get("parameters", {}),
            )

            obs = env.step(action)
            obs_summary = observation_to_summary(obs)

            reward = obs.reward or 0.0
            done = obs.done
            error = obs.last_action_result if reward < 0 else None

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action_dict['action_type']}({json.dumps(action_dict.get('parameters', {}))})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward={reward:.2f}, result={obs.last_action_result}")

            if done:
                break

        if rewards:
            score = max(rewards[-1], 0.0)
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    llm_client = None
    if API_KEY:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_IDS:
        max_steps = int(TASK_REGISTRY[task_id]["max_steps"])
        if IMAGE_NAME:
            env = await LlmRewardLabEnv.from_docker_image(IMAGE_NAME)
            await run_task(env, llm_client, task_id, max_steps)
        else:
            await run_task_local(llm_client, task_id, max_steps)


if __name__ == "__main__":
    asyncio.run(main())