"""Baseline inference entrypoint for llm-regression-detector."""

from __future__ import annotations

import json
import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from models import LlmRewardLabAction
from server.environment import LlmRewardLabEnvironment
from server.tasks import TASK_REGISTRY

TASK_IDS = ["task_detect_localize", "task_diagnose", "task_multi_drift"]


def _fallback_guess(task_id: str) -> tuple[list[str], list[str]]:
    if task_id == "task_detect_localize":
        drifts = ["data_contamination"]
    elif task_id == "task_diagnose":
        drifts = ["quantization_applied"]
    else:
        drifts = ["prompt_template_change", "safety_filter_misconfig"]

    remediations = [
        {
            "data_contamination": "rollback_finetune_checkpoint",
            "quantization_applied": "revert_quantization",
            "prompt_template_change": "rollback_prompt_template",
            "safety_filter_misconfig": "recalibrate_safety_filter",
        }[drift]
        for drift in drifts
    ]
    return drifts, remediations


def _llm_guess(task_id: str, stats: dict) -> tuple[list[str], list[str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return _fallback_guess(task_id)

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are diagnosing LLM regressions. "
        "Return strict JSON with keys drift_events and remediations.\n"
        f"task_id={task_id}\n"
        f"quality_stats={json.dumps(stats, sort_keys=True)}\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return _fallback_guess(task_id)

    drifts = payload.get("drift_events", [])
    remediations = payload.get("remediations", [])
    if not isinstance(drifts, list) or not isinstance(remediations, list):
        return _fallback_guess(task_id)
    return [str(item) for item in drifts], [str(item) for item in remediations]


def run_baseline() -> dict:
    env = LlmRewardLabEnvironment()
    scores: dict[str, float] = {}

    for task_id in TASK_IDS:
        if task_id not in TASK_REGISTRY:
            continue

        obs = env.reset(task_id=task_id, seed=42)
        obs = env.step(
            LlmRewardLabAction(action_type="inspect_samples", parameters={"limit": 20}),
        )
        drifts, remediations = _llm_guess(task_id, obs.quality_stats)
        obs = env.step(
            LlmRewardLabAction(
                action_type="submit_diagnosis",
                parameters={
                    "drift_events": drifts,
                    "remediations": remediations,
                },
            )
        )
        scores[task_id] = float(obs.reward or 0.0)

    mean_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
    return {"seed": 42, "scores": scores, "mean_score": mean_score}


if __name__ == "__main__":
    result = run_baseline()
    print(json.dumps(result, indent=2, sort_keys=True))
