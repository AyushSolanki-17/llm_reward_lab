"""FastAPI application for llm-regression-detector."""

from fastapi import Body
from pydantic import BaseModel, Field

from models import LlmRewardLabAction, LlmRewardLabObservation
from openenv.core.env_server.http_server import create_app
from server.environment import LlmRewardLabEnvironment
from server.graders import grade_submission
from server.tasks import TASK_REGISTRY, list_tasks

app = create_app(
    LlmRewardLabEnvironment,
    LlmRewardLabAction,
    LlmRewardLabObservation,
    env_name="llm-regression-detector",
    max_concurrent_envs=4,
)


class GraderRequest(BaseModel):
    task_id: str
    drift_events: list[str] = Field(default_factory=list)
    remediations: list[str] = Field(default_factory=list)
    budget_used: int = 0
    budget_total: int | None = None


@app.get("/tasks")
def tasks_endpoint():
    return {
        "tasks": list_tasks(),
        "action_schema": LlmRewardLabAction.model_json_schema(),
    }


@app.post("/grader")
def grader_endpoint(payload: GraderRequest = Body(...)):
    if payload.task_id not in TASK_REGISTRY:
        return {"error": f"Unknown task_id: {payload.task_id}"}

    cfg = TASK_REGISTRY[payload.task_id]
    score = grade_submission(
        task_id=payload.task_id,
        submitted_drifts=payload.drift_events,
        submitted_remediations=payload.remediations,
        active_drifts=cfg["active_drifts"],
        budget_used=payload.budget_used,
        budget_total=payload.budget_total or cfg["budget"],
    )
    return {"task_id": payload.task_id, "score": score}


@app.post("/baseline")
def baseline_endpoint():
    from inference import run_baseline

    return run_baseline()


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
