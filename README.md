---
title: LLM Regression Detector
emoji: "📉"
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - llm
  - evaluation
---

# LLM Regression Detector

<p align="left">
  <a href="https://github.com/meta-pytorch/OpenEnv"><img alt="OpenEnv" src="https://img.shields.io/badge/OpenEnv-Compatible-1877F2?logo=meta&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0A9B8E?logo=fastapi&logoColor=white"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://www.docker.com/"><img alt="Docker" src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://huggingface.co/spaces"><img alt="Hugging Face Spaces" src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black"></a>
</p>

An OpenEnv environment for evaluating how well an agent can detect and diagnose silent production regressions in LLM quality.

The environment models a realistic reliability workflow: inspect output quality, test root-cause hypotheses under budget constraints, and submit final drift diagnoses with remediations.

## Why this environment

Production LLM systems fail in nuanced ways:
- quality drops can be task-specific
- regressions may appear only for long inputs
- multiple drifts can interact and hide each other

This environment turns that workflow into a deterministic, reproducible benchmark suitable for OpenEnv evaluation.

## Environment design

### Action space (`LlmRewardLabAction`)

- `inspect_samples` with optional filters: `task_type`, `input_length`, `limit`
- `test_hypothesis` with `hypothesis_id`
- `request_probe` with targeted synthetic sampling
- `submit_diagnosis` with:
  - `drift_events: list[str]`
  - `remediations: list[str]`

### Observation space (`LlmRewardLabObservation`)

- sampled outputs with quality scores
- per-task aggregate quality stats (`mean`, `std`, `sample_count`)
- investigation state: `budget_remaining`, `budget_total`
- available and tested hypotheses
- episode state: `step_count`, `task_id`, `last_action_result`, `done`, `reward`

### Reward shaping

- dense intermediate signal for useful investigative actions
- penalties for invalid actions and wasteful budget usage
- terminal deterministic score from the task grader

## Tasks

Three tasks with increasing difficulty:

1. `task_detect_localize` (easy): isolate a single obvious drift
2. `task_diagnose` (medium): identify a subtler root cause under budget constraints
3. `task_multi_drift` (hard): recover two interacting drifts with tighter trade-offs

Each task is deterministic for a given seed, and graders return normalized scores in `[0.0, 1.0]`.

## Scoring logic

Final score combines:
- drift detection quality (F1)
- remediation correctness
- budget efficiency
- false-positive penalties

This produces meaningful partial credit while still punishing noisy or overfit submissions.

## API endpoints

OpenEnv endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`

Evaluation endpoints:
- `GET /tasks` returns available tasks + action schema
- `POST /grader` returns task score for a submitted diagnosis
- `POST /baseline` runs the reference inference flow and returns reproducible scores

## Quickstart

```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Run baseline locally:

```bash
python inference.py
```

The baseline is deterministic with `seed=42`.  
If `OPENAI_API_KEY` is available, the script can use OpenAI for diagnosis proposals as a fallback path.

## Docker

```bash
docker build -t llm-regression-detector .
docker run --rm -p 8000:8000 llm-regression-detector
```

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id <your-username>/llm-regression-detector
```

## Minimal API examples

Reset:

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_detect_localize","seed":42}'
```

Step:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type":"inspect_samples",
    "parameters":{"task_type":"summarization","limit":10}
  }'
```

Grade:

```bash
curl -X POST http://localhost:8000/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"task_detect_localize",
    "drift_events":["data_contamination"],
    "remediations":["rollback_finetune_checkpoint"],
    "budget_used":5
  }'
```

## Project structure

```text
server/
  app.py            # FastAPI wiring + OpenEnv routes + eval routes
  environment.py    # Core environment state machine
  simulation.py     # Deterministic drift simulator
  graders.py        # Programmatic scoring
  tasks.py          # Task registry and configs
inference.py        # Baseline interaction script
openenv.yaml        # OpenEnv metadata/spec wiring
```

## Validation checklist

- deterministic behavior with fixed seeds
- three graded tasks (easy/medium/hard)
- normalized scores in `[0.0, 1.0]`
- working Docker setup
- baseline script runnable end-to-end
