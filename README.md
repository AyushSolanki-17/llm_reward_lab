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

A real-world OpenEnv environment where an agent triages silent production regressions in LLM quality.

The agent inspects sampled outputs, spends a limited hypothesis-testing budget, and submits a final diagnosis with remediations.

## Problem Modeled

This environment simulates a reliability workflow used by ML platform engineers:
- detect a degradation pattern
- identify likely root causes
- recommend remediations quickly under investigation constraints

## Action Space

`LlmRewardLabAction`
- `inspect_samples` with optional filters (`task_type`, `input_length`, `limit`)
- `test_hypothesis` with `hypothesis_id`
- `request_probe` with targeted sample generation
- `submit_diagnosis` with:
  - `drift_events: list[str]`
  - `remediations: list[str]`

## Observation Space

`LlmRewardLabObservation` includes:
- `samples`: visible output slice
- `quality_stats`: per-task aggregate quality statistics
- `budget_remaining`, `budget_total`
- `available_hypotheses`, `tested_hypotheses`
- `step_count`, `task_id`, `last_action_result`
- standard OpenEnv fields `reward`, `done`, `metadata`

## Tasks and Graders

Three deterministic tasks with increasing difficulty:
- `task_detect_localize` (easy): one obvious drift
- `task_diagnose` (medium): subtle drift under budget
- `task_multi_drift` (hard): multiple interacting drifts under tight budget

The grader returns a score in `[0.0, 1.0]` using:
- drift identification quality (F1)
- remediation correctness
- budget efficiency
- false-positive penalties

## Reward Design

Dense trajectory signal:
- small positive reward for useful inspection/probing actions
- penalties for invalid or wasteful actions
- terminal reward from deterministic grader score

## API Endpoints

OpenEnv default endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`

Additional endpoints required for evaluation:
- `GET /tasks` returns task list + action schema
- `POST /grader` computes task score from submitted diagnosis
- `POST /baseline` runs baseline and returns reproducible task scores

## Local Development

```bash
uv sync
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Run baseline:

```bash
python inference.py
```

If `OPENAI_API_KEY` is set, the baseline uses OpenAI (`gpt-4o-mini`) for diagnosis proposals; otherwise it falls back to a deterministic rule-based policy.

## Docker

```bash
docker build -t llm-regression-detector .
docker run -p 8000:8000 llm-regression-detector
```

## Deploy

```bash
openenv push --repo-id <username>/llm-regression-detector
```
