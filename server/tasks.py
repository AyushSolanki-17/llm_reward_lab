from .simulation import DriftType

TASK_REGISTRY = {
    "task_detect_localize": {
        "id": "task_detect_localize",
        "difficulty": "easy",
        "pool_size": 120,
        "budget": 20,
        "max_steps": 12,
        "active_drifts": [DriftType.DATA_CONTAMINATION],
        "hypotheses": {
            "data_contamination": {
                "description": "Fine-tune data contamination",
                "cost": 5,
            },
            "prompt_template_change": {
                "description": "Prompt template changed",
                "cost": 5,
            },
        },
        "description": "Find and report a single obvious drift from clear quality signals.",
    },
    "task_diagnose": {
        "id": "task_diagnose",
        "difficulty": "medium",
        "pool_size": 170,
        "budget": 60,
        "max_steps": 16,
        "active_drifts": [DriftType.QUANTIZATION_APPLIED],
        "hypotheses": {
            "prompt_template_change": {
                "description": "Prompt template changed",
                "cost": 8,
            },
            "quantization_applied": {
                "description": "Model quantization side effects",
                "cost": 12,
            },
            "safety_filter_misconfig": {
                "description": "Safety filter false positives",
                "cost": 8,
            },
            "data_contamination": {
                "description": "Fine-tune data contamination",
                "cost": 10,
            },
            "context_window_bug": {
                "description": "Long-context truncation bug",
                "cost": 14,
            },
            "infra_latency": {
                "description": "Serving latency spike",
                "cost": 5,
            },
        },
        "description": "Use budgeted hypothesis testing to isolate a subtle, task-specific drift.",
    },
    "task_multi_drift": {
        "id": "task_multi_drift",
        "difficulty": "hard",
        "pool_size": 240,
        "budget": 55,
        "max_steps": 18,
        "active_drifts": [
            DriftType.PROMPT_TEMPLATE_CHANGE,
            DriftType.SAFETY_FILTER_MISCONFIG,
        ],
        "hypotheses": {
            "prompt_template_change": {
                "description": "Prompt template changed",
                "cost": 10,
            },
            "quantization_applied": {
                "description": "Model quantization side effects",
                "cost": 12,
            },
            "safety_filter_misconfig": {
                "description": "Safety filter false positives",
                "cost": 10,
            },
            "data_contamination": {
                "description": "Fine-tune data contamination",
                "cost": 8,
            },
            "context_window_bug": {
                "description": "Long-context truncation bug",
                "cost": 15,
            },
            "router_bug": {
                "description": "Traffic routed to a stale checkpoint",
                "cost": 8,
            },
        },
        "description": "Resolve two interacting drifts under a tight budget — one uniform, one stochastic.",
    },
}


def list_tasks() -> list[dict]:
    return [
        {
            "id": cfg["id"],
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "budget": cfg["budget"],
            "max_steps": cfg["max_steps"],
        }
        for cfg in TASK_REGISTRY.values()
    ]