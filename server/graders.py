from typing import Iterable

from .simulation import DRIFT_CATALOG


# Per-task false-positive penalty: harder tasks punish noise more
_FP_PENALTY = {
    "task_detect_localize": 0.08,
    "task_diagnose": 0.10,
    "task_multi_drift": 0.12,
}

# Per-task score weights: [f1, remediation, efficiency]
_TASK_WEIGHTS = {
    "task_detect_localize": (0.75, 0.25, 0.00),
    "task_diagnose": (0.55, 0.35, 0.10),
    "task_multi_drift": (0.50, 0.35, 0.15),
}


def grade_submission(
    task_id: str,
    submitted_drifts: list[str],
    submitted_remediations: list[str],
    active_drifts: Iterable,
    budget_used: int,
    budget_total: int,
) -> float:
    true_drifts = {drift.value for drift in active_drifts}
    submitted_set = set(submitted_drifts)
    remediations_set = set(submitted_remediations)

    # --- Drift detection F1 ---
    matched_drifts = true_drifts & submitted_set
    false_positives = submitted_set - true_drifts

    precision = len(matched_drifts) / len(submitted_set) if submitted_set else 0.0
    recall = len(matched_drifts) / len(true_drifts) if true_drifts else 1.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # --- Remediation correctness ---
    needed_remediations = {
        DRIFT_CATALOG[next(k for k in DRIFT_CATALOG if k.value == drift)][
            "remediation"
        ]
        for drift in true_drifts
    }
    matched_remediations = len(needed_remediations & remediations_set)
    remediation_score = (
        matched_remediations / len(needed_remediations) if needed_remediations else 0.0
    )

    # --- Budget efficiency ---
    efficiency = 0.0
    if budget_total > 0:
        efficiency = max(0.0, 1.0 - (budget_used / budget_total))

    # --- Composite score ---
    w_f1, w_rem, w_eff = _TASK_WEIGHTS.get(task_id, (0.50, 0.35, 0.15))
    base = w_f1 * f1 + w_rem * remediation_score + w_eff * efficiency

    # --- Penalties ---
    fp_rate = _FP_PENALTY.get(task_id, 0.10)
    false_positive_penalty = fp_rate * len(false_positives)

    # Penalty for submitting spurious remediations (not matching any true drift)
    spurious_remediations = remediations_set - needed_remediations
    spurious_penalty = 0.04 * len(spurious_remediations)

    # False fix penalty: submitting wrong remediations for detected drifts
    # (you found the drift but prescribed the wrong fix — costly in production)
    wrong_fix_count = 0
    if matched_drifts and needed_remediations:
        # Agent identified correct drifts but may have wrong remediations
        correct_remediations = needed_remediations & remediations_set
        # Each needed remediation not provided counts as a wrong/missing fix
        wrong_fix_count = len(needed_remediations) - len(correct_remediations)
    false_fix_penalty = 0.10 * wrong_fix_count

    return round(
        max(0.0, min(1.0, base - false_positive_penalty - spurious_penalty - false_fix_penalty)),
        4,
    )