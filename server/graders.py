from typing import Iterable

from .simulation import DRIFT_CATALOG


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

    matched_drifts = true_drifts & submitted_set
    false_positives = submitted_set - true_drifts

    precision = len(matched_drifts) / len(submitted_set) if submitted_set else 0.0
    recall = len(matched_drifts) / len(true_drifts) if true_drifts else 1.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    needed_remediations = {
        DRIFT_CATALOG[next(k for k in DRIFT_CATALOG if k.value == drift)]["remediation"]
        for drift in true_drifts
    }
    matched_remediations = len(needed_remediations & remediations_set)
    remediation_score = (
        matched_remediations / len(needed_remediations) if needed_remediations else 0.0
    )

    efficiency = 0.0
    if budget_total > 0:
        efficiency = max(0.0, 1.0 - (budget_used / budget_total))

    base = 0.0
    if task_id == "task_detect_localize":
        base = 0.75 * f1 + 0.25 * remediation_score
    elif task_id == "task_diagnose":
        base = 0.55 * f1 + 0.35 * remediation_score + 0.10 * efficiency
    else:
        base = 0.50 * f1 + 0.35 * remediation_score + 0.15 * efficiency

    false_positive_penalty = 0.08 * len(false_positives)
    return round(max(0.0, min(1.0, base - false_positive_penalty)), 4)
