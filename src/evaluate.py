"""Evaluation: metrics that map to how a clinic would actually use the model.

Staff can only call so many patients per day. So the operative question is:
"if we rank tomorrow's appointments by risk and act on the top k, how many of
those are true no-shows?" -> precision@k. Global accuracy is deliberately absent.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


def precision_recall_at_k(y_true, scores, k: int) -> tuple[float, float]:
    """Precision and recall when we act on the k highest-risk appointments."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    top_k = np.argsort(-scores)[:k]
    hits = y_true[top_k].sum()
    precision = hits / k
    recall = hits / y_true.sum() if y_true.sum() else 0.0
    return float(precision), float(recall)


def evaluate_scores(name: str, y_true, scores, k: int) -> dict:
    """Metrics for a model that outputs risk scores/probabilities."""
    p_at_k, r_at_k = precision_recall_at_k(y_true, scores, k)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y_true, scores), 3),
        "pr_auc": round(average_precision_score(y_true, scores), 3),
        f"precision@{k}": round(p_at_k, 3),
        f"recall@{k}": round(r_at_k, 3),
    }


def evaluate_rule(name: str, y_true, flags, k: int) -> dict:
    """Metrics for a binary rule (no scores -> no AUC).

    The rule usually flags far more than k appointments; precision/recall at its
    natural operating point is the fair way to report it.
    """
    return {
        "model": name,
        "roc_auc": None,
        "pr_auc": None,
        f"precision@{k}": round(precision_score(y_true, flags), 3),
        f"recall@{k}": round(recall_score(y_true, flags), 3),
        "note": f"rule flags {int(np.sum(flags))} appts (own operating point, not top-k)",
    }


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model")


def business_value(y_true, scores, k: int, revenue_per_slot: float = 200.0,
                   intervention_success_rate: float = 0.3, cost_per_call: float = 5.0) -> dict:
    """Translate precision@k into money. ALL THREE PARAMETERS ARE ASSUMPTIONS —
    say so out loud whenever you present this. The point is the framework, not
    the specific dollar figure.
    """
    p_at_k, _ = precision_recall_at_k(y_true, scores, k)
    true_noshows_reached = p_at_k * k
    slots_recovered = true_noshows_reached * intervention_success_rate
    return {
        "calls_made": k,
        "true_noshows_reached": round(true_noshows_reached, 1),
        "expected_slots_recovered": round(slots_recovered, 1),
        "expected_revenue_recovered": round(slots_recovered * revenue_per_slot, 0),
        "outreach_cost": k * cost_per_call,
        "assumptions": f"${revenue_per_slot}/slot, {intervention_success_rate:.0%} of reached "
                       f"no-shows convert to attendance, ${cost_per_call}/call",
    }
