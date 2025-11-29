from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    if np.allclose(rank_a.std(), 0) or np.allclose(rank_b.std(), 0):
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


@dataclass
class EvaluationSummary:
    weight_delta_max: float
    weight_delta_mean: float
    ranking_correlation: float
    score_shift_mean: float
    top_changes: Sequence[str]


class FuzzyPipelineEvaluator:
    def __init__(
        self,
        standard_weights: np.ndarray,
        ga_weights: np.ndarray,
        standard_scores: np.ndarray,
        ga_scores: np.ndarray,
        suppliers: Sequence[str],
        commodities: Sequence[str],
    ) -> None:
        self.standard_weights = np.asarray(standard_weights, dtype=float)
        self.ga_weights = np.asarray(ga_weights, dtype=float)
        self.standard_scores = np.asarray(standard_scores, dtype=float)
        self.ga_scores = np.asarray(ga_scores, dtype=float)
        self.suppliers = list(suppliers)
        self.commodities = list(commodities)

    def _top_rank_names(self, scores: np.ndarray, limit: int = 5) -> Sequence[str]:
        order = np.argsort(scores)[::-1][:limit]
        labels = []
        for idx in order:
            supplier = self.suppliers[idx] if idx < len(self.suppliers) else "Unknown"
            commodity = self.commodities[idx] if idx < len(self.commodities) else "Unknown"
            labels.append(f"{supplier} - {commodity}")
        return labels

    def generate_summary(self, top_k: int = 5) -> EvaluationSummary:
        weight_delta = np.abs(self.standard_weights - self.ga_weights)
        score_shift = np.abs(self.standard_scores - self.ga_scores)
        correlation = _spearman(self.standard_scores, self.ga_scores)

        std_top = set(self._top_rank_names(self.standard_scores, top_k))
        ga_top = set(self._top_rank_names(self.ga_scores, top_k))
        top_changes = list(ga_top.symmetric_difference(std_top))

        return EvaluationSummary(
            weight_delta_max=float(weight_delta.max(initial=0.0)),
            weight_delta_mean=float(weight_delta.mean() if len(weight_delta) else 0.0),
            ranking_correlation=float(correlation),
            score_shift_mean=float(score_shift.mean() if len(score_shift) else 0.0),
            top_changes=top_changes,
        )