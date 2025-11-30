import json
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from mcdm.genetic_algorithm import GAParameters, WeightGAOptimizer
from utils.fuzzy_operations import TriangularFuzzyNumber, calculate_consistency_ratio

ENGINEERED_DATA_PATH = Path(config.ENGINEERED_DATA_PATH)
CRITICAL_FEATURES_PATH = PROJECT_ROOT / "outputs" / "critical_features.json"
VISUALIZATION_PATH = PROJECT_ROOT / "outputs" / "visualizations" / "fuzzy_ahp_results.png"
WEIGHTS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "fuzzy_ahp_weights.json"
RANDOM_STATE = 42
IDENTIFIER_COLUMNS = {
    "SC_ID",
    "Supplier_Name",
    "Commodity_Name",
    "Risk_Classification",
}

def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def print_progress(message: str) -> None:
    print(f"[Fuzzy AHP] {message}")

def print_section_header(title: str) -> None:
    banner = "=" * len(title)
    print(f"\n{banner}\n{title}\n{banner}")

RATIO_THRESHOLDS: List[Tuple[float, int]] = [
    (1.0, 1),           # Equal importance
    (1.5, 2),           # Weak importance
    (2.5, 3),           # Moderate importance
    (3.5, 4),           # Moderate to strong importance
    (4.5, 5),           # Strong importance
    (6.0, 6),           # Strong to very strong importance
    (7.5, 7),           # Very strong importance
    (8.5, 8),           # Very very strong importance
    (float('inf'), 9),  # Extreme importance
]

def ratio_to_fuzzy_saathy(ratio: float) -> TriangularFuzzyNumber:
    """Convert numerical ratio to fuzzy Saaty scale (1-9)"""
    if ratio <= 0:
        raise ValueError("Pairwise ratios must be positive.")
    
    # Handle equal importance
    if np.isclose(ratio, 1.0, atol=0.1):
        return TriangularFuzzyNumber.from_saaty_scale(1)
    
    # Handle reciprocal (inverse comparisons)
    if ratio < 1:
        return ratio_to_fuzzy_saathy(1 / ratio).reciprocal()
    
    # Map ratio to Saaty scale
    for threshold, scale in RATIO_THRESHOLDS:
        if ratio <= threshold:
            return TriangularFuzzyNumber.from_saaty_scale(scale)
    
    return TriangularFuzzyNumber.from_saaty_scale(9)

@dataclass
class FuzzyAHPResult:
    feature: str
    baseline_weight: float
    standard_weight: float
    ga_weight: float


def ratio_to_fuzzy_saathy(ratio: float) -> TriangularFuzzyNumber:
    if ratio <= 0:
        raise ValueError("Pairwise ratios must be positive.")
    if np.isclose(ratio, 1.0, atol=1e-3):
        return TriangularFuzzyNumber.from_saaty_scale(1)
    if ratio < 1:
        return ratio_to_fuzzy_saathy(1 / ratio).reciprocal()
    for threshold, scale in RATIO_THRESHOLDS:
        if ratio < threshold:
            return TriangularFuzzyNumber.from_saaty_scale(scale)
    return TriangularFuzzyNumber.from_saaty_scale(9)

def load_baseline_importances(
    dataset_path: Path = ENGINEERED_DATA_PATH,
) -> Tuple[List[str], np.ndarray]:

    if not dataset_path.exists():
        raise FileNotFoundError(f"Engineered dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    candidate_features = [
        col for col in getattr(config, "FEATURE_COLUMNS", []) if col in df.columns
    ]

    if not candidate_features:
        numeric_cols = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in IDENTIFIER_COLUMNS
        ]
        candidate_features = numeric_cols

    if not candidate_features:
        raise ValueError("No numeric features available for fuzzy AHP analysis.")

    # Use equal weights as neutral baseline
    n_features = len(candidate_features)
    baseline = np.ones(n_features) / n_features
    
    # If there is expert judgement and/or domain knowledge
    # baseline = np.array([get_expert_weight(f) for f in candidate_features])
    # baseline /= baseline.sum()

    feature_snapshot = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(candidate_features, baseline)
    ]
    ensure_directory(CRITICAL_FEATURES_PATH.parent)
    with CRITICAL_FEATURES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(feature_snapshot, handle, indent=2)

    return candidate_features, baseline

def build_fuzzy_pairwise_matrix(weights: Sequence[float]) -> List[List[TriangularFuzzyNumber]]:
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    matrix: List[List[TriangularFuzzyNumber]] = [
        [TriangularFuzzyNumber.from_saaty_scale(1) for _ in range(n)] for _ in range(n)
    ]
    for i in range(n):
        for j in range(i + 1, n):
            ratio = float(weights[i] / weights[j])
            fuzzy_value = ratio_to_fuzzy_saathy(ratio)
            matrix[i][j] = fuzzy_value
            matrix[j][i] = fuzzy_value.reciprocal()
    return matrix

def fuzzy_geometric_means(matrix: List[List[TriangularFuzzyNumber]]) -> List[TriangularFuzzyNumber]:
    n = len(matrix)
    geometric_means: List[TriangularFuzzyNumber] = []
    for row in matrix:
        l_product = np.prod([entry.l for entry in row])
        m_product = np.prod([entry.m for entry in row])
        u_product = np.prod([entry.u for entry in row])
        gm = TriangularFuzzyNumber(
            l_product ** (1 / n),
            m_product ** (1 / n),
            u_product ** (1 / n),
        )
        geometric_means.append(gm)
    return geometric_means

# Compute FAHP weights with Chang's extent analysis (2020)
def compute_fuzzy_weights(matrix: List[List[TriangularFuzzyNumber]]) -> Tuple[List[TriangularFuzzyNumber], np.ndarray]:
    gms = fuzzy_geometric_means(matrix)
    total = reduce(lambda acc, val: acc + val, gms[1:], gms[0])
    fuzzy_weights = [gm / total for gm in gms]
    crisp = np.array([fw.defuzzify() for fw in fuzzy_weights], dtype=float)
    crisp /= crisp.sum()
    
    return fuzzy_weights, crisp

def crisp_pairwise_from_fuzzy(matrix: List[List[TriangularFuzzyNumber]]) -> np.ndarray:
    return np.array([[entry.defuzzify() for entry in row] for row in matrix], dtype=float)

def evaluate_consistency(matrix: List[List[TriangularFuzzyNumber]], weights: np.ndarray) -> float:
    crisp_matrix = crisp_pairwise_from_fuzzy(matrix)
    return calculate_consistency_ratio(crisp_matrix, weights)

def build_results_table(results: List[FuzzyAHPResult], standard_cr: float, ga_cr: float) -> PrettyTable:
    table = PrettyTable()
    table.field_names = ["Feature", "Baseline Importance", "Fuzzy AHP Weight", "GA Fuzzy AHP Weight"]
    for item in results:
        table.add_row(
            [
                item.feature,
                f"{item.baseline_weight:.4f}",
                f"{item.standard_weight:.4f}",
                f"{item.ga_weight:.4f}",
            ]
        )
    table.add_row(["Consistency Ratio", "", f"{standard_cr:.4f}", f"{ga_cr:.4f}"])
    return table

def save_results_table_plot(results: List[FuzzyAHPResult], standard_cr: float, ga_cr: float, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(12, 0.5 * (len(results) + 3)))
    ax.axis("off")
    row_data = [
        [
            res.feature,
            f"{res.baseline_weight:.4f}",
            f"{res.standard_weight:.4f}",
            f"{res.ga_weight:.4f}",
        ]
        for res in results
    ]
    row_data.append(["Consistency Ratio", "", f"{standard_cr:.4f}", f"{ga_cr:.4f}"])
    table = ax.table(
        cellText=row_data,
        colLabels=["Feature", "Baseline Importance", "Fuzzy AHP Weight", "GA Fuzzy AHP Weight"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print_progress(f"Saved Fuzzy AHP table to {output_path}")


def build_weights_payload(
    results: List[FuzzyAHPResult],
    standard_cr: float,
    ga_cr: float,
) -> Dict[str, Dict[str, Sequence[Dict[str, float]] | float]]:
    return {
        "standard": {
            "weights": [
                {"feature": res.feature, "weight": res.standard_weight} for res in results
            ],
            "consistency_ratio": standard_cr,
        },
        "ga_optimised": {
            "weights": [{"feature": res.feature, "weight": res.ga_weight} for res in results],
            "consistency_ratio": ga_cr,
        },
    }


def save_weights_json(payload: Dict[str, Any], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print_progress(f"Saved fuzzy AHP weights to {output_path}")


def summarise_weight_differences(
    standard: np.ndarray,
    ga: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    deltas = np.abs(standard - ga)
    summary = {
        "max_delta": float(deltas.max(initial=0.0)),
        "mean_delta": float(deltas.mean() if len(deltas) else 0.0),
        "changed_features": [
            {"feature": name, "delta": float(delta)}
            for name, delta in sorted(
                zip(feature_names, deltas), key=lambda item: item[1], reverse=True
            )
            if delta > 1e-4
        ],
    }
    return summary

# Execute FAHP
def run_fuzzy_ahp_analysis(
    dataset_path: Path = ENGINEERED_DATA_PATH,
) -> Dict[str, Any]:
    print_section_header("Fuzzy AHP and GA-optimised Analysis")
    print_progress("Deriving baseline importances from engineered dataset")
    feature_names, baseline_importances = load_baseline_importances(dataset_path)

    print_progress("Computing standard fuzzy AHP weights")
    fuzzy_matrix = build_fuzzy_pairwise_matrix(baseline_importances)
    _, standard_weights = compute_fuzzy_weights(fuzzy_matrix)
    standard_cr = evaluate_consistency(fuzzy_matrix, standard_weights)

    print_progress("Optimising weights via Genetic Algorithm")
    ga_seed_weights = np.ones(len(feature_names)) / (len(feature_names))

    def _fitness(candidate: np.ndarray) -> float:
        weights = candidate / candidate.sum()
        # Build a pairwise matrix for FAHP weights
        n = len(weights)
        matrix = np.zeros((n,n))
        for i in range (n):
            for j in range (n):
                if weights[j] > 0:
                    matrix[i,j] = weights[i] / weights[j]
                else:
                    matrix[i,j] = 1.0
        cr = calculate_consistency_ratio(matrix, weights)

        # Penalise for deviating too much from basline for GA
        deviation = np.linalg.norm(weights - baseline_importances)
        return cr + 0.05 * deviation # minimise CR and control the deviation

    optimizer = WeightGAOptimizer(GAParameters(random_state=RANDOM_STATE))
    ga_weights = optimizer.optimise(ga_seed_weights, _fitness)
    ga_matrix = build_fuzzy_pairwise_matrix(ga_weights)
    _, ga_fuzzy_weights = compute_fuzzy_weights(ga_matrix)
    ga_cr = evaluate_consistency(ga_matrix, ga_fuzzy_weights)

    results = [
        FuzzyAHPResult(name, base, std, ga)
        for name, base, std, ga in zip(
            feature_names, baseline_importances, standard_weights, ga_fuzzy_weights
        )
    ]

    table = build_results_table(results, standard_cr, ga_cr)
    print(table)

    weights_payload = build_weights_payload(results, standard_cr, ga_cr)
    save_results_table_plot(results, standard_cr, ga_cr, VISUALIZATION_PATH)
    save_weights_json(weights_payload, WEIGHTS_OUTPUT_PATH)

    diff_summary = summarise_weight_differences(standard_weights, ga_fuzzy_weights, feature_names)
    if diff_summary["changed_features"]:
        top_feature = diff_summary["changed_features"][0]
        print_progress(
            f"Greatest GA shift observed on '{top_feature['feature']}' "
            f"with ={top_feature['delta']:.4f}"
        )
    else:
        print_progress("GA optimisation produced negligible changes to weights")

    return {
        "feature_names": feature_names,
        "baseline_importances": baseline_importances,
        "standard_weights": standard_weights,
        "ga_weights": ga_fuzzy_weights,
        "consistency": {"standard": standard_cr, "ga": ga_cr},
        "weights_payload": weights_payload,
        "difference_summary": diff_summary,
    }


def main() -> None:
    run_fuzzy_ahp_analysis()


if __name__ == "__main__":
    main()