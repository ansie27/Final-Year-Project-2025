import json
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import ensure_directory, print_progress, print_section_header
from utils.fuzzy_operations import TriangularFuzzyNumber, calculate_consistency_ratio

CRITICAL_FEATURES_PATH = PROJECT_ROOT / "outputs" / "critical_features.json"
VISUALIZATION_PATH = PROJECT_ROOT / "outputs" / "visualizations" / "fuzzy_ahp_results.png"
WEIGHTS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "fuzzy_ahp_weights.json"
RANDOM_STATE = 42

RATIO_THRESHOLDS: List[Tuple[float, int]] = [
    (1.15, 1),
    (1.35, 2),
    (1.55, 3),
    (1.75, 4),
    (2.0, 5),
    (2.5, 6),
    (3.5, 7),
    (4.5, 8),
]


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


def load_critical_features(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Critical features file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError("Critical features list is empty.")
    return data


def normalise_weights(importance_values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(importance_values, dtype=float)
    total = arr.sum()
    if total == 0:
        raise ValueError("Importance values sum to zero; cannot normalise.")
    return arr / total


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


def ga_optimise_weights(
    baseline_weights: np.ndarray,
    generations: int = 60,
    population_size: int = 40,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(baseline_weights)

    def initialise_population() -> np.ndarray:
        perturbation = rng.lognormal(mean=0.0, sigma=0.3, size=(population_size, n))
        population = baseline_weights * perturbation
        population += 1e-6
        return population

    def fitness(candidate: np.ndarray) -> float:
        weights = candidate / candidate.sum()
        matrix = weights[:, None] / weights[None, :]
        cr = calculate_consistency_ratio(matrix, weights)
        penalty = 0.1 * np.linalg.norm(weights - baseline_weights)
        return cr + penalty

    def tournament_select(population: np.ndarray, scores: np.ndarray, k: int = 3) -> np.ndarray:
        idx = rng.choice(len(population), size=k, replace=False)
        best_idx = idx[np.argmin(scores[idx])]
        return population[best_idx]

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        alpha = rng.random()
        return alpha * parent1 + (1 - alpha) * parent2

    def mutate(candidate: np.ndarray) -> np.ndarray:
        noise = rng.lognormal(mean=0.0, sigma=0.2, size=candidate.shape)
        mutated = candidate * (1 + mutation_rate * (noise - 1))
        mutated[mutated <= 0] = 1e-6
        return mutated

    population = initialise_population()
    best_candidate = population[0]
    best_score = fitness(best_candidate)

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in population])
        gen_best_idx = np.argmin(scores)
        if scores[gen_best_idx] < best_score:
            best_score = scores[gen_best_idx]
            best_candidate = population[gen_best_idx]

        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)

            child = parent1.copy()
            if rng.random() < crossover_rate:
                child = crossover(parent1, parent2)
            if rng.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = np.array(new_population)

    optimised = best_candidate / best_candidate.sum()
    return optimised


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


def save_weights_json(
    results: List[FuzzyAHPResult],
    standard_cr: float,
    ga_cr: float,
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)
    payload = {
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
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print_progress(f"Saved fuzzy AHP weights to {output_path}")


def run_fuzzy_ahp_analysis() -> None:
    print_section_header("Fuzzy AHP & GA-optimised Analysis")
    print_progress("Loading critical features")
    features = load_critical_features(CRITICAL_FEATURES_PATH)
    feature_names = [item["feature"] for item in features]
    baseline_importances = normalise_weights([item["importance"] for item in features])

    print_progress("Computing standard fuzzy AHP weights")
    fuzzy_matrix = build_fuzzy_pairwise_matrix(baseline_importances)
    _, standard_weights = compute_fuzzy_weights(fuzzy_matrix)
    standard_cr = evaluate_consistency(fuzzy_matrix, standard_weights)

    print_progress("Optimising weights via Genetic Algorithm")
    ga_weights = ga_optimise_weights(baseline_importances)
    ga_matrix = build_fuzzy_pairwise_matrix(ga_weights)
    _, ga_fuzzy_weights = compute_fuzzy_weights(ga_matrix)
    ga_cr = evaluate_consistency(ga_matrix, ga_fuzzy_weights)

    results = [
        FuzzyAHPResult(name, base, std, ga)
        for name, base, std, ga in zip(feature_names, baseline_importances, standard_weights, ga_fuzzy_weights)
    ]

    table = build_results_table(results, standard_cr, ga_cr)
    print(table)

    save_results_table_plot(results, standard_cr, ga_cr, VISUALIZATION_PATH)
    save_weights_json(results, standard_cr, ga_cr, WEIGHTS_OUTPUT_PATH)


def main() -> None:
    run_fuzzy_ahp_analysis()


if __name__ == "__main__":
    main()