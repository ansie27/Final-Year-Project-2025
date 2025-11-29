import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from prettytable import PrettyTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from mcdm.genetic_algorithm import GAParameters, WeightGAOptimizer
from utils.fuzzy_operations import TriangularFuzzyNumber, vertex_distance

DATA_PATH = Path(config.ENGINEERED_DATA_PATH)
WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "fuzzy_ahp_weights.json"
RANKING_OUTPUT_PATH = PROJECT_ROOT / "data" / "ranked" / "final_supplier_commodity_ranking.json"
RANDOM_STATE = 42

IDENTIFIER_COLUMNS = [
    "SC_ID",
    "Supplier_ID",
    "Supplier_Name",
    "Commodity_ID",
    "Commodity_Name",
]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_progress(message: str, **_: object) -> None:
    print(f"[Fuzzy TOPSIS] {message}")


def print_section_header(title: str) -> None:
    banner = "=" * len(title)
    print(f"\n{banner}\n{title}\n{banner}")


@dataclass
class RankingResult:
    supplier: str
    commodity: str
    closeness: float


def load_weights(path: Path) -> Dict[str, Dict[str, Sequence[Dict[str, float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Fuzzy AHP weights not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "standard" not in data or "ga_optimised" not in data:
        raise ValueError("Weights file must include 'standard' and 'ga_optimised' sections.")
    return data


def load_dataset(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    unique_columns = list(dict.fromkeys(columns))
    missing = [col for col in unique_columns if col not in df.columns]
    if missing:
        raise ValueError(f"The dataset is missing required columns: {', '.join(missing)}")
    return df


def fill_missing_values(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in feature_names:
        if df_copy[col].dtype == object or str(df_copy[col].dtype).startswith("category"):
            mode_value = df_copy[col].mode().iloc[0]
            df_copy[col] = df_copy[col].fillna(mode_value)
        else:
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    return df_copy


def encode_features(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in feature_names:
        if df_copy[col].dtype == object or str(df_copy[col].dtype).startswith("category"):
            df_copy[col] = df_copy[col].astype("category").cat.codes.astype(float)
        elif df_copy[col].dtype == bool:
            df_copy[col] = df_copy[col].astype(float)
        else:
            df_copy[col] = df_copy[col].astype(float)
    return df_copy


def normalise_matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    values = df[feature_names].to_numpy(dtype=float)
    min_vals = values.min(axis=0)
    max_vals = values.max(axis=0)
    denom = np.where(np.isclose(max_vals - min_vals, 0), 1.0, max_vals - min_vals)
    print_progress("Matrix normalised", step=3)
    return (values - min_vals) / denom


def build_fuzzy_matrix(normalised: np.ndarray, weights: np.ndarray) -> List[List[TriangularFuzzyNumber]]:
    weight_tfns = [TriangularFuzzyNumber(w, w, w) for w in weights]
    matrix: List[List[TriangularFuzzyNumber]] = []
    for row in normalised:
        fuzzy_row: List[TriangularFuzzyNumber] = []
        for value, weight_tfn in zip(row, weight_tfns):
            value_tfn = TriangularFuzzyNumber(value, value, value)
            fuzzy_row.append(value_tfn * weight_tfn)
        matrix.append(fuzzy_row)
    print_progress("Fuzzy decision matrix constructed")
    return matrix


def determine_ideal_solutions(
    weighted_matrix: List[List[TriangularFuzzyNumber]]
) -> Tuple[List[TriangularFuzzyNumber], List[TriangularFuzzyNumber]]:
    transposed = list(zip(*weighted_matrix))
    fpis: List[TriangularFuzzyNumber] = []
    fnis: List[TriangularFuzzyNumber] = []
    for column in transposed:
        l_values = [tfn.l for tfn in column]
        m_values = [tfn.m for tfn in column]
        u_values = [tfn.u for tfn in column]
        fpis.append(TriangularFuzzyNumber(max(l_values), max(m_values), max(u_values)))
        fnis.append(TriangularFuzzyNumber(min(l_values), min(m_values), min(u_values)))
    print_progress("FPIS and FNIS determined")
    return fpis, fnis


def calculate_distances(
    weighted_matrix: List[List[TriangularFuzzyNumber]],
    ideals: List[TriangularFuzzyNumber],
) -> np.ndarray:
    distances = []
    for row in weighted_matrix:
        total = sum(vertex_distance(cell, ideal) for cell, ideal in zip(row, ideals))
        distances.append(total)
    print_progress(f"Calculated distances for {len(distances)} alternatives")
    return np.asarray(distances, dtype=float)


def perform_fuzzy_topsis(
    normalised_matrix: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, List[TriangularFuzzyNumber], List[TriangularFuzzyNumber]]:
    print_progress("Building weighted matrix")
    weighted_matrix = build_fuzzy_matrix(normalised_matrix, weights)
    fpis, fnis = determine_ideal_solutions(weighted_matrix)
    d_plus = calculate_distances(weighted_matrix, fpis)
    d_minus = calculate_distances(weighted_matrix, fnis)
    print_progress("Computed closeness coefficients")
    closeness = d_minus / np.where((d_plus + d_minus) == 0, 1e-12, d_plus + d_minus)
    return closeness, fpis, fnis


def rank_results(closeness: np.ndarray, suppliers: Sequence[str], commodities: Sequence[str]) -> List[RankingResult]:
    order = np.argsort(closeness)[::-1]
    return [
        RankingResult(suppliers[idx], commodities[idx], float(closeness[idx])) for idx in order
    ]


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Arrays must be the same length for correlation.")
    if len(a) == 0:
        return 0.0

    def rankdata(values: np.ndarray) -> np.ndarray:
        temp = values.argsort(kind="mergesort")
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(values), dtype=float)
        unique, inverse = np.unique(values, return_inverse=True)
        for idx, _ in enumerate(unique):
            mask = inverse == idx
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    rank_a = rankdata(a)
    rank_b = rankdata(b)
    if np.allclose(rank_a.std(), 0) or np.allclose(rank_b.std(), 0):
        return 0.0
    corr_matrix = np.corrcoef(rank_a, rank_b)
    return float(corr_matrix[0, 1])


def ga_optimise_weights(
    normalised_matrix: np.ndarray,
    target_scores: np.ndarray,
    seed_weights: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    params = GAParameters(random_state=random_state, generations=40)
    optimizer = WeightGAOptimizer(params)

    def fitness(candidate: np.ndarray) -> float:
        candidate = candidate / candidate.sum()
        scores, _, _ = perform_fuzzy_topsis(normalised_matrix, candidate)
        correlation = spearman_correlation(scores, target_scores)
        if np.isnan(correlation):
            correlation = -1.0
        return -correlation  # minimise negative correlation (maximise stability)

    print_progress("GA optimisation started")
    return optimizer.optimise(seed_weights, fitness)


def serialize_tfn(tfn: TriangularFuzzyNumber) -> Tuple[float, float, float]:
    return (tfn.l, tfn.m, tfn.u)


def build_pretty_table(
    standard_results: List[RankingResult],
    ga_results: List[RankingResult],
    limit: int = 10,
) -> PrettyTable:
    limit = min(limit, max(len(standard_results), len(ga_results)))
    table = PrettyTable()
    table.field_names = [
        "Rank",
        "Std Supplier",
        "Std Commodity",
        "Std Closeness",
        "GA Supplier",
        "GA Commodity",
        "GA Closeness",
    ]
    for idx in range(limit):
        std_res = standard_results[idx] if idx < len(standard_results) else None
        ga_res = ga_results[idx] if idx < len(ga_results) else None
        table.add_row(
            [
                idx + 1,
                std_res.supplier if std_res else "-",
                std_res.commodity if std_res else "-",
                f"{std_res.closeness:.4f}" if std_res else "-",
                ga_res.supplier if ga_res else "-",
                ga_res.commodity if ga_res else "-",
                f"{ga_res.closeness:.4f}" if ga_res else "-",
            ]
        )
    return table


def save_rankings_json(
    standard_results: List[RankingResult],
    ga_results: List[RankingResult],
    fpis: List[TriangularFuzzyNumber],
    fnis: List[TriangularFuzzyNumber],
    output_path: Path,
) -> None:
    ensure_directory(output_path.parent)

    def serialize_results(results: List[RankingResult]) -> List[Dict[str, float | str]]:
        payload = []
        for rank, item in enumerate(results, start=1):
            payload.append(
                {
                    "rank": rank,
                    "supplier": item.supplier,
                    "commodity": item.commodity,
                    "closeness": item.closeness,
                }
            )
        return payload

    payload = {
        "standard": {
            "ranking": serialize_results(standard_results),
            "fpis": [serialize_tfn(tfn) for tfn in fpis],
            "fnis": [serialize_tfn(tfn) for tfn in fnis],
        },
        "ga_optimised": {
            "ranking": serialize_results(ga_results),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_progress(f"Saved final ranking to {output_path}")


def run_fuzzy_topsis(
    weights_data: Optional[Dict[str, Dict[str, Sequence[Dict[str, float]]]]] = None,
    data_path: Path = DATA_PATH,
) -> Dict[str, Any]:
    print_section_header("Fuzzy TOPSIS Analysis")
    print_progress("Loading weights and dataset")
    weights_data = weights_data or load_weights(WEIGHTS_PATH)
    standard_weight_entries = weights_data["standard"]["weights"]
    features = [entry["feature"] for entry in standard_weight_entries]

    required_columns = features + IDENTIFIER_COLUMNS
    df = load_dataset(data_path, required_columns)

    prepared_df = fill_missing_values(df, features)
    encoded_df = encode_features(prepared_df, features)
    normalised_matrix = normalise_matrix(encoded_df, features)

    supplier_series = (
        df["Supplier_Name"]
        if "Supplier_Name" in df.columns
        else df["Supplier_ID"]
    ).fillna("Unknown Supplier")
    commodity_series = (
        df["Commodity_Name"]
        if "Commodity_Name" in df.columns
        else df["Commodity_ID"]
    ).fillna("Unknown Commodity")

    suppliers = supplier_series.astype(str).tolist()
    commodities = commodity_series.astype(str).tolist()

    standard_weights = np.array([entry["weight"] for entry in standard_weight_entries], dtype=float)
    standard_weights = standard_weights / standard_weights.sum()

    print_progress("Running standard fuzzy TOPSIS")
    standard_closeness, fpis, fnis = perform_fuzzy_topsis(normalised_matrix, standard_weights)
    standard_rankings = rank_results(standard_closeness, suppliers, commodities)

    ga_weight_entries = weights_data["ga_optimised"]["weights"]
    ga_seed_weights = np.array([entry["weight"] for entry in ga_weight_entries], dtype=float)
    ga_seed_weights = ga_seed_weights / ga_seed_weights.sum()
    ga_target_scores, _, _ = perform_fuzzy_topsis(normalised_matrix, ga_seed_weights)

    print_progress("Optimising weights for GA fuzzy TOPSIS")
    ga_optimised_weights = ga_optimise_weights(
        normalised_matrix,
        ga_target_scores,
        seed_weights=ga_seed_weights,
        random_state=RANDOM_STATE,
    )
    ga_closeness, _, _ = perform_fuzzy_topsis(normalised_matrix, ga_optimised_weights)
    ga_rankings = rank_results(ga_closeness, suppliers, commodities)

    table = build_pretty_table(standard_rankings, ga_rankings, limit=10)
    print(table)

    save_rankings_json(standard_rankings, ga_rankings, fpis, fnis, RANKING_OUTPUT_PATH)

    return {
        "features": features,
        "suppliers": suppliers,
        "commodities": commodities,
        "standard_scores": standard_closeness,
        "ga_scores": ga_closeness,
        "standard_rankings": standard_rankings,
        "ga_rankings": ga_rankings,
        "weights_used": {
            "standard": standard_weights,
            "ga_seed": ga_seed_weights,
            "ga_optimised": ga_optimised_weights,
        },
    }


def main() -> None:
    run_fuzzy_topsis()


if __name__ == "__main__":
    main()

