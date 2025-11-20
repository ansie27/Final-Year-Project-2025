# This modules selects the best-performing data generation model

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH

EVALUATION_RESULTS_PATH = PROJECT_ROOT / "outputs" / "synthetic_data_evaluation.json"
SYNTHETIC_OUTPUT_PATH = PROJECT_ROOT / "data" / "synthetic" / "syn_20000_data.csv"
SYNTH_ROWS = 20_000
IDENTIFIER_COLUMNS = ("Supplier_ID", "Supplier_Name", "Commodity_ID", "Commodity_Name")
ORDINAL_INT_COLUMNS = (
    "Supplier_Tier",
    "Compliance_Level",
    "Sustainability_Report_Availability",
    "Incident_History_Count",
)
EXCLUDED_COLUMNS = (
    "Supplier_ID",
    "Supplier_Name",
    "Commodity_ID",
    "Commodity_Name",
)

MODEL_REGISTRY = {
    "CTGAN": {
        "class": CTGANSynthesizer,
        "kwargs": {
            "epochs": 300,
            "verbose": True,
        },
    },
    "TVAE": {
        "class": TVAESynthesizer,
        "kwargs": {
            "epochs": 300,
            "verbose": True,
        },
    },
}


def load_processed_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.convert_dtypes()
    drop_cols = [col for col in EXCLUDED_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    int_cols = df.select_dtypes(include=["Int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype("int64")
    return df


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    for col in ORDINAL_INT_COLUMNS:
        if col in df.columns:
            metadata.update_column(column_name=col, sdtype="numerical")
    return metadata


METRIC_PATHS = {
    "ks": ("statistical", "ks"),
    "chi_square": ("statistical", "chi_square"),
    "js_divergence": ("statistical", "js_divergence"),
    "pearson_diff": ("correlation", "pearson_diff"),
    "spearman_diff": ("correlation", "spearman_diff"),
    "wasserstein": ("distribution", "wasserstein"),
    "mse": ("distribution", "mse"),
    "tstr_accuracy": ("ml_utility", "tstr_accuracy"),
    "trts_accuracy": ("ml_utility", "trts_accuracy"),
}

METRIC_DIRECTIONS = {
    "ks": "lower",
    "chi_square": "lower",
    "js_divergence": "lower",
    "pearson_diff": "lower",
    "spearman_diff": "lower",
    "wasserstein": "lower",
    "mse": "lower",
    "tstr_accuracy": "higher",
    "trts_accuracy": "higher",
}


def load_evaluation_results(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation results not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_metrics(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, List[float]]] = {}
    for entry in results:
        model = entry["model_type"]
        aggregates.setdefault(model, {metric: [] for metric in METRIC_PATHS})
        metrics = entry["metrics"]
        for metric, path in METRIC_PATHS.items():
            value = metrics
            for key in path:
                value = value[key]
            aggregates[model][metric].append(value)

    summary: Dict[str, Dict[str, float]] = {}
    for model, metric_values in aggregates.items():
        summary[model] = {
            metric: float(np.nanmean(values)) if values else np.nan
            for metric, values in metric_values.items()
        }
    return summary


def determine_best_model(summary: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, int]]:
    if len(summary) < 2:
        raise ValueError("Need at least two models to compare.")

    scores = {model: 0 for model in summary}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = {model: summary[model][metric] for model in summary}
        if direction == "lower":
            best_value = min(values.values())
            winners = [m for m, v in values.items() if np.isclose(v, best_value)]
        else:
            best_value = max(values.values())
            winners = [m for m, v in values.items() if np.isclose(v, best_value)]
        for winner in winners:
            scores[winner] += 1

    def tie_breaker(model: str) -> float:
        return summary[model]["tstr_accuracy"] + summary[model]["trts_accuracy"]

    best_model = max(scores, key=lambda m: (scores[m], tie_breaker(m)))
    return best_model, scores


def add_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    supplier_ids = []
    commodity_ids = []
    supplier_names = []
    commodity_names = []
    for i in range(n_rows):
        idx = i + 1
        supplier_ids.append(f"SYN_SUP_{idx:05d}")
        commodity_ids.append(f"SYN_COM_{idx:05d}")
        supplier_names.append(f"Synthetic Supplier {idx}")
        commodity_names.append(f"Synthetic Commodity {idx}")

    df["Supplier_ID"] = supplier_ids
    df["Commodity_ID"] = commodity_ids
    df["Supplier_Name"] = supplier_names
    df["Commodity_Name"] = commodity_names
    return df


def generate_synthetic_rows(model_name: str, num_rows: int) -> pd.DataFrame:
    clean_df = load_processed_data(PROCESSED_DATA_PATH)
    metadata = build_metadata(clean_df)

    registry_entry = MODEL_REGISTRY[model_name]
    model_class = registry_entry["class"]
    kwargs = registry_entry["kwargs"]
    synthesizer = model_class(metadata, **kwargs)
    synthesizer.fit(clean_df)
    synthetic_df = synthesizer.sample(num_rows=num_rows)
    return synthetic_df


def integrate_and_save(best_model: str) -> Path:
    synthetic_core = generate_synthetic_rows(best_model, SYNTH_ROWS)
    synthetic_full = add_identifier_columns(synthetic_core)
    SYNTHETIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic_full.to_csv(SYNTHETIC_OUTPUT_PATH, index=False)
    return SYNTHETIC_OUTPUT_PATH


def main() -> None:
    results = load_evaluation_results(EVALUATION_RESULTS_PATH)
    summary = aggregate_metrics(results)
    best_model, scores = determine_best_model(summary)

    print("Model comparison summary:")
    for model, metrics in summary.items():
        print(f"- {model}: score={scores[model]}, metrics={metrics}")

    print(f"\nSelected model: {best_model}")
    output_path = integrate_and_save(best_model)
    print(f"Synthetic + real dataset saved to {output_path}")


if __name__ == "__main__":
    main()