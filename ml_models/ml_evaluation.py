import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import yaml
from prettytable import PrettyTable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RESULTS = {
    "ANN": PROJECT_ROOT / "outputs" / "models" / "ann_results.yaml",
    "Random Forest": PROJECT_ROOT / "outputs" / "models" / "random_forest_results.yaml",
    "XGBoost": PROJECT_ROOT / "outputs" / "models" / "xgboost_results.yaml",
}

METRIC_SPECS = [
    ("accuracy", "Accuracy", True),
    ("balanced_accuracy", "Balanced Accuracy", True),
    ("f1_macro", "F1 (Macro)", True),
    ("matthews_corrcoef", "MCC", True),
    ("roc_auc_macro", "ROC AUC (Macro)", True),
    ("precision_at_k", "Precision@K", True),
    ("recall_at_k", "Recall@K", True),
    ("top_k", "Top-K", False),
]


def _load_yaml_metrics(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_metrics(data: Dict) -> Dict[str, float]:
    metrics = data.get("metrics", {})
    if "test" in metrics:
        metrics = metrics["test"]
    extracted: Dict[str, float] = {}
    for key, _, _ in METRIC_SPECS:
        value = metrics.get(key)
        extracted[key] = float(value) if value is not None else float("nan")
    return extracted


def build_summary_table(results_map: Dict[str, Path]) -> Tuple[PrettyTable, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, float]] = {}
    table = PrettyTable()
    table.field_names = ["Model"] + [label for _, label, _ in METRIC_SPECS]
    table.float_format = ".4"

    for model_name, path in results_map.items():
        data = _load_yaml_metrics(Path(path))
        metrics = _extract_metrics(data)
        summary[model_name] = metrics
        row = [model_name] + [metrics.get(key, float("nan")) for key, _, _ in METRIC_SPECS]
        table.add_row(row)
    return table, summary


def highlight_best_models(summary: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    best: Dict[str, str] = {}
    for key, label, eligible in METRIC_SPECS:
        if not eligible:
            continue
        candidates = []
        for model, metrics in summary.items():
            value = metrics.get(key)
            if value is not None and not math.isnan(value):
                candidates.append((model, value))
        if not candidates:
            continue
        best_model = max(candidates, key=lambda item: item[1])[0]
        best[label] = best_model
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ANN, Random Forest, and XGBoost risk prediction results.")
    parser.add_argument(
        "--ann-results",
        type=Path,
        default=DEFAULT_RESULTS["ANN"],
        help="Path to ANN results YAML file.",
    )
    parser.add_argument(
        "--rf-results",
        type=Path,
        default=DEFAULT_RESULTS["Random Forest"],
        help="Path to Random Forest results YAML file.",
    )
    parser.add_argument(
        "--xgb-results",
        type=Path,
        default=DEFAULT_RESULTS["XGBoost"],
        help="Path to XGBoost results YAML file.",
    )
    args = parser.parse_args()

    results_paths = {
        "ANN": args.ann_results,
        "Random Forest": args.rf_results,
        "XGBoost": args.xgb_results,
    }

    comparison_table, summary = build_summary_table(results_paths)
    best_models = highlight_best_models(summary)
    overall_best = max(
        summary.items(),
        key=lambda item: sum(
            value
            for key, _, eligible in METRIC_SPECS
            if eligible and not math.isnan(value := item[1].get(key, float("nan")))
        ),
    )[0]

    print("\nRISK PREDICTION MODEL COMPARISON")
    print(comparison_table)
    if best_models:
        print("Best models per metric:")
        for metric_label, model in best_models.items():
            print(f"  - {metric_label}: {model}")
    print(f"\nOverall best-suited model for risk prediction: {overall_best}")


if __name__ == "__main__":
    main()
