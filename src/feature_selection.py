import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Ensure the project root is available on the Python path when executing this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from ml_models.ann_model import ANNTrainingConfig, run_ann_training
from ml_models.random_forest_model import (
    RandomForestTrainingConfig,
    run_random_forest_training,
)
from ml_models.xgboost_model import XGBoostTrainingConfig, run_xgboost_training
from src.utils import ensure_directory, print_progress, print_section_header


DATA_PATH = Path(config.ENGINEERED_DATA_PATH)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
PLOT_PATH = OUTPUT_DIR / "top_10_critical_features.png"
CRITICAL_FEATURES_PATH = PROJECT_ROOT / "outputs" / "critical_features.json"
SELECTED_FEATURE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "engineered_top10_features_dataset.csv"
)
MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model_comparison" / "feature_subset_comparison.json"
TARGET_COLUMN = "Risk_Classification"
IDENTIFIER_COLUMNS = [
    "Supplier_ID",
    "Commodity_ID",
    "Supplier_Name",
    "Commodity_Name",
]
RANDOM_STATE = 42
MAX_PFI_ITERATIONS = 5
PFI_STABILITY_TOLERANCE = 5e-4
PRIMARY_METRICS_ORDER = [
    "f1_macro",
    "balanced_accuracy",
    "accuracy",
    "r2",
    "rmse",
    "mae",
]
LOWER_IS_BETTER = {"rmse", "mae", "mse"}
BASELINE_RESULT_PATHS: Dict[str, Path] = {
    "random_forest": config.MODELS_DIR / "random_forest_results.yaml",
    "xgboost": config.MODELS_DIR / "xgboost_results.yaml",
    "ann": config.MODELS_DIR / "ann_results.yaml",
}
TOP_FEATURE_RESULT_PATHS: Dict[str, Path] = {
    "random_forest": config.MODELS_DIR / "random_forest_top_features.yaml",
    "xgboost": config.MODELS_DIR / "xgboost_top_features.yaml",
    "ann": config.MODELS_DIR / "ann_top_features.yaml",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the engineered dataset and validate its presence."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate predictors and target column while dropping identifier fields."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")
    X = df.drop(columns=[TARGET_COLUMN])
    removed_cols = [col for col in IDENTIFIER_COLUMNS if col in X.columns]
    if removed_cols:
        X = X.drop(columns=removed_cols)
    y = df[TARGET_COLUMN]
    return X, y, removed_cols


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify categorical and numerical features."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    if not categorical_cols and not numeric_cols:
        raise ValueError("No features found for modeling.")

    return categorical_cols, numeric_cols


def build_model_pipeline(
    categorical_cols: List[str], numeric_cols: List[str], model_random_state: int | None = None
) -> Pipeline:
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))

    preprocessor = ColumnTransformer(transformers)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=model_random_state if model_random_state is not None else RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

def aggregate_feature_importances(
    importances: np.ndarray,
    categorical_cols: List[str],
    numeric_cols: List[str],
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    aggregated: Dict[str, float] = {}
    processed_index = 0

    if categorical_cols:
        encoder: OneHotEncoder = preprocessor.named_transformers_["categorical"]  # type: ignore[assignment]
        categories_per_feature = encoder.categories_
        for col_idx, col in enumerate(categorical_cols):
            n_categories = len(categories_per_feature[col_idx])
            col_importances = importances[processed_index : processed_index + n_categories]
            aggregated[col] = float(np.abs(col_importances).mean()) if n_categories else 0.0
            processed_index += n_categories

    if numeric_cols:
        numeric_importances = importances[processed_index : processed_index + len(numeric_cols)]
        for col, importance in zip(numeric_cols, numeric_importances):
            aggregated[col] = float(abs(importance))

    importance_df = (
        pd.DataFrame({"feature": list(aggregated.keys()), "importance": list(aggregated.values())})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df

def compute_permutation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> pd.DataFrame:
    """Fit the pipeline and calculate permutation feature importance."""

    pipeline.fit(X_train, y_train)
    result = permutation_importance(
        pipeline,
        X_valid,
        y_valid,
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring=None,
    )

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return aggregate_feature_importances(
        result.importances_mean, categorical_cols, numeric_cols, preprocessor
    )

def calculate_importance_delta(previous: pd.DataFrame, current: pd.DataFrame) -> float:
    """Calculate mean absolute change between two importance rankings."""
    merged = (
        previous.merge(current, on="feature", how="outer", suffixes=("_prev", "_curr"))
        .fillna(0.0)
    )
    return float(np.abs(merged["importance_prev"] - merged["importance_curr"]).mean())

def compute_stable_permutation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
    max_iterations: int = MAX_PFI_ITERATIONS,
    tolerance: float = PFI_STABILITY_TOLERANCE,
) -> pd.DataFrame:
    """
    Re-run permutation importance until results stabilize or the iteration cap is reached.

    The final importance values are the mean importances across all completed iterations.
    """

    history: List[pd.DataFrame] = []
    previous_run: pd.DataFrame | None = None

    for iteration in range(1, max_iterations + 1):
        iteration_pipeline = clone(pipeline)
        model = iteration_pipeline.named_steps.get("model")
        if hasattr(model, "random_state"):
            setattr(model, "random_state", RANDOM_STATE + iteration)

        print_progress(f"PFI iteration {iteration}/{max_iterations}")
        current_run = compute_permutation_importance(
            iteration_pipeline,
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_cols,
            numeric_cols,
        )

        history.append(current_run.assign(iteration=iteration))

        if previous_run is not None:
            delta = calculate_importance_delta(previous_run, current_run)
            print_progress(f"Delta importance vs previous iteration: {delta:.6f}")
            if delta < tolerance:
                print_progress("Stability criterion met; stopping iterations.")
                break
        else:
            print_progress("Baseline PFI established.")

        previous_run = current_run

    combined = pd.concat(history, ignore_index=True)
    aggregated = (
        combined.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    print_progress(f"Averaged permutation importance across {len(history)} iteration(s).")
    return aggregated


def plot_top_features(top_features: pd.DataFrame, output_path: Path) -> None:
    """Create and save a horizontal bar chart of feature importances."""
    ensure_directory(output_path.parent)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_features["feature"][::-1], top_features["importance"][::-1], color="#2f7ed8")
    plt.xlabel("Permutation Importance (absolute mean)")
    plt.title("Top 10 Critical Features by Permutation Feature Importance")
    plt.tight_layout()

    # Annotate bars with values
    for bar, value in zip(bars, top_features["importance"][::-1]):
        plt.text(
            bar.get_width() + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {output_path}")

def display_summary(top_features: pd.DataFrame) -> None:
    """Print the top features to the terminal."""
    print_section_header("Top 10 Critical Features (Permutation Importance)")
    for rank, row in enumerate(top_features.itertuples(index=False), start=1):
        print(f"{rank:>2}. {row.feature:<40} Importance: {row.importance:.5f}")


def save_top_features_json(top_features: pd.DataFrame, output_path: Path) -> None:
    """Persist the top feature importances as JSON."""
    ensure_directory(output_path.parent)
    records = top_features.to_dict(orient="records")
    output_path.write_text(json.dumps(records, indent=2))
    print_progress(f"Saved critical features to {output_path}")


def save_selected_feature_dataset(
    df: pd.DataFrame,
    selected_features: Sequence[str],
    target_column: str,
    output_path: Path,
) -> Path:
    missing = [feature for feature in selected_features if feature not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from dataset: {', '.join(missing)}")
    columns = list(dict.fromkeys([*selected_features, target_column]))
    subset = df[columns].copy()
    ensure_directory(output_path.parent)
    subset.to_csv(output_path, index=False)
    print_progress(f"Saved selected-feature dataset to {output_path}")
    return output_path


def load_model_results(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} results file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def retrain_models_with_dataset(dataset_path: Path, suffix: str) -> Dict[str, Dict[str, Any]]:
    label_fragment = suffix.replace(" ", "_")
    print_progress(f"Retraining models using dataset '{dataset_path.name}' ({suffix})")

    rf_cfg = RandomForestTrainingConfig(
        data_path=dataset_path,
        output_path=TOP_FEATURE_RESULT_PATHS["random_forest"],
    )
    rf_result = run_random_forest_training(rf_cfg)

    xgb_cfg = XGBoostTrainingConfig(
        data_path=dataset_path,
        output_path=TOP_FEATURE_RESULT_PATHS["xgboost"],
    )
    xgb_result = run_xgboost_training(xgb_cfg)

    ann_cfg = ANNTrainingConfig(
        data_path=dataset_path,
        output_path=TOP_FEATURE_RESULT_PATHS["ann"],
    )
    ann_result = run_ann_training(ann_cfg)

    return {
        "random_forest": rf_result,
        "xgboost": xgb_result,
        "ann": ann_result,
    }


def load_baseline_model_suite() -> Dict[str, Dict[str, Any]]:
    return {
        "random_forest": load_model_results(BASELINE_RESULT_PATHS["random_forest"], "Random Forest"),
        "xgboost": load_model_results(BASELINE_RESULT_PATHS["xgboost"], "XGBoost"),
        "ann": load_model_results(BASELINE_RESULT_PATHS["ann"], "ANN"),
    }


def extract_primary_metric(result: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    metrics = result.get("metrics", {}).get("test", {})
    for metric in PRIMARY_METRICS_ORDER:
        if metric in metrics:
            return metric, float(metrics[metric])
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            return metric, float(value)
    return None, None


def compare_model_performance(
    baseline_suite: Dict[str, Dict[str, Any]],
    subset_suite: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    comparisons: Dict[str, Dict[str, Any]] = {}
    improvement_notes: List[str] = []
    degradation_notes: List[str] = []

    for model_key, baseline_result in baseline_suite.items():
        subset_result = subset_suite.get(model_key)
        if subset_result is None:
            continue
        metric_name, baseline_value = extract_primary_metric(baseline_result)
        _, subset_value = extract_primary_metric(subset_result)
        if metric_name is None or baseline_value is None or subset_value is None:
            continue
        direction = -1.0 if metric_name in LOWER_IS_BETTER else 1.0
        raw_delta = subset_value - baseline_value
        directional_delta = raw_delta * direction

        if directional_delta > 0:
            improvement_notes.append(f"{model_key} (+{abs(raw_delta):.4f} {metric_name})")
        elif directional_delta < 0:
            degradation_notes.append(f"{model_key} (-{abs(raw_delta):.4f} {metric_name})")

        comparisons[model_key] = {
            "metric": metric_name,
            "all_features": baseline_value,
            "top_features": subset_value,
            "difference": raw_delta,
            "directional_improvement": directional_delta,
        }

    if improvement_notes and not degradation_notes:
        conclusion = "All models improved when restricted to the top-10 features."
    elif degradation_notes and not improvement_notes:
        conclusion = "All models underperformed when limited to the top-10 features."
    elif improvement_notes or degradation_notes:
        conclusion = (
            "Mixed impact: "
            + ("improvements in " + ", ".join(improvement_notes) if improvement_notes else "")
            + ("; degradations in " + ", ".join(degradation_notes) if degradation_notes else "")
        )
    else:
        conclusion = "No measurable performance change between full and top-10 feature sets."

    return {
        "models": comparisons,
        "conclusion": conclusion,
    }


def persist_comparison_summary(summary: Dict[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(summary, indent=2))
    print_progress(f"Performance comparison saved to {path}")


def run_feature_selection() -> None:
    """Execute the full permutation feature importance workflow."""
    print_section_header("Permutation Feature Importance Analysis")
    print_progress("Loading dataset")
    df = load_dataset(DATA_PATH)
    print_progress(f"Dataset loaded with shape {df.shape}")

    print_progress("Separating features and target", step=1, total=4)
    X, y, dropped_cols = split_features_and_target(df)
    if dropped_cols:
        print_progress(f"Ignored identifier columns: {', '.join(dropped_cols)}")
    categorical_cols, numeric_cols = identify_feature_types(X)

    print_progress("Preparing train/test split", step=2, total=4)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print_progress("Building modeling pipeline template", step=3, total=4)
    pipeline = build_model_pipeline(categorical_cols, numeric_cols)

    print_progress("Computing stabilized permutation feature importance", step=4, total=4)
    importance_df = compute_stable_permutation_importance(
        pipeline,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_cols,
        numeric_cols,
    )

    top_10 = importance_df.head(10).reset_index(drop=True)
    save_top_features_json(top_10, CRITICAL_FEATURES_PATH)
    display_summary(top_10)
    plot_top_features(top_10, PLOT_PATH)

    selected_features = top_10["feature"].tolist()
    save_selected_feature_dataset(df, selected_features, TARGET_COLUMN, SELECTED_FEATURE_DATASET_PATH)

    baseline_suite = load_baseline_model_suite()
    subset_suite = retrain_models_with_dataset(SELECTED_FEATURE_DATASET_PATH, "top_features")
    comparison_summary = compare_model_performance(baseline_suite, subset_suite)
    persist_comparison_summary(comparison_summary, MODEL_COMPARISON_PATH)

    print_section_header("Model performance: All features vs Top-10 subset")
    for model_key, details in comparison_summary["models"].items():
        delta = details["difference"]
        metric = details["metric"]
        print(
            f"{model_key.title():<15} | {metric}: full={details['all_features']:.4f} "
            f"vs top10={details['top_features']:.4f} (Δ={delta:+.4f})"
        )
    print_progress(comparison_summary["conclusion"])


def main() -> None:
    run_feature_selection()


if __name__ == "__main__":
    main()

