import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

try:
    from src.utils import ensure_directory, print_section_header, print_progress
except ModuleNotFoundError:
    from utils import ensure_directory, print_section_header, print_progress


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "syn_20000_engineered_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
PLOT_PATH = OUTPUT_DIR / "top_10_critical_features.png"
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
    """Create the preprocessing and modeling pipeline."""

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
    """
    Aggregate permutation importances to the original feature level.

    For categorical columns, the mean absolute importance across all one-hot encoded
    categories is used to avoid bias toward features with many categories.
    """

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

    top_10 = importance_df.head(10)
    display_summary(top_10)
    plot_top_features(top_10, PLOT_PATH)


def main() -> None:
    run_feature_selection()


if __name__ == "__main__":
    main()

