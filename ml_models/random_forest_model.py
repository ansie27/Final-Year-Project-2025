import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

DEFAULT_DATA_PATH = config.PROCESSED_DATA_DIR / "syn_20000_engineered_features.csv"
DEFAULT_OUTPUT_PATH = config.MODELS_DIR / "random_forest_results.yaml"
DEFAULT_TARGET = "Risk_Classification"
CLASS_THRESHOLD = 15
IDENTIFIER_COLUMNS = [
    "Supplier_ID",
    "Commodity_ID",
    "Supplier_Name",
    "Commodity_Name",
]


def set_random_seed(seed: Optional[int] = None) -> None:
    resolved_seed = config.RANDOM_SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def detect_task_type(series: pd.Series, classification_threshold: int = CLASS_THRESHOLD) -> str:
    if series.dtype == object or series.dtype == "category":
        return "classification"
    if pd.api.types.is_bool_dtype(series):
        return "classification"
    unique_values = series.nunique()
    if unique_values <= classification_threshold:
        return "classification"
    return "regression"


def prepare_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    exclude_from_config = [col for col in config.EXCLUDE_COLUMNS if col in df.columns and col != target_column]
    exclude_identifiers = [col for col in IDENTIFIER_COLUMNS if col in df.columns and col != target_column]
    drop_columns = [target_column] + exclude_from_config + exclude_identifiers
    features = df.drop(columns=drop_columns, errors="ignore").copy()
    if features.shape[1] == 0:
        raise ValueError("No feature columns remain after applying exclusion rules.")
    target = df[target_column].copy()

    categorical_features = features.select_dtypes(include=["object", "category"]).columns
    if len(categorical_features) > 0:
        features = pd.get_dummies(features, columns=categorical_features, drop_first=True)

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return features, target


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    task_type: str
    n_classes: Optional[int] = None
    class_names: Optional[List[str]] = None


def create_data_bundle(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> DatasetBundle:
    if test_size + val_size >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0")

    feature_names = list(features.columns)
    task_type = detect_task_type(target)
    label_encoder: Optional[LabelEncoder] = None
    class_names: Optional[List[str]] = None

    if task_type == "classification":
        label_encoder = LabelEncoder()
        target = pd.Series(label_encoder.fit_transform(target), index=target.index)
        class_names = list(label_encoder.classes_)

    stratify_target = target if task_type == "classification" else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val if stratify_target is not None else None,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()

    if task_type == "regression":
        y_train_arr = y_train_arr.astype(np.float32)
        y_val_arr = y_val_arr.astype(np.float32)
        y_test_arr = y_test_arr.astype(np.float32)
    else:
        y_train_arr = y_train_arr.astype(np.int64)
        y_val_arr = y_val_arr.astype(np.int64)
        y_test_arr = y_test_arr.astype(np.int64)

    return DatasetBundle(
        X_train=X_train_s.astype(np.float32),
        X_val=X_val_s.astype(np.float32),
        X_test=X_test_s.astype(np.float32),
        y_train=y_train_arr,
        y_val=y_val_arr,
        y_test=y_test_arr,
        feature_names=feature_names,
        task_type=task_type,
        n_classes=len(class_names) if class_names else None,
        class_names=class_names,
    )


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: Optional[int] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if task_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    return metrics


def train_random_forest(
    bundle: DatasetBundle,
    n_estimators: int,
    max_depth: Optional[int],
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str,
    n_jobs: int,
) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    if bundle.task_type == "regression":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=config.RANDOM_SEED,
            n_jobs=n_jobs,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=n_jobs,
        )

    model.fit(bundle.X_train, bundle.y_train)

    val_predictions = model.predict(bundle.X_val)
    test_predictions = model.predict(bundle.X_test)

    val_metrics = compute_metrics(
        bundle.task_type,
        bundle.y_val,
        val_predictions,
        bundle.n_classes,
    )
    test_metrics = compute_metrics(
        bundle.task_type,
        bundle.y_test,
        test_predictions,
        bundle.n_classes,
    )
    return model, val_metrics, test_metrics


def serialize_feature_importance(model: object, feature_names: List[str]) -> List[Dict[str, float]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def save_results_yaml(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Random Forest on the engineered synthetic dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to syn_20000_engineered_features.csv")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination YAML file for results.")
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET, help="Target column to predict.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for hold-out test set.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction for validation set (taken from train split).")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth.")
    parser.add_argument("--min-samples-split", type=int, default=2, help="Minimum samples required to split an internal node.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required to be at a leaf node.")
    parser.add_argument("--max-features", type=str, default="sqrt", help="Number of features to consider when looking for the best split.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for training.")
    args = parser.parse_args()

    set_random_seed(config.RANDOM_SEED)

    dataset = load_dataset(args.data_path)
    if args.target_column not in dataset.columns:
        fallback_column = "Risk_Classification" if "Risk_Classification" in dataset.columns else None
        if fallback_column is None:
            raise ValueError(f"Target column '{args.target_column}' not found and no fallback target is available.")
        print(f"[!] Target '{args.target_column}' not found. Falling back to '{fallback_column}'.")
        args.target_column = fallback_column

    features, target = prepare_features(dataset, args.target_column)
    bundle = create_data_bundle(
        features=features,
        target=target,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=config.RANDOM_SEED,
    )

    model, val_metrics, test_metrics = train_random_forest(
        bundle=bundle,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
    )

    payload: Dict[str, Any] = {
        "dataset": {
            "path": str(Path(args.data_path).resolve()),
            "num_rows": int(len(dataset)),
            "num_features": len(bundle.feature_names),
            "target_column": args.target_column,
            "task_type": bundle.task_type,
            "class_names": bundle.class_names,
        },
        "training": {
            "model": "RandomForestRegressor" if bundle.task_type == "regression" else "RandomForestClassifier",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
            "n_jobs": args.n_jobs,
            "random_seed": config.RANDOM_SEED,
            "test_size": args.test_size,
            "val_size": args.val_size,
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "feature_importance": serialize_feature_importance(model, bundle.feature_names),
        "feature_names": bundle.feature_names,
    }

    save_results_yaml(args.output_path, payload)
    print(f"[+] Random Forest training complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()

