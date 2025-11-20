import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
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

if str(PROJECT_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from model_evaluation_utils import (
    ensure_probability_matrix,
    evaluate_classification_metrics,
)

DEFAULT_DATA_PATH = config.PROCESSED_DATA_DIR / "syn_20000_engineered_features.csv"
DEFAULT_OUTPUT_PATH = config.MODELS_DIR / "xgboost_results.yaml"
DEFAULT_TARGET = "Risk_Classification"
CLASS_THRESHOLD = 15
IDENTIFIER_COLUMNS = [
    "Supplier_ID",
    "Commodity_ID",
    "Supplier_Name",
    "Commodity_Name",
]
MODEL_LABEL = "XGBoost"


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
        if n_classes is None or n_classes == 2:
            predicted_classes = (y_pred >= 0.5).astype(int)
        else:
            predicted_classes = np.argmax(y_pred, axis=1)
        metrics["accuracy"] = float(accuracy_score(y_true, predicted_classes))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, predicted_classes))
        metrics["f1_macro"] = float(f1_score(y_true, predicted_classes, average="macro"))
    return metrics


def train_xgboost(
    bundle: DatasetBundle,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    n_estimators: int,
    reg_lambda: float,
    gamma: float,
    n_jobs: int,
) -> Tuple[xgb.Booster, Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    train_dmatrix = xgb.DMatrix(bundle.X_train, label=bundle.y_train, feature_names=bundle.feature_names)
    val_dmatrix = xgb.DMatrix(bundle.X_val, label=bundle.y_val, feature_names=bundle.feature_names)
    test_dmatrix = xgb.DMatrix(bundle.X_test, label=bundle.y_test, feature_names=bundle.feature_names)

    if bundle.task_type == "regression":
        objective = "reg:squarederror"
        eval_metric = "rmse"
    else:
        if bundle.n_classes is None or bundle.n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "lambda": reg_lambda,
        "gamma": gamma,
        "nthread": n_jobs,
        "seed": config.RANDOM_SEED,
    }
    if bundle.task_type == "classification" and bundle.n_classes and bundle.n_classes > 2:
        params["num_class"] = bundle.n_classes

    evals_result: Dict[str, List[float]] = {}
    watchlist = [(train_dmatrix, "train"), (val_dmatrix, "validation")]

    booster = xgb.train(
        params=params,
        dtrain=train_dmatrix,
        evals=watchlist,
        num_boost_round=n_estimators,
        early_stopping_rounds=15,
        evals_result=evals_result,
        verbose_eval=False,
    )

    val_preds = booster.predict(val_dmatrix)
    test_preds = booster.predict(test_dmatrix)

    if bundle.task_type == "classification" and (bundle.n_classes is None or bundle.n_classes == 2):
        test_preds_for_metric = test_preds
        val_preds_for_metric = val_preds
    elif bundle.task_type == "classification":
        test_preds_for_metric = test_preds
        val_preds_for_metric = val_preds
    else:
        test_preds_for_metric = test_preds
        val_preds_for_metric = val_preds

    val_metrics = compute_metrics(bundle.task_type, bundle.y_val, val_preds_for_metric, bundle.n_classes)
    test_metrics = compute_metrics(bundle.task_type, bundle.y_test, test_preds_for_metric, bundle.n_classes)

    return booster, evals_result, val_metrics, test_metrics


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_results_yaml(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = _to_builtin(payload)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable_payload, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost model on the engineered synthetic dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to syn_20000_engineered_features.csv")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination YAML file for results.")
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET, help="Target column to predict.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for hold-out test set.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction for validation set (taken from train split).")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate (eta).")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of the training instances.")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Subsample ratio of columns when constructing each tree.")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Minimum sum of instance weight (hessian) needed in a child.")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization term on weights.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Minimum loss reduction required to make a further partition.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel threads used to run XGBoost.")
    parser.add_argument("--positive-class", type=str, default="High", help="Label treated as positive for Precision@K/Recall@K.")
    parser.add_argument("--top-k", type=int, default=500, help="Number of top predictions for Precision@K/Recall@K.")
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

    booster, evals_result, val_metrics, test_metrics = train_xgboost(
        bundle=bundle,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        n_estimators=args.n_estimators,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
    )
    evaluation_details = None
    if bundle.task_type == "classification":
        test_dmatrix = xgb.DMatrix(bundle.X_test, label=bundle.y_test, feature_names=bundle.feature_names)
        probabilities_raw = booster.predict(test_dmatrix)
        probabilities = ensure_probability_matrix(probabilities_raw, bundle.n_classes)
        extra_metrics, evaluation_details = evaluate_classification_metrics(
            MODEL_LABEL,
            bundle.y_test,
            probabilities,
            bundle.class_names,
            args.positive_class,
            args.top_k,
        )
        test_metrics.update(extra_metrics)

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
            "model": "XGBoost",
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_lambda": args.reg_lambda,
            "gamma": args.gamma,
            "n_jobs": args.n_jobs,
            "random_seed": config.RANDOM_SEED,
            "test_size": args.test_size,
            "val_size": args.val_size,
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "history": evals_result,
        "feature_names": bundle.feature_names,
        "best_iteration": booster.best_iteration,
    }
    if evaluation_details:
        payload["evaluation"] = evaluation_details

    save_results_yaml(args.output_path, payload)
    print(f"[+] XGBoost training complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()