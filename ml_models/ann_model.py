import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
utils_path = PROJECT_ROOT / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from model_evaluation_utils import (
    ensure_probability_matrix,
    evaluate_classification_metrics,
)

import config
DEFAULT_DATA_PATH = config.PROCESSED_DATA_DIR / "syn_20000_engineered_features.csv"
DEFAULT_OUTPUT_PATH = config.MODELS_DIR / "ann_results.yaml"
DEFAULT_TARGET = "Risk_Classification"
CLASS_THRESHOLD = 15
IDENTIFIER_COLUMNS = [
    "Supplier_ID",
    "Commodity_ID",
    "Supplier_Name",
    "Commodity_Name",
]
MODEL_LABEL = "ANN"


def set_random_seeds(seed: Optional[int] = None) -> None:
    resolved_seed = config.RANDOM_SEED if seed is None else seed
    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    if tf is not None:
        tf.random.set_seed(resolved_seed)
    if torch is not None:
        torch.manual_seed(resolved_seed)
        torch.cuda.manual_seed_all(resolved_seed)


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


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
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


def build_keras_model(
    input_dim: int,
    task_type: str,
    output_dim: int,
    hidden_layers: Iterable[int],
    dropout: float,
    learning_rate: float,
) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))

    if task_type == "regression":
        model.add(keras.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = ["mae"]
    else:
        if output_dim == 1:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(keras.layers.Dense(output_dim, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


class TorchMLP(nn.Module):  # pragma: no cover - torch optional
    def __init__(self, input_dim: int, hidden_layers: Iterable[int], dropout: float, output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_units = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_units = units
        layers.append(nn.Linear(prev_units, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs)


def tensorflow_training(
    data: DatasetBundle,
    hidden_layers: Tuple[int, ...],
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if tf is None or keras is None:
        raise ImportError("TensorFlow/Keras is not available. Install tensorflow to use this backend.")

    output_dim = 1 if data.task_type == "regression" or (data.n_classes == 2 or data.n_classes is None) else int(data.n_classes)
    model = build_keras_model(
        input_dim=data.X_train.shape[1],
        task_type=data.task_type,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    predictions = model.predict(data.X_test, verbose=0)

    history_dict = {key: [float(v) for v in values] for key, values in history.history.items()}

    return predictions.squeeze(), history_dict


def torch_training(
    data: DatasetBundle,
    hidden_layers: Tuple[int, ...],
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    patience: int,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is not available. Install torch to use this backend.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim: int
    if data.task_type == "regression":
        output_dim = 1
    elif data.n_classes is None or data.n_classes == 2:
        output_dim = 1
    else:
        output_dim = int(data.n_classes)

    model = TorchMLP(
        input_dim=data.X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        output_dim=output_dim,
    ).to(device)

    if data.task_type == "regression":
        loss_fn = nn.MSELoss()
    elif data.n_classes == 2 or data.n_classes is None:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _feature_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).float()

    def _target_tensor(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if data.task_type == "regression":
            return tensor.float().unsqueeze(-1)
        if data.n_classes == 2 or data.n_classes is None:
            return tensor.float().unsqueeze(-1)
        return tensor.long()

    train_loader = DataLoader(
        TensorDataset(
            _feature_tensor(data.X_train),
            _target_tensor(data.y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            _feature_tensor(data.X_val),
            _target_tensor(data.y_val),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    history: Dict[str, List[float]] = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y_tensor = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            if data.task_type == "regression":
                loss = loss_fn(outputs, batch_y_tensor)
            elif data.n_classes == 2 or data.n_classes is None:
                loss = loss_fn(outputs, batch_y_tensor)
            else:
                loss = loss_fn(outputs, batch_y_tensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)
        history["loss"].append(float(avg_train_loss))

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y_tensor = batch_y.to(device)
                outputs = model(batch_X)
                val_loss_accum += loss_fn(outputs, batch_y_tensor).item()
        avg_val_loss = val_loss_accum / max(len(val_loader), 1)
        history["val_loss"].append(float(avg_val_loss))

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if "best_state" in locals():
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_inputs = _feature_tensor(data.X_test).to(device)
        outputs = model(test_inputs)
        if data.task_type == "regression":
            predictions = outputs.cpu().numpy().squeeze()
        elif data.n_classes == 2 or data.n_classes is None:
            predictions = torch.sigmoid(outputs).cpu().numpy().squeeze()
        else:
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()

    return predictions, history


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


def serialize_history(history: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {key: [float(v) for v in values] for key, values in history.items()}


def save_results_yaml(
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ANN on synthetic engineered dataset.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to syn_20000_engineered_features.csv")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination YAML file for results.")
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET, help="Target column to predict.")
    parser.add_argument("--backend", type=str, choices=["tensorflow", "pytorch"], default="tensorflow", help="Deep learning backend.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for hold-out test set.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction for validation set (taken from train split).")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 64], help="Hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--positive-class", type=str, default="High", help="Label treated as positive for Precision@K/Recall@K.")
    parser.add_argument("--top-k", type=int, default=500, help="Number of top predictions for Precision@K/Recall@K.")
    args = parser.parse_args()

    backend = args.backend
    if backend == "tensorflow" and tf is None:
        backend = "pytorch"
    if backend == "pytorch" and torch is None:
        backend = "tensorflow"
    if backend == "tensorflow" and tf is None:
        raise EnvironmentError("Neither TensorFlow nor PyTorch is available.")
    if backend == "pytorch" and torch is None:
        raise EnvironmentError("Neither PyTorch nor TensorFlow is available.")

    set_random_seeds(config.RANDOM_SEED)

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

    hidden_layers = tuple(args.hidden_layers)

    if backend == "tensorflow":
        raw_predictions, history = tensorflow_training(
            data=bundle,
            hidden_layers=hidden_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    else:
        raw_predictions, history = torch_training(
            data=bundle,
            hidden_layers=hidden_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        )

    metrics = compute_metrics(
        task_type=bundle.task_type,
        y_true=bundle.y_test,
        y_pred=raw_predictions,
        n_classes=bundle.n_classes,
    )
    evaluation_details = None
    if bundle.task_type == "classification":
        probabilities = ensure_probability_matrix(raw_predictions, bundle.n_classes)
        extra_metrics, evaluation_details = evaluate_classification_metrics(
            MODEL_LABEL,
            bundle.y_test,
            probabilities,
            bundle.class_names,
            args.positive_class,
            args.top_k,
        )
        metrics.update(extra_metrics)

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
            "backend": backend,
            "hidden_layers": list(hidden_layers),
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "random_seed": config.RANDOM_SEED,
        },
        "metrics": metrics,
        "history": serialize_history(history),
        "feature_names": bundle.feature_names,
    }
    if evaluation_details:
        payload["evaluation"] = evaluation_details

    save_results_yaml(args.output_path, payload)
    print(f"[+] ANN training complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
