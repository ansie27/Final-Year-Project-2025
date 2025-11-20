import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from xgboost import XGBClassifier
import logging
logging.getLogger("shap").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

ID_COLUMNS = ["Supplier_ID", "Commodity_ID", "Supplier_Name", "Commodity_Name"]
TARGET_COLUMN = "Risk_Classification"
TOP_FEATURES = 10


def load_dataset() -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    data_path = config.PROCESSED_DATA_DIR / "syn_20000_engineered_features.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    exclude_cols = set(config.EXCLUDE_COLUMNS + ID_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude_cols | {TARGET_COLUMN}]
    features = df[feature_cols].copy()
    target = df[TARGET_COLUMN].copy()

    categorical_cols = features.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(target.values)
    class_names = list(encoder.classes_)

    return features, y, list(features.columns), class_names


def build_ann(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summarize_top_features(shap_values, feature_names: List[str], top_n: int) -> List[Tuple[str, float]]:
    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(values) for values in shap_values], axis=0)
    else:
        shap_abs = np.abs(shap_values)
    mean_importance = shap_abs.mean(axis=0)
    indices = np.argsort(mean_importance)[::-1][:top_n]
    return [(feature_names[i], float(mean_importance[i])) for i in indices]


def save_shap_plots(model_name: str, shap_values, sample_data: np.ndarray, feature_names: List[str]) -> None:
    output_dir = config.VISUALIZATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = model_name.lower().replace(" ", "_")

    plt.figure()
    shap.summary_plot(shap_values, sample_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_shap_beeswarm.png", dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample_data, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_shap_bar.png", dpi=300)
    plt.close()


def compute_tree_shap(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    model_name: str,
) -> List[Tuple[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    save_shap_plots(model_name, shap_values, X_sample, feature_names)
    return summarize_top_features(shap_values, feature_names, TOP_FEATURES)


def compute_deep_shap(
    model: keras.Model,
    X_background: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    model_name: str,
) -> List[Tuple[str, float]]:
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_sample)
    save_shap_plots(model_name, shap_values, X_sample, feature_names)
    return summarize_top_features(shap_values, feature_names, TOP_FEATURES)


def build_comparison_table(top_features: Dict[str, List[Tuple[str, float]]]) -> PrettyTable:
    table = PrettyTable()
    headers = ["Rank", "ANN", "Random Forest", "XGBoost"]
    table.field_names = headers

    for idx in range(TOP_FEATURES):
        row = [idx + 1]
        for model in ["ANN", "Random Forest", "XGBoost"]:
            items = top_features.get(model, [])
            if idx < len(items):
                feature, value = items[idx]
                row.append(f"{feature} ({value:.4f})")
            else:
                row.append("-")
        table.add_row(row)
    return table


def main() -> None:
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)

    features, labels, feature_names, class_names = load_dataset()
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=labels,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train_df, y_train)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        gamma=0.0,
        n_jobs=-1,
        random_state=config.RANDOM_SEED,
        tree_method="hist",
        objective="multi:softprob",
        num_class=len(class_names),
    )
    xgb_model.fit(X_train_df, y_train)

    scaler = StandardScaler()
    X_train_ann = scaler.fit_transform(X_train_df.values)
    X_test_ann = scaler.transform(X_test_df.values)

    ann_model = build_ann(X_train_ann.shape[1], len(class_names))
    ann_model.fit(
        X_train_ann,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=128,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    sample_size = min(1000, len(X_test_df))
    sample_df = X_test_df.sample(n=sample_size, random_state=config.RANDOM_SEED)
    sample_ann = scaler.transform(sample_df.values)
    sample_tree = sample_df.values

    top_features: Dict[str, List[Tuple[str, float]]] = {}
    top_features["Random Forest"] = compute_tree_shap(rf_model, sample_tree, feature_names, "Random Forest")
    top_features["XGBoost"] = compute_tree_shap(xgb_model, sample_tree, feature_names, "XGBoost")
    top_features["ANN"] = compute_deep_shap(
        ann_model,
        X_train_ann[: min(200, len(X_train_ann))],
        sample_ann,
        feature_names,
        "ANN",
    )

    ann_probs = ann_model.predict(X_test_ann, verbose=0)
    ann_preds = np.argmax(ann_probs, axis=1)
    ann_accuracy = accuracy_score(y_test, ann_preds)

    print(f"ANN evaluation accuracy: {ann_accuracy:.4f}")
    print("Top risk-driving features per model (mean |SHAP|):")
    comparison_table = build_comparison_table(top_features)
    print(comparison_table)
    print(f"SHAP visualizations saved to: {config.VISUALIZATIONS_DIR}")


if __name__ == "__main__":
    main()

