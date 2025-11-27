# Oversampling has been chosen since the imbalance ratio is moderate
# Undersampling would not be favourable since it would result in data loss
# Techniques to be tested:
# 1. SMOTENC
# 2. ADASYN
# 3. SMOTE + ENN
# 4. CTGAN
# 5. TVAE

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import (
    FEATURE_COLUMNS,
    MODEL_CONFIG,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    VISUALIZATIONS_DIR,
)
from imb.adasyn_oversampler import create_adasyn_sampler
from imb.smote_enn_oversampler import create_smote_enn_sampler
from imb.smotenc_oversampler import create_smotenc_sampler
from imb.tvae_oversampler import create_tvae_sampler
from imb.ctgan_oversampler import create_ctgan_sampler

warnings.filterwarnings("ignore")

TARGET_COLUMN = "Risk_Classification"
GAN_EPOCHS = 150
CLASSIFIER_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    name: str
    macro_f1: float
    macro_auc_pr: float
    per_class_recall: Dict[str, float]
    confusion_matrix: np.ndarray
    visualization_path: Path

def load_dataset() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {PROCESSED_DATA_PATH}. "
            "Generate it via the preprocessing pipeline first."
        )
    logger.info("Loading dataset from %s", PROCESSED_DATA_PATH)
    return pd.read_csv(PROCESSED_DATA_PATH)

def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    available_features = [
        col
        for col in FEATURE_COLUMNS
        if col in df.columns and col != TARGET_COLUMN
    ]
    if not available_features:
        raise ValueError(
            "None of the configured FEATURE_COLUMNS are present in the dataset."
        )
    missing = sorted(set(FEATURE_COLUMNS) - set(available_features))
    if missing:
        logger.warning("The following configured features are missing: %s", ", ".join(missing))

    X = df[available_features].copy()
    id_columns = ['SC_ID', 'Supplier_Name', 'Commodity_Name']
    X = X.drop(columns=[col for col in id_columns if col in X.columns], errors='ignore')
    
    y = df[TARGET_COLUMN].astype(str).copy()
    return X, y

def ensure_directories() -> None:
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def determine_categorical_features(X: pd.DataFrame) -> List[str]:
    categorical_candidates = [
        "Compliance_Level",
        "Sustainability_Report_Availability",
    ]
    auto_detected = X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical = sorted(
        set(categorical_candidates).intersection(X.columns).union(auto_detected)
    )
    return categorical


def encode_categorical_for_sampling(
    X: pd.DataFrame, categorical_columns: List[str]
) -> tuple[pd.DataFrame, Dict[str, Union[pd.Index, np.ndarray]]]:
    if not categorical_columns:
        encoded = X.copy().apply(pd.to_numeric, errors="ignore")
        return encoded.astype(float), {}
    encoded = X.copy()
    mappings: Dict[str, Union[pd.Index, np.ndarray]] = {}
    for col in categorical_columns:
        codes, uniques = pd.factorize(encoded[col], sort=True)
        encoded[col] = pd.Series(codes, index=encoded.index, dtype=float)
        mappings[col] = uniques
    encoded = encoded.apply(pd.to_numeric, errors="coerce").astype(float)
    return encoded, mappings


def decode_categorical_from_sampling(
    df: pd.DataFrame, mappings: Dict[str, Union[pd.Index, np.ndarray]]
) -> pd.DataFrame:
    if not mappings:
        return df
    decoded = df.copy()
    for col, uniques in mappings.items():
        if col not in decoded.columns:
            continue
        values = decoded[col].to_numpy(copy=True)
        values = np.nan_to_num(values, nan=-1.0)
        codes = np.rint(values).astype(int)
        codes[codes < -1] = -1
        max_code = len(uniques) - 1
        if max_code >= 0:
            codes[codes > max_code] = max_code
        decoded_values = np.empty_like(codes, dtype=object)
        valid_mask = codes >= 0
        if valid_mask.any():
            decoded_values[valid_mask] = uniques.take(codes[valid_mask])
        decoded_values[~valid_mask] = pd.NA
        decoded[col] = decoded_values
    return decoded

def build_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(**CLASSIFIER_CONFIG)

def aggregate_confusion_matrix(
    matrices: List[np.ndarray],
) -> np.ndarray:
    return np.sum(matrices, axis=0)

def save_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: List[str],
    technique_name: str,
) -> Path:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{technique_name} - Aggregated Confusion Matrix")
    plt.tight_layout()
    filename = f"{technique_name.lower().replace(' ', '_')}_confusion_matrix.png"
    output_path = VISUALIZATIONS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def macro_auc_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    clf_classes: np.ndarray,
) -> float:
    class_to_index = {label: idx for idx, label in enumerate(clf_classes)}
    scores = []
    for idx, label in enumerate(classes):
        if label not in class_to_index:
            continue
        label_idx = class_to_index[label]
        true_binary = (y_true == label).astype(int)
        proba = y_proba[:, label_idx]
        if len(np.unique(true_binary)) < 2:
            continue
        score = average_precision_score(true_binary, proba)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0

def evaluate_sampler(
    name: str,
    sampler_fn: Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]],
    X: pd.DataFrame,
    y: pd.Series,
    class_labels: List[str],
    n_splits: int,
) -> EvaluationResult:
    logger.info("Evaluating %s with %d-fold stratified CV", name, n_splits)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )

    macro_f1_scores: List[float] = []
    macro_auc_scores: List[float] = []
    per_class_recall_sum = np.zeros(len(class_labels), dtype=float)
    confusion_matrices: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()

        X_resampled, y_resampled = sampler_fn(X_train, y_train)

        clf = build_classifier()
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        recalls = recall_score(
            y_test,
            y_pred,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
        auc_pr = macro_auc_pr(y_test.to_numpy(), y_proba, class_labels, clf.classes_)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)

        macro_f1_scores.append(macro_f1)
        macro_auc_scores.append(auc_pr)
        per_class_recall_sum += recalls
        confusion_matrices.append(cm)

        logger.debug(
            "%s - Fold %d: Macro F1=%.4f, Macro AUC-PR=%.4f",
            name,
            fold_idx,
            macro_f1,
            auc_pr,
        )

    avg_macro_f1 = float(np.mean(macro_f1_scores))
    avg_macro_auc = float(np.mean(macro_auc_scores))
    avg_recalls = (per_class_recall_sum / len(macro_f1_scores)).tolist()
    per_class_recall = {
        label: recall for label, recall in zip(class_labels, avg_recalls)
    }
    confusion_agg = aggregate_confusion_matrix(confusion_matrices)
    viz_path = save_confusion_matrix_plot(confusion_agg, class_labels, name)

    return EvaluationResult(
        name=name,
        macro_f1=avg_macro_f1,
        macro_auc_pr=avg_macro_auc,
        per_class_recall=per_class_recall,
        confusion_matrix=confusion_agg,
        visualization_path=viz_path,
    )

def create_ctgan_sampler(
    feature_columns: List[str],
    discrete_columns: List[str],
) -> Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Create CTGAN sampler (kept local as requested)."""

    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        train_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)

        for col in set(discrete_columns + [TARGET_COLUMN]):
            if col in train_df.columns:
                metadata.update_column(column_name=col, sdtype="categorical")

        model = CTGANSynthesizer(
            metadata=metadata,
            epochs=GAN_EPOCHS,
            verbose=False,
            batch_size=min(512, len(train_df)),
        )
        model.fit(train_df)

        counts = y.value_counts()
        max_count = counts.max()
        synthetic_parts = []

        for label, count in counts.items():
            deficit = int(max_count - count)
            if deficit <= 0:
                continue
            condition_df = pd.DataFrame({TARGET_COLUMN: [label] * deficit})
            try:
                synthetic = model.sample_from_conditions(
                    conditions=condition_df
                )
            except Exception:
                synthetic = model.sample(deficit)
                synthetic = synthetic[synthetic[TARGET_COLUMN] == label]
                if synthetic.empty:
                    continue
                synthetic = synthetic.head(deficit)
            synthetic_parts.append(synthetic)

        if synthetic_parts:
            synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
            augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        else:
            augmented_df = train_df

        X_aug = augmented_df[feature_columns].copy()
        y_aug = augmented_df[TARGET_COLUMN].copy()

        for col in X_aug.columns:
            X_aug[col] = pd.to_numeric(X_aug[col], errors="coerce")

        return X_aug, y_aug

    return _sampler

def evaluate_all(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_indices: List[int],
    discrete_columns: List[str],
    n_splits: int,
) -> Tuple[List[EvaluationResult], Dict[str, Callable]]:

    samplers: Dict[str, Callable[[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]] = {
        "SMOTENC": create_smotenc_sampler(categorical_indices or []),
        "ADASYN": create_adasyn_sampler(),
        "SMOTE + ENN": create_smote_enn_sampler(),
        "CTGAN": create_ctgan_sampler(list(X.columns), discrete_columns),
        "TVAE": create_tvae_sampler(
            list(X.columns),
            discrete_columns,
            TARGET_COLUMN,
            GAN_EPOCHS,
        ),
    }

    class_labels = sorted(y.unique())
    results: List[EvaluationResult] = []
    for name, sampler_fn in samplers.items():
        logger.info("\n========== %s OVERSAMPLING ==========", name.upper())
        try:
            result = evaluate_sampler(name, sampler_fn, X, y, class_labels, n_splits)
        except Exception as exc:
            logger.exception("Failed to evaluate %s: %s", name, exc)
            continue
        results.append(result)

    return results, samplers

def summarize_results(results: List[EvaluationResult]) -> None:
    table = PrettyTable()
    table.field_names = [
        "Technique",
        "Macro F1",
        "Macro AUC-PR",
        "Per-Class Recall",
        "Confusion Matrix Plot",
    ]
    for res in results:
        recall_summary = ", ".join(
            f"{cls}: {score:.2f}" for cls, score in res.per_class_recall.items()
        )
        table.add_row(
            [
                res.name,
                f"{res.macro_f1:.3f}",
                f"{res.macro_auc_pr:.3f}",
                recall_summary,
                res.visualization_path.name,
            ]
        )
    print(table)

def choose_best_result(results: List[EvaluationResult]) -> EvaluationResult:
    if not results:
        raise RuntimeError("No successful evaluation results were produced.")
    results_sorted = sorted(
        results,
        key=lambda res: (res.macro_f1, res.macro_auc_pr),
        reverse=True,
    )
    best = results_sorted[0]
    logger.info(
        "Best technique: %s (Macro F1=%.4f, Macro AUC-PR=%.4f)",
        best.name,
        best.macro_f1,
        best.macro_auc_pr,
    )
    return best

def run_oversampling(
    df: pd.DataFrame | None = None,
    *,
    split_before: bool = True,
) -> pd.DataFrame:
    ensure_directories()
    data = df.copy() if df is not None else load_dataset()
    X, y = select_features(data)
    categorical_columns = determine_categorical_features(X)
    encoded_X, categorical_mappings = encode_categorical_for_sampling(
        X, categorical_columns
    )
    categorical_indices = [
        encoded_X.columns.get_loc(col) for col in categorical_columns if col in encoded_X
    ]
    discrete_columns = categorical_columns.copy()

    if split_before:
        test_size = MODEL_CONFIG.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_X,
            y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        n_splits = MODEL_CONFIG.get("cv_folds", 5)

        results, samplers = evaluate_all(
            X_train,
            y_train,
            categorical_indices,
            discrete_columns,
            n_splits,
        )
        summarize_results(results)
        best_result = choose_best_result(results)
        best_sampler = samplers[best_result.name]
        X_best, y_best = best_sampler(X_train.copy(), y_train.copy())
        best_df = pd.concat(
            [X_best.reset_index(drop=True), y_best.reset_index(drop=True)],
            axis=1,
        )
        decoded_best_df = decode_categorical_from_sampling(best_df, categorical_mappings)

        logger.info(
            "Oversampling completed. Train size: %d (oversampled), Test size (untouched): %d",
            len(X_best),
            len(X_test),
        )
        return decoded_best_df

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Column '{TARGET_COLUMN}' must be present in the dataframe for oversampling."
        )

    X_train, y_train = encoded_X, y
    n_splits = MODEL_CONFIG.get("cv_folds", 5)

    results, samplers = evaluate_all(
        X_train,
        y_train,
        categorical_indices,
        discrete_columns,
        n_splits,
    )
    summarize_results(results)
    best_result = choose_best_result(results)
    best_sampler = samplers[best_result.name]
    X_best, y_best = best_sampler(X_train.copy(), y_train.copy())
    best_df = pd.concat(
        [X_best.reset_index(drop=True), y_best.reset_index(drop=True)],
        axis=1,
    )
    decoded_best_df = decode_categorical_from_sampling(best_df, categorical_mappings)

    logger.info(
        "Oversampling completed on provided dataset. Rows before: %d, after: %d",
        len(X_train),
        len(best_df),
    )
    return decoded_best_df

if __name__ == "__main__":
    run_oversampling()