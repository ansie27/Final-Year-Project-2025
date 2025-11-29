# Oversampling has been chosen since the imbalance ratio is moderate
# Undersampling would not be favourable since it would result in data loss
# Techniques to be tested:
# 1. SMOTENC
# 2. ADASYN

import logging
import warnings
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN, SMOTENC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_COLUMNS,
    MODEL_CONFIG,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    VISUALIZATIONS_DIR,
)

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
TARGET_COLUMN = "Risk_Classification"

CLASSIFIER_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

IMBLEARN_SAMPLER_CONFIG = {
    "random_state": RANDOM_SEED,
    "sampling_strategy": "auto",
}

# Constraint columns that must maintain valid combinations
CONSTRAINT_COLUMNS = ["Supplier_Name", "Country", "Region"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== DATA CLASSES ====================
@dataclass
class EvaluationResult:
    name: str
    macro_f1: float
    macro_auc_pr: float
    per_class_recall: Dict[str, float]
    confusion_matrix: np.ndarray
    visualization_path: Path

# ==================== UTILITY FUNCTIONS ====================
def load_dataset() -> pd.DataFrame:
    """Load preprocessed dataset."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {PROCESSED_DATA_PATH}. "
            "Run preprocessing pipeline first."
        )
    logger.info("Loading dataset from %s", PROCESSED_DATA_PATH)
    return pd.read_csv(PROCESSED_DATA_PATH)

def ensure_directories() -> None:
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    available_features = [
        col for col in FEATURE_COLUMNS
        if col in df.columns and col != TARGET_COLUMN
    ]
    
    if not available_features:
        raise ValueError("No valid feature columns found in dataset.")
    
    X = df[available_features].copy()
    
    # Remove SC_ID
    id_columns = ['SC_ID']
    X = X.drop(columns=[col for col in id_columns if col in X.columns], errors='ignore')
    
    y = df[TARGET_COLUMN].astype(str).copy()
    
    return X, y

def determine_categorical_features(X: pd.DataFrame) -> List[str]:
    """Identify categorical columns in the dataset."""
    # Manually specified categorical columns
    categorical_candidates = [
        "Supplier_Name",
        "Country",
        "Region",
        "Industry_Sector",
        "Supplier_Tier",
        "Commodity_Name",
        "Compliance_Level",
        "Sustainability_Report_Availability",
        "Certifications_Active",
    ]
    
    # Auto-detect object/category types
    auto_detected = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Combine and filter to existing columns
    categorical = sorted(
        set(categorical_candidates).intersection(X.columns).union(auto_detected)
    )
    
    return categorical

# Encode categorical columns for oversampling purposes
def encode_categorical_features(
    X: pd.DataFrame,
    categorical_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, pd.Index]]:
    if not categorical_columns:
        return X.copy().astype(float), {}
    
    encoded = X.copy()
    mappings = {}
    
    for col in categorical_columns:
        if col not in encoded.columns:
            continue
        codes, uniques = pd.factorize(encoded[col], sort=True)
        encoded[col] = codes
        mappings[col] = uniques
    
    # Ensure all numeric
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    
    return encoded, mappings

def decode_categorical_features(
    df: pd.DataFrame,
    mappings: Dict[str, pd.Index]
) -> pd.DataFrame:
    """Decode integer-encoded categorical columns back to original values."""
    if not mappings:
        return df
    
    decoded = df.copy()
    
    for col, uniques in mappings.items():
        if col not in decoded.columns:
            continue
        
        codes = decoded[col].round().astype(int)
        codes = codes.clip(0, len(uniques) - 1)  # Clip to valid range
        decoded[col] = uniques.take(codes)
    
    return decoded


def validate_supplier_location_combinations(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove synthetic samples with invalid Supplier-Country-Region combinations."""
    constraint_cols = [col for col in CONSTRAINT_COLUMNS if col in df.columns]
    
    if not constraint_cols:
        logger.warning("No constraint columns found for validation")
        return df
    
    # Get valid combinations from original data
    valid_combos = original_df[constraint_cols].drop_duplicates()
    
    # Merge and filter
    merged = df.merge(
        valid_combos,
        on=constraint_cols,
        how='left',
        indicator=True
    )
    
    valid_df = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])
    invalid_count = len(df) - len(valid_df)
    
    if invalid_count > 0:
        logger.warning(
            "Removed %d synthetic samples with invalid %s combinations (%.1f%%)",
            invalid_count,
            '-'.join(constraint_cols),
            (invalid_count / len(df)) * 100
        )
    
    return valid_df

# Reassign SC_ID columns after oversampling
def add_sc_id_column(df: pd.DataFrame, start_from: int = 1) -> pd.DataFrame:

    num_records = len(df)
    sc_ids = [f"SC{str(i).zfill(7)}" for i in range(start_from, start_from + num_records)]

    df_with_id = df.copy()
    df_with_id.insert(0, 'SC_ID', sc_ids)
    
    logger.info("Added SC_ID column: %s to %s", sc_ids[0], sc_ids[-1])
    
    return df_with_id
# ==================== EVALUATION ====================
def build_classifier() -> RandomForestClassifier:
    """Build Random Forest classifier for evaluation."""
    return RandomForestClassifier(**CLASSIFIER_CONFIG)

def save_confusion_matrix_plot(
    matrix: np.ndarray,
    labels: List[str],
    technique_name: str,
) -> Path:
    """Save confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
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
    plt.title(f"{technique_name} - Confusion Matrix")
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
    """Calculate macro-averaged AUC-PR score."""
    class_to_index = {label: idx for idx, label in enumerate(clf_classes)}
    scores = []
    
    for label in classes:
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
    """Evaluate oversampling technique using stratified k-fold CV."""
    logger.info("Evaluating %s with %d-fold CV", name, n_splits)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    macro_f1_scores = []
    macro_auc_scores = []
    per_class_recall_sum = np.zeros(len(class_labels))
    confusion_matrices = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()
        
        # Apply oversampling
        X_resampled, y_resampled = sampler_fn(X_train, y_train)
        
        # Train classifier
        clf = build_classifier()
        clf.fit(X_resampled, y_resampled)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        # Calculate metrics
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        recalls = recall_score(y_test, y_pred, labels=class_labels, average=None, zero_division=0)
        auc_pr = macro_auc_pr(y_test.to_numpy(), y_proba, class_labels, clf.classes_)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        
        macro_f1_scores.append(macro_f1)
        macro_auc_scores.append(auc_pr)
        per_class_recall_sum += recalls
        confusion_matrices.append(cm)
        
        logger.debug("%s - Fold %d: F1=%.4f, AUC-PR=%.4f", name, fold_idx, macro_f1, auc_pr)
    
    # Aggregate results
    avg_macro_f1 = float(np.mean(macro_f1_scores))
    avg_macro_auc = float(np.mean(macro_auc_scores))
    avg_recalls = (per_class_recall_sum / n_splits).tolist()
    per_class_recall = {label: recall for label, recall in zip(class_labels, avg_recalls)}
    confusion_agg = np.sum(confusion_matrices, axis=0)
    
    # Save visualization
    viz_path = save_confusion_matrix_plot(confusion_agg, class_labels, name)
    
    return EvaluationResult(
        name=name,
        macro_f1=avg_macro_f1,
        macro_auc_pr=avg_macro_auc,
        per_class_recall=per_class_recall,
        confusion_matrix=confusion_agg,
        visualization_path=viz_path,
    )

def calculate_balanced_sampling_strategy(
    y: pd.Series,
    min_total_samples: int = 50000,
    balance_method: str = "proportional"
) -> Dict[str, int]:
    class_counts = y.value_counts().to_dict()
    n_classes = len(class_counts)
    
    logger.info("Original class distribution:")
    for cls, count in class_counts.items():
        logger.info("  %s: %d (%.1f%%)", cls, count, (count/len(y))*100)
    
    if balance_method == "equal":
        target_per_class = max(
            max(class_counts.values()),  
            min_total_samples // n_classes  
        )
        
        sampling_strategy = {cls: target_per_class for cls in class_counts.keys()}
        
    elif balance_method == "proportional":
        majority_count = max(class_counts.values())
        target_minority = int(majority_count * 0.8)
        
        sampling_strategy = {}
        for cls, count in class_counts.items():
            if count < target_minority:
                sampling_strategy[cls] = target_minority
            else:
                sampling_strategy[cls] = count
        
        # Check if we need more samples to reach min_total_samples
        total_after = sum(sampling_strategy.values())
        if total_after < min_total_samples:
            # Scale up all classes proportionally
            scale_factor = min_total_samples / total_after
            sampling_strategy = {
                cls: int(count * scale_factor) 
                for cls, count in sampling_strategy.items()
            }
    
    else:
        raise ValueError(f"Unknown balance_method: {balance_method}")
    
    total_target = sum(sampling_strategy.values())
    logger.info("\nTarget class distribution:")
    for cls, count in sampling_strategy.items():
        logger.info("  %s: %d (%.1f%%)", cls, count, (count/total_target)*100)
    logger.info("Total target samples: %d", total_target)
    
    return sampling_strategy

# ==================== OVERSAMPLING TECHNIQUES ====================
def create_smotenc_sampler(
    categorical_indices: List[int],
    sampling_strategy: Dict[str, int],
    random_state: int,
) -> Callable:
    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=min(5, len(y) - 1),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    
    return _sampler

def create_adasyn_sampler(
    sampling_strategy: Dict[str, int],
    random_state: int,
) -> Callable:
    """Create ADASYN sampler with custom sampling strategy."""
    def _sampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        sampler = ADASYN(
            random_state=random_state,
            sampling_strategy=sampling_strategy,
            n_neighbors=min(5, len(y) - 1),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
    
    return _sampler

# ==================== MAIN WORKFLOW ====================
def evaluate_all_samplers(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_indices: List[int],
    sampling_strategy: Dict[str, int],
    n_splits: int,
) -> Tuple[List[EvaluationResult], Dict[str, Callable]]:
    """Evaluate all oversampling techniques."""
    
    samplers = {
        "SMOTENC": create_smotenc_sampler(
            categorical_indices, 
            sampling_strategy,
            RANDOM_SEED
        ),
        "ADASYN": create_adasyn_sampler(
            sampling_strategy,
            RANDOM_SEED
        ),
    }
    
    class_labels = sorted(y.unique())
    results = []
    
    for name, sampler_fn in samplers.items():
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING: %s", name.upper())
        logger.info("=" * 60)

        try:
            result = evaluate_sampler(name, sampler_fn, X, y, class_labels, n_splits)
            results.append(result)
        except Exception as exc:
            logger.exception("Failed to evaluate %s: %s", name, exc)
    
    return results, samplers

def summarise_results(results: List[EvaluationResult]) -> None:
    if not results:
        logger.error("No results to display")
        return
    
    table = PrettyTable()
    table.field_names = [
        "Technique",
        "Macro F1",
        "Macro AUC-PR",
        "Per-Class Recall",
    ]
    
    for res in results:
        recall_str = ", ".join(f"{cls[:3]}:{score:.2f}" for cls, score in res.per_class_recall.items())
        table.add_row([
            res.name,
            f"{res.macro_f1:.4f}",
            f"{res.macro_auc_pr:.4f}",
            recall_str,
        ])
    
    print("\n" + "="*80)
    print("OVERSAMPLING EVALUATION RESULTS")
    print("="*80)
    print(table)
    print("="*80)

def choose_best_technique(results: List[EvaluationResult]) -> EvaluationResult:
    """Select best performing technique based on Macro F1 score."""
    if not results:
        raise RuntimeError("No results available to choose from")
    
    best = max(results, key=lambda r: r.macro_f1)
    
    logger.info("\n" + "="*60)
    logger.info("BEST TECHNIQUE: %s", best.name)
    logger.info("Macro F1: %.4f", best.macro_f1)
    logger.info("Macro AUC-PR: %.4f", best.macro_auc_pr)
    logger.info("="*60 + "\n")
    
    return best

def run_oversampling(
    df: pd.DataFrame = None,
    split_before: bool = True,
    evaluate_synthetic: bool = True,
    min_total_samples: int = 50000, 
    balance_method: str = "equal"    
) -> pd.DataFrame:

    ensure_directories()
    
    # Load data
    data = df.copy() if df is not None else load_dataset()
    original_data = data.copy()
    
    logger.info("Dataset loaded: %d rows, %d columns", len(data), len(data.columns))
    logger.info("Class distribution:\n%s", data[TARGET_COLUMN].value_counts())
    
    # Extract features
    X, y = select_features(data)
    
    # Identify categorical features
    categorical_columns = determine_categorical_features(X)
    logger.info("Categorical columns: %s", categorical_columns)
    
    # Encode categorical features
    encoded_X, categorical_mappings = encode_categorical_features(X, categorical_columns)
    
    # Get categorical indices for SMOTENC
    categorical_indices = [
        encoded_X.columns.get_loc(col)
        for col in categorical_columns
        if col in encoded_X.columns
    ]
    
    # Split data if requested
    if split_before:
        test_size = MODEL_CONFIG.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_X,
            y,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        logger.info("Split: Train=%d, Test=%d", len(X_train), len(X_test))
    else:
        X_train, y_train = encoded_X, y
    
    sampling_strategy = calculate_balanced_sampling_strategy(
        y_train,
        min_total_samples=min_total_samples,
        balance_method=balance_method
    )
    
    n_splits = MODEL_CONFIG.get("cv_folds", 5)
    results, samplers = evaluate_all_samplers(
        X_train,
        y_train,
        categorical_indices,
        sampling_strategy,
        n_splits,
    )
    
    # Display results
    summarise_results(results)
    
    # Choose best
    best_result = choose_best_technique(results)
    best_sampler = samplers[best_result.name]
    
    # Apply best sampler to full training data
    logger.info("Applying best technique (%s) to dataset...", best_result.name)
    X_resampled, y_resampled = best_sampler(X_train.copy(), y_train.copy())
    
    logger.info("Resampled: %d rows", len(X_resampled))
    logger.info("Resampled class distribution:\n%s", y_resampled.value_counts())
    
    # Decode categorical features
    X_decoded = decode_categorical_features(X_resampled, categorical_mappings)
    
    # Validate constraint combinations
    combined_df = pd.concat([X_decoded, y_resampled], axis=1)
    validated_df = validate_supplier_location_combinations(combined_df, original_data)
    
    logger.info("After validation: %d rows", len(validated_df))
    logger.info("Final class distribution:\n%s", validated_df[TARGET_COLUMN].value_counts())
    
    # Add back SC_ID column
    validated_df = add_sc_id_column(validated_df, start_from=1)
    
    # Save results
    output_path = PROCESSED_DATA_DIR / f"oversampled_{best_result.name.lower()}.csv"
    validated_df.to_csv(output_path, index=False)
    logger.info("Saved oversampled dataset to: %s", output_path)

    return validated_df

if __name__ == "__main__":
    oversampled_data = run_oversampling(split_before=True, 
                                        evaluate_synthetic=True,
                                        min_total_samples=50000,
                                        balance_method="proportional")