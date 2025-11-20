from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    auc,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROC_DATA_DIR = PROJECT_ROOT / "outputs" / "models" / "roc_data"
VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "visualizations"
ROC_PLOT_PATH = VISUALIZATION_DIR / "ROC_AUC_3Models.png"


def _ensure_dirs() -> None:
    ROC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def ensure_probability_matrix(predictions: np.ndarray, n_classes: Optional[int]) -> np.ndarray:
    preds = np.array(predictions)
    if preds.ndim == 1:
        if n_classes is None or n_classes <= 2:
            preds = preds.reshape(-1, 1)
            preds = np.hstack([1 - preds, preds])
        else:
            raise ValueError("Expected multi-dimensional prediction array for multi-class problem.")
    elif preds.ndim == 2 and preds.shape[1] == 1:
        preds = np.hstack([1 - preds, preds])
    return np.clip(preds, 0.0, 1.0)


def precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    positive_index: int,
    k: int,
) -> Tuple[float, float]:
    probabilities = y_prob[:, positive_index]
    limit = max(1, min(k, len(probabilities)))
    order = np.argsort(probabilities)[::-1][:limit]
    positives = (y_true == positive_index)
    tp = positives[order].sum()
    total_positive = positives.sum()
    precision = float(tp) / limit if limit > 0 else 0.0
    recall = float(tp) / total_positive if total_positive > 0 else 0.0
    return precision, recall


def compute_macro_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if n_classes < 2:
        raise ValueError("ROC curve requires at least two classes.")
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, roc_auc_macro


def persist_roc_curve(model_name: str, fpr: np.ndarray, tpr: np.ndarray, roc_auc_value: float) -> Path:
    _ensure_dirs()
    file_path = ROC_DATA_DIR / f"{model_name.lower().replace(' ', '_')}_roc.npz"
    np.savez(
        file_path,
        fpr=fpr,
        tpr=tpr,
        auc=np.array([roc_auc_value], dtype=float),
        label=np.array([model_name]),
    )
    plot_combined_roc_curves()
    return file_path


def plot_combined_roc_curves() -> None:
    if not ROC_DATA_DIR.exists():
        return
    files = sorted(ROC_DATA_DIR.glob("*.npz"))
    if not files:
        return
    _ensure_dirs()
    plt.figure(figsize=(8, 6))
    plotted = False
    for file in files:
        data = np.load(file, allow_pickle=True)
        fpr = data.get("fpr")
        tpr = data.get("tpr")
        auc_values = data.get("auc")
        label_data = data.get("label")
        if fpr is None or tpr is None:
            continue
        label = ""
        if label_data is not None:
            try:
                label = str(label_data.item())
            except Exception:
                label = str(label_data)
        if not label:
            label = file.stem.replace("_", " ").title()
        auc_value = float(auc_values.flatten()[0]) if auc_values is not None else float("nan")
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_value:.3f})")
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=300)
    plt.close()


def evaluate_classification_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]],
    positive_label: str,
    top_k: int,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    n_classes = y_prob.shape[1]
    metrics: Dict[str, float] = {}
    details: Dict[str, str] = {}
    y_pred = np.argmax(y_prob, axis=1)
    metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    try:
        metrics["roc_auc_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        metrics["roc_auc_macro"] = float("nan")
    labels = class_names or [str(i) for i in range(n_classes)]
    if positive_label in labels:
        positive_index = labels.index(positive_label)
        resolved_positive_label = positive_label
    else:
        positive_index = 0
        resolved_positive_label = labels[0]
    k_value = max(1, min(top_k, len(y_true)))
    precision_k, recall_k = precision_recall_at_k(y_true, y_prob, positive_index, k_value)
    metrics["precision_at_k"] = float(precision_k)
    metrics["recall_at_k"] = float(recall_k)
    metrics["top_k"] = float(k_value)
    try:
        fpr, tpr, macro_auc_curve = compute_macro_roc_curve(y_true, y_prob, n_classes)
        roc_file = persist_roc_curve(model_name, fpr, tpr, macro_auc_curve)
        details["roc_curve_file"] = str(roc_file)
    except ValueError:
        details["roc_curve_file"] = ""
    details["roc_plot_path"] = str(ROC_PLOT_PATH)
    details["positive_class"] = resolved_positive_label
    details["top_k"] = str(k_value)
    return metrics, details

