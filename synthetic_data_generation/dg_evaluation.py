# Evaluation of CTGAN and TVAE
# Metrics:
# Statistical metrics: KS test, Chi-Square test, Jensen-Shannon divergence
# Correlation metrics: Pearson and Spearman correlations
# Distribution metrics: Wasserstein distance, MSE
# Machine learning utilities to assess synthtic data quality
# - Train Synthetic Test Real (TSTR)
# - Train Real Test Synthetic (TRTS)

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp, wasserstein_distance, chi2_contingency, spearmanr, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "synthetic_data_evaluation.json"
EXCLUDED_COLUMNS = {"Supplier_ID", "Supplier_Name", "Commodity_ID", "Commodity_Name"}
TARGET_COLUMN = "Risk_Classification"
NUMERIC_HIST_BINS = 30
RANDOM_STATE = 42

@dataclass
class EvaluationResult:
    dataset: str
    model_type: str
    rows: int
    metrics: Dict[str, Dict[str, float]]

def load_real_data() -> pd.DataFrame:
    df = pd.read_csv(REAL_DATA_PATH)
    return df.drop(columns=[col for col in EXCLUDED_COLUMNS if col in df.columns])

def load_synthetic_data(path: Path, real_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if col not in real_columns], errors="ignore")
    missing_cols = [col for col in real_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = np.nan
    return df[real_columns]


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.select_dtypes(include=["object", "category"]).columns if col != TARGET_COLUMN]


def ks_statistic(real: pd.Series, synthetic: pd.Series) -> float:
    real_clean = real.dropna()
    synth_clean = synthetic.dropna()
    if real_clean.empty or synth_clean.empty:
        return np.nan
    return float(ks_2samp(real_clean, synth_clean, alternative="two-sided").statistic)


def chi_square_stat(real: pd.Series, synthetic: pd.Series) -> float:
    categories = real.value_counts().index.union(synthetic.value_counts().index)
    real_counts = real.value_counts().reindex(categories, fill_value=0)
    synth_counts = synthetic.value_counts().reindex(categories, fill_value=0)
    if real_counts.sum() == 0 or synth_counts.sum() == 0:
        return np.nan
    expected = synth_counts / synth_counts.sum() * real_counts.sum()
    expected = expected.replace(0, 1e-6)
    return float(chisquare(f_obs=real_counts, f_exp=expected)[0])


def js_divergence(real: pd.Series, synthetic: pd.Series) -> float:
    combined = pd.concat([real, synthetic]).dropna()
    if combined.nunique() <= 1:
        return 0.0
    r_hist, bin_edges = np.histogram(real.dropna(), bins=NUMERIC_HIST_BINS, range=(combined.min(), combined.max()))
    s_hist, _ = np.histogram(synthetic.dropna(), bins=NUMERIC_HIST_BINS, range=(combined.min(), combined.max()))
    r_prob = r_hist + 1e-8
    s_prob = s_hist + 1e-8
    r_prob = r_prob / r_prob.sum()
    s_prob = s_prob / s_prob.sum()
    return float(jensenshannon(r_prob, s_prob, base=2))


def wasserstein(real: pd.Series, synthetic: pd.Series) -> float:
    real_clean = real.dropna()
    synth_clean = synthetic.dropna()
    if real_clean.empty or synth_clean.empty:
        return np.nan
    return float(wasserstein_distance(real_clean, synth_clean))


def mean_squared_error_means(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, columns: List[str]) -> float:
    if not columns:
        return np.nan
    real_means = real_df[columns].mean()
    synth_means = synthetic_df[columns].mean()
    return float(((real_means - synth_means) ** 2).mean())


def correlation_diff(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, method: str) -> float:
    corr_real = real_df.corr(method=method)
    corr_synth = synthetic_df.corr(method=method)
    diff = (corr_real - corr_synth).abs()
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(diff.values))


def build_classifier(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    preprocessor = ColumnTransformer(transformers)
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def ml_utility(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> float:
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        return np.nan
    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    test_df = test_df.dropna(subset=[TARGET_COLUMN])
    if train_df.empty or test_df.empty:
        return np.nan
    features = [col for col in train_df.columns if col != TARGET_COLUMN]
    pipeline = build_classifier(numeric_cols, categorical_cols)
    pipeline.fit(train_df[features], train_df[TARGET_COLUMN])
    preds = pipeline.predict(test_df[features])
    return float(accuracy_score(test_df[TARGET_COLUMN], preds))


def evaluate_dataset(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, dataset_name: str) -> EvaluationResult:
    numeric_cols = numeric_columns(real_df.drop(columns=[TARGET_COLUMN], errors="ignore"))
    categorical_cols = categorical_columns(real_df)

    ks_stats = [ks_statistic(real_df[col], synthetic_df[col]) for col in numeric_cols]
    chi_stats = [chi_square_stat(real_df[col], synthetic_df[col]) for col in categorical_cols]
    js_stats = [js_divergence(real_df[col], synthetic_df[col]) for col in numeric_cols]
    wass_dists = [wasserstein(real_df[col], synthetic_df[col]) for col in numeric_cols]

    metrics = {
        "statistical": {
            "ks": float(np.nanmean(ks_stats)) if ks_stats else np.nan,
            "chi_square": float(np.nanmean(chi_stats)) if chi_stats else np.nan,
            "js_divergence": float(np.nanmean(js_stats)) if js_stats else np.nan,
        },
        "correlation": {
            "pearson_diff": correlation_diff(real_df[numeric_cols], synthetic_df[numeric_cols], "pearson")
            if numeric_cols
            else np.nan,
            "spearman_diff": correlation_diff(real_df[numeric_cols], synthetic_df[numeric_cols], "spearman")
            if numeric_cols
            else np.nan,
        },
        "distribution": {
            "wasserstein": float(np.nanmean(wass_dists)) if wass_dists else np.nan,
            "mse": mean_squared_error_means(real_df, synthetic_df, numeric_cols),
        },
    }

    if TARGET_COLUMN in real_df.columns:
        tstr_acc = ml_utility(synthetic_df, real_df, numeric_cols, categorical_cols)
        trts_acc = ml_utility(real_df, synthetic_df, numeric_cols, categorical_cols)
    else:
        tstr_acc = trts_acc = np.nan

    metrics["ml_utility"] = {
        "tstr_accuracy": tstr_acc,
        "trts_accuracy": trts_acc,
    }

    model_type = "CTGAN" if "CTGAN" in dataset_name.upper() else "TVAE"
    return EvaluationResult(
        dataset=dataset_name,
        model_type=model_type,
        rows=len(synthetic_df),
        metrics=metrics,
    )


def print_summary_table(results: List[EvaluationResult]) -> None:
    table = PrettyTable()
    table.field_names = [
        "Dataset",
        "Rows",
        "KS",
        "Chi-Square",
        "JS Divergence",
        "Pearson_Diff",
        "Spearman_Diff",
        "Wasserstein",
        "MSE",
        "TSTR Acc",
        "TRTS Acc",
    ]

    def fmt(value: float) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{value:.4f}"

    for result in results:
        metrics = result.metrics
        table.add_row(
            [
                result.dataset,
                result.rows,
                fmt(metrics["statistical"]["ks"]),
                fmt(metrics["statistical"]["chi_square"]),
                fmt(metrics["statistical"]["js_divergence"]),
                fmt(metrics["correlation"]["pearson_diff"]),
                fmt(metrics["correlation"]["spearman_diff"]),
                fmt(metrics["distribution"]["wasserstein"]),
                fmt(metrics["distribution"]["mse"]),
                fmt(metrics["ml_utility"]["tstr_accuracy"]),
                fmt(metrics["ml_utility"]["trts_accuracy"]),
            ]
        )

    print(table)


def save_results(results: List[EvaluationResult]) -> None:
    data = [
        {
            "dataset": res.dataset,
            "model_type": res.model_type,
            "rows": res.rows,
            "metrics": res.metrics,
        }
        for res in results
    ]
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved evaluation details to {OUTPUT_JSON}")


def main() -> None:
    real_df = load_real_data()
    real_columns = real_df.columns.tolist()
    synthetic_files = sorted(SYNTHETIC_DIR.glob("*.csv"))
    if not synthetic_files:
        raise FileNotFoundError(f"No synthetic datasets found in {SYNTHETIC_DIR}")

    results: List[EvaluationResult] = []
    for path in synthetic_files:
        print(f"Evaluating {path.name}...")
        synthetic_df = load_synthetic_data(path, real_columns)
        result = evaluate_dataset(real_df, synthetic_df, path.name)
        results.append(result)

    print_summary_table(results)
    save_results(results)


if __name__ == "__main__":
    main()