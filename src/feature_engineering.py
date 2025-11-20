from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

SYNTHETIC_DATA_PATH = config.DATA_DIR / "synthetic" / "syn_20000_data.csv"
FEATURE_OUTPUT_PATH = config.PROCESSED_DATA_DIR / "syn_20000_engineered_features.csv"

COMPLIANCE_MAX = 3
SOCIAL_SCORE_MAX = 100
FINANCIAL_SCORE_MAX = 100
ESG_SCORE_MAX = 100
DEFAULT_FILL_VALUE = np.nan

np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float).replace(0, np.nan)
    result = numerator.astype(float).divide(denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def _min_max_scale(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    min_val = values.min(skipna=True)
    max_val = values.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
        return pd.Series(0.0, index=values.index)
    return (values - min_val) / (max_val - min_val)


def _normalize(series: pd.Series, max_value: float) -> pd.Series:
    if max_value == 0:
        return pd.Series(0.0, index=series.index)
    return (series.astype(float) / max_value).clip(lower=0.0, upper=1.0)


def load_synthetic_dataset(path: Path = SYNTHETIC_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {path}")
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    engineered = df.copy()
    added: List[str] = []
    skipped: List[str] = []

    available_columns = set(engineered.columns)

    def _add_feature(name: str, values: pd.Series) -> None:
        engineered[name] = values
        added.append(name)
        available_columns.add(name)

    def _require(columns: Iterable[str], feature_name: str) -> bool:
        missing = [col for col in columns if col not in available_columns]
        if missing:
            skipped.append(f"{feature_name} (missing columns: {', '.join(missing)})")
            return False
        return True

    # ------------------------------------------------------------------ #
    if _require({"Carbon_Emission_Intensity", "Production_Capacity"}, "Emission_to_Capacity_Ratio"):
        _add_feature(
            "Emission_to_Capacity_Ratio",
            _safe_divide(
            engineered["Carbon_Emission_Intensity"],
            engineered["Production_Capacity"],
            ),
        )

    if _require({"On_Time_Delivery_Rate", "Defect_Rate"}, "Operational_Reliability_Index"):
        on_time = engineered["On_Time_Delivery_Rate"].astype(float).clip(0.0, 1.0)
        defects = engineered["Defect_Rate"].astype(float).clip(0.0, 1.0)
        _add_feature("Operational_Reliability_Index", on_time * (1.0 - defects))

    if _require({"Lead_Time_Days", "Production_Capacity"}, "LeadTime_Capacity_Ratio"):
        _add_feature(
            "LeadTime_Capacity_Ratio",
            _safe_divide(
            engineered["Lead_Time_Days"],
            engineered["Production_Capacity"],
            ),
        )

    if _require(
        {"Waste_Management_Efficiency", "Renewable_Energy_Usage"},
        "Green_Efficiency_Score",
    ):
        _add_feature(
            "Green_Efficiency_Score",
            (
                engineered["Waste_Management_Efficiency"].astype(float).clip(0.0, 1.0)
                + engineered["Renewable_Energy_Usage"].astype(float).clip(0.0, 1.0)
            )
            / 2.0,
        )

    if _require(
        {"Operational_Reliability_Index", "Financial_Stability_Score", "Compliance_Level"},
        "Resilience_Score",
    ):
        operational = engineered.get(
            "Operational_Reliability_Index", pd.Series(DEFAULT_FILL_VALUE, index=engineered.index)
        ).fillna(0.0)
        financial = _normalize(engineered["Financial_Stability_Score"], FINANCIAL_SCORE_MAX)
        compliance = _normalize(engineered["Compliance_Level"], COMPLIANCE_MAX)
        _add_feature("Resilience_Score", operational * financial * compliance)

    if _require(
        {"ESG_Score", "Compliance_Level", "Carbon_Emission_Intensity", "Cost_Index"},
        "Sustainability_Risk_Index",
    ):
        esg_component = 1.0 - _normalize(engineered["ESG_Score"], ESG_SCORE_MAX)
        compliance_component = 1.0 - _normalize(engineered["Compliance_Level"], COMPLIANCE_MAX)
        emission_component = engineered["Carbon_Emission_Intensity"].astype(float).clip(lower=0.0)
        cost_component = engineered["Cost_Index"].astype(float).clip(lower=0.0)
        risk_score = (
            0.35 * esg_component
            + 0.25 * compliance_component
            + 0.25 * emission_component
            + 0.15 * cost_component
        )
        _add_feature("Sustainability_Risk_Index", (risk_score * 100).clip(0.0, 100.0))

    if _require({"ESG_Score", "Compliance_Level"}, "ESG_Compliance_Composite"):
        esg_norm = _normalize(engineered["ESG_Score"], ESG_SCORE_MAX)
        compliance_norm = _normalize(engineered["Compliance_Level"], COMPLIANCE_MAX)
        _add_feature("ESG_Compliance_Composite", (0.6 * esg_norm + 0.4 * compliance_norm) * 100)

    if _require(
        {"Social_Score", "Labour_Compliance_Score", "Diversity_Index"},
        "Social_Responsibility_Score",
    ):
        social_norm = _normalize(engineered["Social_Score"], SOCIAL_SCORE_MAX)
        labour_norm = engineered["Labour_Compliance_Score"].astype(float).clip(0.0, 1.0)
        diversity_norm = engineered["Diversity_Index"].astype(float).clip(0.0, 1.0)
        _add_feature(
            "Social_Responsibility_Score",
            (0.5 * social_norm + 0.3 * labour_norm + 0.2 * diversity_norm) * 100,
        )

    if _require({"Carbon_Emission_Intensity"}, "Regional_Risk_Index"):
        region_metric = engineered["Carbon_Emission_Intensity"].astype(float)
        if "Region" in engineered.columns:
            region_metric = (
                engineered.groupby("Region")["Carbon_Emission_Intensity"].transform("mean").astype(float)
            )
        country_metric = engineered["Carbon_Emission_Intensity"].astype(float)
        if "Country" in engineered.columns:
            country_metric = (
                engineered.groupby("Country")["Carbon_Emission_Intensity"].transform("mean").astype(float)
            )
        combined_metric = 0.6 * region_metric + 0.4 * country_metric
        _add_feature("Regional_Risk_Index", (_min_max_scale(combined_metric) * 100).fillna(0.0))

    if _require({"Carbon_Emission_Intensity", "Cost_Index"}, "Emission_Per_Dollar"):
        _add_feature(
            "Emission_Per_Dollar",
            _safe_divide(
            engineered["Carbon_Emission_Intensity"],
            engineered["Cost_Index"],
            ),
        )

    if _require(
        {"Carbon_Emission_Intensity", "Water_Intensity", "Logistics_Distance_km"},
        "Emission_Votality",
    ):
        emission_norm = engineered["Carbon_Emission_Intensity"].astype(float).clip(lower=0.0)
        water_norm = _min_max_scale(engineered["Water_Intensity"])
        logistics_norm = _min_max_scale(engineered["Logistics_Distance_km"])
        _add_feature(
            "Emission_Votality",
            0.6 * emission_norm + 0.25 * water_norm + 0.15 * logistics_norm
        )

    ghg_columns = [col for col in engineered.columns if "ghg" in col.lower()]
    if ghg_columns:
        ghg_frame = engineered[ghg_columns].astype(float)
        dominance_score = ghg_frame.div(ghg_frame.sum(axis=1).replace(0, np.nan), axis=0).max(axis=1)
        _add_feature("GHG_Type_Dominance_Score", dominance_score.fillna(0.0))
    else:
        skipped.append("GHG_Type_Dominance_Score (no GHG-related columns found)")

    return engineered, added, skipped


def save_engineered_dataset(df: pd.DataFrame, path: Path = FEATURE_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_feature_engineering_pipeline(
    input_path: Path = SYNTHETIC_DATA_PATH,
    output_path: Path = FEATURE_OUTPUT_PATH,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    dataset = load_synthetic_dataset(input_path)
    engineered, added, skipped = engineer_features(dataset)
    save_engineered_dataset(engineered, output_path)
    return engineered, added, skipped


def display_feature_engineering_summary(
    features_added: Iterable[str],
    features_skipped: Iterable[str],
    dataset_name: str = "Synthetic Supplier Dataset",
) -> None:
    print("\n" + "=" * 70)
    print(f"FEATURE ENGINEERING SUMMARY - {dataset_name}")
    print("=" * 70)
    added_list = list(features_added)
    skipped_list = list(features_skipped)
    print(f"Features engineered: {len(added_list)}")
    for feature in added_list:
        print(f"  [+] {feature}")
    if skipped_list:
        print("\nSkipped / unavailable features:")
        for feature in skipped_list:
            print(f"  [!] {feature}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    engineered_df, added_features, skipped_features = run_feature_engineering_pipeline()
    display_feature_engineering_summary(
        added_features,
        skipped_features,
        dataset_name="syn_20000_data.csv",
    )

