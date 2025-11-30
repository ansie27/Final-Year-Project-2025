import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except ImportError:
    ARIMA = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENGINEERED_DATA_PATH

# Region definitions
REGION_EAST_ASIA_PACIFIC = "East Asia and Pacific"
REGION_EUROPE_CENTRAL_ASIA = "Europe and Central Asia"
REGION_NORTH_AMERICA = "North America"
REGION_MENA_AP = "Middle East, North Africa, Afghanistan and Pakistan"
REGION_SUB_SAHARAN = "Sub-Saharan Africa"
REGION_LATAM_CARIB = "Latin America and Caribbean"
REGION_SOUTH_ASIA = "South Asia"

REGION_ORDER = [
    REGION_EAST_ASIA_PACIFIC,
    REGION_EUROPE_CENTRAL_ASIA,
    REGION_NORTH_AMERICA,
    REGION_MENA_AP,
    REGION_SUB_SAHARAN,
    REGION_LATAM_CARIB,
    REGION_SOUTH_ASIA,
]

REGION_COLOR_MAP = {
    REGION_EAST_ASIA_PACIFIC: "#2E86AB",
    REGION_EUROPE_CENTRAL_ASIA: "#6A0572",
    REGION_NORTH_AMERICA: "#1F618D",
    REGION_MENA_AP: "#D7263D",
    REGION_SUB_SAHARAN: "#1B998B",
    REGION_LATAM_CARIB: "#F6C667",
    REGION_SOUTH_ASIA: "#C084FC",
}

REGION_NAME_TO_CODE = {
    REGION_EAST_ASIA_PACIFIC.lower(): "EAP",
    REGION_EUROPE_CENTRAL_ASIA.lower(): "ECA",
    REGION_NORTH_AMERICA.lower(): "NA",
    REGION_MENA_AP.lower(): "MENA-AP",
    REGION_SUB_SAHARAN.lower(): "SSA",
    REGION_LATAM_CARIB.lower(): "LAC",
    REGION_SOUTH_ASIA.lower(): "SA",
}

# Country mappings
NORTH_AMERICA_COUNTRIES = {"Canada", "United States"}

LATAM_AND_CARIBBEAN_COUNTRIES = {
    "Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize",
    "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba",
    "Dominican Republic", "Ecuador", "El Salvador", "Grenada", "Guatemala",
    "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
    "Panama", "Paraguay", "Peru", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Suriname", "Trinidad and Tobago", "Uruguay", "Venezuela",
}

SOUTH_ASIA_COUNTRIES = {
    "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Sri Lanka",
}

MENA_AP_COUNTRIES = {
    "Afghanistan", "Algeria", "Bahrain", "Djibouti", "Egypt", "Iran", "Iraq",
    "Israel", "Jordan", "Kuwait", "Lebanon", "Libya", "Morocco", "Oman",
    "Pakistan", "Qatar", "Saudi Arabia", "Syrian Arab Republic", "Tunisia",
    "United Arab Emirates", "West Bank and Gaza", "Yemen, Rep.",
}

CENTRAL_ASIA_AND_EU_COUNTRIES = {
    "Armenia", "Azerbaijan", "Georgia", "Kazakhstan", "Kyrgyz Republic",
    "Tajikistan", "Turkmenistan", "Uzbekistan",
}

REGION_DESCRIPTIONS = {
    REGION_EAST_ASIA_PACIFIC: "Rapidly industrializing economies with diversified manufacturing bases and evolving ESG regulations.",
    REGION_EUROPE_CENTRAL_ASIA: "Highly regulated supply chains with strong sustainability mandates and advanced compliance regimes.",
    REGION_NORTH_AMERICA: "Mature suppliers with sophisticated risk management and complex multi-tier logistics networks.",
    REGION_MENA_AP: "Energy-dense trade routes balancing geopolitical considerations with critical mineral exports.",
    REGION_SUB_SAHARAN: "Resource-rich suppliers with emerging governance practices and infrastructure investment needs.",
    REGION_LATAM_CARIB: "Agricultural and extractive hubs powering global food and metals supply, often exposed to climate extremes.",
    REGION_SOUTH_ASIA: "High-growth manufacturing corridor with large labour-intensive supplier bases and maturing ESG standards.",
}


def build_region_dataframe() -> pd.DataFrame:
    """Build dataframe mapping countries to regions using gapminder data."""
    try:
        import plotly.express as px
        gapminder = px.data.gapminder().query("year == 2007")[
            ["country", "iso_alpha", "continent"]
        ].copy()
    except Exception:
        # Fallback if plotly not available
        return pd.DataFrame(columns=["country", "iso_alpha", "region"])

    def assign_region(row: pd.Series) -> str:
        country = row["country"]
        continent = row["continent"]

        if country in MENA_AP_COUNTRIES:
            return REGION_MENA_AP
        if country in SOUTH_ASIA_COUNTRIES:
            return REGION_SOUTH_ASIA
        if country in CENTRAL_ASIA_AND_EU_COUNTRIES:
            return REGION_EUROPE_CENTRAL_ASIA

        if continent == "Americas":
            if country in NORTH_AMERICA_COUNTRIES:
                return REGION_NORTH_AMERICA
            return REGION_LATAM_CARIB

        if continent == "Africa":
            return REGION_SUB_SAHARAN

        if continent == "Europe":
            return REGION_EUROPE_CENTRAL_ASIA

        if continent in ["Oceania", "Asia"]:
            return REGION_EAST_ASIA_PACIFIC

        return REGION_EAST_ASIA_PACIFIC

    gapminder["region"] = gapminder.apply(assign_region, axis=1)
    world_regions = gapminder[["country", "iso_alpha", "region"]].dropna()
    world_regions = world_regions.drop_duplicates(subset="iso_alpha")
    return world_regions

def load_engineered_dataset() -> pd.DataFrame:
    """Load the engineered supplier/commodity dataset used across the dashboard."""
    if ENGINEERED_DATA_PATH.exists():
        try:
            df = pd.read_csv(ENGINEERED_DATA_PATH)
            return df
        except Exception as exc:  # pragma: no cover - log best-effort
            print(f"[dashboard] Failed to load engineered dataset: {exc}")
    return pd.DataFrame()

# Initialise data
WORLD_REGION_DATA = build_region_dataframe()
ENGINEERED_DATASET = load_engineered_dataset()

RISK_LABELS = ["High", "Moderate", "Low"]
RISK_COLOR_MAP = {
    "High": "#fa7066",
    "Moderate": "#fcaf56",
    "Low": "#adcf7a",
}
RISK_SCORE_CANDIDATES = [
    "Sustainability_Risk_Index",
    "Overall_Risk_Score",
    "Green_Efficiency_Score",
    "ESG_Score",
]
COMPLIANCE_CANDIDATES = ["Compliance_Level", "Certifications_Active"]
TREND_SERIES = [
    ("Carbon Emission Intensity", "Carbon_Emission_Intensity", "#1abc9c"),
    ("GHG Scope 1", "GHG_Scope1_Intensity", "#3498db"),
    ("GHG Scope 2", "GHG_Scope2_Intensity", "#9b59b6"),
    ("Renewable Energy Usage", "Renewable_Energy_Usage", "#f39c12"),
]
CARBON_TRAJECTORY_FEATURES = [
    "Carbon_Emission_Intensity",
    "Renewable_Energy_Usage",
    "Emission_Per_Dollar",
    "Green_Efficiency_Score",
]
TREND_FEATURES = [
    ("ESG Score", "ESG_Score", "#1f77b4"),
    ("Environmental Score", "Environmental_Score", "#2ca02c"),
    ("Social Score", "Social_Score", "#ff7f0e"),
    ("Governance Score", "Governance_Score", "#9467bd"),
]
SUPPLIER_PROFILE_COLUMNS = [
    "SC_ID",
    "Supplier_Name",
    "Country",
    "Region",
    "Commodity_Name",
    "Industry_Sector",
    "Supplier_Tier",
    "Certifications_Active",
    "ESG_Score",
    "Compliance_Level",
    "Financial_Stability_Score",
    "Environmental_Score",
    "Social_Score",
    "Governance_Score",
    "On_Time_Delivery_Rate",
    "Defect_Rate",
    "Lead_Time_Days",
    "Carbon_Emission_Intensity",
]
FALLBACK_COMMODITIES = [
    {"name": "Commodity Y", "sustainability": 72, "ghg_score": 0.89, "cost": 4.8},
    {"name": "Commodity X", "sustainability": 47, "ghg_score": 0.89, "cost": 5.3},
    {"name": "Commodity A", "sustainability": 68, "ghg_score": 0.88, "cost": 6.3},
    {"name": "Commodity T", "sustainability": 23, "ghg_score": 0.38, "cost": 7.1},
    {"name": "Commodity E", "sustainability": 64, "ghg_score": 0.89, "cost": 6.3},
]
FALLBACK_MOVERS = [
    {"name": "Supplier G", "country": "Unknown", "resilience": 0.63, "risk_delta": 0.07},
    {"name": "Supplier Q", "country": "Unknown", "resilience": 0.54, "risk_delta": 0.06},
    {"name": "Supplier B", "country": "Unknown", "resilience": 0.51, "risk_delta": 0.04},
    {"name": "Supplier L", "country": "Unknown", "resilience": 0.45, "risk_delta": 0.03},
    {"name": "Supplier C", "country": "Unknown", "resilience": 0.35, "risk_delta": 0.02},
]


def _first_available_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((col for col in candidates if col in df.columns), None)


def _safe_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_dashboard_dataset() -> pd.DataFrame:
    """Return a copy of the engineered dataset with risk classifications ensured."""
    return compute_risk_scores(ENGINEERED_DATASET)


def filter_by_region(dataset: pd.DataFrame, region: Optional[str]) -> pd.DataFrame:
    if not region or region == "All" or dataset.empty:
        return dataset
    if "Region" not in dataset.columns:
        return dataset

    region_value = str(region).strip()
    if not region_value:
        return dataset

    normalized_code = REGION_NAME_TO_CODE.get(region_value.lower(), region_value.upper())

    series = dataset["Region"].fillna("").astype(str)
    mask = series.str.upper() == normalized_code.upper()
    if not mask.any():
        mask = series.str.lower() == region_value.lower()

    filtered = dataset[mask]
    return filtered if not filtered.empty else dataset

# Flask app setup
DASHBOARD_DIR = Path(__file__).resolve().parent
HTML_DIR = DASHBOARD_DIR / "html"
CSS_DIR = DASHBOARD_DIR / "css"
JS_DIR = DASHBOARD_DIR / "js"
TABS_DIR = HTML_DIR / "tabs"

app = Flask(__name__)

# ============================================================================
# STATIC FILE SERVING
# ============================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return send_from_directory(HTML_DIR, "index.html")

@app.route("/tabs/<path:filename>")
def serve_tab(filename: str):
    """Serve HTML fragments for tabs."""
    return send_from_directory(TABS_DIR, filename)

@app.route("/css/<path:filename>")
def serve_css(filename: str):
    """Serve CSS assets."""
    return send_from_directory(CSS_DIR, filename)

@app.route("/js/<path:filename>")
def serve_js(filename: str):
    """Serve JavaScript assets."""
    return send_from_directory(JS_DIR, filename)

# ============================================================================
# DATA PROCESSING AND MACHINE LEARNING MODELS
# ============================================================================

def compute_risk_scores(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()

    if df.empty:
        return df

    if "Risk_Classification" in df.columns:
        return df

    metric_col = _first_available_column(df, RISK_SCORE_CANDIDATES)
    if metric_col:
        metric_series = _safe_series(df[metric_col])
        q1, q2 = metric_series.quantile([0.33, 0.66]).tolist()

        def _classify(value: float) -> str:
            if pd.isna(value):
                return "Moderate"
            if value <= q1:
                return "Low"
            if value <= q2:
                return "Moderate"
            return "High"

        df["Risk_Classification"] = metric_series.apply(_classify)
    else:
        np.random.seed(42)
        df["Risk_Classification"] = np.random.choice(
            RISK_LABELS, size=len(df), p=[0.2, 0.5, 0.3]
        )

    return df

def compute_global_risk_score(df: pd.DataFrame) -> float:
    metric_col = _first_available_column(df, RISK_SCORE_CANDIDATES)
    if metric_col:
        series = _safe_series(df[metric_col]).dropna()
        if not series.empty:
            return round(float(series.mean()), 1)
    return round(random.uniform(60, 95), 1)


def calculate_compliance_rate(df: pd.DataFrame) -> float:
    if "Compliance_Level" in df.columns:
        values = _safe_series(df["Compliance_Level"]).dropna()
        if not values.empty:
            max_value = max(values.max(), 1.0)
            return float(min(100.0, (values.mean() / max_value) * 100))

    if "Certifications_Active" in df.columns:
        series = df["Certifications_Active"].fillna("").astype(str).str.lower()
        total = len(series)
        if total:
            compliant = series[series != "none"]
            return float((len(compliant) / total) * 100)

    return 0.0


def build_risk_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "Risk_Classification" not in df.columns:
        counts = pd.Series([1, 1, 1], index=RISK_LABELS)
    else:
        counts = df["Risk_Classification"].value_counts().reindex(RISK_LABELS, fill_value=0)

    labels = counts.index.tolist()
    values = counts.tolist()
    colors = [RISK_COLOR_MAP.get(label, "#d9d9d9") for label in labels]

    return {"values": values, "labels": labels, "colors": colors}


def _forecast_series(series: pd.Series, steps: int = 2) -> Tuple[List[int], List[float], List[float], List[float]]:
    series = series.dropna()
    if series.empty:
        return [], [], [], []

    last_year = int(series.index.astype(int).max())
    forecast_years = [last_year + i for i in range(1, steps + 1)]

    def _fallback(value: float) -> Tuple[List[int], List[float], List[float], List[float]]:
        base = float(value)
        values = []
        upper = []
        lower = []
        current = base
        for _ in range(steps):
            current = max(0.0, min(100.0, current - 1.5))
            values.append(round(current, 2))
            upper.append(min(100.0, current + 3))
            lower.append(max(0.0, current - 3))
        return forecast_years, values, upper, lower

    if ARIMA is None or len(series) < 3:
        return _fallback(series.iloc[-1])

    try:
        model = ARIMA(series, order=(1, 1, 0))
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=steps)
        predicted = forecast.predicted_mean.clip(0, 100).round(2).tolist()
        conf_int = forecast.conf_int(alpha=0.2)
        lower = conf_int.iloc[:, 0].clip(0, 100).round(2).tolist()
        upper = conf_int.iloc[:, 1].clip(0, 100).round(2).tolist()
        return forecast_years, predicted, upper, lower
    except Exception:
        return _fallback(series.iloc[-1])


def build_overview_summary(dataset: pd.DataFrame, region: Optional[str] = None) -> Dict[str, Any]:
    dataset = filter_by_region(dataset, region)
    if dataset.empty:
        return {
            "risk_score": round(random.uniform(60, 95), 1),
            "total_suppliers": 0,
            "risk_distribution": build_risk_distribution(dataset),
            "compliance_rate": 0.0,
        }

    risk_score = compute_global_risk_score(dataset)
    total_suppliers = (
        int(dataset["Supplier_Name"].nunique()) if "Supplier_Name" in dataset.columns else len(dataset)
    )
    risk_distribution = build_risk_distribution(dataset)
    compliance_rate = round(calculate_compliance_rate(dataset), 1)

    return {
        "risk_score": risk_score,
        "total_suppliers": total_suppliers,
        "risk_distribution": risk_distribution,
        "compliance_rate": compliance_rate,
    }


def build_top_commodities(dataset: pd.DataFrame, limit: int = 5, region: Optional[str] = None) -> List[Dict[str, Any]]:
    dataset = filter_by_region(dataset, region)
    if dataset.empty or "Commodity_Name" not in dataset.columns:
        return FALLBACK_COMMODITIES

    required_cols = ["Commodity_Name", "ESG_Score", "Carbon_Emission_Intensity", "Cost_Index"]
    missing = [col for col in required_cols if col not in dataset.columns]
    if missing:
        return FALLBACK_COMMODITIES

    risk_metric = _first_available_column(dataset, RISK_SCORE_CANDIDATES)
    grouped = (
        dataset.groupby("Commodity_Name")
        .agg(
            sustainability=("ESG_Score", "mean"),
            ghg_score=("Carbon_Emission_Intensity", "mean"),
            cost=("Cost_Index", "mean"),
            risk_metric=(risk_metric, "mean") if risk_metric else ("ESG_Score", "mean"),
        )
        .reset_index()
    )

    if risk_metric:
        grouped = grouped.sort_values("risk_metric", ascending=False)
    else:
        grouped = grouped.sort_values("sustainability", ascending=False)

    return [
        {
            "name": row["Commodity_Name"],
            "sustainability": round(float(row["sustainability"]), 2),
            "ghg_score": round(float(row["ghg_score"]), 2),
            "cost": round(float(row["cost"]), 2),
        }
        for _, row in grouped.head(limit).iterrows()
    ]


def build_top_movers(dataset: pd.DataFrame, limit: int = 5, region: Optional[str] = None) -> List[Dict[str, Any]]:
    dataset = filter_by_region(dataset, region)
    if dataset.empty or "Supplier_Name" not in dataset.columns:
        return FALLBACK_MOVERS

    resilience_col = _first_available_column(dataset, ["Resilience_Score", "Operational_Reliability_Index"])
    risk_col = _first_available_column(dataset, RISK_SCORE_CANDIDATES)
    if not resilience_col or not risk_col:
        return FALLBACK_MOVERS

    country_col = "Country" if "Country" in dataset.columns else None

    agg_spec = {
        "resilience": (resilience_col, "mean"),
        "risk_min": (risk_col, "min"),
        "risk_max": (risk_col, "max"),
    }
    if country_col:
        agg_spec["country"] = (country_col, lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])

    grouped = dataset.groupby("Supplier_Name").agg(**agg_spec)
    grouped["risk_delta"] = (grouped["risk_max"] - grouped["risk_min"]).abs()
    grouped = grouped.sort_values("risk_delta", ascending=False)

    movers = []
    for supplier, row in grouped.head(limit).iterrows():
        movers.append(
            {
                "name": supplier,
                "country": row.get("country", "N/A"),
                "resilience": round(float(row["resilience"]), 3),
                "risk_delta": round(float(row["risk_delta"]), 3),
            }
        )

    return movers or FALLBACK_MOVERS


def build_forecast_payload(dataset: pd.DataFrame) -> Dict[str, Any]:
    if dataset.empty:
        return {
            "values": [1, 1, 1],
            "labels": RISK_LABELS,
            "colors": [RISK_COLOR_MAP[label] for label in RISK_LABELS],
        }

    dataset = compute_risk_scores(dataset)

    required_cols = [
        "Risk_Classification",
        "ESG_Score",
        "Compliance_Level",
        "Financial_Stability_Score",
    ]
    missing = [col for col in required_cols if col not in dataset.columns]
    if missing:
        counts = dataset["Risk_Classification"].value_counts().reindex(RISK_LABELS, fill_value=0)
        colors = [RISK_COLOR_MAP.get(label, "#d9d9d9") for label in counts.index.tolist()]
        return {"values": counts.tolist(), "labels": counts.index.tolist(), "colors": colors}

    data = dataset[required_cols].dropna()
    data["Risk_Flag"] = data["Risk_Classification"].map(
        {"High": 2, "Moderate": 1, "Low": 0}
    )
    data = data.dropna(subset=["Risk_Flag"])

    weights = np.array([0.4, 0.3, 0.3])
    min_values = data[["ESG_Score", "Compliance_Level", "Financial_Stability_Score"]].min()
    max_values = data[["ESG_Score", "Compliance_Level", "Financial_Stability_Score"]].max()
    normalized = (data[["ESG_Score", "Compliance_Level", "Financial_Stability_Score"]] - min_values) / (
        max_values - min_values
    )
    normalized = normalized.fillna(0.5)
    composite = normalized.to_numpy() @ weights

    thresholds = np.quantile(composite, [0.33, 0.66])

    projections = pd.Series(index=RISK_LABELS, data=0, dtype=float)
    for value in composite:
        if value <= thresholds[0]:
            projections["Low"] += 1
        elif value <= thresholds[1]:
            projections["Moderate"] += 1
        else:
            projections["High"] += 1

    labels = projections.index.tolist()
    colors = [RISK_COLOR_MAP.get(label, "#d9d9d9") for label in labels]
    return {"values": projections.tolist(), "labels": labels, "colors": colors}


def build_carbon_trajectory_payload(dataset: pd.DataFrame) -> Dict[str, Any]:
    if dataset.empty:
        months = [f"M{i}" for i in range(1, 13)]
        values = [max(0, 100 - i * 3) for i in range(12)]
        return {"months": months, "values": values}

    missing = [col for col in CARBON_TRAJECTORY_FEATURES if col not in dataset.columns]
    if missing:
        months = [f"M{i}" for i in range(1, 13)]
        values = [max(0, 100 - i * 3) for i in range(12)]
        return {"months": months, "values": values}

    data = dataset[CARBON_TRAJECTORY_FEATURES].dropna()
    if data.empty:
        months = [f"M{i}" for i in range(1, 13)]
        values = [max(0, 100 - i * 3) for i in range(12)]
        return {"months": months, "values": values}

    normalized = data.copy()
    for column in CARBON_TRAJECTORY_FEATURES:
        series = _safe_series(normalized[column])
        min_val, max_val = series.min(), series.max()
        if np.isclose(max_val - min_val, 0):
            normalized[column] = 0.5
        else:
            normalized[column] = (series - min_val) / (max_val - min_val)

    base_intensity = normalized["Carbon_Emission_Intensity"].mean()
    renewable_factor = normalized["Renewable_Energy_Usage"].mean()
    emission_per_dollar = normalized["Emission_Per_Dollar"].mean()
    efficiency = normalized["Green_Efficiency_Score"].mean()
    base_intensity = 1.0 if np.isnan(base_intensity) else base_intensity

    months = [f"M{i}" for i in range(1, 13)]
    values = []
    current = base_intensity if not np.isnan(base_intensity) else 1.0
    for idx in range(12):
        reduction = 0.02 + renewable_factor * 0.03 + efficiency * 0.02
        penalty = emission_per_dollar * 0.02
        current = max(0.05, current * (1 - reduction + penalty))
        values.append(round(float(current * 100), 2))

    return {"months": months, "values": values}


def select_supplier_row(
    dataset: pd.DataFrame,
    supplier_name: Optional[str] = None,
) -> Optional[pd.Series]:
    if dataset.empty:
        return None

    searchable = dataset.copy()
    searchable["Supplier_Name"] = searchable["Supplier_Name"].fillna("")

    if supplier_name:
        mask = searchable["Supplier_Name"].str.contains(
            supplier_name.strip(), case=False, na=False
        )
        filtered = searchable[mask]
        if not filtered.empty:
            searchable = filtered

    random_idx = random.randint(0, len(searchable) - 1)
    return searchable.iloc[random_idx]


def _format_percentage(value: float, default: str = "--") -> str:
    if value is None or pd.isna(value):
        return default
    return f"{float(value) * 100:.1f}%"


def _format_ratio(value: float, default: str = "--") -> str:
    if value is None or pd.isna(value):
        return default
    return f"{float(value):.2f}"


def _to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(item) for item in value]
    if isinstance(value, (np.generic, pd.Series)):
        try:
            return value.item()
        except Exception:
            return value.tolist()
    return value

def build_supplier_profile_payload(row: pd.Series) -> Dict[str, Any]:
    esg_score = float(row.get("ESG_Score", 0))
    compliance = float(row.get("Compliance_Level", 0))
    financial = float(row.get("Financial_Stability_Score", 0))

    risk_index = round(
        np.mean([esg_score, compliance, financial]).item() if esg_score else random.uniform(60, 85),
        1,
    )
    classification = row.get("Risk_Classification") or (
        "High" if risk_index >= 75 else ("Moderate" if risk_index >= 60 else "Low")
    )

    # Build risk trend series using synthetic smoothing
    base = risk_index
    actual_years = ["2023 Q3", "2023 Q4", "2024 Q1", "2024 Q2"]
    actual_values = [
        round(max(0, min(100, base + offset)), 1)
        for offset in (-4, -2, -1, 0)
    ]
    future_years = ["2024 Q3", "2024 Q4", "2025 Q1", "2025 Q2"]
    forecast_values = [
        round(max(0, min(100, base - (idx + 1) * 1.5)), 1)
        for idx in range(len(future_years))
    ]

    esg_breakdown = {
        "environmental": float(row.get("Environmental_Score", 0)),
        "social": float(row.get("Social_Score", 0)),
        "governance": float(row.get("Governance_Score", 0)),
    }

    reliability = {
        "on_time": _format_percentage(row.get("On_Time_Delivery_Rate", 0.9)),
        "defect_rate": _format_percentage(row.get("Defect_Rate", 0.02)),
        "lead_time": _format_ratio(row.get("Lead_Time_Days", 0.3)),
        "operational_index": _format_ratio((row.get("On_Time_Delivery_Rate", 0.9) + 1 - row.get("Defect_Rate", 0.02)) / 2),
    }

    risk_drivers = [
        {"label": "Carbon Emission", "value": float(row.get("Carbon_Emission_Intensity", 0))},
        {"label": "Compliance Level", "value": compliance},
        {"label": "Financial Stability", "value": financial},
    ]

    return {
        "sc_id": row.get("SC_ID", "N/A"),
        "supplier": row.get("Supplier_Name", "N/A"),
        "tier": row.get("Supplier_Tier", "N/A"),
        "industry": row.get("Industry_Sector", "N/A"),
        "region": row.get("Region", "N/A"),
        "country": row.get("Country", "N/A"),
        "commodity": row.get("Commodity_Name", "N/A"),
        "certifications": row.get("Certifications_Active", "None"),
        "employees": int(random.uniform(5000, 25000)),
        "risk_index": risk_index,
        "classification": classification,
        "metrics": {
            "esg_score": round(esg_score, 1),
            "compliance": round(compliance, 1),
            "financial": round(financial, 1),
            "resilience": round(float(row.get("Resilience_Score", esg_score * 0.9)), 1)
            if not pd.isna(row.get("Resilience_Score"))
            else round(esg_score * 0.9, 1),
        },
        "risk_trend": {
            "actual": {"x": actual_years, "y": actual_values},
            "forecast": {"x": future_years, "y": forecast_values},
        },
        "esg_breakdown": esg_breakdown,
        "reliability": reliability,
        "risk_drivers": risk_drivers,
    }


def build_trend_payload(dataset: pd.DataFrame) -> Dict[str, Any]:
    if dataset.empty:
        return {"series": []}

    year_col = _first_available_column(dataset, ["Year"])
    if not year_col:
        return {"series": []}

    years = pd.to_numeric(_safe_series(dataset[year_col]), errors="coerce")
    dataset = dataset.assign(__year=years.round().astype("Int64"))
    dataset = dataset.dropna(subset=["__year"])

    if dataset.empty:
        return {"series": []}

    grouped = dataset.groupby("__year").mean(numeric_only=True).sort_index()
    series_payload = []

    for label, column, color in TREND_FEATURES:
        if column not in grouped.columns:
            continue
        values = grouped[column].dropna()
        if values.empty:
            continue
        actual_years = values.index.astype(int).tolist()
        actual_values = values.clip(0, 100).round(2).tolist()
        forecast_years, forecast_values, upper, lower = _forecast_series(values)
        series_payload.append(
            {
                "name": label,
                "color": color,
                "actual": {"x": actual_years, "y": actual_values},
                "forecast": {"x": forecast_years, "y": forecast_values},
                "confidence": {"x": forecast_years, "upper": upper, "lower": lower},
            }
        )

    return {"series": series_payload}


def aggregate_region_data(dataset: pd.DataFrame, region_df: pd.DataFrame) -> List[Dict]:
    if dataset.empty or "Region" not in dataset.columns:
        fallback = []
        for region in REGION_ORDER:
            region_subset = region_df[region_df["region"] == region]
            fallback.append(
                {
                    "region": region,
                    "country_count": len(region_subset),
                    "supplier_count": 0,
                    "avg_risk_score": round(random.uniform(45, 85), 1),
                    "high_risk_pct": round(random.uniform(10, 25), 1),
                }
            )
        return fallback

    df = dataset.copy()
    df["Region"] = df["Region"].fillna("Unknown")
    risk_col = _first_available_column(df, RISK_SCORE_CANDIDATES)
    grouped = df.groupby("Region")

    results: List[Dict[str, Any]] = []
    handled_regions = set()

    for region in REGION_ORDER:
        region_slice = grouped.get_group(region) if region in grouped.groups else pd.DataFrame()
        handled_regions.add(region)

        supplier_count = int(region_slice["Supplier_Name"].nunique()) if not region_slice.empty and "Supplier_Name" in region_slice.columns else 0
        country_count = int(region_slice["Country"].nunique()) if not region_slice.empty and "Country" in region_slice.columns else int(region_df[region_df["region"] == region]["country"].nunique())

        if not region_slice.empty and risk_col:
            avg_risk = _safe_series(region_slice[risk_col]).mean()
        else:
            avg_risk = random.uniform(45, 85)
        if pd.isna(avg_risk):
            avg_risk = random.uniform(45, 85)

        if not region_slice.empty and "Risk_Classification" in region_slice.columns:
            high_pct = (
                region_slice["Risk_Classification"].eq("High").mean() * 100
                if len(region_slice)
                else 0.0
            )
        else:
            high_pct = random.uniform(10, 25)

        results.append(
            {
                "region": region,
                "country_count": int(country_count),
                "supplier_count": supplier_count,
                "avg_risk_score": round(float(avg_risk), 1),
                "high_risk_pct": round(float(high_pct), 1),
            }
        )

    # Include any additional regions found in the dataset
    for region in grouped.groups.keys():
        if region in handled_regions:
            continue
        region_slice = grouped.get_group(region)
        supplier_count = int(region_slice["Supplier_Name"].nunique()) if "Supplier_Name" in region_slice.columns else len(region_slice)
        country_count = int(region_slice["Country"].nunique()) if "Country" in region_slice.columns else 0
        avg_risk = _safe_series(region_slice[risk_col]).mean() if risk_col else random.uniform(45, 85)
        if pd.isna(avg_risk):
            avg_risk = random.uniform(45, 85)
        high_pct = (
            region_slice["Risk_Classification"].eq("High").mean() * 100
            if "Risk_Classification" in region_slice.columns
            else random.uniform(10, 25)
        )
        results.append(
            {
                "region": region,
                "country_count": int(country_count),
                "supplier_count": supplier_count,
                "avg_risk_score": round(float(avg_risk), 1),
                "high_risk_pct": round(float(high_pct), 1),
            }
        )

    return results


def list_region_options(dataset: pd.DataFrame) -> List[str]:
    if dataset.empty or "Region" not in dataset.columns:
        return []
    regions = dataset["Region"].dropna().astype(str).unique().tolist()
    normalized = set()
    for value in regions:
        value = value.strip()
        if not value:
            continue
        code = REGION_NAME_TO_CODE.get(value.lower(), value.upper())
        normalized.add(code)
    return sorted(normalized)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/api/map-data")
def api_map_data():
    """
    Return all data needed to render the choropleth map.
    Frontend will handle Plotly rendering.
    """
    selected_region = request.args.get("selected")
    
    df = WORLD_REGION_DATA.copy()
    
    # Prepare display region
    if selected_region:
        df["display_region"] = df["region"].apply(
            lambda r: r if r == selected_region else "Other Regions"
        )
        categories = ["Other Regions", selected_region]
    else:
        df["display_region"] = df["region"]
        categories = REGION_ORDER
    
    # Convert to JSON-serializable format
    map_data = {
        "locations": df["iso_alpha"].tolist(),
        "display_regions": df["display_region"].tolist(),
        "regions": df["region"].tolist(),
        "countries": df["country"].tolist(),
        "category_order": categories,
        "color_map": REGION_COLOR_MAP,
        "selected_region": selected_region,
    }
    
    return jsonify(map_data)

@app.route("/api/region-info")
def api_region_info():
    """Return descriptive info for selected region."""
    region = request.args.get("region")
    
    if not region:
        return jsonify({
            "region": None,
            "description": "Select a region on the map to focus on its risk profile, supplier distribution, and supporting narrative.",
            "countries": [],
            "country_count": 0,
            "default": True,
            "highlights": [
                "Regions outside the selection are muted in grey.",
                "Only the chosen region keeps its palette color.",
                "Country lists update dynamically.",
            ],
        })
    
    subset = WORLD_REGION_DATA[WORLD_REGION_DATA["region"] == region]
    countries = sorted(subset["country"].tolist())
    description = REGION_DESCRIPTIONS.get(
        region, "Regional description not available."
    )
    
    return jsonify({
        "region": region,
        "description": description,
        "countries": countries,
        "country_count": len(countries),
        "default": False,
    })


@app.route("/api/region-from-iso")
def api_region_from_iso():
    """Translate ISO code to region name."""
    iso_code = request.args.get("iso")
    
    if not iso_code:
        return jsonify({"region": None})
    
    match = WORLD_REGION_DATA[WORLD_REGION_DATA["iso_alpha"] == iso_code]
    
    if match.empty:
        return jsonify({"region": None})
    
    return jsonify({"region": match.iloc[0]["region"]})


@app.route("/api/overview-summary")
def api_overview_summary():
    """
    Compute and return overview metrics.
    Includes risk score, supplier count, and risk distribution.
    """
    dataset = get_dashboard_dataset()
    region = request.args.get("region")
    summary = build_overview_summary(dataset, region=region)
    return jsonify(summary)


@app.route("/api/top-commodities")
def api_top_commodities():
    """Return top high-risk commodities."""
    region = request.args.get("region")
    dataset = get_dashboard_dataset()
    commodities = build_top_commodities(dataset, region=region)
    return jsonify({"commodities": commodities})


@app.route("/api/top-movers")
def api_top_movers():
    """Return top suppliers with largest risk changes."""
    region = request.args.get("region")
    dataset = get_dashboard_dataset()
    movers = build_top_movers(dataset, region=region)
    return jsonify({"movers": movers})


@app.route("/api/forecast-data")
def api_forecast_data():
    dataset = get_dashboard_dataset()
    payload = build_forecast_payload(dataset)
    return jsonify(payload)


@app.route("/api/trend-data")
def api_trend_data():
    """Return sustainability trend time series."""
    dataset = get_dashboard_dataset()
    payload = build_trend_payload(dataset)
    return jsonify(payload)


@app.route("/api/carbon-trajectory")
def api_carbon_trajectory():
    """Return projected carbon neutrality trajectory."""
    dataset = get_dashboard_dataset()
    payload = build_carbon_trajectory_payload(dataset)
    return jsonify(payload)


@app.route("/api/supplier-profile")
def api_supplier_profile():
    """Return a supplier snapshot from the engineered dataset."""
    dataset = ENGINEERED_DATASET
    supplier_query = request.args.get("supplier")
    row = select_supplier_row(dataset, supplier_query)
    if row is None:
        return jsonify({"error": "Supplier data unavailable"}), 404
    payload = build_supplier_profile_payload(row)
    return jsonify(_to_native(payload))

@app.route("/api/region-summaries")
def api_region_summaries():
    """Return aggregated data for all regions."""
    dataset = get_dashboard_dataset()
    summaries = aggregate_region_data(dataset, WORLD_REGION_DATA)
    return jsonify({"regions": summaries})


@app.route("/api/regions")
def api_regions():
    """Return list of region codes available for filtering."""
    dataset = get_dashboard_dataset()
    regions = list_region_options(dataset)
    return jsonify({"regions": regions})

if __name__ == "__main__":
    app.run(debug=True)