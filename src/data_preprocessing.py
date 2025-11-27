# Data Preprocessing for dataset
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
from typing import Iterable
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from oversampling import run_oversampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLACEHOLDER_TOKENS = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "unknown",
    "not available",
    "tbd",
    "--",
    "-",
}

EDA_COLUMNS = (
    "SC_ID",
    "Supplier_Name",
    "Commodity_Name",
    "Year",
    "Month",
    "Country",
    "Region",
    "Supplier_Tier",
    "Industry_Sector",
    "ESG_Score",
    "Environmental_Score",
    "Social_Score",
    "Governance_Score",
    "Carbon_Emission_Intensity",
    "GHG_Scope1_Intensity",
    "GHG_Scope2_Intensity",
    "GHG_Scope3_Intensity",
    "Water_Intensity",
    "Waste_Management_Efficiency",
    "Renewable_Energy_Usage",
    "Compliance_Level",
    "Sustainability_Report_Availability",
    "Certifications_Active",
    "Lead_Time_Days",
    "Production_Capacity",
    "On_Time_Delivery_Rate",
    "Defect_Rate",
    "Incident_History_Count",
    "Labour_Compliance_Score",
    "Diversity_Index",
    "Financial_Stability_Score",
    "Freight_Cost",
    "Cost_Index",
    "Commodity_Price_Index",
    "Commodity_Demand",
    "Trade_Volume",
    "Logistics_Distance_km",
    "Risk_Classification",
)

def _filter_known(columns: Iterable[str]) -> list[str]:
    """Return only columns verified in the EDA notebook."""
    return [col for col in columns if col in EDA_COLUMNS]

NUMERIC_COLUMNS = _filter_known([
    "Year",
    "Supplier_Tier",
    "ESG_Score",
    "Environmental_Score",
    "Social_Score",
    "Governance_Score",
    "Carbon_Emission_Intensity",
    "GHG_Scope1_Intensity",
    "GHG_Scope2_Intensity",
    "GHG_Scope3_Intensity",
    "Water_Intensity",
    "Waste_Management_Efficiency",
    "Renewable_Energy_Usage",
    "Compliance_Level",
    "Sustainability_Report_Availability",
    "Lead_Time_Days",
    "Production_Capacity",
    "On_Time_Delivery_Rate",
    "Defect_Rate",
    "Incident_History_Count",
    "Labour_Compliance_Score",
    "Diversity_Index",
    "Financial_Stability_Score",
    "Freight_Cost",
    "Cost_Index",
    "Commodity_Price_Index",
    "Commodity_Demand",
    "Trade_Volume",
    "Logistics_Distance_km",
])

RATIO_COLUMNS = _filter_known([
    "Carbon_Emission_Intensity",
    "Waste_Management_Efficiency",
    "Renewable_Energy_Usage",
    "On_Time_Delivery_Rate",
    "Defect_Rate",
    "Diversity_Index",
])

BINARY_COLUMNS = _filter_known(["Sustainability_Report_Availability"])
INTEGER_COLUMNS = _filter_known([
    "Year",
    "Supplier_Tier",
    "Compliance_Level",
    "Lead_Time_Days",
    "Production_Capacity",
    "Incident_History_Count",
])

NON_NEGATIVE_COLUMNS = _filter_known([
    "ESG_Score",
    "Environmental_Score",
    "Social_Score",
    "Governance_Score",
    "Carbon_Emission_Intensity",
    "GHG_Scope1_Intensity",
    "GHG_Scope2_Intensity",
    "GHG_Scope3_Intensity",
    "Water_Intensity",
    "Waste_Management_Efficiency",
    "Renewable_Energy_Usage",
    "Compliance_Level",
    "Sustainability_Report_Availability",
    "Lead_Time_Days",
    "Production_Capacity",
    "On_Time_Delivery_Rate",
    "Defect_Rate",
    "Freight_Cost",
    "Commodity_Price_Index",
    "Commodity_Demand",
    "Trade_Volume",
    "Financial_Stability_Score",
    "Cost_Index",
    "Labour_Compliance_Score",
    "Diversity_Index",
    "Incident_History_Count",
    "Logistics_Distance_km",
])

REGION_CODE_MAP = {
    "east asia and pacific": "EAP",
    "south asia": "SA",
    "north america": "NA",
    "sub-saharan africa": "SSA",
    "europe and central asia": "ECA",
    "latin america and caribbean": "LAC",
    "middle east, north africa, afghanistan and pakistan": "MENA-AP",
}

MONTH_ABBREVIATIONS = {
    "january": "JAN",
    "february": "FEB",
    "march": "MAR",
    "april": "APR",
    "may": "MAY",
    "june": "JUN",
    "july": "JUL",
    "august": "AUG",
    "september": "SEP",
    "october": "OCT",
    "november": "NOV",
    "december": "DEC",
}

RISK_LEVEL_MAP = {
    "low": "Low",
    "lo": "Low",
    "moderate": "Moderate",
    "medium": "Moderate",
    "med": "Moderate",
    "mod": "Moderate",
    "high": "High",
    "hi": "High",
    "critical": "High",
}

MISSING_VALUE_RULES: dict[str, dict[str, object]] = {
    "Certifications_Active": {"strategy": "constant", "value": "Not Specified"},
    "ESG_Score": {"strategy": "median"},
    "Labour_Compliance_Score": {"strategy": "median"},
    "Governance_Score": {"strategy": "median"},
    "Environmental_Score": {"strategy": "median"},
    "Social_Score": {"strategy": "median"},
    "Financial_Stability_Score": {"strategy": "median"},
}

INDUSTRY_SECTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "Agriculture & Agribusiness": (
        "Agriculture",
        "Agribusiness",
        "Agriculture/Food",
        "Agribusiness/Farming",
        "Agribusiness/Exports",
        "Agriculture/Coffee",
        "Agriculture/Coffee Exports",
        "Agriculture/Cocoa",
        "Agriculture/Cocoa Exports",
        "Agriculture/Sugar",
        "Agriculture/Tea",
        "Agriculture/Tea Exports",
        "Agriculture/Dried Fruits",
        "Agriculture/Spices",
        "Agriculture/Olive Oil",
        "Agriculture/Oils",
        "Agriculture/Horticulture",
        "Agriculture/Cashew and Cotton",
        "Agriculture/Cashew Exports",
        "Agriculture/Coffee and Tea",
        "Agribusiness/Oilseeds",
        "Agribusiness/Oils",
        "Tea/Agribusiness",
        "Cocoa/Agribusiness",
        "Spices/Agriculture",
        "Coffee/Agriculture",
        "Agricultural Trade",
        "Agri Inputs/Social Enterprise",
        "Agri Aggregation/Logistics",
    ),
    "Food & Beverage Processing": (
        "Food Processing",
        "Food Processing/Meat",
        "Food Processing/FMCG",
        "Food Processing/Sugar",
        "Food Processing/Bakery",
        "Food Processing/Exports",
        "Food Processing/Agribusiness",
        "Food Processing/Ingredients",
        "Food Processing/Snacks",
        "Food and Beverage",
        "Beverages",
        "Beverages/Packaging",
        "Beverages/FMCG",
        "Dairy/Food Processing",
        "Dairy/Food",
        "Seafood/Fisheries",
        "Fisheries",
        "Food Ingredients",
        "Agribusiness/Sugar",
        "Agribusiness/Fruit Exports",
    ),
    "Oil, Gas & Refining": (
        "Oil and Gas",
        "Oil and Gas/Refining",
        "Oil and Gas/Petrochemicals",
        "Oil and Gas Distribution",
        "Oil and Gas/Refining/Distribution",
        "Refining/Oil and Gas",
        "Refining",
        "Energy/Refining",
        "LNG/Energy",
        "LNG/Energy (project-level)",
        "Energy/LNG",
        "Oil and Gas/LNG",
        "Gas/Energy",
        "Oil and Timber",
    ),
    "Metals & Mining (Primary)": (
        "Metals and Mining",
        "Metals and Minings",
        "Mining",
        "Mining and Metals",
        "Mining/Metals",
        "Metals",
        "Gold/Mining",
        "Copper/Mining",
        "Copper",
        "Coal/Mining",
        "Uranium/Mining",
        "Bauxite/Mining",
        "Mining (Tin)",
        "PGMs/Mining",
        "Mining/Iron Ore",
        "Mining and Steel",
        "Copper and Cobalt",
        "Base Metals",
        "Mining/Graphite",
        "Mining/Timber",
        "Mining/Aggregates",
        "Mining/Commodities",
        "Minerals/Heavy Minerals",
        "Mining Investments",
        "Rare Earth Materials",
    ),
    "Steel & Metallurgical": (
        "Steel and Materials",
        "Steel/Materials",
        "Steel",
        "Steel and Heavy Industry",
        "Steel/Industrial Pipes",
        "Steel/Metals",
        "Steel/Mining",
        "Metallurigical",
        "Aluminium/Metals",
    ),
    "Chemicals & Petrochemicals": (
        "Chemicals",
        "Petrochemicals",
        "Petrochemicals/Polymers",
        "Petrochemicals and Polymers",
        "Petrochemicals/Fertilizers",
        "Chemicals and Fuels",
        "Chemicals/Energy",
        "Chemicals/Fibers",
        "Chemicals/Activated Carbon",
        "Chemicals/Coatings",
        "Chemicals/Paints",
        "Mining and Specialty Chemicals",
        "Agrochemicals",
        "Agriculture/Chemicals",
    ),
    "Fertilizers": (
        "Fertilizers",
        "Fertilizers/Mining",
        "Fertilizers/Petrochemicals",
        "Fertilizers/Chemicals",
        "Agrochemicals/Fertilizers",
    ),
    "Energy & Utilities": (
        "Energy",
        "Energy Materials",
        "Energy/Utilities",
        "Energy/Mining",
        "Energy/Investment",
        "Energy Equipment",
        "Energy/Hydropower",
        "Renewable Energy",
    ),
    "Automotive & Transportation": (
        "Automotive",
        "Automotive/EV",
        "Automotive Components",
        "Automotive Suppliers",
        "Automotive/Equipment",
        "Automotive/Components",
        "Automotive/Electronics",
        "Automotive/Energy",
        "Automotive/Agriculture",
        "Machinery/Agriculture",
        "Agriculture Machinery",
        "Heavy Machinery",
    ),
    "Electronics, Semiconductors & Technology": (
        "Semiconductors",
        "Electronics",
        "Electronics/Semiconductors",
        "Electronics/IT",
        "Electronics/Manufacturing",
        "Electronics/Materials",
        "Semiconductors/Electronics",
        "Technology",
        "Consumer Electronics",
        "Networking/IT",
        "IT/Enterprise Solutions",
        "IT/Software Hardware",
        "Industrial IoT",
        "Telecommunications",
    ),
    "Industrial Manufacturing & Construction": (
        "Industrial Manufacturing",
        "Materials",
        "Materials/Manufacturing",
        "Industrial Components",
        "Furniture/Manufacturing",
        "Packaging/Manufacturing",
        "Glass Manufacturing",
        "Cement and Construction Materials",
        "Cement",
        "Cement/Construction",
        "Cement and Building Materials",
        "Construction Materials",
        "Construction/Materials",
        "Industrial Tools/Manufacturing",
        "HVAC/Components",
        "Forestry/Materials",
        "Forestry/Packaging",
        "Pulp and Paper",
    ),
    "Pharmaceuticals & Healthcare": (
        "Pharmaceuticals",
        "Pharmaceutical",
        "Pharmaceutical Manufacturing",
        "Biotechnology",
        "Healthcare/Manufacturing",
        "FMCG/Pharma",
    ),
    "Consumer Goods, Retail & Logistics": (
        "FMCG",
        "FMCG/Manufacturing",
        "FMCG/Industrial",
        "FMCG/Agriculture",
        "FMCG/Agribusiness",
        "Textiles and Apparel",
        "Textiles/Apparel",
        "Textiles/Handicrafts",
        "Textiles/Materials",
        "Retail",
        "Retail/Distribution",
        "Retail/Consumer",
        "Retail/FMCG",
        "Footwear/Consumer Goods",
        "Logistics",
        "Logistics/Transportation",
        "Logistics/Ports",
        "Aviation/Logistics",
        "Distribution/Services",
        "Aerospace",
        "Aerospace/Defense",
        "Aerospace and Defense",
        "Aerospace/Transportation",
        "Defense",
        "Oilfield Services",
        "Mining Services/Explosives",
        "Commodities Trading/Agribusiness",
        "Commodities Trading/Mining",
        "Commodities Trading/Metals",
        "Conglomerate/Exports",
        "Diamonds/Mining",
        "Diamonds/Trading",
        "Diamonds",
    ),
}

def _build_industry_sector_lookup(
    groups: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for category, labels in groups.items():
        for label in labels:
            key = label.strip().lower()
            if key:
                mapping[key] = category
    return mapping

INDUSTRY_SECTOR_LOOKUP = _build_industry_sector_lookup(INDUSTRY_SECTOR_GROUPS)

# Main preprocessing function
def preprocess_data(
    raw_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path)
    df = (
        df.copy()
        .pipe(_standardise_column_names)
        .pipe(_clean_text_columns)
        .pipe(_normalise_risk_classification)
        .pipe(_uppercase_country)
        .pipe(_abbreviate_month)
        .pipe(_map_region_codes)
        .pipe(_map_industry_sector)
    )
    df = _coerce_numeric_columns(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = _impute_missing_values(df)
    df = _winsorize_numeric_outliers(df)
    df = _enforce_value_ranges(df)
    df = _finalize_dtypes(df)

    processed_dir = output_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path} with {len(df)} rows.")
    try:
        oversampled_path = run_oversampling()
        print(
            "Oversampling completed successfully. "
            f"Oversampled dataset saved to {oversampled_path}."
        )
    except Exception as exc:
        print(f"Oversampling step failed: {exc}")
    return df

def _standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_columns = df.select_dtypes(include="object").columns
    for col in object_columns:
        df[col] = df[col].map(_clean_text_value)
    return df

def _clean_text_value(value: object) -> object:
    if isinstance(value, str):
        cleaned = re.sub(r"\s+", " ", value.strip())
        lowered = cleaned.lower()
        if lowered in PLACEHOLDER_TOKENS:
            return np.nan
        return cleaned
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return value

def _normalise_risk_classification(df: pd.DataFrame) -> pd.DataFrame:
    if "Risk_Classification" not in df.columns:
        return df
    df = df.copy()

    def _normalise(value: object) -> object:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in RISK_LEVEL_MAP:
                return RISK_LEVEL_MAP[lowered]
            return value.strip().title()
        return value

    df["Risk_Classification"] = df["Risk_Classification"].map(_normalise)
    return df

# Uppercase all country names
def _uppercase_country(df: pd.DataFrame) -> pd.DataFrame:
    if "Country" not in df.columns:
        return df
    df = df.copy()
    df["Country"] = df["Country"].map(
        lambda value: value.upper() if isinstance(value, str) else value
    )
    return df

# Change months from full name to short name
# Eg. "December" > "DEC"
def _abbreviate_month(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize month names to three-letter uppercase abbreviations."""
    if "Month" not in df.columns:
        return df

    def _map_month(value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if not cleaned:
                return value
            return MONTH_ABBREVIATIONS.get(cleaned, cleaned[:3].upper())
        return value

    df = df.copy()
    df["Month"] = df["Month"].map(_map_month)
    return df

# Map all 7 regions to region codes
# Dictionary:
# "East Asia and Pacific" > "EAP"
# "South Asia" > "SA"
# "North America" > "NA"
# "Sub-Saharan Africa" > "SSA"
# "Europe and Central Asia" > "ECA"
# "Latin America and Caribbean" > "LAC"
# "Middle East, North Africa, Afghanistan and Pakistan" > "MENA-AP"
def _map_region_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Map verbose region labels to their short codes."""
    region_columns = [col for col in ("Region", "Regions") if col in df.columns]
    if not region_columns:
        return df

    def _map_value(value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            return REGION_CODE_MAP.get(cleaned.lower(), cleaned)
        return value

    df = df.copy()
    for col in region_columns:
        df[col] = df[col].map(_map_value)
    return df

# Map Industry_Sector
def _map_industry_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Map granular industry labels to consolidated sectors."""
    column = next(
        (col for col in ("Industry_Sector", "Industry") if col in df.columns), None
    )
    if not column:
        return df

    def _map_value(value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return value
            return INDUSTRY_SECTOR_LOOKUP.get(cleaned.lower(), cleaned)
        return value

    df = df.copy()
    df[column] = df[column].map(_map_value)
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present_numeric = [col for col in NUMERIC_COLUMNS if col in df.columns]
    for col in present_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if present_numeric:
        df[present_numeric] = df[present_numeric].replace([np.inf, -np.inf], np.nan)
    return df

# Impute the columns with missing values only (based on EDA)
def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, config in MISSING_VALUE_RULES.items():
        if column not in df.columns:
            continue
        series = df[column]
        if not series.isna().any():
            continue
        strategy = config.get("strategy")
        if strategy == "constant":
            df[column] = series.fillna(config.get("value"))
        elif strategy == "median":
            median = series.median()
            df[column] = series.fillna(median)
    return df

def _winsorize_numeric_outliers(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        series = df[col]
        if series.isna().all():
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        df[col] = series.clip(lower=lower, upper=upper)
    return df


def _enforce_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    for col in RATIO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    if "Supplier_Tier" in df.columns:
        df["Supplier_Tier"] = df["Supplier_Tier"].clip(1, 3)
    if "Compliance_Level" in df.columns:
        df["Compliance_Level"] = df["Compliance_Level"].clip(0, 3)
    return df


def _finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    return df

if __name__ == "__main__":
    preprocess_data()