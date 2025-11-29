import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH, RAW_DATA_PATH

def _load_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Load CSV while gracefully handling non-UTF8 characters."""
    encodings = ("utf-8", "utf-8-sig", "latin1", "cp1252")
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding=encodings[-1], encoding_errors="replace")

SUPPLIER_COUNTRY_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "BHP GROUP": ("Australia", "East Asia and Pacific"),
    "COMINOR/OIL AND TIMBER EXPORTERS": (
        "Republic of the Congo",
        "Sub-Saharan Africa",
    ),
    "FURUKAWA ELECTRIC CO.": ("Japan", "East Asia and Pacific"),
    "GANFENG LITHIUM CO.": ("China", "East Asia and Pacific"),
    "MURATA MANUFACTURING CO.": ("Japan", "East Asia and Pacific"),
    "MONÓMEROS COLOMBO VENEZOLANOS": ("Colombia", "Latin America and Caribbean"),
    "MONÃMEROS COLOMBO VENEZOLANOS": ("Colombia", "Latin America and Caribbean"),
    "NIPPON STEEL CORPORATION": ("Japan", "East Asia and Pacific"),
    "PT PUPUK INDONESIA": ("Indonesia", "East Asia and Pacific"),
    "RIO TINTO GROUP": ("United Kingdom", "Europe and Central Asia"),
    "SIAM CEMENT GROUP (SCG)": ("Thailand", "East Asia and Pacific"),
    "SUMITOMO METAL MINING": ("Japan", "East Asia and Pacific"),
    "ZHEJIANG HUAYOU COBALT CO.": ("China", "East Asia and Pacific"),
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

REGION_CODE_MAP = {
    "east asia and pacific": "EAP",
    "south asia": "SA",
    "north america": "NA",
    "sub-saharan africa": "SSA",
    "europe and central asia": "ECA",
    "latin america and caribbean": "LAC",
    "middle east, north africa, afghanistan and pakistan": "MENA-AP",
}

INDUSTRY_SECTOR_GROUPS: Dict[str, Tuple[str, ...]] = {
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

GEO_MEDIAN_IMPUTE_COLUMNS = [
    "ESG_Score",
    "Labour_Compliance_Score",
    "Governance_Score",
    "Environmental_Score",
    "Social_Score",
    "Financial_Stability_Score",
]

def preprocess_data(
    raw_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Load raw data, apply preprocessing helpers, and save the processed CSV."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df = _load_csv_with_fallback(raw_path)
    processed_df = (
        df.pipe(_fix_supplier_country_records)
        .pipe(_uppercase_country)
        .pipe(_abbreviate_month)
        .pipe(_map_region_codes)
        .pipe(_map_industry_sector)
        .pipe(_impute_missing_values)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    return processed_df

def _build_industry_sector_lookup(
    groups: Dict[str, Tuple[str, ...]]
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for category, labels in groups.items():
        for label in labels:
            key = label.strip().lower()
            if key:
                mapping[key] = category
    return mapping

INDUSTRY_SECTOR_LOOKUP = _build_industry_sector_lookup(INDUSTRY_SECTOR_GROUPS)

def _normalise_supplier_name(value: object) -> object:
    if isinstance(value, str):
        cleaned = re.sub(r"\s+", " ", value.strip())
        return cleaned.upper()
    return value

def _fix_supplier_country_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uppercase supplier names, realign country + region overrides, and return tmp_df.
    """
    required_columns = {"Supplier_Name", "Country"}
    if not required_columns.issubset(df.columns):
        return df

    tmp_df = df.copy()
    tmp_df["Supplier_Name"] = tmp_df["Supplier_Name"].map(_normalise_supplier_name)

    mask = tmp_df["Supplier_Name"].isin(SUPPLIER_COUNTRY_OVERRIDES)
    if mask.any():
        overrides = tmp_df.loc[mask, "Supplier_Name"].map(
            SUPPLIER_COUNTRY_OVERRIDES
        )
        tmp_df.loc[mask, "Country"] = overrides.map(lambda pair: pair[0])
        if "Region" in tmp_df.columns:
            tmp_df.loc[mask, "Region"] = overrides.map(lambda pair: pair[1])

    return tmp_df


def _uppercase_country(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase all country labels if the column is available."""
    if "Country" not in df.columns:
        return df
    tmp_df = df.copy()
    tmp_df["Country"] = tmp_df["Country"].map(
        lambda value: value.upper() if isinstance(value, str) else value
    )
    return tmp_df


def _abbreviate_month(df: pd.DataFrame) -> pd.DataFrame:
    """Convert full month names to 3-letter abbreviations."""
    if "Month" not in df.columns:
        return df

    def _map_month(value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if not cleaned:
                return value
            return MONTH_ABBREVIATIONS.get(cleaned, cleaned[:3].upper())
        return value

    tmp_df = df.copy()
    tmp_df["Month"] = tmp_df["Month"].map(_map_month)
    return tmp_df


def _map_region_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Map human-readable regions to their canonical codes."""
    region_columns = [col for col in ("Region", "Regions") if col in df.columns]
    if not region_columns:
        return df

    def _map_value(value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            return REGION_CODE_MAP.get(cleaned.lower(), cleaned)
        return value

    tmp_df = df.copy()
    for col in region_columns:
        tmp_df[col] = tmp_df[col].map(_map_value)
    return tmp_df


def _map_industry_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise industry sector labels using INDUSTRY_SECTOR_GROUPS."""
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

    tmp_df = df.copy()
    tmp_df[column] = tmp_df[column].map(_map_value)
    return tmp_df

# Impute missing values with median
# Handle 'Certifications_Active' column
def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with geographic medians falling back to global median."""
    tmp_df = df.copy()
    for column in GEO_MEDIAN_IMPUTE_COLUMNS:
        if column not in tmp_df.columns or not tmp_df[column].isna().any():
            continue
        tmp_df = _fill_with_geographic_median(tmp_df, column)
    for column in ["Certifications_Active"]:
        if column not in tmp_df.columns or not tmp_df[column].isna().any():
            continue
        tmp_df[column] = tmp_df[column].fillna("None")
    return tmp_df

def _fill_with_geographic_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    tmp_df = df.copy()
    if "Country" in tmp_df.columns:
        country_medians = tmp_df.groupby("Country")[column].transform("median")
        tmp_df[column] = tmp_df[column].fillna(country_medians)
    if "Region" in tmp_df.columns:
        region_medians = tmp_df.groupby("Region")[column].transform("median")
        tmp_df[column] = tmp_df[column].fillna(region_medians)
    tmp_df[column] = tmp_df[column].fillna(tmp_df[column].median())
    return tmp_df

if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing complete!")