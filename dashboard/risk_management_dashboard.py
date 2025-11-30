import random
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH

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
    except:
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

def load_preprocessed_dataset() -> pd.DataFrame:
    """Load the preprocessed supplier/commodity dataset."""
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return pd.DataFrame()

# Initialise data
WORLD_REGION_DATA = build_region_dataframe()
PREPROCESSED_DATASET = load_preprocessed_dataset()

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
    
    # Simulate risk scoring
    if not df.empty and "Risk_Classification" not in df.columns:
        np.random.seed(42)
        df["Risk_Classification"] = np.random.choice(
            ["High", "Moderate", "Low"], 
            size=len(df), 
            p=[0.15, 0.45, 0.40]
        )
    return df

def aggregate_region_data(dataset: pd.DataFrame, region_df: pd.DataFrame) -> List[Dict]:
    # Merge dataset with region info if country column exists
    results = []
    
    for region in REGION_ORDER:
        region_subset = region_df[region_df["region"] == region]
        
        summary = {
            "region": region,
            "country_count": len(region_subset),
            "supplier_count": 0,
            "avg_risk_score": round(random.uniform(45, 85), 1),
            "high_risk_pct": round(random.uniform(10, 25), 1),
        }
        
        results.append(summary)
    
    return results

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
    # Process dataset
    dataset = compute_risk_scores(PREPROCESSED_DATASET)
    
    # Calculate metrics
    random_score = round(random.uniform(60, 95), 1)
    
    if not dataset.empty and "Supplier_Name" in dataset.columns:
        total_suppliers = int(dataset["Supplier_Name"].nunique())
    else:
        total_suppliers = 0
    
    if not dataset.empty and "Risk_Classification" in dataset.columns:
        counts = dataset["Risk_Classification"].value_counts().reindex(
            ["High", "Moderate", "Low"], fill_value=0
        )
    else:
        counts = pd.Series([15, 48, 37], index=["High", "Moderate", "Low"])
    
    # Return data for frontend to render donut chart
    return jsonify({
        "risk_score": random_score,
        "total_suppliers": total_suppliers,
        "risk_distribution": {
            "values": counts.tolist(),
            "labels": counts.index.tolist(),
            "colors": ["#fa7066", "#fcaf56", "#adcf7a"],
        },
        "compliance_rate": 92,
    })


@app.route("/api/top-commodities")
def api_top_commodities():
    """Return top high-risk commodities."""
    # This would normally query the processed dataset
    commodities = [
        {"name": "Commodity Y", "sustainability": 72, "ghg_score": 0.89, "cost": 4.8},
        {"name": "Commodity X", "sustainability": 47, "ghg_score": 0.89, "cost": 5.3},
        {"name": "Commodity A", "sustainability": 68, "ghg_score": 0.88, "cost": 6.3},
        {"name": "Commodity T", "sustainability": 23, "ghg_score": 0.38, "cost": 7.1},
        {"name": "Commodity E", "sustainability": 64, "ghg_score": 0.89, "cost": 6.3},
    ]
    
    return jsonify({"commodities": commodities})


@app.route("/api/top-movers")
def api_top_movers():
    """Return top suppliers with largest risk changes."""
    movers = [
        {"name": "Supplier G", "resilience": 0.63, "risk_delta": 0.07},
        {"name": "Supplier Q", "resilience": 0.54, "risk_delta": 0.06},
        {"name": "Supplier B", "resilience": 0.51, "risk_delta": 0.04},
        {"name": "Supplier L", "resilience": 0.45, "risk_delta": 0.03},
        {"name": "Supplier C", "resilience": 0.35, "risk_delta": 0.02},
    ]
    
    return jsonify({"movers": movers})


@app.route("/api/forecast-data")
def api_forecast_data():
    """Return 2026 forecast distribution."""
    return jsonify({
        "values": [14, 48, 38],
        "labels": ["High", "Medium", "Low"],
        "colors": ["#fa7066", "#fcaf56", "#adcf7a"],
    })


@app.route("/api/trend-data")
def api_trend_data():
    """Return sustainability trend time series."""
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    series = [
        {
            "name": "Carbon Emission Intensity",
            "color": "#1abc9c",
            "x": years,
            "y": [12, 13, 14, 16, 18, 21],
        },
        {
            "name": "GHG Scopes 1",
            "color": "#3498db",
            "x": years,
            "y": [8, 9, 11, 12, 13, 14],
        },
        {
            "name": "GHG Scopes 2",
            "color": "#9b59b6",
            "x": years,
            "y": [5, 6, 6, 7, 8, 9],
        },
        {
            "name": "Renewable Energy Usage",
            "color": "#f39c12",
            "x": years,
            "y": [3, 4, 5, 6, 7, 8],
        },
    ]
    
    return jsonify({"series": series})

@app.route("/api/region-summaries")
def api_region_summaries():
    """Return aggregated data for all regions."""
    summaries = aggregate_region_data(PREPROCESSED_DATASET, WORLD_REGION_DATA)
    return jsonify({"regions": summaries})

if __name__ == "__main__":
    app.run(debug=True)