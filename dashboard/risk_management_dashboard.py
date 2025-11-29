import random
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_PATH

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

NORTH_AMERICA_COUNTRIES = {"Canada", "United States"}

LATAM_AND_CARIBBEAN_COUNTRIES = {
    "Antigua and Barbuda",
    "Argentina",
    "Bahamas",
    "Barbados",
    "Belize",
    "Bolivia",
    "Brazil",
    "Chile",
    "Colombia",
    "Costa Rica",
    "Cuba",
    "Dominican Republic",
    "Ecuador",
    "El Salvador",
    "Grenada",
    "Guatemala",
    "Guyana",
    "Haiti",
    "Honduras",
    "Jamaica",
    "Mexico",
    "Nicaragua",
    "Panama",
    "Paraguay",
    "Peru",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Suriname",
    "Trinidad and Tobago",
    "Uruguay",
    "Venezuela",
}

SOUTH_ASIA_COUNTRIES = {
    "Bangladesh",
    "Bhutan",
    "India",
    "Maldives",
    "Nepal",
    "Sri Lanka",
}

MENA_AP_COUNTRIES = {
    "Afghanistan",
    "Algeria",
    "Bahrain",
    "Djibouti",
    "Egypt",
    "Iran",
    "Iraq",
    "Israel",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Pakistan",
    "Qatar",
    "Saudi Arabia",
    "Syrian Arab Republic",
    "Tunisia",
    "United Arab Emirates",
    "West Bank and Gaza",
    "Yemen, Rep.",
}

CENTRAL_ASIA_AND_EU_COUNTRIES = {
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Kazakhstan",
    "Kyrgyz Republic",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
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
    """Return a dataframe with ISO codes mapped to the custom seven regions."""
    gapminder = px.data.gapminder().query("year == 2007")[
        ["country", "iso_alpha", "continent"]
    ].copy()

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

        if continent == "Oceania":
            return REGION_EAST_ASIA_PACIFIC

        if continent == "Asia":
            return REGION_EAST_ASIA_PACIFIC

        return REGION_EAST_ASIA_PACIFIC

    gapminder["region"] = gapminder.apply(assign_region, axis=1)
    world_regions = gapminder[["country", "iso_alpha", "region"]].dropna()
    world_regions = world_regions.drop_duplicates(subset="iso_alpha")
    return world_regions


WORLD_REGION_DATA = build_region_dataframe()

def load_preprocessed_dataset() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    return pd.DataFrame()

PREPROCESSED_DATASET = load_preprocessed_dataset()


def _figure_to_serialisable(data: Dict) -> Dict:
    """Recursively convert numpy arrays in Plotly figure to lists."""
    if isinstance(data, dict):
        return {key: _figure_to_serialisable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_figure_to_serialisable(item) for item in data]
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def build_world_map(
    data: pd.DataFrame,
    selected_region: Optional[str] = None,
) -> Dict:
    """Create a choropleth map that highlights a selected region."""
    df = data.copy()
    muted_label = "Other Regions"

    if selected_region:
        df["display_region"] = df["region"].apply(
            lambda region: region if region == selected_region else muted_label
        )
        color_map = {muted_label: "#d9d9d9", selected_region: REGION_COLOR_MAP[selected_region]}
        category_order = [muted_label, selected_region]
        title = f"{selected_region} Highlighted"
    else:
        df["display_region"] = df["region"]
        color_map = {muted_label: "#d9d9d9", **REGION_COLOR_MAP}
        category_order = REGION_ORDER
        title = "Global Regions Overview"

    fig = px.choropleth(
        data_frame=df,
        locations="iso_alpha",
        color="display_region",
        hover_name="country",
        hover_data={"region": True},
        custom_data=["region"],
        category_orders={"display_region": category_order},
        color_discrete_map=color_map,
        title=title,
        projection="natural earth",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Region",
        height=420,
        title=dict(
            text=title,
            font=dict(size=20),
            x=0.01,
            xanchor="left",
        ),
    )
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="#F5F5F5",
    )
    return _figure_to_serialisable(fig.to_dict())


def build_region_info(selected_region: Optional[str]) -> Dict:
    """Return descriptive data for the selected region."""
    if not selected_region:
        return {
            "region": None,
            "description": "Select a region on the map to focus on its risk profile, supplier distribution, and supporting narrative.",
            "countries": [],
            "countryCount": 0,
            "default": True,
            "highlights": [
                "Regions outside the selection are muted in grey.",
                "Only the chosen region keeps its palette color.",
                "Country lists update dynamically.",
            ],
        }

    subset = WORLD_REGION_DATA[WORLD_REGION_DATA["region"] == selected_region]
    countries = sorted(subset["country"].tolist())
    description = REGION_DESCRIPTIONS.get(
        selected_region, "Regional description not available."
    )
    return {
        "region": selected_region,
        "description": description,
        "countries": countries,
        "countryCount": len(countries),
        "default": False,
    }


def resolve_region_from_iso(iso_code: Optional[str]) -> Optional[str]:
    """Return the region name associated with a given ISO alpha-3 code."""
    if not iso_code:
        return None
    match = WORLD_REGION_DATA[WORLD_REGION_DATA["iso_alpha"] == iso_code]
    if match.empty:
        return None
    return match.iloc[0]["region"]


DASHBOARD_DIR = Path(__file__).resolve().parent
HTML_DIR = DASHBOARD_DIR / "html"
CSS_DIR = DASHBOARD_DIR / "css"
JS_DIR = DASHBOARD_DIR / "js"
TABS_DIR = HTML_DIR / "tabs"

app = Flask(__name__)


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


@app.route("/api/world-map")
def api_world_map():
    """Return the Plotly figure for the world map."""
    selected_region = request.args.get("selected")
    figure = build_world_map(WORLD_REGION_DATA, selected_region)
    return jsonify({"figure": figure})


@app.route("/api/region-info")
def api_region_info():
    """Return descriptive info for the selected region."""
    region = request.args.get("region")
    info = build_region_info(region)
    return jsonify(info)


@app.route("/api/region-from-iso")
def api_region_from_iso():
    """Translate an ISO code into its configured region."""
    iso_code = request.args.get("iso")
    region = resolve_region_from_iso(iso_code)
    return jsonify({"region": region})


def build_overview_summary() -> Dict:
    """Compute summary metrics for the overview panel."""
    random_score = round(random.uniform(60, 95), 1)
    dataset = PREPROCESSED_DATASET
    if not dataset.empty and "Supplier_Name" in dataset.columns:
        total_suppliers = int(dataset["Supplier_Name"].nunique())
    else:
        total_suppliers = 0

    if not dataset.empty and "Risk_Classification" in dataset.columns:
        counts = dataset["Risk_Classification"].value_counts().reindex(
            ["High", "Moderate", "Low"], fill_value=0
        )
    else:
        counts = pd.Series([1, 1, 1], index=["High", "Moderate", "Low"])

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        title="Supplier-Commodity Risk Mix",
        color=counts.index,
        color_discrete_map={
            "High": "#fa7066",
            "Moderate": "#fcaf56",
            "Low": "#adcf7a",
        },
    )
    fig.update_layout(margin=dict(t=60, b=0, l=0, r=0), legend_title="")

    return {
        "risk_score": random_score,
        "total_suppliers": total_suppliers,
        "donut": _figure_to_serialisable(fig.to_dict()),
    }


@app.route("/api/overview-summary")
def api_overview_summary():
    """Provide headline metrics for the overview panel."""
    summary = build_overview_summary()
    return jsonify(
        {
            "riskScore": summary["risk_score"],
            "totalSuppliers": summary["total_suppliers"],
            "donut": summary["donut"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

