from dash import Dash, dcc, html
import pandas as pd
import plotly.express as px

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

NORTH_AMERICA_COUNTRIES = {
    "Canada",
    "United States",
}

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

CENTRAL_ASIA_AND_CAUCAUSUS_COUNTRIES = {
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Kazakhstan",
    "Kyrgyz Republic",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
}

CARD_BASE_STYLE = {
    "border": "1px solid #e5e5e5",
    "borderRadius": "18px",
    "padding": "24px",
    "backgroundColor": "#FFFFFF",
    "boxShadow": "0 20px 40px rgba(15, 57, 95, 0.08)",
}

RISK_LEVELS = [
    ("High", "#D7263D"),
    ("Moderate", "#F5A524"),
    ("Low", "#F7DC6F"),
    ("Neutral", "#27AE60"),
]

KEY_METRICS = [
    ("Total Suppliers by Risk Category", "450"),
    ("Tons Commodities by Risk Category", "320"),
]

TOP_RISK_SUPPLIERS = ["Supplier A", "Supplier B", "Supplier C", "Supplier D"]

MODEL_PERFORMANCE = {
    "Accuracy": "0.85",
    "F1-score": "0.83",
    "AUC": "0.88",
}

SUPPLIER_DETAILS = [
    ("Supplier A", "China", "Electronics", "Tier 1", "High"),
    ("Supplier B", "Vietnam", "Textiles", "Tier 2", "Moderate"),
    ("Supplier C", "Brazil", "Agriculture", "Tier 2", "Low"),
    ("Supplier D", "Germany", "Automotive", "Tier 1", "Moderate"),
]

CRITICAL_ALERTS = [
    "Supplier A – high risk; high emissions",
    "Supplier F – compliance audit overdue",
]

RECOMMENDED_ACTIONS = [
    "Improve supplier audits",
    "Transition to renewable energy",
    "Enhance ESG disclosure requirements",
]


def card_container(children, extra_style=None):
    """Wrap content in a styled dashboard card."""
    style = CARD_BASE_STYLE.copy()
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)


def build_feature_importance_chart():
    """Bar chart summarizing dummy feature importances."""
    df = pd.DataFrame(
        {
            "Feature": [
                "Carbon Emissions",
                "ESG Score",
                "Geographic Region",
                "Commodity Type",
            ],
            "Importance": [0.9, 0.72, 0.4, 0.35],
        }
    )
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Feature",
        color_discrete_sequence=["#1abc9c", "#3498db", "#9b59b6", "#f39c12"],
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="Importance",
        yaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, 1])
    return fig


def build_emission_trend_chart():
    """Line chart for dummy GHG emission trends."""
    df = pd.DataFrame(
        {
            "Year": [2019, 2020, 2021, 2022, 2023],
            "North America": [18, 17, 20, 22, 23],
            "Asia": [12, 14, 15, 17, 18],
            "Europe": [10, 9, 9, 8, 8],
        }
    )
    long_df = df.melt(id_vars="Year", var_name="Region", value_name="GHG Emissions")
    fig = px.line(
        long_df,
        x="Year",
        y="GHG Emissions",
        color="Region",
        markers=True,
        color_discrete_sequence=["#1F618D", "#F39C12", "#27AE60"],
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        yaxis_title="GHG Emissions (mtCO2e)",
        xaxis_title=None,
        legend_title_text="Region",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(range=[5, 25])
    return fig

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
        if country in CENTRAL_ASIA_AND_CAUCAUSUS_COUNTRIES:
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

def build_world_map(data: pd.DataFrame):
    """Create a choropleth map showing the seven regions."""
    fig = px.choropleth(
        data_frame=data,
        locations="iso_alpha",
        color="region",
        hover_name="country",
        category_orders={"region": REGION_ORDER},
        color_discrete_map=REGION_COLOR_MAP,
        title="Global Regions Overview",
        projection="natural earth",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        legend_title_text="Region",
    )
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="#F5F5F5",
    )
    return fig

def create_app() -> Dash:
    """Initialize the Dash application."""
    data = build_region_dataframe()
    fig = build_world_map(data)
    feature_fig = build_feature_importance_chart()
    ghg_fig = build_emission_trend_chart()

    app = Dash(__name__)
    app.title = "Risk Regions Map"
    app.layout = html.Div(
        [
            html.H1(
                "Risk Management in Green Supply Chain Management",
                style={
                    "fontSize": "32px",
                    "marginBottom": "24px",
                    "letterSpacing": "0.03em",
                },
            ),
            html.Div(
                [
                    card_container(
                        [
                            dcc.Graph(
                                id="world-region-map",
                                figure=fig,
                                style={"height": "55vh"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                style={
                                                    "display": "inline-block",
                                                    "width": "12px",
                                                    "height": "12px",
                                                    "borderRadius": "50%",
                                                    "backgroundColor": color,
                                                    "marginRight": "8px",
                                                }
                                            ),
                                            html.Span(label),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    )
                                    for label, color in RISK_LEVELS
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "20px",
                                    "marginTop": "10px",
                                    "color": "#6c6f7d",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        {"padding": "18px"},
                    ),
                    card_container(
                        [
                            html.H2(
                                "Key Risk Metrics",
                                style={"fontSize": "20px", "marginBottom": "16px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P(
                                                label,
                                                style={
                                                    "fontSize": "13px",
                                                    "textTransform": "uppercase",
                                                    "letterSpacing": "0.05em",
                                                    "color": "#7a7d85",
                                                    "marginBottom": "6px",
                                                },
                                            ),
                                            html.Div(
                                                value,
                                                style={
                                                    "fontSize": "44px",
                                                    "fontWeight": "600",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "18px"},
                                    )
                                    for label, value in KEY_METRICS
                                ],
                            ),
                            html.H3(
                                "Top 10 High-Risk Suppliers",
                                style={"fontSize": "16px", "marginTop": "10px"},
                            ),
                            html.Ul(
                                [html.Li(name) for name in TOP_RISK_SUPPLIERS],
                                style={
                                    "listStyle": "none",
                                    "padding": 0,
                                    "margin": "12px 0 0 0",
                                    "lineHeight": "1.6",
                                },
                            ),
                        ],
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 2fr) minmax(0, 1fr)",
                    "gap": "24px",
                    "width": "100%",
                },
            ),
            html.Div(
                [
                    card_container(
                        [
                            html.H2(
                                "Predictive Model Insights",
                                style={"fontSize": "20px", "marginBottom": "12px"},
                            ),
                            dcc.Graph(
                                id="feature-importance-chart",
                                figure=feature_fig,
                                style={"height": "250px"},
                            ),
                            html.Div(
                                [
                                    html.P(
                                        "Model Performance",
                                        style={
                                            "fontWeight": "600",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        metric,
                                                        style={
                                                            "color": "#7a7d85",
                                                            "fontSize": "12px",
                                                            "letterSpacing": "0.08em",
                                                        },
                                                    ),
                                                    html.Strong(
                                                        value,
                                                        style={"fontSize": "20px"},
                                                    ),
                                                ],
                                                style={"display": "flex", "flexDirection": "column"},
                                            )
                                            for metric, value in MODEL_PERFORMANCE.items()
                                        ],
                                        style={
                                            "display": "flex",
                                            "gap": "24px",
                                        },
                                    ),
                                ],
                                style={"marginTop": "10px"},
                            ),
                        ],
                    ),
                    card_container(
                        [
                            html.H2(
                                "Sustainability Trend Analysis",
                                style={"fontSize": "20px", "marginBottom": "12px"},
                            ),
                            html.P(
                                "GHG Emission Trends",
                                style={
                                    "fontSize": "13px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.08em",
                                    "color": "#7a7d85",
                                    "marginBottom": "8px",
                                },
                            ),
                            dcc.Graph(
                                id="ghg-trend-chart",
                                figure=ghg_fig,
                                style={"height": "260px"},
                            ),
                        ],
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                    "gap": "24px",
                    "marginTop": "24px",
                    "width": "100%",
                },
            ),
            html.Div(
                [
                    card_container(
                        [
                            html.H2(
                                "Supplier & Commodity Details",
                                style={"fontSize": "20px", "marginBottom": "12px"},
                            ),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Supplier"),
                                                html.Th("Country"),
                                                html.Th("Commodity"),
                                                html.Th("Tier"),
                                                html.Th("Risk"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(supplier),
                                                    html.Td(country),
                                                    html.Td(commodity),
                                                    html.Td(tier),
                                                    html.Td(risk),
                                                ]
                                            )
                                            for (
                                                supplier,
                                                country,
                                                commodity,
                                                tier,
                                                risk,
                                            ) in SUPPLIER_DETAILS
                                        ]
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "borderCollapse": "collapse",
                                },
                            ),
                        ],
                    ),
                    card_container(
                        [
                            html.H2(
                                "Decision Support / Alerts",
                                style={"fontSize": "20px", "marginBottom": "12px"},
                            ),
                            html.Ul(
                                [html.Li(alert) for alert in CRITICAL_ALERTS],
                                style={
                                    "paddingLeft": "16px",
                                    "lineHeight": "1.8",
                                },
                            ),
                        ],
                    ),
                    card_container(
                        [
                            html.H2(
                                "Recommended Actions",
                                style={"fontSize": "20px", "marginBottom": "12px"},
                            ),
                            html.Ul(
                                [html.Li(action) for action in RECOMMENDED_ACTIONS],
                                style={
                                    "paddingLeft": "16px",
                                    "lineHeight": "1.8",
                                },
                            ),
                        ],
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)",
                    "gap": "24px",
                    "marginTop": "24px",
                    "width": "100%",
                },
            ),
        ],
        style={
            "fontFamily": "'Helvetica Neue', Arial, sans-serif",
            "padding": "40px 60px 80px",
            "backgroundColor": "#f4f6fb",
            "color": "#1c1f24",
            "minHeight": "100vh",
        },
    )
    return app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run(debug=True)