"""
Configuration file for Green Supply Chain Risk Management Project

This file contains all configuration parameters, paths, and settings
for the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# =====================================================================
# DATA PATHS
# =====================================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =====================================================================
# DATA FILES
# =====================================================================
RAW_DATA_FILES = {
    'supplier': RAW_DATA_DIR / "synthetic_supplier_dataset_1.csv",
    'commodity': RAW_DATA_DIR / "SupplyChainGHGEmissionFactors_v1.2_NAICS_byGHG_USD2021.csv",
    'co2': RAW_DATA_DIR / "owid-co2-data.csv",
    'esg': RAW_DATA_DIR / "SP 500 ESG Risk Ratings.csv"
}

PROCESSED_DATA_FILES = {
    'supplier': PROCESSED_DATA_DIR / "preprocessed_supplier_data.csv",
    'commodity': PROCESSED_DATA_DIR / "preprocessed_commodity_data.csv",
    'co2': PROCESSED_DATA_DIR / "preprocessed_co2_data.csv",
    'esg': PROCESSED_DATA_DIR / "preprocessed_esg_data.csv",
    'esg_industry': PROCESSED_DATA_DIR / "preprocessed_esg_by_industry.csv",
    'integrated': PROCESSED_DATA_DIR / "integrated_commodity_dataset.csv",
    'enhanced': PROCESSED_DATA_DIR / "integrated_dataset_with_risk_metrics.csv"
}

# =====================================================================
# REPRODUCIBILITY
# =====================================================================
RANDOM_SEED = 42

# =====================================================================
# RISK CALCULATION PARAMETERS
# =====================================================================
RISK_WEIGHTS = {
    'environmental': 0.35,
    'compliance': 0.30,
    'operational': 0.20,
    'financial': 0.15
}

RISK_THRESHOLDS = {
    'low': 30,
    'medium': 60,
    'high': 80,
    'critical': 100
}

# =====================================================================
# MODEL PARAMETERS
# =====================================================================
MODEL_CONFIG = {
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': RANDOM_SEED,
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# =====================================================================
# VISUALIZATION PARAMETERS
# =====================================================================
VISUALIZATION_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 300,
    'style': 'whitegrid',
    'font_size': 10,
    'color_palette': 'Set2'
}

# =====================================================================
# ANALYSIS PARAMETERS
# =====================================================================
ANALYSIS_CONFIG = {
    'correlation_threshold': 0.7,
    'top_n_features': 15,
    'pca_components': None,  # None means use all components
    'clustering_n_clusters': 4
}

# =====================================================================
# REPORTING PARAMETERS
# =====================================================================
REPORTING_CONFIG = {
    'include_plots': True,
    'include_statistics': True,
    'include_model_results': True,
    'export_formats': ['txt', 'csv', 'json']
}

# =====================================================================
# LOGGING
# =====================================================================
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "project.log"

# =====================================================================
# FEATURE COLUMNS
# =====================================================================
RISK_COLUMNS = [
    'Environmental_Risk_Score',
    'Compliance_Risk_Score',
    'Operational_Risk_Score',
    'Financial_Risk_Score',
    'Overall_Risk_Score'
]

FEATURE_COLUMNS = [
    'Environmental_Score',
    'ESG_Score',
    'Social_Score',
    'Governance_Score',
    'Carbon_Emission_Intensity',
    'Renewable_Energy_Usage',
    'Waste_Management_Efficiency',
    'Compliance_Level',
    'Sustainability_Report_Availability',
    'On_Time_Delivery_Rate',
    'Defect_Rate',
    'Financial_Stability_Score',
    'Labour_Compliance_Score',
    'Supplier_Audit_Score',
    'Incident_History_Count'
]

EXCLUDE_COLUMNS = [
    'Supplier_ID',
    'Supplier_Name',
    'Risk_Classification'
]

# =====================================================================
# FUZZY AHP-TOPSIS-GA PARAMETERS
# =====================================================================
FUZZY_AHP_TOPSIS_GA_CONFIG = {
    'use_ga_optimization': True,
    'population_size': 50,
    'generations': 100,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'top_n_suppliers': 10
}

# Criteria for Fuzzy AHP-TOPSIS analysis
TOPSIS_CRITERIA_COLUMNS = [
    'Environmental_Score',
    'ESG_Score',
    'On_Time_Delivery_Rate',
    'Financial_Stability_Score',
    'Compliance_Level',
    'Renewable_Energy_Usage'
]

# Criteria types: 'benefit' (higher is better) or 'cost' (lower is better)
TOPSIS_CRITERIA_TYPES = [
    'benefit',  # Environmental_Score
    'benefit',  # ESG_Score
    'benefit',  # On_Time_Delivery_Rate
    'benefit',  # Financial_Stability_Score
    'benefit',  # Compliance_Level
    'benefit'   # Renewable_Energy_Usage
]

