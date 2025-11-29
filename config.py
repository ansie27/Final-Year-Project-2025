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
RAW_DATA_PATH = RAW_DATA_DIR / "syn_supplier_commodity_dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "preprocessed_supplier_commodity_dataset.csv"
OVERSAMPLED_DATA_PATH = PROCESSED_DATA_DIR / "oversampled_preprocessed_supplier_commodity_data.csv"
ENGINEERED_DATA_PATH = PROCESSED_DATA_DIR / "engineered_supplier_commodity_features.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

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
    'Year',
    'Month',
    'Country',
    'Region',
    'Industry_Sector',
    'Supplier_Tier',
    'ESG_Score',
    'Environmental_Score',
    'Social_Score',
    'Governance_Score',
    'Carbon_Emission_Intensity',
    'GHG_Scope1_Intensity',
    'GHG_Scope2_Intensity',
    'GHG_Scope3_Intensity',
    'Water_Intensity',
    'Waste_Management_Efficiency',
    'Renewable_Energy_Usage',
    'Compliance_Level',
    'Sustainability_Report_Availability',
    'Certifications_Active',
    'Lead_Time_Days',
    'Production_Capacity',
    'On_Time_Delivery_Rate',
    'Defect_Rate',
    'Incident_History_Count',
    'Labour_Compliance_Score',
    'Diversity_Index',
    'Financial_Stability_Score',
    'Freight_Cost',
    'Cost_Index',
    'Commodity_Price_Index',
    'Commodity_Demand',
    'Trade_Volume',
    'Logistics_Distance_km',
]

EXCLUDE_COLUMNS = [
    'SC_ID',
    'Supplier_Name',
    'Commodity_Name',
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

# =====================================================================
# SYNTHETIC DATA GENERATION PARAMETERS
# =====================================================================
SYNTHETIC_DATA_CONFIG = {
    'enabled': True,  # Enable/disable synthetic data generation in pipeline
    'comparison_epochs': 300,  # Epochs for model comparison
    'generation_epochs': 300,  # Epochs for final generation
    'supplier_target_col': 'Overall_Risk_Score',  # Target for ML utility evaluation
    'commodity_target_col': None,  # Target for commodity ML utility (if applicable)
    'save_models': True,  # Whether to save trained models
    'save_synthetic_data': True,  # Whether to save generated synthetic data
    'use_cached_selection': True,  # Use cached model selection if available
}

# Synthetic data output paths
SYNTHETIC_DATA_DIR = OUTPUT_DIR / "synthetic_data"
SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_DATA_FILES = {
    'supplier': SYNTHETIC_DATA_DIR / "synthetic_supplier_data.csv",
    'commodity': SYNTHETIC_DATA_DIR / "synthetic_commodity_data.csv",
}

SYNTHETIC_MODEL_DIR = MODELS_DIR / "synthetic_generators"
SYNTHETIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

SYNTHETIC_MODEL_FILES = {
    'supplier': SYNTHETIC_MODEL_DIR / "supplier_generator",
    'commodity': SYNTHETIC_MODEL_DIR / "commodity_generator",
}

# Model selection results path
MODEL_SELECTION_DIR = OUTPUT_DIR / "synthetic_data_generation"
MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

