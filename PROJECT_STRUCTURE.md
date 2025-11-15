# Project Structure Documentation

## Overview

This document provides a detailed overview of the project structure and the purpose of each component.

## Directory Structure

```
Final-Year-Project-2025/
│
├── data/                              # Data directory
│   ├── raw/                          # Raw, unprocessed data files
│   │   ├── synthetic_supplier_dataset_1.csv
│   │   ├── SupplyChainGHGEmissionFactors_v1.2_NAICS_byGHG_USD2021.csv
│   │   ├── owid-co2-data.csv
│   │   └── SP 500 ESG Risk Ratings.csv
│   └── processed/                    # Processed and cleaned data files
│       ├── preprocessed_supplier_data.csv
│       ├── preprocessed_commodity_data.csv
│       ├── preprocessed_co2_data.csv
│       ├── preprocessed_esg_data.csv
│       ├── preprocessed_esg_by_industry.csv
│       ├── integrated_commodity_dataset.csv
│       └── integrated_dataset_with_risk_metrics.csv
│
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── data_preprocessing.py         # Data cleaning and preprocessing functions
│   ├── data_integration.py           # Dataset integration and merging
│   ├── analysis.py                   # Statistical analysis functions
│   ├── visualization.py              # Data visualization functions
│   ├── modeling.py                   # Machine learning model functions
│   ├── evaluation.py                 # Model evaluation functions
│   ├── reporting.py                  # Report generation functions
│   └── utils.py                      # Utility helper functions
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_data_preprocessing.py    # Tests for preprocessing module
│   └── test_data_integration.py      # Tests for integration module
│
├── notebooks/                        # Jupyter notebooks
│   └── eda.ipynb                     # Exploratory Data Analysis notebook
│
├── outputs/                          # Generated outputs (created at runtime)
│   ├── visualizations/               # Generated plots and charts
│   ├── reports/                      # Generated reports
│   └── models/                       # Saved ML models
│
├── config.py                         # Configuration file with all settings
├── main.py                           # Main pipeline script (basic)
├── run_full_analysis.py              # Complete analysis pipeline script
├── requirements.txt                  # Python package dependencies
├── README.md                         # Main project documentation
└── PROJECT_STRUCTURE.md              # This file
```

## Module Descriptions

### Core Modules

#### `src/data_preprocessing.py`
- **Purpose**: Data cleaning and preprocessing
- **Key Functions**:
  - `preprocess_supplier_data()`: Clean and standardize supplier data
  - `preprocess_commodity_data()`: Process GHG emission factors
  - `preprocess_co2_data()`: Clean CO2 emissions data
  - `preprocess_esg_data()`: Process ESG ratings
  - `calculate_risk_metrics()`: Calculate risk scores
- **Output**: Cleaned datasets ready for integration

#### `src/data_integration.py`
- **Purpose**: Integrate multiple data sources
- **Key Functions**:
  - `integrate_datasets()`: Merge supplier, commodity, CO2, and ESG data
  - `display_integration_summary()`: Show integration statistics
- **Output**: Integrated dataset with all features

#### `src/analysis.py`
- **Purpose**: Statistical analysis
- **Key Functions**:
  - `calculate_descriptive_statistics()`: Summary statistics
  - `calculate_correlation_matrix()`: Correlation analysis
  - `calculate_feature_importance()`: Feature importance analysis
  - `perform_statistical_tests()`: Hypothesis testing
  - `perform_pca_analysis()`: Principal Component Analysis
- **Output**: Statistical analysis results

#### `src/visualization.py`
- **Purpose**: Create visualizations
- **Key Functions**:
  - `create_risk_distribution_plot()`: Risk score distributions
  - `create_correlation_heatmap()`: Correlation visualizations
  - `create_feature_importance_plot()`: Feature importance charts
  - `create_comprehensive_dashboard()`: Full visualization suite
- **Output**: PNG image files

#### `src/modeling.py`
- **Purpose**: Machine learning models
- **Key Functions**:
  - `train_risk_prediction_model()`: Regression models for risk prediction
  - `train_risk_classification_model()`: Classification models
  - `perform_clustering()`: Supplier clustering
  - `perform_cross_validation()`: Model validation
- **Output**: Trained models and predictions

#### `src/evaluation.py`
- **Purpose**: Model evaluation
- **Key Functions**:
  - `evaluate_regression_model()`: Regression metrics
  - `evaluate_classification_model()`: Classification metrics
  - `plot_prediction_vs_actual()`: Prediction visualization
  - `generate_evaluation_report()`: Evaluation reports
- **Output**: Evaluation metrics and plots

#### `src/reporting.py`
- **Purpose**: Generate reports
- **Key Functions**:
  - `generate_executive_summary()`: Executive summary report
  - `generate_supplier_risk_report()`: Supplier-specific reports
  - `generate_comprehensive_report()`: Full analysis report
  - `export_results_to_json()`: JSON export
- **Output**: Text reports and JSON files

#### `src/utils.py`
- **Purpose**: Utility functions
- **Key Functions**:
  - File I/O helpers
  - Data validation
  - Formatting utilities
  - Progress tracking
- **Output**: Helper functions for other modules

### Configuration

#### `config.py`
- **Purpose**: Centralized configuration
- **Contains**:
  - Data paths
  - Model parameters
  - Risk calculation weights
  - Visualization settings
  - Analysis parameters

### Main Scripts

#### `main.py`
- **Purpose**: Basic data processing pipeline
- **Functionality**:
  - Load raw data
  - Preprocess datasets
  - Integrate data
  - Calculate risk metrics
  - Save processed outputs

#### `run_full_analysis.py`
- **Purpose**: Complete analysis pipeline
- **Functionality**:
  - All steps from `main.py`
  - Statistical analysis
  - Machine learning modeling
  - Visualization generation
  - Report generation
  - Model evaluation

## Data Flow

1. **Raw Data** → `data/raw/`
2. **Preprocessing** → `src/data_preprocessing.py`
3. **Processed Data** → `data/processed/`
4. **Integration** → `src/data_integration.py`
5. **Analysis** → `src/analysis.py`
6. **Modeling** → `src/modeling.py`
7. **Evaluation** → `src/evaluation.py`
8. **Visualization** → `src/visualization.py` → `outputs/visualizations/`
9. **Reporting** → `src/reporting.py` → `outputs/reports/`

## Usage Workflow

### Basic Pipeline
```bash
python main.py
```

### Full Analysis
```bash
python run_full_analysis.py
```

### Individual Modules
```python
from src.analysis import calculate_correlation_matrix
correlation = calculate_correlation_matrix(data)
```

## Output Files

### Processed Data
- All files in `data/processed/` are CSV format
- Ready for analysis and modeling

### Visualizations
- All files in `outputs/visualizations/` are PNG format
- High resolution (300 DPI) suitable for reports

### Reports
- All files in `outputs/reports/` are text format
- Human-readable summaries and analysis results

### Models
- Saved models in `outputs/models/` (if implemented)
- Can be loaded for predictions

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Dependencies

All dependencies are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

## Notes

- All random seeds are set to 42 for reproducibility
- Output directories are created automatically
- Data validation is performed at each step
- Error handling is implemented throughout

