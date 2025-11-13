import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preprocessing import (
    preprocess_supplier_data,
    preprocess_commodity_data,
    preprocess_co2_data,
    preprocess_esg_data,
    aggregate_esg_by_industry,
    calculate_risk_metrics,
    display_preprocessing_summary
)
from data_integration import integrate_datasets, display_integration_summary

# =====================================================================
# CONFIGURATION
# =====================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    print(f"Random seed set to {seed} for reproducibility")


def load_raw_datasets():
    """Load all raw datasets from data/raw directory."""
    print("\n" + "="*70)
    print("LOADING RAW DATASETS")
    print("="*70 + "\n")
    
    try:
        supplier_data = pd.read_csv(f"{RAW_DATA_DIR}/synthetic_supplier_dataset_1.csv")
        print(f"✓ Loaded supplier data: {supplier_data.shape}")
        
        commodity_data = pd.read_csv(
            f"{RAW_DATA_DIR}/SupplyChainGHGEmissionFactors_v1.2_NAICS_byGHG_USD2021.csv"
        )
        print(f"✓ Loaded commodity/GHG data: {commodity_data.shape}")
        
        co2_data = pd.read_csv(f"{RAW_DATA_DIR}/owid-co2-data.csv")
        print(f"✓ Loaded CO2 data: {co2_data.shape}")
        
        esg_data = pd.read_csv(f"{RAW_DATA_DIR}/SP 500 ESG Risk Ratings.csv")
        print(f"✓ Loaded S&P 500 ESG data: {esg_data.shape}")
        
        return supplier_data, commodity_data, co2_data, esg_data
    
    except FileNotFoundError as e:
        print(f"Error: Could not find data file - {e}")
        sys.exit(1)


def save_processed_datasets(supplier_data, commodity_data, co2_data, esg_data, esg_industry):
    """Save preprocessed datasets to processed directory."""
    print("\n" + "="*70)
    print("SAVING PREPROCESSED DATASETS")
    print("="*70 + "\n")
    
    supplier_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv", index=False)
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv")
    
    commodity_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv", index=False)
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv")
    
    co2_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_co2_data.csv", index=False)
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/preprocessed_co2_data.csv")
    
    esg_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_esg_data.csv", index=False)
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/preprocessed_esg_data.csv")
    
    esg_industry.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_esg_by_industry.csv", index=False)
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/preprocessed_esg_by_industry.csv")


def save_integrated_dataset(integrated_data, enhanced_data):
    """Save integrated and enhanced datasets."""
    print("\n" + "="*70)
    print("SAVING INTEGRATED & ENHANCED DATASETS")
    print("="*70 + "\n")
    
    integrated_data.to_csv(
        f"{PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv",
        index=False
    )
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv")
    
    enhanced_data.to_csv(
        f"{PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv",
        index=False
    )
    print(f"✓ Saved: {PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv")


def main():
    """
    Main pipeline orchestrator.
    Runs preprocessing, integration, and risk metric calculation.
    Ensures reproducibility with fixed random seed.
    """
    
    # =====================================================================
    # STEP 0: SET RANDOM SEEDS
    # =====================================================================
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70 + "\n")
    set_random_seeds(RANDOM_SEED)
    
    # =====================================================================
    # STEP 1: LOAD RAW DATASETS
    # =====================================================================
    supplier_data, commodity_data, co2_data, esg_data = load_raw_datasets()
    
    # =====================================================================
    # STEP 2: PREPROCESS DATASETS
    # =====================================================================
    print("\n")
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    esg_data = preprocess_esg_data(esg_data)
    esg_industry = aggregate_esg_by_industry(esg_data)
    
    # Display preprocessing summary
    display_preprocessing_summary(supplier_data, commodity_data, co2_data, esg_industry)
    
    # Save preprocessed datasets
    save_processed_datasets(supplier_data, commodity_data, co2_data, esg_data, esg_industry)
    
    # =====================================================================
    # STEP 3: INTEGRATE DATASETS
    # =====================================================================
    integrated_data = integrate_datasets(supplier_data, commodity_data, co2_data, esg_industry)
    
    # =====================================================================
    # STEP 4: CALCULATE RISK METRICS
    # =====================================================================
    enhanced_data = calculate_risk_metrics(integrated_data)
    
    # =====================================================================
    # STEP 5: SAVE OUTPUTS
    # =====================================================================
    save_integrated_dataset(integrated_data, enhanced_data)
    
    # =====================================================================
    # STEP 6: DISPLAY FINAL SUMMARY
    # =====================================================================
    display_integration_summary(integrated_data, enhanced_data)
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    print("\n" + "="*70)
    print("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    print("📊 Output Files Generated:")
    print(f"  1. {PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv")
    print(f"  2. {PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv")
    print(f"  3. {PROCESSED_DATA_DIR}/preprocessed_co2_data.csv")
    print(f"  4. {PROCESSED_DATA_DIR}/preprocessed_esg_data.csv")
    print(f"  5. {PROCESSED_DATA_DIR}/preprocessed_esg_by_industry.csv")
    print(f"  6. {PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv")
    print(f"  7. {PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv")
    
    print(f"\n🔒 Reproducibility Guarantee:")
    print(f"   Random seed fixed to RANDOM_SEED={RANDOM_SEED}")
    print(f"   All results will be identical on every run.\n")
    
    return enhanced_data


if __name__ == "__main__":
    enhanced_data = main()


def save_processed_datasets(supplier_data, commodity_data, co2_data):
    """Save preprocessed datasets to processed directory."""
    print("\n" + "="*70)
    print("SAVING PREPROCESSED DATASETS")
    print("="*70 + "\n")
    
    supplier_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv", index=False)
    print(f"Saved: {PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv")
    
    commodity_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv", index=False)
    print(f"Saved: {PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv")
    
    co2_data.to_csv(f"{PROCESSED_DATA_DIR}/preprocessed_co2_data.csv", index=False)
    print(f"Saved: {PROCESSED_DATA_DIR}/preprocessed_co2_data.csv")


def save_integrated_dataset(integrated_data, enhanced_data):
    """Save integrated and enhanced datasets."""
    print("\n" + "="*70)
    print("SAVING INTEGRATED & ENHANCED DATASETS")
    print("="*70 + "\n")
    
    integrated_data.to_csv(
        f"{PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv",
        index=False
    )
    print(f"Saved: {PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv")
    
    enhanced_data.to_csv(
        f"{PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv",
        index=False
    )
    print(f"Saved: {PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv")


def main():
    """
    Main pipeline orchestrator.
    Runs preprocessing, integration, and risk metric calculation.
    """
    
    # print("\n")
    # print("=" * 70)
    # print("=" + " " * 68 + "=")
    # print("=" + "  GREEN SUPPLY CHAIN RISK MANAGEMENT - FINAL YEAR PROJECT".center(68) + "█")
    # print("█" + " " * 68 + "█")
    # print("█" * 70)
    
    # =====================================================================
    # STEP 0: SET RANDOM SEEDS
    # =====================================================================
    print("\n" + "="*70)
    print("INITIALIZATION")
    print("="*70 + "\n")
    set_random_seeds(RANDOM_SEED)
    
    # =====================================================================
    # STEP 1: LOAD RAW DATASETS
    # =====================================================================
    supplier_data, commodity_data, co2_data, esg_data = load_raw_datasets()
    
    # =====================================================================
    # STEP 2: PREPROCESS DATASETS
    # =====================================================================
    print("\n")
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    
    # Display preprocessing summary
    display_preprocessing_summary(supplier_data, commodity_data, co2_data)
    
    # Save preprocessed datasets
    save_processed_datasets(supplier_data, commodity_data, co2_data)
    
    # =====================================================================
    # STEP 3: INTEGRATE DATASETS
    # =====================================================================
    integrated_data = integrate_datasets(supplier_data, commodity_data, co2_data)
    
    # =====================================================================
    # STEP 4: CALCULATE RISK METRICS
    # =====================================================================
    enhanced_data = calculate_risk_metrics(integrated_data)
    
    # =====================================================================
    # STEP 5: SAVE OUTPUTS
    # =====================================================================
    save_integrated_dataset(integrated_data, enhanced_data)
    
    # =====================================================================
    # STEP 6: DISPLAY FINAL SUMMARY
    # =====================================================================
    display_integration_summary(integrated_data, enhanced_data)
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    print("📊 Output Files Generated:")
    print(f"  1. {PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv")
    print(f"  2. {PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv")
    print(f"  3. {PROCESSED_DATA_DIR}/preprocessed_co2_data.csv")
    print(f"  4. {PROCESSED_DATA_DIR}/integrated_commodity_dataset.csv")
    print(f"  5. {PROCESSED_DATA_DIR}/integrated_dataset_with_risk_metrics.csv")
    
    print("\nNote: All random seeds fixed to RANDOM_SEED=" + str(RANDOM_SEED))
    print("   This ensures reproducible results on every run.\n")
    
    return enhanced_data


if __name__ == "__main__":
    enhanced_data = main()
