import pandas as pd
import numpy as np

# =====================================================================
# DATA INTEGRATION FOR GREEN SUPPLY CHAIN RISK MANAGEMENT
# =====================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def integrate_supplier_dataset(supplier_data, commodity_data, co2_data, esg_industry=None):
    """
    Integrates supplier, commodity (GHG), CO2, and ESG datasets.
    
    Parameters:
    -----------
    supplier_data : pd.DataFrame
        Preprocessed supplier dataset
    commodity_data : pd.DataFrame
        GHG emission factors dataset
    co2_data : pd.DataFrame
        Country-level CO2 emissions data
    esg_industry : pd.DataFrame, optional
        ESG metrics aggregated by industry
        
    Returns:
    --------
    pd.DataFrame
        Integrated dataset with all sources merged
    """
    
    print("="*70)
    print("DATA INTEGRATION PROCESS")
    print("="*70)
    
    # =====================================================================
    # STEP 1: AGGREGATE GHG FACTORS BY INDUSTRY & GHG TYPE
    # =====================================================================
    print("\n[1/4] Aggregating GHG emission factors by industry...")
    
    # For each industry and GHG type, get the supply chain emission factor with margins
    ghg_factors = commodity_data.groupby(['Industry_Sector', 'GHG']).agg({
        'Supply Chain Emission Factors with Margins': 'mean'
    }).reset_index()
    
    ghg_factors.rename(columns={
        'Supply Chain Emission Factors with Margins': 'Avg_Supply_Chain_Emission_Factor'
    }, inplace=True)
    
    # Create a pivot table for easier merging: Industry x GHG
    ghg_pivot = ghg_factors.pivot_table(
        index='Industry_Sector',
        columns='GHG',
        values='Avg_Supply_Chain_Emission_Factor',
        aggfunc='first'
    )
    ghg_pivot.reset_index(inplace=True)
    ghg_pivot.columns.name = None
    
    print(f"   ✓ Aggregated {len(ghg_pivot)} industries with GHG emission factors")
    
    # =====================================================================
    # STEP 2: MERGE SUPPLIER DATA WITH GHG FACTORS
    # =====================================================================
    print("\n[2/4] Merging supplier data with GHG emission factors...")
    
    supplier_ghg = pd.merge(
        supplier_data,
        ghg_pivot,
        on='Industry_Sector',
        how='left'
    )
    
    print(f"   ✓ Merged {len(supplier_ghg)} supplier records with GHG factors")
    
    # =====================================================================
    # STEP 3: MERGE WITH ESG DATA (if provided)
    # =====================================================================
    if esg_industry is not None:
        print("\n[3/4] Merging with S&P 500 ESG metrics by industry...")
        
        supplier_ghg_esg = pd.merge(
            supplier_ghg,
            esg_industry,
            on='Industry_Sector',
            how='left'
        )
        
        print(f"   ✓ Merged ESG metrics for {esg_industry['Industry_Sector'].nunique()} industries")
        supplier_ghg = supplier_ghg_esg
        step_number = 4
    else:
        step_number = 3
    
    # =====================================================================
    # STEP 4(3): MERGE WITH COUNTRY-LEVEL CO2 DATA (LATEST YEAR)
    # =====================================================================
    print(f"\n[{step_number}/4] Merging with country-level CO2 data...")
    
    # Get the most recent year's CO2 data for each country
    co2_latest = co2_data.sort_values('year').drop_duplicates(
        subset=['Country'],
        keep='last'
    )[['Country', 'year', 'co2', 'gdp', 'population']]
    
    co2_latest.rename(columns={
        'year': 'CO2_Data_Year',
        'co2': 'Country_Total_CO2_Emissions',
        'gdp': 'Country_GDP',
        'population': 'Country_Population'
    }, inplace=True)
    
    integrated_data = pd.merge(
        supplier_ghg,
        co2_latest,
        on='Country',
        how='left'
    )
    
    print(f"   ✓ Merged CO2 data for {co2_latest['Country'].nunique()} countries")
    
    print("\n" + "="*70)
    print(f"INTEGRATION COMPLETE")
    print(f"Output shape: {integrated_data.shape} (rows × columns)")
    print("="*70)
    
    return integrated_data


def display_integration_summary(integrated_data, enhanced_data):
    """
    Display summary of integrated and enhanced datasets.
    
    Parameters:
    -----------
    integrated_data : pd.DataFrame
        Integrated dataset without risk metrics
    enhanced_data : pd.DataFrame
        Integrated dataset with calculated risk metrics
    """
    
    print("\n" + "="*70)
    print("INTEGRATED DATASET SUMMARY")
    print("="*70)
    
    print(f"\nDataset Shape: {integrated_data.shape}")
    print(f"Total Suppliers: {integrated_data['Supplier_ID'].nunique()}")
    print(f"Total Industries: {integrated_data['Industry_Sector'].nunique()}")
    print(f"Countries Represented: {integrated_data['Country'].nunique()}")
    
    print("\n--- Risk Metrics Available ---")
    risk_cols = ['Environmental_Risk_Score', 'Compliance_Risk_Score', 
                 'Operational_Risk_Score', 'Financial_Risk_Score', 
                 'Overall_Risk_Score', 'Risk_Classification']
    for col in risk_cols:
        if col in enhanced_data.columns:
            print(f"  ✓ {col}")
    
    print("\n--- Risk Classification Distribution ---")
    print(enhanced_data['Risk_Classification'].value_counts().sort_index())
    
    print("\n--- Overall Risk Score Statistics ---")
    print(enhanced_data['Overall_Risk_Score'].describe())
    
    print("\n--- Top 5 Highest Risk Suppliers ---")
    top_risk = enhanced_data.nlargest(5, 'Overall_Risk_Score')[
        ['Supplier_ID', 'Supplier_Name', 'Country', 'Overall_Risk_Score', 'Risk_Classification']
    ]
    print(top_risk.to_string(index=False))
    
    print("\n--- Top 5 Lowest Risk Suppliers ---")
    low_risk = enhanced_data.nsmallest(5, 'Overall_Risk_Score')[
        ['Supplier_ID', 'Supplier_Name', 'Country', 'Overall_Risk_Score', 'Risk_Classification']
    ]
    print(low_risk.to_string(index=False))
    
    print("\n--- Data Quality Report ---")
    missing = enhanced_data.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"Missing Values Per Column:")
        for col, count in missing.items():
            pct = (count / len(enhanced_data)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("✓ No missing values detected!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Load datasets
    print("Loading datasets...\n")
    supplier_data = pd.read_csv("data/raw/synthetic_supplier_dataset_1.csv")
    commodity_data = pd.read_csv("data/raw/SupplyChainGHGEmissionFactors_v1.2_NAICS_byGHG_USD2021.csv")
    co2_data = pd.read_csv("data/raw/owid-co2-data.csv")
    esg_industry = pd.read_csv("data/processed/preprocessed_esg_by_industry.csv")
    
    # Integrate datasets
    integrated_data = integrate_supplier_dataset(supplier_data, commodity_data, co2_data, esg_industry)
    
    # Save integrated dataset
    print("\nSaving integrated dataset...")
    integrated_data.to_csv("data/processed/integrated_commodity_dataset.csv", index=False)
    print("Saved to: data/processed/integrated_commodity_dataset.csv")