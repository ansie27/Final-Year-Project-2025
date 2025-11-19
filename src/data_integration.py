import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path as PathlibPath

# Parent directory
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
import config

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --------------------------------------------------
# COMMODITY DATASET INTEGRATION
# --------------------------------------------------

def integrate_commodity_datasets(
    ghg_emission_data: pd.DataFrame,
    co2_data: pd.DataFrame
) -> pd.DataFrame:    
    print("\n" + "="*70)
    print("INTEGRATING COMMODITY DATASETS")
    print("="*70)
    
    commodity_integrated = ghg_emission_data.copy()
    
    print(f"\n[1] Initial GHG Emission Data Shape: {commodity_integrated.shape}")
    print(f"    Columns: {list(commodity_integrated.columns)}")
    
    # Standardise column name for NAICS Title
    if '2017 NAICS Title' in commodity_integrated.columns:
        commodity_integrated.rename(columns={
            '2017 NAICS Title': 'Industry_Sector'
        }, inplace=True)
        print("    ✓ Renamed '2017 NAICS Title' to 'Industry_Sector'")
    
    # Prep CO2 data for integration
    print(f"\n[2] Preparing CO2 Data...")
    print(f"    Initial CO2 Data Shape: {co2_data.shape}")
    
    # Get latest year data for each country
    co2_latest = co2_data.sort_values('year', ascending=False).drop_duplicates(
        subset=['country'], keep='first'
    ).copy()
    
    # Standardise country names
    co2_latest['Country'] = co2_latest['country'].str.strip().str.title()
    
    # Select relevant CO2 columns for integration
    co2_columns = ['Country', 'year', 'co2', 'gdp', 'population', 
                   'co2_per_capita', 'co2_per_gdp', 'co2_growth_prct']
    
    # Only include columns that exist
    available_co2_cols = [col for col in co2_columns if col in co2_latest.columns]
    co2_summary = co2_latest[available_co2_cols].copy()
    co2_summary.rename(columns={
        'year': 'CO2_Data_Year',
        'co2': 'Country_Total_CO2_Emissions',
        'gdp': 'Country_GDP',
        'population': 'Country_Population',
        'co2_per_capita': 'Country_CO2_Per_Capita',
        'co2_per_gdp': 'Country_CO2_Per_GDP',
        'co2_growth_prct': 'Country_CO2_Growth_Percent'
    }, inplace=True)
    
    print(f"    Prepared CO2 summary with {len(co2_summary)} countries")
    print(f"    CO2 columns: {list(co2_summary.columns)}")
    
    # Create a cross-product approach: each GHG emission factor gets enriched
    # with country-level CO2 data (since GHG data is industry-based, not country-based)
    
    print(f"\n[3] Creating Integrated Commodity Dataset...")
    
    # Get unique industries and countries
    unique_industries = commodity_integrated['Industry_Sector'].unique()
    unique_countries = co2_summary['Country'].unique()
    
    # Create a cross-product: each industry-GHG combination with each country
    # For country-specific analysis while maintaining industry GHG factors
    industry_ghg_combinations = commodity_integrated[
        ['Industry_Sector', 'GHG', 'Supply Chain Emission Factors with Margins',
         'Supply Chain Emission Factors without Margins', 'Unit']
    ].drop_duplicates()
    
    # Create country-industry combinations
    # Associate each industry with all countries that have CO2 data
    # This creates a comprehensive dataset for analysis
    integrated_list = []
    
    for _, industry_row in industry_ghg_combinations.iterrows():
        for _, country_row in co2_summary.iterrows():
            integrated_row = {
                'Industry_Sector': industry_row['Industry_Sector'],
                'GHG': industry_row['GHG'],
                'Supply_Chain_Emission_Factor_With_Margins': industry_row['Supply Chain Emission Factors with Margins'],
                'Supply_Chain_Emission_Factor_Without_Margins': industry_row['Supply Chain Emission Factors without Margins'],
                'Emission_Factor_Unit': industry_row['Unit'],
                'Country': country_row['Country'],
                'CO2_Data_Year': country_row['CO2_Data_Year'],
                'Country_Total_CO2_Emissions': country_row['Country_Total_CO2_Emissions'],
                'Country_GDP': country_row['Country_GDP'],
                'Country_Population': country_row['Country_Population']
            }
            
            # Add optional CO2 columns if available
            if 'Country_CO2_Per_Capita' in country_row:
                integrated_row['Country_CO2_Per_Capita'] = country_row['Country_CO2_Per_Capita']
            if 'Country_CO2_Per_GDP' in country_row:
                integrated_row['Country_CO2_Per_GDP'] = country_row['Country_CO2_Per_GDP']
            if 'Country_CO2_Growth_Percent' in country_row:
                integrated_row['Country_CO2_Growth_Percent'] = country_row['Country_CO2_Growth_Percent']
            
            integrated_list.append(integrated_row)
    
    commodity_integrated = pd.DataFrame(integrated_list)
    
    print(f"    Created integrated dataset with {len(commodity_integrated)} records")
    print(f"    Unique Industries: {commodity_integrated['Industry_Sector'].nunique()}")
    print(f"    Unique Countries: {commodity_integrated['Country'].nunique()}")
    print(f"    Unique GHG Types: {commodity_integrated['GHG'].nunique()}")
    
    # Calculate derived metrics
    print(f"\n[4] Calculating Derived Metrics...")
    
    # Calculate emission intensity per country (combining industry factors with country context)
    commodity_integrated['Emission_Intensity_Score'] = (
        commodity_integrated['Supply_Chain_Emission_Factor_With_Margins'] * 
        (commodity_integrated['Country_Total_CO2_Emissions'].fillna(0) / 
         commodity_integrated['Country_Population'].fillna(1))
    )
    
    # Normalise emission intensity
    if commodity_integrated['Emission_Intensity_Score'].max() > 0:
        commodity_integrated['Normalized_Emission_Intensity'] = (
            (commodity_integrated['Emission_Intensity_Score'] - 
             commodity_integrated['Emission_Intensity_Score'].min()) /
            (commodity_integrated['Emission_Intensity_Score'].max() - 
             commodity_integrated['Emission_Intensity_Score'].min())
        ) * 100
    else:
        commodity_integrated['Normalized_Emission_Intensity'] = 0
    
    print(f"    Calculated Emission_Intensity_Score")
    print(f"    Calculated Normalized_Emission_Intensity")
    
    print(f"\n[5] Final Integrated Commodity Dataset Shape: {commodity_integrated.shape}")
    print(f"    Columns: {list(commodity_integrated.columns)}")
    print("="*70)
    
    return commodity_integrated

# --------------------------------------------------
# SUPPLIER DATASET INTEGRATION
# --------------------------------------------------

def integrate_supplier_datasets(
    supplier_data: pd.DataFrame,
    esg_data: pd.DataFrame
) -> pd.DataFrame:
    
    print("\n" + "="*70)
    print("INTEGRATING SUPPLIER DATASETS")
    print("="*70)
    
    supplier_integrated = supplier_data.copy()
    
    print(f"\n[1] Initial Supplier Data Shape: {supplier_integrated.shape}")
    print(f"    Columns: {list(supplier_integrated.columns)}")
    
    # Prepare ESG data for integration
    print(f"\n[2] Preparing ESG Data...")
    print(f"    Initial ESG Data Shape: {esg_data.shape}")
    print(f"    ESG Columns: {list(esg_data.columns)}")
    
    # Standardize industry names in ESG data
    if 'Industry' in esg_data.columns:
        esg_data = esg_data.copy()
        esg_data['Industry_Sector'] = esg_data['Industry'].str.strip().str.title()
        print("    ✓ Created 'Industry_Sector' from 'Industry' column")
    
    # Aggregate ESG data by industry (since multiple companies can be in same industry)
    print(f"\n[3] Aggregating ESG Data by Industry...")
    
    # Select relevant ESG columns
    esg_columns = ['Industry_Sector', 'Total ESG Risk score', 
                   'Environment Risk Score', 'Governance Risk Score', 
                   'Social Risk Score', 'Controversy Score']
    
    # Only include columns that exist and have data
    available_esg_cols = [col for col in esg_columns if col in esg_data.columns]
    
    # Remove rows with missing key ESG scores
    esg_clean = esg_data[available_esg_cols].dropna(
        subset=['Total ESG Risk score']
    ).copy()
    
    # Aggregate by industry
    esg_aggregated = esg_clean.groupby('Industry_Sector').agg({
        'Total ESG Risk score': ['mean', 'std', 'count'],
        'Environment Risk Score': 'mean',
        'Governance Risk Score': 'mean',
        'Social Risk Score': 'mean',
        'Controversy Score': 'mean'
    }).reset_index()
    
    # Flatten
    esg_aggregated.columns = [
        'Industry_Sector',
        'Avg_ESG_Risk_Score',
        'Std_ESG_Risk_Score',
        'Company_Count',
        'Avg_Environment_Risk',
        'Avg_Governance_Risk',
        'Avg_Social_Risk',
        'Avg_Controversy_Score'
    ]
    
    print(f"    Aggregated ESG data for {len(esg_aggregated)} industries")
    print(f"    Total companies represented: {esg_aggregated['Company_Count'].sum():.0f}")
    
    # Merge ESG data with supplier data on Industry_Sector
    print(f"\n[4] Merging ESG Data with Supplier Data...")
    
    # Ensure Industry_Sector is string type in both datasets
    supplier_integrated['Industry_Sector'] = supplier_integrated['Industry_Sector'].astype(str)
    esg_aggregated['Industry_Sector'] = esg_aggregated['Industry_Sector'].astype(str)
    
    # Merge
    supplier_integrated = pd.merge(
        supplier_integrated,
        esg_aggregated,
        on='Industry_Sector',
        how='left'
    )
    
    print(f"    Merged ESG data with supplier data")
    print(f"    Matched industries: {supplier_integrated['Avg_ESG_Risk_Score'].notna().sum()} / {len(supplier_integrated)}")
    
    # Fill missing ESG values with industry median or overall median
    print(f"\n[5] Handling Missing ESG Values...")
    
    esg_cols_to_fill = [
        'Avg_ESG_Risk_Score', 'Avg_Environment_Risk', 
        'Avg_Governance_Risk', 'Avg_Social_Risk', 'Avg_Controversy_Score'
    ]
    
    for col in esg_cols_to_fill:
        if col in supplier_integrated.columns:
            missing_count = supplier_integrated[col].isna().sum()
            if missing_count > 0:
                # Fill with overall median
                median_val = supplier_integrated[col].median()
                supplier_integrated[col].fillna(median_val, inplace=True)
                print(f"    ✓ Filled {missing_count} missing values in {col} with median: {median_val:.2f}")
    
    # Calculate derived ESG metrics
    print(f"\n[6] Calculating Derived ESG Metrics...")
    
    # ESG Compliance Score (inverse of risk - higher is better)
    supplier_integrated['ESG_Compliance_Score'] = (
        100 - supplier_integrated['Avg_ESG_Risk_Score'].fillna(50)
    )
    
    # ESG Risk-Adjusted Score (combining supplier ESG with industry ESG)
    if 'ESG_Score' in supplier_integrated.columns:
        supplier_integrated['ESG_Risk_Adjusted_Score'] = (
            supplier_integrated['ESG_Score'].fillna(50) * 0.6 +
            supplier_integrated['ESG_Compliance_Score'] * 0.4
        )
        print(f"    ✓ Calculated ESG_Risk_Adjusted_Score")
    
    print(f"    ✓ Calculated ESG_Compliance_Score")
    
    print(f"\n[7] Final Integrated Supplier Dataset Shape: {supplier_integrated.shape}")
    print(f"    Columns: {list(supplier_integrated.columns)}")
    print("="*70)
    
    return supplier_integrated


def integrate_datasets(
    supplier_data: pd.DataFrame = None,
    commodity_data: pd.DataFrame = None,
    co2_data: pd.DataFrame = None,
    esg_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Main integration function that integrates all datasets.
    This function can be called with preprocessed data or will load from config.
    
    Parameters:
    -----------
    supplier_data : pd.DataFrame, optional
        Preprocessed supplier data
    commodity_data : pd.DataFrame, optional
        Preprocessed commodity/GHG data
    co2_data : pd.DataFrame, optional
        Preprocessed CO2 data
    esg_data : pd.DataFrame, optional
        Preprocessed ESG data
        
    Returns:
    --------
    pd.DataFrame
        Fully integrated dataset combining supplier and commodity information
    """
    
    print("\n" + "="*70)
    print("MAIN DATA INTEGRATION PIPELINE")
    print("="*70)
    
    # Load data if not provided
    if supplier_data is None or commodity_data is None or co2_data is None or esg_data is None:
        print("\n[Loading data from config paths...]")
        from data_preprocessing import (
            preprocess_supplier_data,
            preprocess_commodity_data,
            preprocess_co2_data,
            preprocess_esg_data
        )
        
        if supplier_data is None:
            supplier_data = pd.read_csv(config.RAW_DATA_FILES['supplier'])
            supplier_data = preprocess_supplier_data(supplier_data)
        
        if commodity_data is None:
            commodity_data = pd.read_csv(config.RAW_DATA_FILES['commodity'])
            commodity_data = preprocess_commodity_data(commodity_data)
        
        if co2_data is None:
            co2_data = pd.read_csv(config.RAW_DATA_FILES['co2'])
            co2_data = preprocess_co2_data(co2_data)
        
        if esg_data is None:
            esg_data = pd.read_csv(config.RAW_DATA_FILES['esg'])
            esg_data = preprocess_esg_data(esg_data)
    
    # Step 1: Integrate commodity datasets
    print("\n" + "="*70)
    print("STEP 1: Integrating Commodity Datasets")
    print("="*70)
    integrated_commodity = integrate_commodity_datasets(commodity_data, co2_data)
    
    # Step 2: Integrate supplier datasets
    print("\n" + "="*70)
    print("STEP 2: Integrating Supplier Datasets")
    print("="*70)
    integrated_supplier = integrate_supplier_datasets(supplier_data, esg_data)
    
    # Step 3: Final integration - merge supplier with commodity on Industry_Sector and Country
    print("\n" + "="*70)
    print("STEP 3: Final Integration - Merging Supplier and Commodity Datasets")
    print("="*70)
    
    # Aggregate commodity data by Industry_Sector and Country for merging
    print("\n[1] Aggregating Commodity Data for Merging...")
    
    commodity_agg = integrated_commodity.groupby(['Industry_Sector', 'Country']).agg({
        'Supply_Chain_Emission_Factor_With_Margins': 'mean',
        'Supply_Chain_Emission_Factor_Without_Margins': 'mean',
        'Country_Total_CO2_Emissions': 'first',
        'Country_GDP': 'first',
        'Country_Population': 'first',
        'Emission_Intensity_Score': 'mean',
        'Normalized_Emission_Intensity': 'mean'
    }).reset_index()
    
    print(f"    ✓ Aggregated commodity data: {len(commodity_agg)} industry-country combinations")
    
    # Merge with supplier data
    print("\n[2] Merging with Supplier Data...")
    
    # Ensure matching data types
    integrated_supplier['Industry_Sector'] = integrated_supplier['Industry_Sector'].astype(str)
    integrated_supplier['Country'] = integrated_supplier['Country'].astype(str)
    commodity_agg['Industry_Sector'] = commodity_agg['Industry_Sector'].astype(str)
    commodity_agg['Country'] = commodity_agg['Country'].astype(str)
    
    final_integrated = pd.merge(
        integrated_supplier,
        commodity_agg,
        on=['Industry_Sector', 'Country'],
        how='left',
        suffixes=('', '_Commodity')
    )
    
    print(f"    ✓ Merged datasets")
    print(f"    ✓ Final integrated dataset shape: {final_integrated.shape}")
    print(f"    ✓ Matched records: {final_integrated['Supply_Chain_Emission_Factor_With_Margins'].notna().sum()} / {len(final_integrated)}")
    
    # Fill missing commodity values with industry-level averages
    print("\n[3] Handling Missing Commodity Values...")
    
    commodity_cols = [
        'Supply_Chain_Emission_Factor_With_Margins',
        'Supply_Chain_Emission_Factor_Without_Margins',
        'Emission_Intensity_Score',
        'Normalized_Emission_Intensity'
    ]
    
    for col in commodity_cols:
        if col in final_integrated.columns:
            missing_count = final_integrated[col].isna().sum()
            if missing_count > 0:
                # Fill with industry-level median
                industry_medians = final_integrated.groupby('Industry_Sector')[col].transform('median')
                final_integrated[col].fillna(industry_medians, inplace=True)
                
                # If still missing, fill with overall median
                overall_median = final_integrated[col].median()
                final_integrated[col].fillna(overall_median, inplace=True)
                
                print(f"    ✓ Filled {missing_count} missing values in {col}")
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print(f"\nFinal Integrated Dataset:")
    print(f"  Shape: {final_integrated.shape}")
    print(f"  Columns: {len(final_integrated.columns)}")
    print(f"  Unique Suppliers: {final_integrated['Supplier_ID'].nunique()}")
    print(f"  Unique Industries: {final_integrated['Industry_Sector'].nunique()}")
    print(f"  Unique Countries: {final_integrated['Country'].nunique()}")
    print("="*70)
    
    return final_integrated


if __name__ == "__main__":
    """
    Main execution block for testing the integration module.
    """
    
    print("="*70)
    print("DATA INTEGRATION MODULE - STANDALONE EXECUTION")
    print("="*70)
    
    # Load and preprocess data
    from data_preprocessing import (
        preprocess_supplier_data,
        preprocess_commodity_data,
        preprocess_co2_data,
        preprocess_esg_data
    )
    
    print("\nLoading raw datasets...")
    supplier_data = pd.read_csv(config.RAW_DATA_FILES['supplier'])
    commodity_data = pd.read_csv(config.RAW_DATA_FILES['commodity'])
    co2_data = pd.read_csv(config.RAW_DATA_FILES['co2'])
    esg_data = pd.read_csv(config.RAW_DATA_FILES['esg'])
    
    print("Preprocessing datasets...")
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    esg_data = preprocess_esg_data(esg_data)
    
    # Integrate datasets
    print("\nIntegrating datasets...")
    integrated_data = integrate_datasets(
        supplier_data=supplier_data,
        commodity_data=commodity_data,
        co2_data=co2_data,
        esg_data=esg_data
    )
    
    # Save integrated datasets
    print("\nSaving integrated datasets...")
    
    # Save commodity dataset
    commodity_integrated = integrate_commodity_datasets(commodity_data, co2_data)
    commodity_integrated.to_csv(
        config.PROCESSED_DATA_FILES['integrated'].parent / 'integrated_commodity_dataset.csv',
        index=False
    )
    print(f"✓ Saved integrated commodity dataset")
    
    # Save supplier dataset
    supplier_integrated = integrate_supplier_datasets(supplier_data, esg_data)
    supplier_integrated.to_csv(
        config.PROCESSED_DATA_FILES['integrated'].parent / 'integrated_supplier_dataset.csv',
        index=False
    )
    print(f"✓ Saved integrated supplier dataset")
    
    # Save final integrated dataset
    integrated_data.to_csv(
        config.PROCESSED_DATA_FILES['integrated'],
        index=False
    )
    print(f"✓ Saved final integrated dataset: {config.PROCESSED_DATA_FILES['integrated']}")
    
    print("\n" + "="*70)
    print("DATA INTEGRATION COMPLETE")
    print("="*70)

