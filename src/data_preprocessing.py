import pandas as pd
import numpy as np

# =====================================================================
# DATA PREPROCESSING FOR GREEN SUPPLY CHAIN RISK MANAGEMENT
# =====================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def preprocess_supplier_data(supplier_data):
    print("="*70)
    print("SUPPLIER DATA PREPROCESSING")
    print("="*70)
    
    supplier_data = supplier_data.copy()
    
    print(f"\nInitial dataset shape: {supplier_data.shape}")
    
    # Standardize country names
    print("\n[1] Standardizing country names...")
    supplier_data['Country'] = supplier_data['Country'].str.strip().str.title()
    print(f"   Standardized country names: {supplier_data['Country'].nunique()} unique countries")
    
    # FIX: Convert Industry_Sector to string type
    print("\n[2] Standardizing Industry_Sector data type...")
    supplier_data['Industry_Sector'] = supplier_data['Industry_Sector'].astype(str)
    print(f"   Industry_Sector converted to string type")
    
    # Handle missing values
    print("\n[3] Handling missing values...")
    missing_before = supplier_data.isnull().sum().sum()
    
    # Fill missing Environmental_Score with median
    if supplier_data['Environmental_Score'].isnull().any():
        median_env = supplier_data['Environmental_Score'].median()
        supplier_data['Environmental_Score'].fillna(median_env, inplace=True)
        print(f"   Filled Environmental_Score with median: {median_env:.2f}")
    
    # Fill missing Labour_Compliance_Score with median
    if supplier_data['Labour_Compliance_Score'].isnull().any():
        median_labour = supplier_data['Labour_Compliance_Score'].median()
        supplier_data['Labour_Compliance_Score'].fillna(median_labour, inplace=True)
        print(f"   Filled Labour_Compliance_Score with median: {median_labour:.2f}")
    
    # Fill missing Logistics_Distance_km with region average
    if supplier_data['Logistics_Distance_km'].isnull().any():
        supplier_data['Logistics_Distance_km'].fillna(
            supplier_data.groupby('Region')['Logistics_Distance_km'].transform('mean'),
            inplace=True
        )
        print(f"   Filled Logistics_Distance_km with region averages")
    
    # Fill remaining Production_Capacity with industry average
    if supplier_data['Production_Capacity'].isnull().any():
        supplier_data['Production_Capacity'].fillna(
            supplier_data.groupby('Industry_Sector')['Production_Capacity'].transform('mean'),
            inplace=True
        )
        print(f"   Filled Production_Capacity with industry averages")
    
    missing_after = supplier_data.isnull().sum().sum()
    print(f"   Reduced missing values from {missing_before} to {missing_after}")
    
    # Data type conversions
    print("\n[4] Converting data types...")
    supplier_data['Supplier_ID'] = supplier_data['Supplier_ID'].astype('string')
    supplier_data['Supplier_Tier'] = supplier_data['Supplier_Tier'].astype('int8')
    print(f"   Data types standardized")
    
    # Remove duplicates if any
    print("\n[5] Checking for duplicates...")
    duplicates = supplier_data.duplicated(subset=['Supplier_ID']).sum()
    if duplicates > 0:
        supplier_data = supplier_data.drop_duplicates(subset=['Supplier_ID'], keep='first')
        print(f"   Removed {duplicates} duplicate supplier records")
    else:
        print(f"   No duplicates found")
    
    # Validate numeric ranges
    print("\n[6] Validating numeric ranges...")
    
    # Ensure scores are in valid ranges (0-100)
    score_columns = ['ESG_Score', 'Environmental_Score', 'Social_Score', 
                     'Governance_Score', 'Financial_Stability_Score']
    for col in score_columns:
        if col in supplier_data.columns:
            invalid = (supplier_data[col] < 0) | (supplier_data[col] > 100)
            if invalid.any():
                supplier_data.loc[invalid, col] = supplier_data[col].clip(0, 100)
                print(f"   Clipped {invalid.sum()} invalid values in {col}")
    
    # Ensure rates are between 0-1
    rate_columns = ['On_Time_Delivery_Rate', 'Defect_Rate', 
                    'Waste_Management_Efficiency', 'Renewable_Energy_Usage']
    for col in rate_columns:
        if col in supplier_data.columns:
            invalid = (supplier_data[col] < 0) | (supplier_data[col] > 1)
            if invalid.any():
                supplier_data.loc[invalid, col] = supplier_data[col].clip(0, 1)
                print(f"   Clipped {invalid.sum()} invalid values in {col}")
    
    print(f"\nFinal dataset shape: {supplier_data.shape}")
    print("="*70)
    
    return supplier_data


def preprocess_commodity_data(commodity_data): 
    print("\n" + "="*70)
    print("COMMODITY (GHG) DATA PREPROCESSING")
    print("="*70)
    
    commodity_data = commodity_data.copy()
    
    print(f"\nInitial dataset shape: {commodity_data.shape}")
    
    # Rename columns for consistency
    print("\n[1] Standardizing column names...")
    commodity_data.rename(columns={
        '2017 NAICS Title': 'Industry_Sector'
    }, inplace=True)
    print(f"   Column names standardized")
    
    # FIX: Convert Industry_Sector to string type
    print("\n[2] Standardizing Industry_Sector data type...")
    commodity_data['Industry_Sector'] = commodity_data['Industry_Sector'].astype(str)
    print(f"   Industry_Sector converted to string type")
    
    # Remove duplicates
    print("\n[3] Checking for duplicates...")
    duplicates = commodity_data.duplicated(
        subset=['Industry_Sector', 'GHG']
    ).sum()
    if duplicates > 0:
        commodity_data = commodity_data.drop_duplicates(
            subset=['Industry_Sector', 'GHG'],
            keep='first'
        )
        print(f"   Removed {duplicates} duplicate records")
    else:
        print(f"   No duplicates found")
    
    # Handle missing values
    print("\n[4] Handling missing values...")
    missing = commodity_data.isnull().sum()
    if missing.sum() > 0:
        print(f"   Found missing values:")
        for col, count in missing[missing > 0].items():
            print(f"      {col}: {count}")
        # Forward fill for emission factors
        commodity_data['Supply Chain Emission Factors with Margins'] = commodity_data['Supply Chain Emission Factors with Margins'].ffill()
    else:
        print(f"   No missing values")
    
    # Validate numeric ranges
    print("\n[5] Validating numeric ranges...")
    emission_col = 'Supply Chain Emission Factors with Margins'
    if (commodity_data[emission_col] < 0).any():
        commodity_data = commodity_data[commodity_data[emission_col] >= 0]
        print(f"   Removed negative emission factor records")
    
    print(f"\nFinal dataset shape: {commodity_data.shape}")
    print("="*70)
    
    return commodity_data


def preprocess_esg_data(esg_data):
    """
    Clean and standardize S&P 500 ESG dataset.
    
    Parameters:
    -----------
    esg_data : pd.DataFrame
        Raw ESG dataset from S&P 500
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and aggregated ESG dataset
    """
    
    print("\n" + "="*70)
    print("S&P 500 ESG DATA PREPROCESSING")
    print("="*70)
    
    esg_data = esg_data.copy()
    
    print(f"\nInitial dataset shape: {esg_data.shape}")
    
    # Rename / create columns for consistency
    print("\n[1] Standardizing column names...")

    # Keep original 'Industry' column (needed by data_integration.integrate_supplier_datasets)
    # but also create a standardized 'Industry_Sector' version used throughout the pipeline.
    if 'Industry' in esg_data.columns and 'Industry_Sector' not in esg_data.columns:
        esg_data['Industry_Sector'] = esg_data['Industry']
    elif 'Industry_Sector' in esg_data.columns and 'Industry' not in esg_data.columns:
        # Ensure downstream functions that expect 'Industry' can still operate
        esg_data['Industry'] = esg_data['Industry_Sector']

    # Standardize sector naming
    if 'Sector' in esg_data.columns and 'Sector_Classification' not in esg_data.columns:
        esg_data.rename(columns={'Sector': 'Sector_Classification'}, inplace=True)

    print(f"   * Column names standardized (Industry / Industry_Sector, Sector_Classification)")
    
    # Handle missing values
    print("\n[2] Handling missing values...")
    
    # Remove records with missing key ESG metrics
    esg_data = esg_data.dropna(subset=['Total ESG Risk score'])
    print(f"   * Removed records with missing ESG Risk Score: {esg_data.shape[0]} records remaining")
    
    # Fill missing individual scores with total score average if available
    score_cols = ['Environment Risk Score', 'Governance Risk Score', 'Social Risk Score']
    for col in score_cols:
        if col in esg_data.columns:
            missing_count = esg_data[col].isnull().sum()
            if missing_count > 0:
                # Fill with ESG component average
                median_val = esg_data[col].median()
                esg_data[col].fillna(median_val, inplace=True)
                print(f"   [+] Filled {col}: {missing_count} records")
    
    # Handle controversy scores
    if 'Controversy Score' in esg_data.columns:
        esg_data['Controversy Score'].fillna(0, inplace=True)
    
    print("\n[3] Standardizing Industry / Industry_Sector names...")
    # Standardize industry names on the Industry_Sector column
    if 'Industry_Sector' in esg_data.columns:
        esg_data['Industry_Sector'] = esg_data['Industry_Sector'].astype(str).str.strip().str.title()
    # Keep 'Industry' in sync for compatibility with integration module
    if 'Industry' in esg_data.columns:
        esg_data['Industry'] = esg_data['Industry'].astype(str).str.strip().str.title()
    print(f"   * Standardized to {esg_data['Industry_Sector'].nunique()} unique industries")
    
    print("\n[4] Validating numeric ranges...")
    # Ensure scores are positive
    numeric_cols = ['Total ESG Risk score', 'Environment Risk Score', 
                    'Governance Risk Score', 'Social Risk Score', 'Controversy Score']
    for col in numeric_cols:
        if col in esg_data.columns:
            esg_data[col] = esg_data[col].clip(lower=0)
    print(f"   * Validated numeric ranges")
    
    print(f"\nFinal dataset shape: {esg_data.shape}")
    print("="*70)
    
    return esg_data


def aggregate_esg_by_industry(esg_data):
    """
    Aggregate ESG metrics by industry sector.
    
    Parameters:
    -----------
    esg_data : pd.DataFrame
        Preprocessed ESG dataset
        
    Returns:
    --------
    pd.DataFrame
        ESG metrics aggregated by industry
    """
    
    print("\nAggregating ESG metrics by Industry_Sector...")
    
    # Group by industry and calculate mean ESG metrics
    esg_industry = esg_data.groupby('Industry_Sector').agg({
        'Total ESG Risk score': ['mean', 'std', 'count'],
        'Environment Risk Score': 'mean',
        'Governance Risk Score': 'mean',
        'Social Risk Score': 'mean',
        'Controversy Score': 'mean'
    }).reset_index()
    
    # Flatten column names
    esg_industry.columns = ['Industry_Sector', 
                            'Avg_ESG_Risk_Score', 'Std_ESG_Risk_Score', 'Company_Count',
                            'Avg_Environment_Risk', 'Avg_Governance_Risk', 'Avg_Social_Risk',
                            'Avg_Controversy_Score']
    
    print(f"   [+] Aggregated {len(esg_industry)} industries with ESG metrics")
    print(f"   [+] Total companies represented: {esg_industry['Company_Count'].sum():.0f}")
    
    return esg_industry


def preprocess_co2_data(co2_data):
    """
    Clean and standardize CO2 emissions dataset.
    
    Parameters:
    -----------
    co2_data : pd.DataFrame
        Raw CO2 dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned CO2 dataset
    """
    
    print("\n" + "="*70)
    print("CO2 EMISSIONS DATA PREPROCESSING")
    print("="*70)
    
    co2_data = co2_data.copy()
    
    print(f"\nInitial dataset shape: {co2_data.shape}")
    
    # Rename / create columns for consistency
    print("\n[1] Standardizing column names...")

    # Keep both 'country' (expected by data_integration.integrate_commodity_datasets)
    # and 'Country' (used in the rest of the pipeline).
    if 'country' in co2_data.columns and 'Country' not in co2_data.columns:
        co2_data['Country'] = co2_data['country']
    elif 'Country' in co2_data.columns and 'country' not in co2_data.columns:
        co2_data['country'] = co2_data['Country']

    print(f"   Column names standardized (country / Country)")
    
    # Standardize country names
    print("\n[2] Standardizing country names...")
    if 'Country' in co2_data.columns:
        co2_data['Country'] = co2_data['Country'].astype(str).str.strip().str.title()
        # Keep lowercase 'country' alias aligned for integration module
        co2_data['country'] = co2_data['Country']
    print(f"   Standardized: {co2_data['Country'].nunique()} unique countries")
    
    # Remove records with missing key columns
    print("\n[3] Handling missing values...")
    missing_before = co2_data.isnull().sum().sum()
    
    # Keep only records with year and co2 data
    co2_data = co2_data[co2_data['year'].notna()]
    co2_data = co2_data[co2_data['co2'].notna()]
    
    missing_after = co2_data.isnull().sum().sum()
    print(f"   Removed {missing_before - missing_after} rows with missing key data")
    
    # Filter for reasonable year range
    print("\n[4] Filtering year range...")
    co2_data = co2_data[(co2_data['year'] >= 1990) & (co2_data['year'] <= 2024)]
    print(f"   Filtered to years 1990-2024")
    
    # Validate numeric ranges
    print("\n[5] Validating numeric ranges...")
    co2_data = co2_data[co2_data['co2'] >= 0]
    if 'gdp' in co2_data.columns:
        co2_data = co2_data[(co2_data['gdp'].isna()) | (co2_data['gdp'] >= 0)]
    if 'population' in co2_data.columns:
        co2_data = co2_data[(co2_data['population'].isna()) | (co2_data['population'] > 0)]
    print(f"   Validated numeric ranges")
    
    print(f"\nFinal dataset shape: {co2_data.shape}")
    print("="*70)
    
    return co2_data


def calculate_risk_metrics(integrated_data):
    """
    Calculate derived risk metrics for integrated dataset.
    
    Parameters:
    -----------
    integrated_data : pd.DataFrame
        Integrated dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with calculated risk metrics
    """
    
    print("\n" + "="*70)
    print("CALCULATING RISK METRICS")
    print("="*70)
    
    integrated_data = integrated_data.copy()
    
    print("\n[1] Calculating Environmental Risk Score...")
    integrated_data['Environmental_Risk_Score'] = (
        (100 - integrated_data['Environmental_Score'].fillna(50)) * 0.4 +
        (integrated_data['Carbon_Emission_Intensity'].fillna(
            integrated_data['Carbon_Emission_Intensity'].median()
        )) * 0.3 +
        ((100 - integrated_data['Renewable_Energy_Usage'].fillna(50) * 100) * 0.3)
    ) / 3
    print(f"   Environmental_Risk_Score calculated")
    
    print("\n[2] Calculating Compliance Risk Score...")
    integrated_data['Compliance_Risk_Score'] = (
        (3 - integrated_data['Compliance_Level'].fillna(1.5)) * 25 +
        ((1 - integrated_data['Sustainability_Report_Availability'].fillna(0.5)) * 50)
    )
    print(f"   Compliance_Risk_Score calculated")
    
    print("\n[3] Calculating Operational Risk Score...")
    integrated_data['Operational_Risk_Score'] = (
        (1 - integrated_data['On_Time_Delivery_Rate'].fillna(0.85)) * 50 +
        (integrated_data['Defect_Rate'].fillna(0.05) * 500)
    )
    print(f"   Operational_Risk_Score calculated")
    
    print("\n[4] Calculating Financial Risk Score...")
    integrated_data['Financial_Risk_Score'] = 100 - integrated_data['Financial_Stability_Score'].fillna(50)
    print(f"   Financial_Risk_Score calculated")
    
    print("\n[5] Calculating Overall Risk Score...")
    integrated_data['Overall_Risk_Score'] = (
        integrated_data['Environmental_Risk_Score'].fillna(50) * 0.35 +
        integrated_data['Compliance_Risk_Score'].fillna(50) * 0.30 +
        integrated_data['Operational_Risk_Score'].fillna(50) * 0.20 +
        integrated_data['Financial_Risk_Score'].fillna(50) * 0.15
    )
    print(f"   Overall_Risk_Score calculated")
    
    print("\n[6] Classifying Risk Levels...")
    def classify_risk(score):
        if pd.isna(score):
            return 'Unknown'
        elif score < 30:
            return 'Low'
        elif score < 60:
            return 'Medium'
        elif score < 80:
            return 'High'
        else:
            return 'Critical'
    
    integrated_data['Risk_Classification'] = integrated_data['Overall_Risk_Score'].apply(classify_risk)
    print(f"   Risk_Classification assigned")
    
    print("\n" + "="*70)
    
    return integrated_data


def display_preprocessing_summary(supplier_data, commodity_data, co2_data, esg_industry=None):
    """
    Display summary of preprocessed datasets.
    
    Parameters:
    -----------
    supplier_data : pd.DataFrame
        Preprocessed supplier data
    commodity_data : pd.DataFrame
        Preprocessed commodity data
    co2_data : pd.DataFrame
        Preprocessed CO2 data
    esg_industry : pd.DataFrame, optional
        Aggregated ESG data by industry
    """
    
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    
    print(f"\nSupplier Data:")
    print(f"  Shape: {supplier_data.shape}")
    print(f"  Unique suppliers: {supplier_data['Supplier_ID'].nunique()}")
    print(f"  Unique industries: {supplier_data['Industry_Sector'].nunique()}")
    print(f"  Unique countries: {supplier_data['Country'].nunique()}")
    
    print(f"\nCommodity/GHG Data:")
    print(f"  Shape: {commodity_data.shape}")
    print(f"  Unique industries: {commodity_data['Industry_Sector'].nunique()}")
    print(f"  Unique GHG types: {commodity_data['GHG'].nunique()}")
    
    print(f"\nCO2 Data:")
    print(f"  Shape: {co2_data.shape}")
    print(f"  Unique countries: {co2_data['Country'].nunique()}")
    print(f"  Year range: {co2_data['year'].min()}-{co2_data['year'].max()}")
    
    if esg_industry is not None:
        print(f"\nS&P 500 ESG Data (Aggregated by Industry):")
        print(f"  Shape: {esg_industry.shape}")
        print(f"  Industries covered: {esg_industry['Industry_Sector'].nunique()}")
        print(f"  Total companies: {esg_industry['Company_Count'].sum():.0f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Load raw datasets
    print("Loading raw datasets...\n")
    supplier_data = pd.read_csv("data/raw/synthetic_supplier_dataset_1.csv")
    commodity_data = pd.read_csv("data/raw/SupplyChainGHGEmissionFactors_v1.2_NAICS_byGHG_USD2021.csv")
    co2_data = pd.read_csv("data/raw/owid-co2-data.csv")
    esg_data = pd.read_csv("data/raw/SP 500 ESG Risk Ratings.csv")
    
    # Preprocess datasets
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    esg_data = preprocess_esg_data(esg_data)
    esg_industry = aggregate_esg_by_industry(esg_data)
    
    # Display summary
    display_preprocessing_summary(supplier_data, commodity_data, co2_data, esg_industry)
    
    # Save preprocessed datasets
    print("\nSaving preprocessed datasets...")
    supplier_data.to_csv("data/processed/preprocessed_supplier_data.csv", index=False)
    commodity_data.to_csv("data/processed/preprocessed_commodity_data.csv", index=False)
    co2_data.to_csv("data/processed/preprocessed_co2_data.csv", index=False)
    esg_data.to_csv("data/processed/preprocessed_esg_data.csv", index=False)
    esg_industry.to_csv("data/processed/preprocessed_esg_by_industry.csv", index=False)
    print("[SUCCESS] Preprocessed datasets saved to data/processed/")