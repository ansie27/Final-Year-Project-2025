import pandas as pd
import numpy as np

# =====================================================================
# FEATURE ENGINEERING FOR GREEN SUPPLY CHAIN RISK MANAGEMENT
# =====================================================================

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def engineer_supplier_features(supplier_data, commodity_data, co2_data, esg_industry):
    """
    Engineer advanced features for supplier dataset.
    
    Parameters:
    -----------
    supplier_data : pd.DataFrame
        Preprocessed supplier data
    commodity_data : pd.DataFrame
        GHG emission factors data
    co2_data : pd.DataFrame
        CO2 emissions data
    esg_industry : pd.DataFrame
        ESG metrics by industry
        
    Returns:
    --------
    pd.DataFrame
        Supplier dataset with engineered features
    """
    
    print("\n" + "="*70)
    print("SUPPLIER FEATURE ENGINEERING")
    print("="*70)
    
    supplier_features = supplier_data.copy()
    
    # =====================================================================
    # 1. EMISSION-CAPACITY FEATURES
    # =====================================================================
    print("\n[1] Engineering Emission-Capacity Features...")
    
    # Avoid division by zero
    supplier_features['Emission_to_Capacity_Ratio'] = supplier_features.apply(
        lambda row: row['Carbon_Emission_Intensity'] / row['Production_Capacity'] 
        if row['Production_Capacity'] > 0 else np.nan,
        axis=1
    )
    supplier_features['Emission_to_Capacity_Ratio'] = supplier_features['Emission_to_Capacity_Ratio'].fillna(
        supplier_features['Emission_to_Capacity_Ratio'].median()
    )
    print("   - Emission_to_Capacity_Ratio")
    
    # =====================================================================
    # 2. OPERATIONAL RELIABILITY FEATURES
    # =====================================================================
    print("\n[2] Engineering Operational Reliability Features...")
    
    # Operational Reliability Index
    supplier_features['Operational_Reliability_Index'] = (
        supplier_features['On_Time_Delivery_Rate'] * 
        (1 - supplier_features['Defect_Rate'])
    )
    supplier_features['Operational_Reliability_Index'] = supplier_features['Operational_Reliability_Index'].clip(0, 1)
    print("   - Operational_Reliability_Index")
    
    # Lead Time to Capacity Ratio
    supplier_features['LeadTime_Capacity_Ratio'] = supplier_features.apply(
        lambda row: row['Lead_Time_Days'] / row['Production_Capacity']
        if row['Production_Capacity'] > 0 else np.nan,
        axis=1
    )
    supplier_features['LeadTime_Capacity_Ratio'] = supplier_features['LeadTime_Capacity_Ratio'].fillna(
        supplier_features['LeadTime_Capacity_Ratio'].median()
    )
    print("   - LeadTime_Capacity_Ratio")
    
    # =====================================================================
    # 3. GREEN EFFICIENCY FEATURES
    # =====================================================================
    print("\n[3] Engineering Green Efficiency Features...")
    
    # Green Efficiency Score (normalized 0-1)
    supplier_features['Green_Efficiency_Score'] = (
        supplier_features['Waste_Management_Efficiency'] + 
        supplier_features['Renewable_Energy_Usage']
    ) / 2
    print("   - Green_Efficiency_Score")
    
    # =====================================================================
    # 4. RESILIENCE FEATURES
    # =====================================================================
    print("\n[4] Engineering Resilience Features...")
    
    # Resilience Score (normalized 0-100)
    supplier_features['Resilience_Score'] = (
        supplier_features['Operational_Reliability_Index'] * 
        (supplier_features['Financial_Stability_Score'] / 100) * 
        (supplier_features['Compliance_Level'] / 3)  # Normalize to 0-1
    ) * 100
    supplier_features['Resilience_Score'] = supplier_features['Resilience_Score'].clip(0, 100)
    print("   - Resilience_Score")
    
    # =====================================================================
    # 5. ESG-COMPLIANCE COMPOSITE
    # =====================================================================
    print("\n[5] Engineering ESG-Compliance Features...")
    
    # Merge with ESG data
    supplier_features = pd.merge(
        supplier_features,
        esg_industry[['Industry_Sector', 'Avg_ESG_Risk_Score', 'Avg_Environment_Risk', 
                      'Avg_Governance_Risk', 'Avg_Social_Risk']],
        on='Industry_Sector',
        how='left'
    )
    
    # ESG Compliance Composite (lower is better for risk score)
    supplier_features['ESG_Compliance_Composite'] = (
        (100 - supplier_features['ESG_Score'].fillna(50)) * 0.4 +  # Lower ESG score = higher risk
        (100 - supplier_features['Compliance_Level'].fillna(1.5) * 33.33) * 0.3 +  # Normalize compliance to 0-100
        supplier_features['Labour_Compliance_Score'].fillna(50) * 0.3  # Direct score
    )
    supplier_features['ESG_Compliance_Composite'] = supplier_features['ESG_Compliance_Composite'].clip(0, 100)
    print("   - ESG_Compliance_Composite")
    
    # Social Responsibility Score
    supplier_features['Social_Responsibility_Score'] = (
        supplier_features['Social_Score'].fillna(50) * 0.4 +
        supplier_features['Labour_Compliance_Score'].fillna(50) * 0.3 +
        (supplier_features['Diversity_Index'].fillna(0.5) * 100) * 0.3  # Normalize
    )
    supplier_features['Social_Responsibility_Score'] = supplier_features['Social_Responsibility_Score'].clip(0, 100)
    print("   - Social_Responsibility_Score")
    
    # =====================================================================
    # 6. SUSTAINABILITY RISK INDEX
    # =====================================================================
    print("\n[6] Engineering Sustainability Risk Index...")
    
    # Normalized Cost Index for weighting
    cost_norm = (supplier_features['Cost_Index'] - supplier_features['Cost_Index'].min()) / \
                (supplier_features['Cost_Index'].max() - supplier_features['Cost_Index'].min())
    
    supplier_features['Sustainability_Risk_Index'] = (
        (100 - supplier_features['Environmental_Score'].fillna(50)) * 0.35 +  # Environmental risk
        (100 - supplier_features['ESG_Compliance_Composite']) * 0.25 +  # ESG/Compliance risk
        supplier_features['Carbon_Emission_Intensity'].fillna(50) * 0.20 +  # Emissions risk
        cost_norm * 100 * 0.20  # Cost/affordability risk
    )
    supplier_features['Sustainability_Risk_Index'] = supplier_features['Sustainability_Risk_Index'].clip(0, 100)
    print("   - Sustainability_Risk_Index")
    
    # =====================================================================
    # 7. GEOGRAPHIC RISK FEATURES
    # =====================================================================
    print("\n[7] Engineering Geographic Risk Features...")
    
    # Merge CO2 data for country-level risk
    co2_latest = co2_data.sort_values('year').drop_duplicates(
        subset=['Country'], keep='last'
    )[['Country', 'co2', 'population']]
    
    co2_latest.rename(columns={
        'co2': 'Country_CO2_Emissions',
        'population': 'Country_Population'
    }, inplace=True)
    
    supplier_features = pd.merge(
        supplier_features,
        co2_latest,
        on='Country',
        how='left'
    )
    
    # Country Carbon Intensity (emissions per capita)
    supplier_features['Country_Carbon_Intensity'] = supplier_features.apply(
        lambda row: row['Country_CO2_Emissions'] / row['Country_Population']
        if pd.notna(row['Country_Population']) and row['Country_Population'] > 0 else np.nan,
        axis=1
    )
    supplier_features['Country_Carbon_Intensity'] = supplier_features['Country_Carbon_Intensity'].fillna(
        supplier_features['Country_Carbon_Intensity'].median()
    )
    print("   - Country_Carbon_Intensity")
    
    print(f"\nFinal Supplier Dataset Shape: {supplier_features.shape}")
    print("="*70)
    
    return supplier_features


def engineer_commodity_features(commodity_data, co2_data, esg_industry):
    """
    Engineer advanced features for commodity (GHG) dataset.
    
    Parameters:
    -----------
    commodity_data : pd.DataFrame
        Preprocessed commodity/GHG data
    co2_data : pd.DataFrame
        CO2 emissions data by country
    esg_industry : pd.DataFrame
        ESG metrics by industry
        
    Returns:
    --------
    pd.DataFrame
        Commodity dataset with engineered features
    """
    
    print("\n" + "="*70)
    print("COMMODITY FEATURE ENGINEERING")
    print("="*70)
    
    commodity_features = commodity_data.copy()
    
    # =====================================================================
    # 1. GHG TYPE ANALYSIS
    # =====================================================================
    print("\n[1] Engineering GHG Type Analysis Features...")
    
    # GHG Type Dominance Score - which GHG dominates in each industry
    ghg_industry_pivot = commodity_features.pivot_table(
        index='Industry_Sector',
        columns='GHG',
        values='Supply Chain Emission Factors with Margins',
        aggfunc='mean'
    )
    
    # Calculate which GHG is dominant (has highest emission factor)
    commodity_features['GHG_Dominance'] = commodity_features.apply(
        lambda row: ghg_industry_pivot.loc[row['Industry_Sector']].idxmax()
        if row['Industry_Sector'] in ghg_industry_pivot.index else 'Unknown',
        axis=1
    )
    
    # GHG Type Dominance Score (0-1, where 1 = the dominant GHG in that industry)
    commodity_features['GHG_Type_Dominance_Score'] = commodity_features.apply(
        lambda row: 1.0 if row['GHG'] == row['GHG_Dominance'] else 0.5,
        axis=1
    )
    print("   - GHG_Type_Dominance_Score")
    print("   - GHG_Dominance")
    
    # =====================================================================
    # 2. EMISSION VOLATILITY FEATURES
    # =====================================================================
    print("\n[2] Engineering Emission Volatility Features...")
    
    # Calculate standard deviation of emissions within each industry
    emission_std = commodity_features.groupby('Industry_Sector')[
        'Supply Chain Emission Factors with Margins'
    ].std().fillna(0)
    
    commodity_features['Emission_Volatility'] = commodity_features['Industry_Sector'].map(emission_std)
    
    # Normalize volatility
    if commodity_features['Emission_Volatility'].max() > 0:
        commodity_features['Emission_Volatility_Score'] = (
            commodity_features['Emission_Volatility'] / 
            commodity_features['Emission_Volatility'].max()
        ) * 100
    else:
        commodity_features['Emission_Volatility_Score'] = 0
    
    print("   - Emission_Volatility")
    print("   - Emission_Volatility_Score")
    
    # =====================================================================
    # 3. REGIONAL RISK INDEX
    # =====================================================================
    print("\n[3] Engineering Regional Risk Index...")
    
    # Merge ESG data for industry-level risk context
    commodity_features = pd.merge(
        commodity_features,
        esg_industry[['Industry_Sector', 'Avg_ESG_Risk_Score', 'Avg_Environment_Risk']],
        on='Industry_Sector',
        how='left'
    )
    
    # Regional (Industry) Risk Index based on ESG environment risk
    commodity_features['Regional_Risk_Index'] = (
        commodity_features['Avg_ESG_Risk_Score'].fillna(50) * 0.6 +
        commodity_features['Avg_Environment_Risk'].fillna(50) * 0.4
    )
    commodity_features['Regional_Risk_Index'] = commodity_features['Regional_Risk_Index'].clip(0, 100)
    print("   - Regional_Risk_Index")
    
    # =====================================================================
    # 4. EMISSION INTENSITY FEATURES
    # =====================================================================
    print("\n[4] Engineering Emission Intensity Features...")
    
    # Normalize Supply Chain Emission Factors for comparison
    emission_min = commodity_features['Supply Chain Emission Factors with Margins'].min()
    emission_max = commodity_features['Supply Chain Emission Factors with Margins'].max()
    
    if emission_max > emission_min:
        commodity_features['Normalized_Emission_Factor'] = (
            (commodity_features['Supply Chain Emission Factors with Margins'] - emission_min) /
            (emission_max - emission_min)
        ) * 100
    else:
        commodity_features['Normalized_Emission_Factor'] = 50
    
    print("   - Normalized_Emission_Factor")
    
    # =====================================================================
    # 5. INDUSTRY-SPECIFIC EMISSION PROFILE
    # =====================================================================
    print("\n[5] Engineering Industry Emission Profile...")
    
    # Count GHG types per industry
    ghg_count_per_industry = commodity_features.groupby('Industry_Sector')['GHG'].nunique()
    commodity_features['GHG_Types_Count'] = commodity_features['Industry_Sector'].map(ghg_count_per_industry)
    
    # Complexity Score - industries with more GHG types have higher complexity
    commodity_features['Industry_Complexity_Score'] = (
        (commodity_features['GHG_Types_Count'] / commodity_features['GHG_Types_Count'].max()) * 100
    )
    print("   - GHG_Types_Count")
    print("   - Industry_Complexity_Score")
    
    # =====================================================================
    # 6. COMBINED COMMODITY RISK SCORE
    # =====================================================================
    print("\n[6] Engineering Combined Commodity Risk Score...")
    
    commodity_features['Commodity_Risk_Score'] = (
        commodity_features['Normalized_Emission_Factor'] * 0.35 +
        commodity_features['Regional_Risk_Index'] * 0.30 +
        commodity_features['Emission_Volatility_Score'].fillna(0) * 0.20 +
        commodity_features['Industry_Complexity_Score'] * 0.15
    )
    commodity_features['Commodity_Risk_Score'] = commodity_features['Commodity_Risk_Score'].clip(0, 100)
    print("   - Commodity_Risk_Score")
    
    print(f"\nFinal Commodity Dataset Shape: {commodity_features.shape}")
    print("="*70)
    
    return commodity_features


def display_feature_engineering_summary(supplier_features, commodity_features):
    """
    Display summary of engineered features.
    
    Parameters:
    -----------
    supplier_features : pd.DataFrame
        Supplier dataset with engineered features
    commodity_features : pd.DataFrame
        Commodity dataset with engineered features
    """
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    
    print("\nSupplier Dataset:")
    print(f"  Shape: {supplier_features.shape}")
    
    supplier_new_features = [
        'Emission_to_Capacity_Ratio', 'Operational_Reliability_Index',
        'LeadTime_Capacity_Ratio', 'Green_Efficiency_Score', 'Resilience_Score',
        'ESG_Compliance_Composite', 'Social_Responsibility_Score',
        'Sustainability_Risk_Index', 'Country_Carbon_Intensity'
    ]
    print(f"  New Features Created: {len(supplier_new_features)}")
    for feat in supplier_new_features:
        if feat in supplier_features.columns:
            print(f"    - {feat}")
    
    print("\nCommodity Dataset:")
    print(f"  Shape: {commodity_features.shape}")
    
    commodity_new_features = [
        'GHG_Type_Dominance_Score', 'Emission_Volatility',
        'Emission_Volatility_Score', 'Regional_Risk_Index',
        'Normalized_Emission_Factor', 'GHG_Types_Count',
        'Industry_Complexity_Score', 'Commodity_Risk_Score'
    ]
    print(f"  New Features Created: {len(commodity_new_features)}")
    for feat in commodity_new_features:
        if feat in commodity_features.columns:
            print(f"    - {feat}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Feature Engineering Module Loaded Successfully")
