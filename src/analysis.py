"""
Statistical Analysis Module for Green Supply Chain Risk Management

This module provides functions for statistical analysis, correlation analysis,
and feature importance analysis of the integrated dataset.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def calculate_descriptive_statistics(data, columns=None):
    """
    Calculate descriptive statistics for specified columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    columns : list, optional
        List of column names to analyze. If None, analyzes all numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        Descriptive statistics summary
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_summary = data[columns].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        'skewness': data[columns].skew(),
        'kurtosis': data[columns].kurtosis(),
        'variance': data[columns].var(),
        'missing_count': data[columns].isnull().sum(),
        'missing_pct': (data[columns].isnull().sum() / len(data)) * 100
    })
    
    stats_summary = pd.concat([stats_summary.T, additional_stats], axis=1)
    
    return stats_summary


def calculate_correlation_matrix(data, method='pearson', risk_columns=None):
    """
    Calculate correlation matrix for risk-related columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    risk_columns : list, optional
        List of risk-related columns. If None, uses default risk columns.
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if risk_columns is None:
        risk_columns = [
            'Environmental_Risk_Score', 'Compliance_Risk_Score',
            'Operational_Risk_Score', 'Financial_Risk_Score', 'Overall_Risk_Score',
            'Environmental_Score', 'ESG_Score', 'Carbon_Emission_Intensity',
            'Renewable_Energy_Usage', 'On_Time_Delivery_Rate', 'Defect_Rate',
            'Financial_Stability_Score', 'Compliance_Level'
        ]
    
    # Filter to available columns
    available_columns = [col for col in risk_columns if col in data.columns]
    numeric_data = data[available_columns].select_dtypes(include=[np.number])
    
    correlation_matrix = numeric_data.corr(method=method)
    
    return correlation_matrix


def identify_high_correlations(correlation_matrix, threshold=0.7):
    """
    Identify highly correlated feature pairs.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    threshold : float
        Correlation threshold (default: 0.7)
        
    Returns:
    --------
    pd.DataFrame
        Pairs of highly correlated features
    """
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'Feature_1': correlation_matrix.columns[i],
                    'Feature_2': correlation_matrix.columns[j],
                    'Correlation': corr_value
                })
    
    return pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)


def calculate_feature_importance(data, target_column='Overall_Risk_Score', method='correlation'):
    """
    Calculate feature importance for predicting target variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target variable column name
    method : str
        Method for calculating importance ('correlation', 'mutual_info')
        
    Returns:
    --------
    pd.DataFrame
        Feature importance scores
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Select numeric features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != target_column]
    
    if method == 'correlation':
        correlations = data[numeric_features + [target_column]].corr()[target_column]
        correlations = correlations.drop(target_column)
        importance_df = pd.DataFrame({
            'Feature': correlations.index,
            'Importance': correlations.abs().values,
            'Correlation': correlations.values
        }).sort_values('Importance', ascending=False)
    
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_regression
        
        # Prepare data
        X = data[numeric_features].fillna(data[numeric_features].median())
        y = data[target_column].fillna(data[target_column].median())
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        importance_df = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': mi_scores
        }).sort_values('Importance', ascending=False)
    
    return importance_df


def perform_statistical_tests(data, group_column, value_column, test_type='t-test'):
    """
    Perform statistical tests between groups.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    group_column : str
        Column to group by
    value_column : str
        Column to test
    test_type : str
        Type of test ('t-test', 'anova', 'mannwhitney')
        
    Returns:
    --------
    dict
        Test results
    """
    groups = data.groupby(group_column)[value_column].apply(list)
    
    if test_type == 't-test' and len(groups) == 2:
        stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
        return {
            'test': 't-test',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    elif test_type == 'anova' and len(groups) > 2:
        stat, p_value = stats.f_oneway(*groups.values)
        return {
            'test': 'ANOVA',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    elif test_type == 'mannwhitney' and len(groups) == 2:
        stat, p_value = stats.mannwhitneyu(groups.iloc[0], groups.iloc[1])
        return {
            'test': 'Mann-Whitney U',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return None


def analyze_risk_by_category(data, category_column, risk_column='Overall_Risk_Score'):
    """
    Analyze risk scores by category.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    category_column : str
        Column to group by (e.g., 'Country', 'Industry_Sector', 'Risk_Classification')
    risk_column : str
        Risk score column to analyze
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics by category
    """
    analysis = data.groupby(category_column)[risk_column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    analysis.columns = ['Count', 'Mean_Risk', 'Median_Risk', 'Std_Risk', 'Min_Risk', 'Max_Risk']
    analysis = analysis.sort_values('Mean_Risk', ascending=False)
    
    return analysis


def perform_pca_analysis(data, n_components=None, risk_columns=None):
    """
    Perform Principal Component Analysis on risk-related features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    n_components : int, optional
        Number of components. If None, uses all components.
    risk_columns : list, optional
        Columns to include in PCA
        
    Returns:
    --------
    dict
        PCA results including explained variance and components
    """
    if risk_columns is None:
        risk_columns = [
            'Environmental_Risk_Score', 'Compliance_Risk_Score',
            'Operational_Risk_Score', 'Financial_Risk_Score'
        ]
    
    # Filter to available columns
    available_columns = [col for col in risk_columns if col in data.columns]
    X = data[available_columns].fillna(data[available_columns].median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    if n_components is None:
        n_components = min(len(available_columns), len(X))
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    results = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'feature_names': available_columns,
        'transformed_data': X_pca
    }
    
    return results


def generate_analysis_report(data, output_path=None):
    """
    Generate comprehensive statistical analysis report.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    output_path : str, optional
        Path to save report. If None, returns report as string.
        
    Returns:
    --------
    str
        Analysis report
    """
    report = []
    report.append("="*70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("="*70)
    report.append("")
    
    # Dataset overview
    report.append("Dataset Overview:")
    report.append(f"  Total Records: {len(data)}")
    report.append(f"  Total Features: {len(data.columns)}")
    report.append(f"  Missing Values: {data.isnull().sum().sum()}")
    report.append("")
    
    # Risk score statistics
    if 'Overall_Risk_Score' in data.columns:
        report.append("Overall Risk Score Statistics:")
        risk_stats = data['Overall_Risk_Score'].describe()
        for stat, value in risk_stats.items():
            report.append(f"  {stat.capitalize()}: {value:.2f}")
        report.append("")
    
    # Risk by classification
    if 'Risk_Classification' in data.columns:
        report.append("Risk Distribution by Classification:")
        risk_dist = data['Risk_Classification'].value_counts()
        for classification, count in risk_dist.items():
            pct = (count / len(data)) * 100
            report.append(f"  {classification}: {count} ({pct:.1f}%)")
        report.append("")
    
    # Top risk factors
    if 'Overall_Risk_Score' in data.columns:
        report.append("Top 10 Features Correlated with Overall Risk:")
        corr_matrix = calculate_correlation_matrix(data)
        if 'Overall_Risk_Score' in corr_matrix.columns:
            risk_corr = corr_matrix['Overall_Risk_Score'].abs().sort_values(ascending=False)
            risk_corr = risk_corr.drop('Overall_Risk_Score')
            for i, (feature, corr) in enumerate(risk_corr.head(10).items(), 1):
                report.append(f"  {i}. {feature}: {corr:.3f}")
        report.append("")
    
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Analysis report saved to: {output_path}")
    
    return report_text


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("Import this module to use analysis functions in your pipeline.")

