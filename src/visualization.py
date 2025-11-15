"""
Visualization Module for Green Supply Chain Risk Management

This module provides functions for creating visualizations including
charts, graphs, and dashboards for risk analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def create_risk_distribution_plot(data, risk_column='Overall_Risk_Score', save_path=None):
    """
    Create distribution plot of risk scores.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    risk_column : str
        Risk score column to plot
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(data[risk_column].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Risk Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {risk_column}')
    axes[0].axvline(data[risk_column].mean(), color='red', linestyle='--', 
                    label=f'Mean: {data[risk_column].mean():.2f}')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(data[risk_column].dropna(), vert=True)
    axes[1].set_ylabel('Risk Score')
    axes[1].set_title(f'Box Plot of {risk_column}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_risk_classification_plot(data, classification_column='Risk_Classification', save_path=None):
    """
    Create bar plot of risk classifications.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    classification_column : str
        Risk classification column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    risk_counts = data[classification_column].value_counts()
    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red', 'Unknown': 'gray'}
    bar_colors = [colors.get(risk, 'blue') for risk in risk_counts.index]
    
    axes[0].bar(risk_counts.index, risk_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Risk Classification')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Risk Classification Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=bar_colors, startangle=90)
    axes[1].set_title('Risk Classification Proportion')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_correlation_heatmap(data, risk_columns=None, save_path=None):
    """
    Create correlation heatmap for risk-related features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    risk_columns : list, optional
        List of risk-related columns
    save_path : str, optional
        Path to save the figure
    """
    if risk_columns is None:
        risk_columns = [
            'Environmental_Risk_Score', 'Compliance_Risk_Score',
            'Operational_Risk_Score', 'Financial_Risk_Score', 'Overall_Risk_Score',
            'Environmental_Score', 'ESG_Score', 'Carbon_Emission_Intensity',
            'Renewable_Energy_Usage', 'On_Time_Delivery_Rate', 'Defect_Rate',
            'Financial_Stability_Score'
        ]
    
    # Filter to available columns
    available_columns = [col for col in risk_columns if col in data.columns]
    corr_data = data[available_columns].select_dtypes(include=[np.number]).corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Risk Factors Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_risk_by_category_plot(data, category_column, risk_column='Overall_Risk_Score', 
                                  top_n=10, save_path=None):
    """
    Create bar plot of average risk scores by category.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    category_column : str
        Column to group by (e.g., 'Country', 'Industry_Sector')
    risk_column : str
        Risk score column
    top_n : int
        Number of top categories to display
    save_path : str, optional
        Path to save the figure
    """
    risk_by_category = data.groupby(category_column)[risk_column].mean().sort_values(ascending=False)
    top_categories = risk_by_category.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top_categories)), top_categories.values, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_categories)))
    ax.set_yticklabels(top_categories.index)
    ax.set_xlabel('Average Risk Score')
    ax.set_title(f'Top {top_n} {category_column} by Average Risk Score')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, val) in enumerate(top_categories.items()):
        ax.text(val, i, f' {val:.2f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_feature_importance_plot(importance_df, top_n=15, save_path=None):
    """
    Create bar plot of feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
                   alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()
    
    # Add value labels
    for i, val in enumerate(top_features['Importance'].values):
        ax.text(val, i, f' {val:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_risk_comparison_plot(data, group_column, risk_column='Overall_Risk_Score', save_path=None):
    """
    Create comparison plot of risk scores across groups.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    group_column : str
        Column to group by
    risk_column : str
        Risk score column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Violin plot
    groups = data[group_column].unique()[:10]  # Limit to top 10 groups
    filtered_data = data[data[group_column].isin(groups)]
    
    sns.violinplot(data=filtered_data, x=group_column, y=risk_column, ax=axes[0])
    axes[0].set_xlabel(group_column)
    axes[0].set_ylabel(risk_column)
    axes[0].set_title(f'Risk Score Distribution by {group_column}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Box plot
    sns.boxplot(data=filtered_data, x=group_column, y=risk_column, ax=axes[1])
    axes[1].set_xlabel(group_column)
    axes[1].set_ylabel(risk_column)
    axes[1].set_title(f'Risk Score Comparison by {group_column}')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_time_series_plot(data, time_column, value_column, group_column=None, save_path=None):
    """
    Create time series plot if time data is available.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    time_column : str
        Time column name
    value_column : str
        Value column to plot
    group_column : str, optional
        Column to group by for multiple series
    save_path : str, optional
        Path to save the figure
    """
    if time_column not in data.columns:
        print(f"Time column '{time_column}' not found in data")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if group_column:
        for group in data[group_column].unique()[:5]:  # Limit to 5 groups
            group_data = data[data[group_column] == group]
            group_data = group_data.sort_values(time_column)
            ax.plot(group_data[time_column], group_data[value_column], 
                   label=group, marker='o', markersize=4)
        ax.legend()
    else:
        sorted_data = data.sort_values(time_column)
        ax.plot(sorted_data[time_column], sorted_data[value_column], marker='o', markersize=4)
    
    ax.set_xlabel(time_column)
    ax.set_ylabel(value_column)
    ax.set_title(f'{value_column} Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_topsis_ranking_plot(ranking_data, top_n=20, save_path=None):
    """
    Create bar plot of top N suppliers ranked by TOPSIS scores.
    
    Parameters:
    -----------
    ranking_data : pd.DataFrame
        DataFrame with 'Supplier_ID' and 'TOPSIS_Score' columns
    top_n : int
        Number of top suppliers to display
    save_path : str, optional
        Path to save the figure
    """
    top_suppliers = ranking_data.head(top_n).sort_values('TOPSIS_Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_suppliers)), top_suppliers['TOPSIS_Score'].values, 
                   alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_yticks(range(len(top_suppliers)))
    ax.set_yticklabels(top_suppliers['Supplier_ID'].values)
    ax.set_xlabel('TOPSIS Score', fontsize=12)
    ax.set_title(f'Top {top_n} Suppliers Ranked by TOPSIS Score', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, val in enumerate(top_suppliers['TOPSIS_Score'].values):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    # Add rank labels
    for i, rank in enumerate(top_suppliers['Rank'].values):
        ax.text(0.01, i, f'#{int(rank)}', va='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_comprehensive_dashboard(data, output_dir='outputs/visualizations', ranking_data=None):
    """
    Create comprehensive visualization dashboard.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    output_dir : str
        Directory to save all visualizations
    ranking_data : pd.DataFrame, optional
        TOPSIS ranking data with 'Supplier_ID', 'TOPSIS_Score', and 'Rank' columns
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Creating comprehensive visualization dashboard...")
    
    # Risk distribution
    if 'Overall_Risk_Score' in data.columns:
        create_risk_distribution_plot(data, save_path=f"{output_dir}/risk_distribution.png")
        plt.close()
    
    # Risk classification
    if 'Risk_Classification' in data.columns:
        create_risk_classification_plot(data, save_path=f"{output_dir}/risk_classification.png")
        plt.close()
    
    # Correlation heatmap
    create_correlation_heatmap(data, save_path=f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # Risk by country
    if 'Country' in data.columns and 'Overall_Risk_Score' in data.columns:
        create_risk_by_category_plot(data, 'Country', save_path=f"{output_dir}/risk_by_country.png")
        plt.close()
    
    # Risk by industry
    if 'Industry_Sector' in data.columns and 'Overall_Risk_Score' in data.columns:
        create_risk_by_category_plot(data, 'Industry_Sector', save_path=f"{output_dir}/risk_by_industry.png")
        plt.close()
    
    # TOPSIS ranking plot
    if ranking_data is not None and 'TOPSIS_Score' in ranking_data.columns:
        create_topsis_ranking_plot(ranking_data, top_n=20, 
                                  save_path=f"{output_dir}/topsis_supplier_ranking.png")
        plt.close()
    
    print(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("Visualization Module")
    print("Import this module to use visualization functions in your pipeline.")

