"""
Reporting Module for Green Supply Chain Risk Management

This module provides functions for generating comprehensive reports,
summaries, and documentation of the analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def generate_executive_summary(data, output_path=None):
    """
    Generate executive summary report.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Integrated dataset with risk metrics
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Executive summary text
    """
    summary = []
    summary.append("="*70)
    summary.append("EXECUTIVE SUMMARY - GREEN SUPPLY CHAIN RISK MANAGEMENT")
    summary.append("="*70)
    summary.append("")
    summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Dataset overview
    summary.append("Dataset Overview:")
    summary.append(f"  Total Suppliers Analyzed: {data['Supplier_ID'].nunique() if 'Supplier_ID' in data.columns else len(data)}")
    summary.append(f"  Total Records: {len(data)}")
    summary.append(f"  Industries Covered: {data['Industry_Sector'].nunique() if 'Industry_Sector' in data.columns else 'N/A'}")
    summary.append(f"  Countries Represented: {data['Country'].nunique() if 'Country' in data.columns else 'N/A'}")
    summary.append("")
    
    # Risk overview
    if 'Overall_Risk_Score' in data.columns:
        summary.append("Risk Score Overview:")
        summary.append(f"  Average Risk Score: {data['Overall_Risk_Score'].mean():.2f}")
        summary.append(f"  Median Risk Score: {data['Overall_Risk_Score'].median():.2f}")
        summary.append(f"  Minimum Risk Score: {data['Overall_Risk_Score'].min():.2f}")
        summary.append(f"  Maximum Risk Score: {data['Overall_Risk_Score'].max():.2f}")
        summary.append(f"  Standard Deviation: {data['Overall_Risk_Score'].std():.2f}")
        summary.append("")
    
    # Risk classification distribution
    if 'Risk_Classification' in data.columns:
        summary.append("Risk Classification Distribution:")
        risk_dist = data['Risk_Classification'].value_counts()
        for classification, count in risk_dist.items():
            pct = (count / len(data)) * 100
            summary.append(f"  {classification}: {count} suppliers ({pct:.1f}%)")
        summary.append("")
    
    # Top risk areas
    if 'Overall_Risk_Score' in data.columns:
        summary.append("Key Findings:")
        high_risk_count = len(data[data['Overall_Risk_Score'] >= 60]) if 'Overall_Risk_Score' in data.columns else 0
        summary.append(f"  - {high_risk_count} suppliers identified as high or critical risk")
        
        if 'Country' in data.columns:
            top_risk_country = data.groupby('Country')['Overall_Risk_Score'].mean().idxmax()
            summary.append(f"  - Highest average risk by country: {top_risk_country}")
        
        if 'Industry_Sector' in data.columns:
            top_risk_industry = data.groupby('Industry_Sector')['Overall_Risk_Score'].mean().idxmax()
            summary.append(f"  - Highest average risk by industry: {top_risk_industry}")
        summary.append("")
    
    # Recommendations
    summary.append("Recommendations:")
    summary.append("  1. Prioritize risk mitigation for suppliers with Critical or High risk classifications")
    summary.append("  2. Conduct detailed audits for suppliers in high-risk countries or industries")
    summary.append("  3. Implement continuous monitoring for medium-risk suppliers")
    summary.append("  4. Develop improvement plans for suppliers with low ESG scores")
    summary.append("  5. Strengthen compliance requirements for suppliers with high compliance risk")
    summary.append("")
    
    summary.append("="*70)
    
    summary_text = "\n".join(summary)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary_text)
        print(f"Executive summary saved to: {output_path}")
    
    return summary_text


def generate_supplier_risk_report(data, supplier_id=None, output_path=None):
    """
    Generate detailed risk report for a specific supplier or all suppliers.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with risk metrics
    supplier_id : str, optional
        Specific supplier ID. If None, generates report for all suppliers.
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Supplier risk report text
    """
    if supplier_id:
        supplier_data = data[data['Supplier_ID'] == supplier_id]
        if len(supplier_data) == 0:
            return f"Supplier {supplier_id} not found in dataset."
    else:
        supplier_data = data
    
    report = []
    report.append("="*70)
    report.append("SUPPLIER RISK ASSESSMENT REPORT")
    report.append("="*70)
    report.append("")
    report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if supplier_id:
        report.append(f"Supplier ID: {supplier_id}")
        if 'Supplier_Name' in supplier_data.columns:
            report.append(f"Supplier Name: {supplier_data['Supplier_Name'].iloc[0]}")
        report.append("")
    
    # Risk scores
    risk_columns = ['Environmental_Risk_Score', 'Compliance_Risk_Score',
                   'Operational_Risk_Score', 'Financial_Risk_Score', 'Overall_Risk_Score']
    
    available_risk_cols = [col for col in risk_columns if col in supplier_data.columns]
    
    if available_risk_cols:
        report.append("Risk Scores:")
        for col in available_risk_cols:
            if supplier_id:
                value = supplier_data[col].iloc[0]
                report.append(f"  {col}: {value:.2f}")
            else:
                mean_val = supplier_data[col].mean()
                report.append(f"  Average {col}: {mean_val:.2f}")
        report.append("")
    
    # Risk classification
    if 'Risk_Classification' in supplier_data.columns:
        if supplier_id:
            classification = supplier_data['Risk_Classification'].iloc[0]
            report.append(f"Risk Classification: {classification}")
        else:
            report.append("Risk Classification Distribution:")
            classification_dist = supplier_data['Risk_Classification'].value_counts()
            for cls, count in classification_dist.items():
                pct = (count / len(supplier_data)) * 100
                report.append(f"  {cls}: {count} ({pct:.1f}%)")
        report.append("")
    
    # Supplier details
    if supplier_id and len(supplier_data) > 0:
        detail_cols = ['Country', 'Region', 'Industry_Sector', 'Supplier_Tier']
        available_detail_cols = [col for col in detail_cols if col in supplier_data.columns]
        
        if available_detail_cols:
            report.append("Supplier Details:")
            for col in available_detail_cols:
                value = supplier_data[col].iloc[0]
                report.append(f"  {col}: {value}")
            report.append("")
    
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Supplier risk report saved to: {output_path}")
    
    return report_text


def generate_comprehensive_report(data, analysis_results=None, model_results=None, 
                                 output_dir='outputs/reports'):
    """
    Generate comprehensive analysis report combining all results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Integrated dataset
    analysis_results : dict, optional
        Statistical analysis results
    model_results : dict, optional
        Machine learning model results
    output_dir : str
        Directory to save reports
        
    Returns:
    --------
    dict
        Paths to generated reports
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # Executive summary
    exec_summary_path = f"{output_dir}/executive_summary.txt"
    generate_executive_summary(data, exec_summary_path)
    report_paths['executive_summary'] = exec_summary_path
    
    # Supplier risk report
    supplier_report_path = f"{output_dir}/supplier_risk_report.txt"
    generate_supplier_risk_report(data, output_path=supplier_report_path)
    report_paths['supplier_risk_report'] = supplier_report_path
    
    # Analysis summary
    if analysis_results:
        analysis_path = f"{output_dir}/analysis_summary.txt"
        with open(analysis_path, 'w') as f:
            f.write(str(analysis_results))
        report_paths['analysis_summary'] = analysis_path
    
    # Model evaluation report
    if model_results:
        model_path = f"{output_dir}/model_evaluation.txt"
        with open(model_path, 'w') as f:
            f.write(str(model_results))
        report_paths['model_evaluation'] = model_path
    
    # Export data summary to CSV
    if 'Overall_Risk_Score' in data.columns:
        summary_df = data.groupby('Risk_Classification').agg({
            'Overall_Risk_Score': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        summary_path = f"{output_dir}/risk_summary_statistics.csv"
        summary_df.to_csv(summary_path)
        report_paths['summary_statistics'] = summary_path
    
    print(f"All reports generated and saved to: {output_dir}")
    
    return report_paths


def export_results_to_json(data, output_path='outputs/results.json'):
    """
    Export analysis results to JSON format.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to export
    output_path : str
        Path to save JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(data),
            'total_suppliers': data['Supplier_ID'].nunique() if 'Supplier_ID' in data.columns else len(data)
        },
        'summary_statistics': {}
    }
    
    # Add risk score statistics
    if 'Overall_Risk_Score' in data.columns:
        results['summary_statistics']['overall_risk'] = {
            'mean': float(data['Overall_Risk_Score'].mean()),
            'median': float(data['Overall_Risk_Score'].median()),
            'std': float(data['Overall_Risk_Score'].std()),
            'min': float(data['Overall_Risk_Score'].min()),
            'max': float(data['Overall_Risk_Score'].max())
        }
    
    # Add risk classification distribution
    if 'Risk_Classification' in data.columns:
        risk_dist = data['Risk_Classification'].value_counts().to_dict()
        results['summary_statistics']['risk_classification'] = {
            k: int(v) for k, v in risk_dist.items()
        }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results exported to JSON: {output_path}")


if __name__ == "__main__":
    print("Reporting Module")
    print("Import this module to use reporting functions in your pipeline.")

