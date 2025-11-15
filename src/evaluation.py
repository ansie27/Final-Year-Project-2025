"""
Model Evaluation Module for Green Supply Chain Risk Management

This module provides functions for evaluating model performance,
validation, and generating evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def evaluate_regression_model(y_true, y_pred, model_name='Model'):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }
    
    return metrics


def evaluate_classification_model(y_true, y_pred, y_proba=None, model_name='Model'):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target labels
    y_pred : array-like
        Predicted target labels
    y_proba : array-like, optional
        Predicted probabilities
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
        except:
            metrics['roc_auc'] = None
    
    return metrics


def compare_models(model_results_list):
    """
    Compare multiple model results.
    
    Parameters:
    -----------
    model_results_list : list
        List of model result dictionaries
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for result in model_results_list:
        if 'test_metrics' in result:
            metrics = result['test_metrics'].copy()
            if 'model_name' not in metrics:
                metrics['model_name'] = result.get('model_name', 'Unknown')
            comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def plot_prediction_vs_actual(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, model_name='Model', save_path=None):
    """
    Plot confusion matrix for classification model.
    
    Parameters:
    -----------
    y_true : array-like
        True target labels
    y_pred : array-like
        Predicted target labels
    class_names : list, optional
        Names of classes
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name}: Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def generate_evaluation_report(model_results, output_path=None):
    """
    Generate comprehensive evaluation report.
    
    Parameters:
    -----------
    model_results : dict
        Model evaluation results
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Evaluation report text
    """
    report = []
    report.append("="*70)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*70)
    report.append("")
    
    # Model information
    if 'model_name' in model_results:
        report.append(f"Model: {model_results['model_name']}")
        report.append("")
    
    # Training metrics
    if 'train_metrics' in model_results:
        report.append("Training Metrics:")
        for metric, value in model_results['train_metrics'].items():
            if isinstance(value, float):
                report.append(f"  {metric.upper()}: {value:.4f}")
            else:
                report.append(f"  {metric.upper()}: {value}")
        report.append("")
    
    # Test metrics
    if 'test_metrics' in model_results:
        report.append("Test Metrics:")
        for metric, value in model_results['test_metrics'].items():
            if isinstance(value, float):
                report.append(f"  {metric.upper()}: {value:.4f}")
            else:
                report.append(f"  {metric.upper()}: {value}")
        report.append("")
    
    # Feature importance
    if 'feature_importance' is not None and model_results.get('feature_importance') is not None:
        report.append("Top 10 Most Important Features:")
        top_features = model_results['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            report.append(f"  {row['Feature']}: {row['Importance']:.4f}")
        report.append("")
    
    # Classification report
    if 'classification_report' in model_results:
        report.append("Classification Report:")
        report.append(model_results['classification_report'])
        report.append("")
    
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Evaluation report saved to: {output_path}")
    
    return report_text


def calculate_model_confidence_intervals(y_true, y_pred, confidence=0.95):
    """
    Calculate confidence intervals for model predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    confidence : float
        Confidence level (default: 0.95)
        
    Returns:
    --------
    dict
        Confidence interval statistics
    """
    residuals = y_true - y_pred
    std_error = np.std(residuals)
    n = len(residuals)
    
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_error / np.sqrt(n)
    
    mean_error = np.mean(residuals)
    
    intervals = {
        'mean_error': mean_error,
        'std_error': std_error,
        'margin_error': margin_error,
        'confidence_level': confidence,
        'lower_bound': mean_error - margin_error,
        'upper_bound': mean_error + margin_error
    }
    
    return intervals


if __name__ == "__main__":
    print("Evaluation Module")
    print("Import this module to use evaluation functions in your pipeline.")

