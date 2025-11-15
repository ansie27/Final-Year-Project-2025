"""
Full Analysis Pipeline Script

This script runs the complete analysis pipeline including:
1. Data preprocessing and integration
2. Statistical analysis
3. Machine learning modeling
4. Visualization
5. Report generation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

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
from analysis import (
    calculate_descriptive_statistics,
    calculate_correlation_matrix,
    calculate_feature_importance,
    generate_analysis_report
)
from visualization import create_comprehensive_dashboard
from modeling import (
    train_risk_prediction_model,
    train_risk_classification_model,
    perform_clustering
)
from evaluation import (
    evaluate_regression_model,
    generate_evaluation_report,
    plot_prediction_vs_actual,
    plot_confusion_matrix
)
from reporting import (
    generate_executive_summary,
    generate_comprehensive_report,
    export_results_to_json
)
import config


def main():
    """
    Run the complete analysis pipeline.
    """
    print("="*70)
    print("GREEN SUPPLY CHAIN RISK MANAGEMENT - FULL ANALYSIS PIPELINE")
    print("="*70)
    print()
    
    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    
    # =====================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # =====================================================================
    print("STEP 1: Loading and Preprocessing Data")
    print("-" * 70)
    
    try:
        supplier_data = pd.read_csv(config.RAW_DATA_FILES['supplier'])
        commodity_data = pd.read_csv(config.RAW_DATA_FILES['commodity'])
        co2_data = pd.read_csv(config.RAW_DATA_FILES['co2'])
        esg_data = pd.read_csv(config.RAW_DATA_FILES['esg'])
        
        print("✓ Raw data loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error loading data: {e}")
        print("Please ensure all data files are in the data/raw directory")
        return
    
    # Preprocess data
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    esg_data = preprocess_esg_data(esg_data)
    esg_industry = aggregate_esg_by_industry(esg_data)
    
    print("✓ Data preprocessing completed")
    print()
    
    # =====================================================================
    # STEP 2: INTEGRATE DATASETS
    # =====================================================================
    print("STEP 2: Integrating Datasets")
    print("-" * 70)
    
    integrated_data = integrate_datasets(
        supplier_data, commodity_data, co2_data, esg_industry
    )
    
    # Calculate risk metrics
    enhanced_data = calculate_risk_metrics(integrated_data)
    
    print("✓ Dataset integration and risk calculation completed")
    print()
    
    # =====================================================================
    # STEP 3: STATISTICAL ANALYSIS
    # =====================================================================
    print("STEP 3: Performing Statistical Analysis")
    print("-" * 70)
    
    # Descriptive statistics
    stats = calculate_descriptive_statistics(enhanced_data)
    print("✓ Descriptive statistics calculated")
    
    # Correlation analysis
    correlation_matrix = calculate_correlation_matrix(enhanced_data)
    print("✓ Correlation matrix calculated")
    
    # Feature importance
    feature_importance = calculate_feature_importance(
        enhanced_data, 
        target_column='Overall_Risk_Score',
        method='correlation'
    )
    print("✓ Feature importance calculated")
    
    # Generate analysis report
    analysis_report = generate_analysis_report(
        enhanced_data,
        output_path=str(config.REPORTS_DIR / "statistical_analysis_report.txt")
    )
    print("✓ Statistical analysis report generated")
    print()
    
    # =====================================================================
    # STEP 4: MACHINE LEARNING MODELING
    # =====================================================================
    print("STEP 4: Building Machine Learning Models")
    print("-" * 70)
    
    # Train risk prediction model
    print("Training risk prediction model (Random Forest)...")
    prediction_model = train_risk_prediction_model(
        enhanced_data,
        target_column='Overall_Risk_Score',
        model_type='random_forest',
        test_size=config.MODEL_CONFIG['test_size'],
        random_state=config.RANDOM_SEED
    )
    print(f"✓ Prediction model trained - R² Score: {prediction_model['test_metrics']['r2']:.4f}")
    
    # Train risk classification model
    print("Training risk classification model (Random Forest)...")
    classification_model = train_risk_classification_model(
        enhanced_data,
        target_column='Risk_Classification',
        model_type='random_forest',
        test_size=config.MODEL_CONFIG['test_size'],
        random_state=config.RANDOM_SEED
    )
    print(f"✓ Classification model trained - Accuracy: {classification_model['test_metrics']['accuracy']:.4f}")
    
    # Perform clustering
    print("Performing supplier clustering...")
    clustering_results = perform_clustering(
        enhanced_data,
        n_clusters=config.ANALYSIS_CONFIG['clustering_n_clusters'],
        method='kmeans'
    )
    print(f"✓ Clustering completed - Silhouette Score: {clustering_results['silhouette_score']:.4f}")
    print()
    
    # =====================================================================
    # STEP 5: MODEL EVALUATION
    # =====================================================================
    print("STEP 5: Evaluating Models")
    print("-" * 70)
    
    # Generate evaluation reports
    pred_eval_report = generate_evaluation_report(
        prediction_model,
        output_path=str(config.REPORTS_DIR / "prediction_model_evaluation.txt")
    )
    print("✓ Prediction model evaluation report generated")
    
    class_eval_report = generate_evaluation_report(
        classification_model,
        output_path=str(config.REPORTS_DIR / "classification_model_evaluation.txt")
    )
    print("✓ Classification model evaluation report generated")
    
    # Create evaluation plots
    plot_prediction_vs_actual(
        prediction_model['y_test'],
        prediction_model['y_test_pred'],
        model_name='Risk Prediction Model',
        save_path=str(config.VISUALIZATIONS_DIR / "prediction_vs_actual.png")
    )
    print("✓ Prediction vs actual plot saved")
    
    if 'label_encoder' in classification_model:
        class_names = classification_model['label_encoder'].classes_
        plot_confusion_matrix(
            classification_model['y_test'],
            classification_model['y_test_pred'],
            class_names=class_names,
            model_name='Risk Classification Model',
            save_path=str(config.VISUALIZATIONS_DIR / "confusion_matrix.png")
        )
        print("✓ Confusion matrix plot saved")
    print()
    
    # =====================================================================
    # STEP 6: VISUALIZATION
    # =====================================================================
    print("STEP 6: Creating Visualizations")
    print("-" * 70)
    
    create_comprehensive_dashboard(
        enhanced_data,
        output_dir=str(config.VISUALIZATIONS_DIR)
    )
    print("✓ Comprehensive visualization dashboard created")
    print()
    
    # =====================================================================
    # STEP 7: REPORT GENERATION
    # =====================================================================
    print("STEP 7: Generating Reports")
    print("-" * 70)
    
    # Executive summary
    exec_summary = generate_executive_summary(
        enhanced_data,
        output_path=str(config.REPORTS_DIR / "executive_summary.txt")
    )
    print("✓ Executive summary generated")
    
    # Comprehensive report
    report_paths = generate_comprehensive_report(
        enhanced_data,
        analysis_results={'correlation_matrix': correlation_matrix, 'feature_importance': feature_importance},
        model_results={'prediction': prediction_model, 'classification': classification_model},
        output_dir=str(config.REPORTS_DIR)
    )
    print("✓ Comprehensive reports generated")
    
    # Export to JSON
    export_results_to_json(
        enhanced_data,
        output_path=str(config.OUTPUT_DIR / "results.json")
    )
    print("✓ Results exported to JSON")
    print()
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    print("="*70)
    print("✓ FULL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print()
    print("Output Summary:")
    print(f"  - Processed Data: {config.PROCESSED_DATA_DIR}")
    print(f"  - Visualizations: {config.VISUALIZATIONS_DIR}")
    print(f"  - Reports: {config.REPORTS_DIR}")
    print(f"  - Models: {config.MODELS_DIR}")
    print()
    print("Key Results:")
    print(f"  - Total Suppliers Analyzed: {enhanced_data['Supplier_ID'].nunique()}")
    print(f"  - Average Risk Score: {enhanced_data['Overall_Risk_Score'].mean():.2f}")
    print(f"  - Prediction Model R²: {prediction_model['test_metrics']['r2']:.4f}")
    print(f"  - Classification Accuracy: {classification_model['test_metrics']['accuracy']:.4f}")
    print()


if __name__ == "__main__":
    main()

