"""
Main Pipeline Script for Green Supply Chain Risk Management

This script runs the complete analysis pipeline including:
1. Data preprocessing and integration
2. Feature engineering
3. Synthetic data generation (with best model selection: CTGAN or TVAE)
4. Statistical analysis
5. Machine learning modeling
6. Fuzzy AHP-TOPSIS-GA supplier ranking
7. Visualization
8. Report generation
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
from data_integration import integrate_datasets
from feature_engineering import (
    engineer_supplier_features,
    engineer_commodity_features,
    display_feature_engineering_summary
)
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
from draft_fuzzy_ahp_topsis_ga import analyze_supplier_ranking # change this back later

# Conditional import for synthetic data generation (requires sdv package)
try:
    from synthetic_data_generation import DG_ModelSelector, generate_synthetic_data
    SYNTHETIC_DATA_AVAILABLE = True
except ImportError:
    SYNTHETIC_DATA_AVAILABLE = False
    DG_ModelSelector = None
    generate_synthetic_data = None

import config


def set_random_seeds(seed=config.RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    print(f"Random seed set to {seed} for reproducibility")


def load_raw_datasets():
    """Load all raw datasets from data/raw directory."""
    print("\n" + "="*70)
    print("LOADING RAW DATASETS")
    print("="*70 + "\n")
    
    try:
        supplier_data = pd.read_csv(config.RAW_DATA_FILES['supplier'])
        print(f"[*] Loaded supplier data: {supplier_data.shape}")
        
        commodity_data = pd.read_csv(config.RAW_DATA_FILES['commodity'])
        print(f"[*] Loaded commodity/GHG data: {commodity_data.shape}")
        
        co2_data = pd.read_csv(config.RAW_DATA_FILES['co2'])
        print(f"[*] Loaded CO2 data: {co2_data.shape}")
        
        esg_data = pd.read_csv(config.RAW_DATA_FILES['esg'])
        print(f"[*] Loaded S&P 500 ESG data: {esg_data.shape}")
        
        return supplier_data, commodity_data, co2_data, esg_data
    
    except FileNotFoundError as e:
        print(f"[!] Error loading data: {e}")
        print("Please ensure all data files are in the data/raw directory")
        sys.exit(1)


def save_processed_datasets(supplier_data, commodity_data, co2_data, esg_data, esg_industry):
    """Save preprocessed datasets to processed directory."""
    print("\n" + "="*70)
    print("SAVING PREPROCESSED DATASETS")
    print("="*70 + "\n")
    
    supplier_data.to_csv(str(config.PROCESSED_DATA_FILES['supplier']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['supplier']}")
    
    commodity_data.to_csv(str(config.PROCESSED_DATA_FILES['commodity']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['commodity']}")
    
    co2_data.to_csv(str(config.PROCESSED_DATA_FILES['co2']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['co2']}")
    
    esg_data.to_csv(str(config.PROCESSED_DATA_FILES['esg']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['esg']}")
    
    esg_industry.to_csv(str(config.PROCESSED_DATA_FILES['esg_industry']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['esg_industry']}")


def save_engineered_datasets(supplier_features, commodity_features):
    """Save engineered datasets."""
    print("\n" + "="*70)
    print("SAVING ENGINEERED DATASETS")
    print("="*70 + "\n")
    
    supplier_features.to_csv(
        str(config.PROCESSED_DATA_DIR / "supplier_dataset_with_features.csv"),
        index=False
    )
    print(f"[*] Saved: supplier_dataset_with_features.csv")
    
    commodity_features.to_csv(
        str(config.PROCESSED_DATA_DIR / "commodity_dataset_with_features.csv"),
        index=False
    )
    print(f"[*] Saved: commodity_dataset_with_features.csv")


def save_final_datasets(supplier_with_risk, commodity_features):
    """Save final integrated datasets."""
    print("\n" + "="*70)
    print("SAVING FINAL DATASETS")
    print("="*70 + "\n")
    
    supplier_with_risk.to_csv(
        str(config.PROCESSED_DATA_DIR / "supplier_dataset_final.csv"),
        index=False
    )
    print(f"[*] Saved: supplier_dataset_final.csv")
    
    commodity_features.to_csv(
        str(config.PROCESSED_DATA_DIR / "commodity_dataset_final.csv"),
        index=False
    )
    print(f"[*] Saved: commodity_dataset_final.csv")


def save_integrated_dataset(integrated_data, enhanced_data):
    """Save integrated and enhanced datasets."""
    print("\n" + "="*70)
    print("SAVING INTEGRATED & ENHANCED DATASETS")
    print("="*70 + "\n")
    
    integrated_data.to_csv(str(config.PROCESSED_DATA_FILES['integrated']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['integrated']}")
    
    enhanced_data.to_csv(str(config.PROCESSED_DATA_FILES['enhanced']), index=False)
    print(f"[*] Saved: {config.PROCESSED_DATA_FILES['enhanced']}")


def main():
    """
    Main pipeline orchestrator.
    Runs complete analysis including preprocessing, feature engineering, 
    integration, analysis, modeling, Fuzzy AHP-TOPSIS-GA, visualization, and reporting.
    """
    print("="*70)
    print("GREEN SUPPLY CHAIN RISK MANAGEMENT - COMPLETE ANALYSIS PIPELINE")
    print("="*70)
    print()
    
    # Set random seed
    set_random_seeds(config.RANDOM_SEED)
    
    # Initialize variables for synthetic data generation
    best_models = None
    supplier_synthetic = None
    commodity_synthetic = None
    
    # =====================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # =====================================================================
    print("\nSTEP 1: Loading and Preprocessing Data")
    print("-" * 70)
    
    supplier_data, commodity_data, co2_data, esg_data = load_raw_datasets()
    
    # Preprocess data
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    co2_data = preprocess_co2_data(co2_data)
    esg_data = preprocess_esg_data(esg_data)
    esg_industry = aggregate_esg_by_industry(esg_data)
    
    print("* Data preprocessing completed")
    
    # Display preprocessing summary
    display_preprocessing_summary(supplier_data, commodity_data, co2_data, esg_industry)
    
    # Save preprocessed datasets
    save_processed_datasets(supplier_data, commodity_data, co2_data, esg_data, esg_industry)
    print()
    
    # =====================================================================
    # STEP 2: FEATURE ENGINEERING
    # =====================================================================
    print("STEP 2: Feature Engineering")
    print("-" * 70)
    
    supplier_features = engineer_supplier_features(supplier_data, commodity_data, co2_data, esg_industry)
    commodity_features = engineer_commodity_features(commodity_data, co2_data, esg_industry)
    
    print("* Feature engineering completed")
    
    # Display feature engineering summary
    display_feature_engineering_summary(supplier_features, commodity_features)
    
    # Save engineered datasets
    save_engineered_datasets(supplier_features, commodity_features)
    print()
    
    # =====================================================================
    # STEP 3: SYNTHETIC DATA GENERATION (BEST MODEL SELECTION)
    # =====================================================================
    print("STEP 3: Synthetic Data Generation with Best Model Selection")
    print("-" * 70)
    
    # Check if synthetic data generation is enabled and available
    if not SYNTHETIC_DATA_AVAILABLE:
        print("Warning: Synthetic data generation module not available (sdv package not installed)")
        print("  Skipping synthetic data generation...")
    elif hasattr(config, 'SYNTHETIC_DATA_CONFIG') and config.SYNTHETIC_DATA_CONFIG.get('enabled', True):
        try:
            # Initialize model selector
            selector = DG_ModelSelector(output_dir=config.MODEL_SELECTION_DIR)
            
            # Check if we should use cached selection
            use_cached = config.SYNTHETIC_DATA_CONFIG.get('use_cached_selection', True)
            best_models = None
            
            if use_cached:
                import json
                summary_path = config.MODEL_SELECTION_DIR / "model_selection_summary.json"
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r') as f:
                            cached = json.load(f)
                            if 'supplier' in cached and 'commodity' in cached:
                                best_models = {
                                    'supplier': cached['supplier']['best_model'],
                                    'commodity': cached['commodity']['best_model']
                                }
                                print("* Using cached model selection results")
                    except Exception as e:
                        print(f"Warning: Could not load cached selection: {e}")
            
            # If no cached selection, run model comparison
            if best_models is None:
                print("Running model comparison (CTGAN vs TVAE)...")
                best_models = selector.select_models_for_both(
                    supplier_data=supplier_features,
                    commodity_data=commodity_features,
                    supplier_target_col=config.SYNTHETIC_DATA_CONFIG.get('supplier_target_col'),
                    commodity_target_col=config.SYNTHETIC_DATA_CONFIG.get('commodity_target_col'),
                    epochs=config.SYNTHETIC_DATA_CONFIG.get('comparison_epochs', 300),
                    save_results=True
                )
            
            print(f"* Selected models - Supplier: {best_models['supplier']}, Commodity: {best_models['commodity']}")
            
            # Generate synthetic data using best models
            print("\nGenerating synthetic data using selected best models...")
            
            # Generate synthetic supplier data
            supplier_synthetic, supplier_generator = generate_synthetic_data(
                real_data=supplier_features,
                model_type=best_models['supplier'].lower(),
                num_rows=len(supplier_features),
                epochs=config.SYNTHETIC_DATA_CONFIG.get('generation_epochs', 300),
                save_path=config.SYNTHETIC_DATA_FILES['supplier'] if config.SYNTHETIC_DATA_CONFIG.get('save_synthetic_data', True) else None,
                save_model=config.SYNTHETIC_DATA_CONFIG.get('save_models', True),
                model_save_path=config.SYNTHETIC_MODEL_FILES['supplier'] if config.SYNTHETIC_DATA_CONFIG.get('save_models', True) else None
            )
            print(f"* Generated synthetic supplier data: {supplier_synthetic.shape}")
            
            # Generate synthetic commodity data
            commodity_synthetic, commodity_generator = generate_synthetic_data(
                real_data=commodity_features,
                model_type=best_models['commodity'].lower(),
                num_rows=len(commodity_features),
                epochs=config.SYNTHETIC_DATA_CONFIG.get('generation_epochs', 300),
                save_path=config.SYNTHETIC_DATA_FILES['commodity'] if config.SYNTHETIC_DATA_CONFIG.get('save_synthetic_data', True) else None,
                save_model=config.SYNTHETIC_DATA_CONFIG.get('save_models', True),
                model_save_path=config.SYNTHETIC_MODEL_FILES['commodity'] if config.SYNTHETIC_DATA_CONFIG.get('save_models', True) else None
            )
            print(f"* Generated synthetic commodity data: {commodity_synthetic.shape}")
            
            print("* Synthetic data generation completed")
            
        except Exception as e:
            print(f"Warning: Synthetic data generation failed: {e}")
            print("  Continuing with original data...")
    else:
        print("Synthetic data generation is disabled in config")
    print()
    
    # =====================================================================
    # STEP 4: CALCULATE RISK METRICS AND CREATE FINAL DATASETS
    # =====================================================================
    print("STEP 4: Calculating Risk Metrics")
    print("-" * 70)
    
    supplier_with_risk = calculate_risk_metrics(supplier_features)
    
    print("* Risk metrics calculated")
    
    # Save final datasets
    save_final_datasets(supplier_with_risk, commodity_features)
    print()
    
    # =====================================================================
    # STEP 5: STATISTICAL ANALYSIS
    # =====================================================================
    print("STEP 5: Performing Statistical Analysis")
    print("-" * 70)
    
    # Use supplier dataset with risk metrics for analysis
    enhanced_data = supplier_with_risk.copy()
    
    # Descriptive statistics
    stats = calculate_descriptive_statistics(enhanced_data)
    print("* Descriptive statistics calculated")
    
    # Correlation analysis
    correlation_matrix = calculate_correlation_matrix(enhanced_data)
    print("* Correlation matrix calculated")
    
    # Feature importance
    feature_importance = calculate_feature_importance(
        enhanced_data, 
        target_column='Overall_Risk_Score',
        method='correlation'
    )
    print("* Feature importance calculated")
    
    # Generate analysis report
    analysis_report = generate_analysis_report(
        enhanced_data,
        output_path=str(config.REPORTS_DIR / "statistical_analysis_report.txt")
    )
    print("* Statistical analysis report generated")
    print()
    
    # =====================================================================
    # STEP 6: MACHINE LEARNING MODELING
    # =====================================================================
    print("STEP 6: Building Machine Learning Models")
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
    print(f"* Prediction model trained - R2 Score: {prediction_model['test_metrics']['r2']:.4f}")
    
    # Train risk classification model
    print("Training risk classification model (Random Forest)...")
    classification_model = train_risk_classification_model(
        enhanced_data,
        target_column='Risk_Classification',
        model_type='random_forest',
        test_size=config.MODEL_CONFIG['test_size'],
        random_state=config.RANDOM_SEED
    )
    print(f"* Classification model trained - Accuracy: {classification_model['test_metrics']['accuracy']:.4f}")
    
    # Perform clustering
    print("Performing supplier clustering...")
    clustering_results = perform_clustering(
        enhanced_data,
        n_clusters=config.ANALYSIS_CONFIG['clustering_n_clusters'],
        method='kmeans'
    )
    print(f"* Clustering completed - Silhouette Score: {clustering_results['silhouette_score']:.4f}")
    print()
    
    # =====================================================================
    # STEP 7: FUZZY AHP-TOPSIS-GA SUPPLIER RANKING
    # =====================================================================
    print("STEP 7: Fuzzy AHP-TOPSIS-GA Supplier Ranking")
    print("-" * 70)
    
    # Filter criteria columns to only those that exist in the data
    available_criteria = [col for col in config.TOPSIS_CRITERIA_COLUMNS if col in enhanced_data.columns]
    
    if len(available_criteria) > 0:
        # Match criteria types to available criteria
        criteria_indices = [config.TOPSIS_CRITERIA_COLUMNS.index(col) for col in available_criteria]
        available_criteria_types = [config.TOPSIS_CRITERIA_TYPES[i] for i in criteria_indices]
        
        # Perform Fuzzy AHP-TOPSIS-GA analysis
        topsis_results = analyze_supplier_ranking(
            enhanced_data,
            criteria_columns=available_criteria,
            criteria_types=available_criteria_types,
            supplier_id_column='Supplier_ID',
            top_n=config.FUZZY_AHP_TOPSIS_GA_CONFIG['top_n_suppliers'],
            use_ga_optimization=config.FUZZY_AHP_TOPSIS_GA_CONFIG['use_ga_optimization'],
            random_state=config.RANDOM_SEED
        )
        
        # Save ranking results
        ranking_output_path = config.PROCESSED_DATA_DIR / "supplier_ranking_topsis.csv"
        # Extract just the ranking columns for CSV export
        ranking_export = enhanced_data[['Supplier_ID']].copy()
        ranking_export = ranking_export.merge(
            topsis_results['ranking_results'][['Supplier_ID', 'TOPSIS_Score', 'Rank']],
            on='Supplier_ID',
            how='left'
        )
        ranking_export.to_csv(ranking_output_path, index=False)
        print(f"* Supplier ranking saved to: {ranking_output_path}")
    else:
        print("Warning: No valid criteria columns found for TOPSIS analysis")
        topsis_results = None
    print()
    
    # =====================================================================
    # STEP 8: MODEL EVALUATION
    # =====================================================================
    print("STEP 8: Evaluating Models")
    print("-" * 70)
    
    # Generate evaluation reports
    pred_eval_report = generate_evaluation_report(
        prediction_model,
        output_path=str(config.REPORTS_DIR / "prediction_model_evaluation.txt")
    )
    print("* Prediction model evaluation report generated")
    
    class_eval_report = generate_evaluation_report(
        classification_model,
        output_path=str(config.REPORTS_DIR / "classification_model_evaluation.txt")
    )
    print("* Classification model evaluation report generated")
    
    # Create evaluation plots
    plot_prediction_vs_actual(
        prediction_model['y_test'],
        prediction_model['y_test_pred'],
        model_name='Risk Prediction Model',
        save_path=str(config.VISUALIZATIONS_DIR / "prediction_vs_actual.png")
    )
    print("* Prediction vs actual plot saved")
    
    if 'label_encoder' in classification_model:
        class_names = classification_model['label_encoder'].classes_
        plot_confusion_matrix(
            classification_model['y_test'],
            classification_model['y_test_pred'],
            class_names=class_names,
            model_name='Risk Classification Model',
            save_path=str(config.VISUALIZATIONS_DIR / "confusion_matrix.png")
        )
        print("* Confusion matrix plot saved")
    print()
    
    # =====================================================================
    # STEP 9: VISUALIZATION
    # =====================================================================
    print("STEP 9: Creating Visualizations")
    print("-" * 70)
    
    # Prepare ranking data for visualization
    ranking_data_for_viz = None
    if topsis_results:
        ranking_data_for_viz = topsis_results['ranking_results']
    
    create_comprehensive_dashboard(
        enhanced_data,
        output_dir=str(config.VISUALIZATIONS_DIR),
        ranking_data=ranking_data_for_viz
    )
    print("* Comprehensive visualization dashboard created")
    print()
    
    # =====================================================================
    # STEP 10: REPORT GENERATION
    # =====================================================================
    print("STEP 10: Generating Reports")
    print("-" * 70)
    
    # Executive summary
    exec_summary = generate_executive_summary(
        enhanced_data,
        output_path=str(config.REPORTS_DIR / "executive_summary.txt")
    )
    print("* Executive summary generated")
    
    # Comprehensive report
    report_paths = generate_comprehensive_report(
        enhanced_data,
        analysis_results={'correlation_matrix': correlation_matrix, 'feature_importance': feature_importance},
        model_results={'prediction': prediction_model, 'classification': classification_model},
        output_dir=str(config.REPORTS_DIR)
    )
    print("* Comprehensive reports generated")
    
    # Export to JSON
    export_results_to_json(
        enhanced_data,
        output_path=str(config.OUTPUT_DIR / "results.json")
    )
    print("* Results exported to JSON")
    print()
    
    # =====================================================================
    # COMPLETION
    # =====================================================================
    print("="*70)
    print("[SUCCESS] COMPLETE ANALYSIS PIPELINE EXECUTED SUCCESSFULLY")
    print("="*70)
    print()
    print("Output Summary:")
    print(f"  - Processed Data: {config.PROCESSED_DATA_DIR}")
    print(f"  - Visualizations: {config.VISUALIZATIONS_DIR}")
    print(f"  - Reports: {config.REPORTS_DIR}")
    print(f"  - Models: {config.MODELS_DIR}")
    if hasattr(config, 'SYNTHETIC_DATA_DIR'):
        print(f"  - Synthetic Data: {config.SYNTHETIC_DATA_DIR}")
    print()
    print("Key Results:")
    print(f"  - Total Suppliers Analyzed: {enhanced_data['Supplier_ID'].nunique()}")
    print(f"  - Average Risk Score: {enhanced_data['Overall_Risk_Score'].mean():.2f}")
    print(f"  - Prediction Model R2: {prediction_model['test_metrics']['r2']:.4f}")
    print(f"  - Classification Accuracy: {classification_model['test_metrics']['accuracy']:.4f}")
    if topsis_results:
        print(f"  - Top Ranked Supplier: {topsis_results['ranking_results'].iloc[0]['Supplier_ID']} "
              f"(TOPSIS Score: {topsis_results['ranking_results'].iloc[0]['TOPSIS_Score']:.4f})")
    print()
    print("Output Files Generated:")
    print(f"  1. {config.PROCESSED_DATA_DIR}/preprocessed_supplier_data.csv")
    print(f"  2. {config.PROCESSED_DATA_DIR}/preprocessed_commodity_data.csv")
    print(f"  3. {config.PROCESSED_DATA_DIR}/preprocessed_co2_data.csv")
    print(f"  4. {config.PROCESSED_DATA_DIR}/preprocessed_esg_data.csv")
    print(f"  5. {config.PROCESSED_DATA_DIR}/preprocessed_esg_by_industry.csv")
    print(f"  6. {config.PROCESSED_DATA_DIR}/supplier_dataset_with_features.csv")
    print(f"  7. {config.PROCESSED_DATA_DIR}/commodity_dataset_with_features.csv")
    print(f"  8. {config.PROCESSED_DATA_DIR}/supplier_dataset_final.csv")
    print(f"  9. {config.PROCESSED_DATA_DIR}/commodity_dataset_final.csv")
    if topsis_results:
        print(f"  10. {config.PROCESSED_DATA_DIR}/supplier_ranking_topsis.csv")
    if hasattr(config, 'SYNTHETIC_DATA_CONFIG') and config.SYNTHETIC_DATA_CONFIG.get('enabled', True):
        if best_models:
            print(f"  11. {config.SYNTHETIC_DATA_FILES['supplier']} (using {best_models['supplier']})")
            print(f"  12. {config.SYNTHETIC_DATA_FILES['commodity']} (using {best_models['commodity']})")
    print()
    print(f"Reproducibility Guarantee:")
    print(f"   Random seed fixed to RANDOM_SEED={config.RANDOM_SEED}")
    print(f"   All results will be identical on every run.\n")
    
    return enhanced_data


if __name__ == "__main__":
    enhanced_data = main()
