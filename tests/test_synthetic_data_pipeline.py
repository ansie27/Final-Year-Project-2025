import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.synthetic_data_generation.dg_models import CTGANSyntheticDataGenerator, TVAESyntheticDataGenerator
from src.synthetic_data_generation.dg_evaluation import SyntheticDataEvaluator
from src.synthetic_data_generation.dg_model_selector import SyntheticDataModelSelector
from src.synthetic_data_generation.final_dg_model import SyntheticDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_synthetic_data_pipeline(
    supplier_data=None,
    commodity_data=None,
    output_dir='outputs/test_run',
    epochs=50,  # Reduced for testing
    evaluation_sizes=None,  # Smaller sizes for testing
    final_generation_size=500,  # Smaller final size for testing
    supplier_path="data/processed/integrated_supplier_dataset.csv",
    commodity_path="data/processed/integrated_commodity_dataset.csv"
):
    """
    Test the complete synthetic data generation pipeline.
    
    Parameters
    ----------
    supplier_data : pd.DataFrame, optional
        Real supplier data. If None, will load from supplier_path.
    commodity_data : pd.DataFrame, optional
        Real commodity data. If None, will load from commodity_path.
    output_dir : str
        Output directory for results
    epochs : int
        Training epochs (use smaller value for testing)
    evaluation_sizes : list, optional
        Sizes for evaluation phase. Default: [100, 200]
    final_generation_size : int
        Final generation size
    supplier_path : str
        Path to supplier CSV file (used if supplier_data is None)
    commodity_path : str
        Path to commodity CSV file (used if commodity_data is None)
    """
    
    if evaluation_sizes is None:
        evaluation_sizes = [100, 200]
    
    print("="*80)
    print("SYNTHETIC DATA GENERATION PIPELINE TEST".center(80))
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare data
    print("\n[Step 1/6] Preparing Data...")
    print("-"*80)
    
    # Load data if not provided
    if supplier_data is None:
        print(f"Loading supplier data from {supplier_path}...")
        supplier_data = pd.read_csv(supplier_path)
    
    if commodity_data is None:
        print(f"Loading commodity data from {commodity_path}...")
        commodity_data = pd.read_csv(commodity_path)
    
    print(f"✓ Supplier data: {supplier_data.shape}")
    print(f"✓ Commodity data: {commodity_data.shape}")
    
    # Save real data for reference
    real_supplier_path = output_path / 'real_supplier_data.csv'
    real_commodity_path = output_path / 'real_commodity_data.csv'
    
    supplier_data.to_csv(real_supplier_path, index=False)
    commodity_data.to_csv(real_commodity_path, index=False)
    
    print(f"✓ Real data saved to {output_path}")
    
    # Step 2: Import modules
    print("\n[Step 2/6] Modules Already Imported...")
    print("-"*80)
    print("✓ All modules imported successfully")
    
    # Step 3: Train CTGAN
    print("\n[Step 3/6] Training CTGAN Models...")
    print("-"*80)
    print(f"Training CTGAN on supplier data ({supplier_data.shape[0]} rows)...")
    
    ctgan_supplier = CTGANSyntheticDataGenerator(epochs=epochs, random_state=42)
    ctgan_supplier.fit(supplier_data)
    
    ctgan_commodity = CTGANSyntheticDataGenerator(epochs=epochs, random_state=42)
    ctgan_commodity.fit(commodity_data)
    
    print("✓ CTGAN models trained")
    
    # Step 4: Train TVAE
    print("\n[Step 4/6] Training TVAE Models...")
    print("-"*80)
    
    tvae_supplier = TVAESyntheticDataGenerator(epochs=epochs, random_state=42)
    tvae_supplier.fit(supplier_data)
    
    tvae_commodity = TVAESyntheticDataGenerator(epochs=epochs, random_state=42)
    tvae_commodity.fit(commodity_data)
    
    print("✓ TVAE models trained")
    
    # Step 5: Generate evaluation data
    print("\n[Step 5/6] Generating Synthetic Data for Evaluation...")
    print("-"*80)
    
    ctgan_supplier_synthetic = {}
    tvae_supplier_synthetic = {}
    ctgan_commodity_synthetic = {}
    tvae_commodity_synthetic = {}
    
    for size in evaluation_sizes:
        print(f"  Generating {size} rows...")
        ctgan_supplier_synthetic[size] = ctgan_supplier.generate(size)
        tvae_supplier_synthetic[size] = tvae_supplier.generate(size)
        ctgan_commodity_synthetic[size] = ctgan_commodity.generate(size)
        tvae_commodity_synthetic[size] = tvae_commodity.generate(size)
    
    print("✓ Evaluation data generated")
    
    # Step 6: Evaluate models
    print("\n[Step 6/6] Evaluating and Selecting Best Model...")
    print("-"*80)
    
    evaluator = SyntheticDataEvaluator(random_state=42)
    
    # Evaluate CTGAN
    print("\nEvaluating CTGAN...")
    ctgan_supplier_eval = evaluator.evaluate_single_dataset(
        supplier_data,
        ctgan_supplier_synthetic,
        'CTGAN_Supplier',
        target_col='risk_level'
    )
    
    ctgan_commodity_eval = evaluator.evaluate_single_dataset(
        commodity_data,
        ctgan_commodity_synthetic,
        'CTGAN_Commodity',
        target_col='sustainability_rating'
    )
    
    # Evaluate TVAE
    print("\nEvaluating TVAE...")
    tvae_supplier_eval = evaluator.evaluate_single_dataset(
        supplier_data,
        tvae_supplier_synthetic,
        'TVAE_Supplier',
        target_col='risk_level'
    )
    
    tvae_commodity_eval = evaluator.evaluate_single_dataset(
        commodity_data,
        tvae_commodity_synthetic,
        'TVAE_Commodity',
        target_col='sustainability_rating'
    )
    
    # Combine results
    ctgan_combined = pd.concat([ctgan_supplier_eval, ctgan_commodity_eval], ignore_index=True)
    tvae_combined = pd.concat([tvae_supplier_eval, tvae_commodity_eval], ignore_index=True)
    
    # Select best model
    print("\nSelecting best model...")
    selector = SyntheticDataModelSelector(output_dir=output_path)
    
    best_model, analysis = selector.select_best_model(
        ctgan_combined,
        tvae_combined,
        ctgan_supplier_eval,
        ctgan_commodity_eval,
        save_results=True
    )
    
    # Generate final data with best model
    print(f"\n[Final Generation] Using {best_model} to generate {final_generation_size} rows...")
    print("-"*80)
    
    if best_model == 'CTGAN':
        final_supplier = ctgan_supplier.generate(final_generation_size)
        final_commodity = ctgan_commodity.generate(final_generation_size)
    else:
        final_supplier = tvae_supplier.generate(final_generation_size)
        final_commodity = tvae_commodity.generate(final_generation_size)
    
    # Save final results
    final_supplier_path = output_path / f'final_synthetic_supplier_{best_model.lower()}.csv'
    final_commodity_path = output_path / f'final_synthetic_commodity_{best_model.lower()}.csv'
    
    final_supplier.to_csv(final_supplier_path, index=False)
    final_commodity.to_csv(final_commodity_path, index=False)
    
    print(f"✓ Final supplier data saved: {final_supplier_path}")
    print(f"  Shape: {final_supplier.shape}")
    print(f"✓ Final commodity data saved: {final_commodity_path}")
    print(f"  Shape: {final_commodity.shape}")
    
    # Save evaluation results
    eval_output_dir = output_path / 'evaluations'
    eval_output_dir.mkdir(exist_ok=True)
    
    ctgan_combined.to_csv(eval_output_dir / 'ctgan_evaluation.csv', index=False)
    tvae_combined.to_csv(eval_output_dir / 'tvae_evaluation.csv', index=False)
    
    print(f"\n✓ Evaluation results saved to: {eval_output_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY".center(80))
    print("="*80)
    print(f"\nSelected Model: {best_model}")
    print(f"Overall Score: {analysis['overall_score']:.4f}")
    print(f"\nOutput Directory: {output_path}")
    print(f"  - Real data: real_supplier_data.csv, real_commodity_data.csv")
    print(f"  - Final synthetic data: final_synthetic_supplier_{best_model.lower()}.csv, final_synthetic_commodity_{best_model.lower()}.csv")
    print(f"  - Evaluations: evaluations/")
    print(f"  - Selection results: model_selection_results.json")
    print("="*80 + "\n")
    
    return {
        'best_model': best_model,
        'analysis': analysis,
        'final_supplier': final_supplier,
        'final_commodity': final_commodity,
        'ctgan_evaluation': ctgan_combined,
        'tvae_evaluation': tvae_combined,
        'output_dir': str(output_path)
    }


# Simple one-liner for Colab
def run_quick_test():
    """Quick test with minimal parameters (fast execution)."""
    return test_synthetic_data_pipeline(
        epochs=20,  # Very fast for testing
        evaluation_sizes=[50, 100],
        final_generation_size=200
    )


# Full test with your actual data
def run_with_real_data(supplier_csv_path, commodity_csv_path, output_dir='outputs/full_run'):
    """Run pipeline with your actual datasets."""
    print(f"Loading data from CSV files...")
    supplier_data = pd.read_csv(supplier_csv_path)
    commodity_data = pd.read_csv(commodity_csv_path)
    
    print(f"Starting pipeline with {supplier_data.shape[0]} supplier and {commodity_data.shape[0]} commodity records...")
    
    return test_synthetic_data_pipeline(
        supplier_data=supplier_data,
        commodity_data=commodity_data,
        output_dir=output_dir,
        epochs=50,  # Reduced for testing
        evaluation_sizes=[100, 200],  # Smaller for testing
        final_generation_size=500  # Smaller for testing
    )

if __name__ == "__main__":
    # Run quick test
    print("Running quick test...")
    results = run_with_real_data("data/processed/integrated_supplier_dataset.csv", "data/processed/integrated_commodity_dataset.csv")
    
    if results:
        print(f"\n✓ Test completed successfully!")
        print(f"Best model: {results['best_model']}")
