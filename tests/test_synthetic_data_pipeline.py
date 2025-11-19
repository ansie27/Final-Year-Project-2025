import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_sample_data():
    """Create sample supplier and commodity datasets for testing."""
    np.random.seed(42)
    
    # Sample Supplier Data
    supplier_data = pd.DataFrame({
        'supplier_id': range(1, 101),
        'cost': np.random.uniform(100, 1000, 100),
        'quality_score': np.random.uniform(0, 100, 100),
        'delivery_time': np.random.randint(1, 30, 100),
        'co2_emissions': np.random.uniform(10, 500, 100),
        'esg_score': np.random.uniform(0, 100, 100),
        'certification': np.random.choice(['ISO9001', 'ISO14001', 'None'], 100),
        'risk_level': np.random.choice(['Low', 'Medium', 'High'], 100)
    })
    
    # Sample Commodity Data
    commodity_data = pd.DataFrame({
        'commodity_id': range(1, 101),
        'naics_code': np.random.randint(100000, 999999, 100),
        'ghg_emissions': np.random.uniform(0.001, 0.1, 100),
        'price_per_unit': np.random.uniform(10, 500, 100),
        'supply_chain_risk': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['Manufacturing', 'Agriculture', 'Mining', 'Services'], 100),
        'sustainability_rating': np.random.choice(['A', 'B', 'C', 'D'], 100)
    })
    
    return supplier_data, commodity_data


def test_synthetic_data_pipeline(
    supplier_data=None,
    commodity_data=None,
    output_dir='outputs/test_run',
    epochs=50,  # Reduced for testing
    evaluation_sizes=[100, 200],  # Smaller sizes for testing
    final_generation_size=500  # Smaller final size for testing
):
    """
    Test the complete synthetic data generation pipeline.
    
    Parameters
    ----------
    supplier_data : pd.DataFrame, optional
        Real supplier data. If None, sample data will be created.
    commodity_data : pd.DataFrame, optional
        Real commodity data. If None, sample data will be created.
    output_dir : str
        Output directory for results
    epochs : int
        Training epochs (use smaller value for testing)
    evaluation_sizes : list
        Sizes for evaluation phase
    final_generation_size : int
        Final generation size
    """
    
    print("="*80)
    print("SYNTHETIC DATA GENERATION PIPELINE TEST".center(80))
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare data
    print("\n[Step 1/6] Preparing Data...")
    print("-"*80)
    
    if supplier_data is None or commodity_data is None:
        print("Creating sample datasets...")
        supplier_data, commodity_data = create_sample_data()
    
    # Save real data
    supplier_path = output_path / 'real_supplier_data.csv'
    commodity_path = output_path / 'real_commodity_data.csv'
    
    supplier_data.to_csv(supplier_path, index=False)
    commodity_data.to_csv(commodity_path, index=False)
    
    print(f"✓ Supplier data: {supplier_data.shape}")
    print(f"✓ Commodity data: {commodity_data.shape}")
    
    # Step 2: Import modules (after data is ready)
    print("\n[Step 2/6] Importing Modules...")
    print("-"*80)
    
    try:
        from src.synthetic_data_generation.dg_models import CTGANSyntheticDataGenerator, TVAESyntheticDataGenerator
        from src.synthetic_data_generation.dg_evaluation import SyntheticDataEvaluator
        from src.synthetic_data_generation.dg_model_selector import SyntheticDataModelSelector
        print("✓ All modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure all module files are in the same directory:")
        print("  - dg_models.py")
        print("  - dg_evaluation.py")
        print("  - dg_model_selector.py")
        return None
    
    # Step 3: Train CTGAN
    print("\n[Step 3/6] Training CTGAN Models...")
    print("-"*80)
    
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
    supplier_data = pd.read_csv(supplier_csv_path)
    commodity_data = pd.read_csv(commodity_csv_path)
    
    return test_synthetic_data_pipeline(
        supplier_data=supplier_data,
        commodity_data=commodity_data,
        output_dir=output_dir,
        epochs=300,  # Full training
        evaluation_sizes=[500, 1000, 5000, 10000],
        final_generation_size=50000
    )


if __name__ == "__main__":
    # Run quick test
    print("Running quick test...")
    results = run_quick_test()
    
    if results:
        print(f"\n✓ Test completed successfully!")
        print(f"Best model: {results['best_model']}")