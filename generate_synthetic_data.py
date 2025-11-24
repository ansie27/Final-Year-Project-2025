"""
Synthetic Data Generation Script

This script:
1. Compares CTGAN and TVAE models for supplier and commodity data
2. Selects the most efficient model for each dataset type
3. Generates synthetic data using only the selected best models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from synthetic_data_generation import DG_ModelSelector, generate_synthetic_data
from data_preprocessing import (
    preprocess_supplier_data,
    preprocess_commodity_data
)
import config


def load_and_preprocess_data():
    """Load and preprocess supplier and commodity data."""
    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70 + "\n")
    
    # Load raw data
    supplier_data = pd.read_csv(config.RAW_DATA_FILES['supplier'])
    commodity_data = pd.read_csv(config.RAW_DATA_FILES['commodity'])
    
    print(f"✓ Loaded supplier data: {supplier_data.shape}")
    print(f"✓ Loaded commodity data: {commodity_data.shape}")
    
    # Preprocess
    supplier_data = preprocess_supplier_data(supplier_data)
    commodity_data = preprocess_commodity_data(commodity_data)
    
    print(f"✓ Preprocessed supplier data: {supplier_data.shape}")
    print(f"✓ Preprocessed commodity data: {commodity_data.shape}")
    
    return supplier_data, commodity_data


def select_best_models(supplier_data, commodity_data):
    """Select the best models for supplier and commodity data."""
    print("\n" + "="*70)
    print("SELECTING BEST MODELS")
    print("="*70)
    
    selector = DG_ModelSelector(output_dir=config.MODEL_SELECTION_DIR)
    
    # Check if we should use cached selection
    if config.SYNTHETIC_DATA_CONFIG['use_cached_selection']:
        summary_path = config.MODEL_SELECTION_DIR / "model_selection_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path, 'r') as f:
                cached = json.load(f)
                if 'supplier' in cached and 'commodity' in cached:
                    print("\n✓ Using cached model selection results")
                    return {
                        'supplier': cached['supplier']['best_model'],
                        'commodity': cached['commodity']['best_model']
                    }
    
    # Run model selection
    best_models = selector.select_models_for_both(
        supplier_data=supplier_data,
        commodity_data=commodity_data,
        supplier_target_col=config.SYNTHETIC_DATA_CONFIG.get('supplier_target_col'),
        commodity_target_col=config.SYNTHETIC_DATA_CONFIG.get('commodity_target_col'),
        epochs=config.SYNTHETIC_DATA_CONFIG['comparison_epochs'],
        save_results=True
    )
    
    return best_models


def generate_synthetic_datasets(supplier_data, commodity_data, best_models):
    """Generate synthetic data using the selected best models."""
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATA")
    print("="*70)
    
    results = {}
    
    # Generate supplier synthetic data
    print(f"\nGenerating synthetic supplier data using {best_models['supplier']}...")
    supplier_synthetic, supplier_generator = generate_synthetic_data(
        real_data=supplier_data,
        model_type=best_models['supplier'].lower(),
        num_rows=len(supplier_data),
        epochs=config.SYNTHETIC_DATA_CONFIG['generation_epochs'],
        save_path=config.SYNTHETIC_DATA_FILES['supplier'] if config.SYNTHETIC_DATA_CONFIG['save_synthetic_data'] else None,
        save_model=config.SYNTHETIC_DATA_CONFIG['save_models'],
        model_save_path=config.SYNTHETIC_MODEL_FILES['supplier'] if config.SYNTHETIC_DATA_CONFIG['save_models'] else None
    )
    results['supplier'] = {
        'data': supplier_synthetic,
        'generator': supplier_generator,
        'model_type': best_models['supplier']
    }
    
    # Generate commodity synthetic data
    print(f"\nGenerating synthetic commodity data using {best_models['commodity']}...")
    commodity_synthetic, commodity_generator = generate_synthetic_data(
        real_data=commodity_data,
        model_type=best_models['commodity'].lower(),
        num_rows=len(commodity_data),
        epochs=config.SYNTHETIC_DATA_CONFIG['generation_epochs'],
        save_path=config.SYNTHETIC_DATA_FILES['commodity'] if config.SYNTHETIC_DATA_CONFIG['save_synthetic_data'] else None,
        save_model=config.SYNTHETIC_DATA_CONFIG['save_models'],
        model_save_path=config.SYNTHETIC_MODEL_FILES['commodity'] if config.SYNTHETIC_DATA_CONFIG['save_models'] else None
    )
    results['commodity'] = {
        'data': commodity_synthetic,
        'generator': commodity_generator,
        'model_type': best_models['commodity']
    }
    
    return results


def main():
    """Main execution function."""
    print("="*70)
    print("SYNTHETIC DATA GENERATION - BEST MODEL SELECTION & GENERATION")
    print("="*70)
    
    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    print(f"\nRandom seed set to {config.RANDOM_SEED} for reproducibility")
    
    # Load and preprocess data
    supplier_data, commodity_data = load_and_preprocess_data()
    
    # Select best models
    best_models = select_best_models(supplier_data, commodity_data)
    
    # Generate synthetic data
    synthetic_results = generate_synthetic_datasets(
        supplier_data,
        commodity_data,
        best_models
    )
    
    # Summary
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nSelected Models:")
    print(f"  Supplier: {best_models['supplier']}")
    print(f"  Commodity: {best_models['commodity']}")
    print(f"\nGenerated Data:")
    print(f"  Supplier: {synthetic_results['supplier']['data'].shape}")
    print(f"  Commodity: {synthetic_results['commodity']['data'].shape}")
    
    if config.SYNTHETIC_DATA_CONFIG['save_synthetic_data']:
        print(f"\nSaved Files:")
        print(f"  Supplier: {config.SYNTHETIC_DATA_FILES['supplier']}")
        print(f"  Commodity: {config.SYNTHETIC_DATA_FILES['commodity']}")
    
    if config.SYNTHETIC_DATA_CONFIG['save_models']:
        print(f"\nSaved Models:")
        print(f"  Supplier: {config.SYNTHETIC_MODEL_FILES['supplier']}")
        print(f"  Commodity: {config.SYNTHETIC_MODEL_FILES['commodity']}")
    
    print("\n" + "="*70)
    
    return synthetic_results


if __name__ == "__main__":
    results = main()












