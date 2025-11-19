"""
Final Synthetic Data Generation Module.

Uses the selected best model (CTGAN or TVAE) from dg_model_selector.py to generate
large-scale synthetic data (50,000 rows) for supplier and commodity datasets.

Workflow:
1. Load real supplier and commodity data
2. Train both CTGAN and TVAE models
3. Evaluate performance using SyntheticDataEvaluator
4. Select best model using SyntheticDataModelSelector
5. Use selected model to generate 50,000 rows for each dataset
6. Save results to CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from .dg_models import CTGANSyntheticDataGenerator, TVAESyntheticDataGenerator
from .dg_evaluation import SyntheticDataEvaluator, evaluate_synthetic_data_models
from .dg_model_selector import SyntheticDataModelSelector

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic data using the selected best model (CTGAN or TVAE).
    """
    
    def __init__(self, model_type: str = 'ctgan'):
        """
        Parameters:
        -----------
        model_type : str
            'ctgan' or 'tvae' - the model to use for generation
        """
        self.model_type = model_type.lower()
        if self.model_type not in ['ctgan', 'tvae']:
            raise ValueError("model_type must be 'ctgan' or 'tvae'")
        
        self.metadata = None
        self.synthesizer = None
        self.is_fitted = False
    
    def fit(
        self,
        real_data: pd.DataFrame,
        epochs: int = 300,
        verbose: bool = True
    ):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.model_type.upper()} synthesizer...")
            print(f"{'='*60}")
        
        # Setup metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(real_data)
        
        # Create synthesizer
        if self.model_type == 'ctgan':
            self.synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                verbose=verbose
            )
        else:  # TVAE
            self.synthesizer = TVAESynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                verbose=verbose
            )
        
        # Train
        self.synthesizer.fit(real_data)
        self.is_fitted = True
        
        if verbose:
            print(f"✓ {self.model_type.upper()} training completed")
    
    def generate(
        self,
        num_rows: int,
        verbose: bool = True
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Synthesizer must be fitted before generating data. Call fit() first.")
        
        if verbose:
            print(f"\nGenerating {num_rows} synthetic rows using {self.model_type.upper()}...")
        
        synthetic_data = self.synthesizer.sample(num_rows=num_rows)
        
        if verbose:
            print(f"✓ Generated {len(synthetic_data)} synthetic rows")
        
        return synthetic_data
    
    def fit_and_generate(
        self,
        real_data: pd.DataFrame,
        num_rows: Optional[int] = None,
        epochs: int = 300,
        verbose: bool = True
    ) -> pd.DataFrame:
        if num_rows is None:
            num_rows = len(real_data)
        
        self.fit(real_data, epochs=epochs, verbose=verbose)
        return self.generate(num_rows, verbose=verbose)
    
    def save_model(self, filepath: Path):
        """Save the trained synthesizer to disk."""
        if not self.is_fitted:
            raise ValueError("No fitted model to save. Call fit() first.")
        
        # SDV save method expects a string path
        self.synthesizer.save(str(filepath))
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath: Path):
        """Load a trained synthesizer from disk."""
        if self.model_type == 'ctgan':
            self.synthesizer = CTGANSynthesizer.load(str(filepath))
        else:
            self.synthesizer = TVAESynthesizer.load(str(filepath))
        
        self.metadata = self.synthesizer.metadata
        self.is_fitted = True
        print(f"✓ Model loaded from: {filepath}")


def generate_synthetic_data(
    real_data: pd.DataFrame,
    model_type: str,
    num_rows: Optional[int] = None,
    epochs: int = 300,
    save_path: Optional[Path] = None,
    save_model: bool = False,
    model_save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, SyntheticDataGenerator]:

    generator = SyntheticDataGenerator(model_type=model_type)
    
    if num_rows is None:
        num_rows = len(real_data)
    
    synthetic_data = generator.fit_and_generate(
        real_data,
        num_rows=num_rows,
        epochs=epochs,
        verbose=True
    )
    
    # Save synthetic data
    if save_path:
        synthetic_data.to_csv(save_path, index=False)
        print(f"✓ Synthetic data saved to: {save_path}")
    
    # Save model
    if save_model and model_save_path:
        generator.save_model(model_save_path)
    
    return synthetic_data, generator


def generate_final_synthetic_data(
    supplier_real_path: str,
    commodity_real_path: str,
    output_dir: Optional[str] = None,
    num_rows: int = 50000,
    epochs: int = 300,
    evaluation_sizes: Optional[list] = None,
    target_supplier_col: Optional[str] = None,
    target_commodity_col: Optional[str] = None
) -> Dict:
    """
    Generate final large-scale synthetic data using the selected best model.
    
    This function:
    1. Loads real supplier and commodity data
    2. Generates synthetic data at multiple evaluation sizes using both CTGAN and TVAE
    3. Evaluates both models using SyntheticDataEvaluator
    4. Selects the best model using SyntheticDataModelSelector
    5. Uses the selected model to generate 50,000 rows (or specified num_rows)
    6. Saves final synthetic data to CSV files
    
    Parameters
    ----------
    supplier_real_path : str
        Path to real supplier dataset CSV
    commodity_real_path : str
        Path to real commodity dataset CSV
    output_dir : str, optional
        Output directory for results. Default: data/processed/
    num_rows : int
        Number of rows to generate. Default: 50000
    epochs : int
        Training epochs for synthesizers. Default: 300
    evaluation_sizes : list, optional
        Sizes for evaluation. Default: [500, 1000, 5000, 10000]
    target_supplier_col : str, optional
        Target column for supplier ML utility evaluation
    target_commodity_col : str, optional
        Target column for commodity ML utility evaluation
        
    Returns
    -------
    dict
        Summary of generation results with keys:
        - selected_model: Name of best model ('CTGAN' or 'TVAE')
        - supplier_synthetic: Generated supplier data
        - commodity_synthetic: Generated commodity data
        - supplier_output_path: Path to saved supplier data
        - commodity_output_path: Path to saved commodity data
        - selection_analysis: Model selection details
    """
    
    print(f"\n{'='*80}")
    print(f"FINAL SYNTHETIC DATA GENERATION PIPELINE".center(80))
    print(f"{'='*80}\n")
    
    output_dir = Path(output_dir or "data/processed/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if evaluation_sizes is None:
        evaluation_sizes = [500, 1000, 5000, 10000]
    
    # =====================================================================
    # Step 1: Load real data
    # =====================================================================
    print("\n[Step 1/5] Loading real datasets...")
    print("-" * 80)
    
    supplier_real = pd.read_csv(supplier_real_path)
    commodity_real = pd.read_csv(commodity_real_path)
    
    print(f"Supplier data: {supplier_real.shape[0]} rows, {supplier_real.shape[1]} columns")
    print(f"Commodity data: {commodity_real.shape[0]} rows, {commodity_real.shape[1]} columns")
    
    # =====================================================================
    # Step 2: Generate synthetic data at evaluation sizes for comparison
    # =====================================================================
    print("\n[Step 2/5] Generating synthetic data for model evaluation...")
    print("-" * 80)
    
    # CTGAN generation
    print("\nTraining CTGAN models...")
    ctgan_supplier_gen = CTGANSyntheticDataGenerator(random_state=42)
    ctgan_supplier_gen.fit(supplier_real, epochs=epochs)
    
    ctgan_commodity_gen = CTGANSyntheticDataGenerator(random_state=42)
    ctgan_commodity_gen.fit(commodity_real, epochs=epochs)
    
    # TVAE generation
    print("\nTraining TVAE models...")
    tvae_supplier_gen = TVAESyntheticDataGenerator(random_state=42)
    tvae_supplier_gen.fit(supplier_real, epochs=epochs)
    
    tvae_commodity_gen = TVAESyntheticDataGenerator(random_state=42)
    tvae_commodity_gen.fit(commodity_real, epochs=epochs)
    
    # Generate data at evaluation sizes
    ctgan_supplier_data = {}
    tvae_supplier_data = {}
    ctgan_commodity_data = {}
    tvae_commodity_data = {}
    
    for size in evaluation_sizes:
        print(f"\nGenerating {size} rows...")
        ctgan_supplier_data[size] = ctgan_supplier_gen.generate(size)
        tvae_supplier_data[size] = tvae_supplier_gen.generate(size)
        ctgan_commodity_data[size] = ctgan_commodity_gen.generate(size)
        tvae_commodity_data[size] = tvae_commodity_gen.generate(size)
    
    # =====================================================================
    # Step 3: Evaluate models
    # =====================================================================
    print("\n[Step 3/5] Evaluating CTGAN and TVAE models...")
    print("-" * 80)
    
    evaluator = SyntheticDataEvaluator()
    
    # Evaluate supplier data
    ctgan_supplier_eval = evaluator.evaluate_single_dataset(
        supplier_real,
        ctgan_supplier_data,
        'CTGAN_Supplier',
        target_supplier_col
    )
    
    tvae_supplier_eval = evaluator.evaluate_single_dataset(
        supplier_real,
        tvae_supplier_data,
        'TVAE_Supplier',
        target_supplier_col
    )
    
    # Evaluate commodity data
    ctgan_commodity_eval = evaluator.evaluate_single_dataset(
        commodity_real,
        ctgan_commodity_data,
        'CTGAN_Commodity',
        target_commodity_col
    )
    
    tvae_commodity_eval = evaluator.evaluate_single_dataset(
        commodity_real,
        tvae_commodity_data,
        'TVAE_Commodity',
        target_commodity_col
    )
    
    print(f"\nCTGAN Supplier evaluation:\n{ctgan_supplier_eval}")
    print(f"\nTVAE Supplier evaluation:\n{tvae_supplier_eval}")
    print(f"\nCTGAN Commodity evaluation:\n{ctgan_commodity_eval}")
    print(f"\nTVAE Commodity evaluation:\n{tvae_commodity_eval}")
    
    # =====================================================================
    # Step 4: Select best model
    # =====================================================================
    print("\n[Step 4/5] Selecting best model...")
    print("-" * 80)
    
    selector = SyntheticDataModelSelector(output_dir=output_dir)
    
    best_model, analysis = selector.select_best_model(
        ctgan_supplier_eval,  # CTGAN results
        tvae_supplier_eval,   # TVAE results
        ctgan_supplier_eval,  # For combined results
        ctgan_commodity_eval,
        save_results=True
    )
    
    # =====================================================================
    # Step 5: Generate final large-scale data using best model
    # =====================================================================
    print(f"\n[Step 5/5] Generating {num_rows} rows using {best_model}...")
    print("-" * 80)
    
    if best_model.upper() == 'CTGAN':
        print(f"\nGenerating {num_rows} supplier rows with CTGAN...")
        final_supplier_synthetic = ctgan_supplier_gen.generate(num_rows)
        
        print(f"Generating {num_rows} commodity rows with CTGAN...")
        final_commodity_synthetic = ctgan_commodity_gen.generate(num_rows)
    else:
        print(f"\nGenerating {num_rows} supplier rows with TVAE...")
        final_supplier_synthetic = tvae_supplier_gen.generate(num_rows)
        
        print(f"Generating {num_rows} commodity rows with TVAE...")
        final_commodity_synthetic = tvae_commodity_gen.generate(num_rows)
    
    # =====================================================================
    # Save Results
    # =====================================================================
    print(f"\nSaving results...")
    
    supplier_output_path = output_dir / 'final_synthetic_supplier_data.csv'
    commodity_output_path = output_dir / 'final_synthetic_commodity_data.csv'
    
    final_supplier_synthetic.to_csv(supplier_output_path, index=False)
    final_commodity_synthetic.to_csv(commodity_output_path, index=False)
    
    print(f"✓ Supplier synthetic data saved: {supplier_output_path}")
    print(f"  Shape: {final_supplier_synthetic.shape}")
    
    print(f"✓ Commodity synthetic data saved: {commodity_output_path}")
    print(f"  Shape: {final_commodity_synthetic.shape}")
    
    # =====================================================================
    # Summary
    # =====================================================================
    results = {
        'selected_model': best_model,
        'supplier_synthetic': final_supplier_synthetic,
        'commodity_synthetic': final_commodity_synthetic,
        'supplier_output_path': str(supplier_output_path),
        'commodity_output_path': str(commodity_output_path),
        'num_rows_generated': num_rows,
        'selection_analysis': analysis,
        'supplier_evaluation': ctgan_supplier_eval if best_model.upper() == 'CTGAN' else tvae_supplier_eval,
        'commodity_evaluation': ctgan_commodity_eval if best_model.upper() == 'CTGAN' else tvae_commodity_eval
    }
    
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE".center(80))
    print(f"{'='*80}")
    print(f"\nSelected Model: {results['selected_model']}")
    print(f"Supplier Data: {supplier_output_path}")
    print(f"Commodity Data: {commodity_output_path}")
    print(f"Rows Generated: {num_rows}")
    print(f"\n{'='*80}\n")
    
    return results

