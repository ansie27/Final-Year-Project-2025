"""
Model Selector for Synthetic Data Generation

This module compares CTGAN and TVAE models and selects the most efficient one
for generating synthetic supplier and commodity data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

from .comparison import SyntheticDataComparison


class ModelSelector:
    """
    Selects the most efficient synthetic data generation model (CTGAN or TVAE)
    for supplier and commodity datasets.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Parameters:
        -----------
        output_dir : Path, optional
            Directory to save model selection results
        """
        self.output_dir = output_dir or Path("outputs/synthetic_data_generation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.selection_results = {}
        self.best_models = {}
    
    def select_best_model(
        self,
        real_data: pd.DataFrame,
        data_type: str = 'supplier',
        target_col: Optional[str] = None,
        epochs: int = 300,
        save_results: bool = True
    ) -> str:
        """
        Compare CTGAN and TVAE and select the most efficient model.
        
        Parameters:
        -----------
        real_data : pd.DataFrame
            Real dataset to use for comparison
        data_type : str
            'supplier' or 'commodity'
        target_col : str, optional
            Target column for ML utility evaluation
        epochs : int
            Number of training epochs for comparison
        save_results : bool
            Whether to save comparison results
            
        Returns:
        --------
        str
            'CTGAN' or 'TVAE' - the selected best model
        """
        print(f"\n{'='*70}")
        print(f"MODEL SELECTION FOR {data_type.upper()} DATA".center(70))
        print(f"{'='*70}\n")
        
        # Create comparison instance
        comparator = SyntheticDataComparison(real_data, data_type=data_type)
        
        # Run comparison
        results = comparator.compare_generators(
            target_col=target_col,
            epochs=epochs,
            save_outputs=save_results
        )
        
        # Determine best model
        best_model = comparator._determine_winner()
        
        # Store results
        self.selection_results[data_type] = {
            'best_model': best_model,
            'comparison_results': results,
            'statistical_similarity': results['statistical_similarity'],
            'ml_utility': results.get('ml_utility', {})
        }
        
        self.best_models[data_type] = best_model
        
        print(f"\n{'='*70}")
        print(f"SELECTED MODEL FOR {data_type.upper()}: {best_model}")
        print(f"{'='*70}\n")
        
        # Save selection results
        if save_results:
            self._save_selection_results(data_type)
        
        return best_model
    
    def _save_selection_results(self, data_type: str):
        """Save model selection results to JSON."""
        output_path = self.output_dir / f"{data_type}_model_selection.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.selection_results[data_type], f, indent=4, default=str)
        
        print(f"✓ Model selection results saved: {output_path}")
    
    def get_best_model(self, data_type: str) -> Optional[str]:
        """Get the selected best model for a data type."""
        return self.best_models.get(data_type)
    
    def select_models_for_both(
        self,
        supplier_data: pd.DataFrame,
        commodity_data: pd.DataFrame,
        supplier_target_col: Optional[str] = None,
        commodity_target_col: Optional[str] = None,
        epochs: int = 300,
        save_results: bool = True
    ) -> Dict[str, str]:
        """
        Select best models for both supplier and commodity data.
        
        Parameters:
        -----------
        supplier_data : pd.DataFrame
            Supplier dataset
        commodity_data : pd.DataFrame
            Commodity dataset
        supplier_target_col : str, optional
            Target column for supplier ML utility evaluation
        commodity_target_col : str, optional
            Target column for commodity ML utility evaluation
        epochs : int
            Number of training epochs
        save_results : bool
            Whether to save results
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with 'supplier' and 'commodity' keys mapping to best models
        """
        print(f"\n{'='*70}")
        print("SELECTING BEST MODELS FOR SUPPLIER AND COMMODITY DATA".center(70))
        print(f"{'='*70}\n")
        
        # Select for supplier
        supplier_model = self.select_best_model(
            supplier_data,
            data_type='supplier',
            target_col=supplier_target_col,
            epochs=epochs,
            save_results=save_results
        )
        
        # Select for commodity
        commodity_model = self.select_best_model(
            commodity_data,
            data_type='commodity',
            target_col=commodity_target_col,
            epochs=epochs,
            save_results=save_results
        )
        
        # Save overall summary
        if save_results:
            summary = {
                'supplier': {
                    'best_model': supplier_model,
                    'selection_details': self.selection_results.get('supplier', {})
                },
                'commodity': {
                    'best_model': commodity_model,
                    'selection_details': self.selection_results.get('commodity', {})
                }
            }
            
            summary_path = self.output_dir / "model_selection_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
            
            print(f"\n✓ Overall model selection summary saved: {summary_path}")
        
        return {
            'supplier': supplier_model,
            'commodity': commodity_model
        }





