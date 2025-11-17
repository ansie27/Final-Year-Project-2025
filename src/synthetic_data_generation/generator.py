"""
Synthetic Data Generator

This module generates synthetic data using the most efficient model (CTGAN or TVAE)
as determined by the model selector.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pickle
import json

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata


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
        """
        Train the synthesizer on real data.
        
        Parameters:
        -----------
        real_data : pd.DataFrame
            Real dataset to learn from
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print training progress
        """
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
        else:  # tvae
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
        """
        Generate synthetic data.
        
        Parameters:
        -----------
        num_rows : int
            Number of synthetic rows to generate
        verbose : bool
            Whether to print generation progress
            
        Returns:
        --------
        pd.DataFrame
            Generated synthetic data
        """
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
        """
        Fit the model and generate synthetic data in one step.
        
        Parameters:
        -----------
        real_data : pd.DataFrame
            Real dataset to learn from
        num_rows : int, optional
            Number of synthetic rows to generate. If None, uses length of real_data
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        pd.DataFrame
            Generated synthetic data
        """
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
        # SDV load is a class method
        if self.model_type == 'ctgan':
            self.synthesizer = CTGANSynthesizer.load(str(filepath))
        else:  # tvae
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
    """
    Convenience function to generate synthetic data using specified model.
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset to learn from
    model_type : str
        'ctgan' or 'tvae'
    num_rows : int, optional
        Number of synthetic rows to generate. If None, uses length of real_data
    epochs : int
        Number of training epochs
    save_path : Path, optional
        Path to save generated synthetic data
    save_model : bool
        Whether to save the trained model
    model_save_path : Path, optional
        Path to save the trained model
        
    Returns:
    --------
    Tuple[pd.DataFrame, SyntheticDataGenerator]
        Generated synthetic data and the generator instance
    """
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

