# CTGAN and TVAE data generation models
# Used on not yet preprocessed datasets

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration for synthetic data generation
SYNTHETIC_DATA_SIZES = [500, 1000, 5000, 10000]
DEFAULT_EPOCHS = 300
RANDOM_SEED = 42

class BaseSyntheticDataGenerator:
    """Base class for synthetic data generators."""
    
    def __init__(self, model_name: str, epochs: int = DEFAULT_EPOCHS, random_state: int = RANDOM_SEED):
        self.model_name = model_name
        self.epochs = epochs
        self.random_state = random_state
        self.metadata = None
        self.synthesizer = None
        self.is_fitted = False
        
        logger.info(f"Initialized {self.model_name} generator (epochs={epochs})")
    
    def fit(self, real_data: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement fit() method")
    
    def generate(self, num_rows: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before generating data")
        
        logger.info(f"{self.model_name}: Generating {num_rows} synthetic rows...")
        synthetic_data = self.synthesizer.sample(num_rows=num_rows)
        logger.info(f"{self.model_name}: Generated {len(synthetic_data)} rows")
        
        return synthetic_data
    
    def generate_multiple_sizes(self, sizes: list = None) -> Dict[int, pd.DataFrame]:
        if sizes is None:
            sizes = SYNTHETIC_DATA_SIZES
        
        logger.info(f"{self.model_name}: Generating synthetic data at multiple sizes: {sizes}")
        
        results = {}
        for size in sizes:
            results[size] = self.generate(size)
        
        return results


class CTGANSyntheticDataGenerator(BaseSyntheticDataGenerator):
    def __init__(
        self,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = 128,
        generator_dim: Tuple[int, ...] = (256, 256),
        discriminator_dim: Tuple[int, ...] = (256, 256),
        learning_rate: float = 2e-4,
        random_state: int = RANDOM_SEED
    ):
        super().__init__('CTGAN', epochs, random_state)
        
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.learning_rate = learning_rate
    
    def fit(self, real_data: pd.DataFrame) -> None:
        logger.info(f"CTGAN: Fitting on data with shape {real_data.shape}")
        
        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(real_data)
        
        logger.debug(f"CTGAN: Detected metadata for {len(real_data.columns)} columns")
        
        # Create and fit synthesizer
        self.synthesizer = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_seed=self.random_state,
            verbose=True
        )
        
        logger.info("CTGAN: Starting training...")
        self.synthesizer.fit(real_data)
        
        self.is_fitted = True
        logger.info("CTGAN: Training completed")


class TVAESyntheticDataGenerator(BaseSyntheticDataGenerator):
    def __init__(
        self,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = 128,
        encoder_dim: Tuple[int, ...] = (128, 128),
        decoder_dim: Tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        random_state: int = RANDOM_SEED
    ):
        super().__init__('TVAE', epochs, random_state)
        
        self.batch_size = batch_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.learning_rate = learning_rate
    
    def fit(self, real_data: pd.DataFrame) -> None:
        logger.info(f"TVAE: Fitting on data with shape {real_data.shape}")
        
        # Create metadata
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(real_data)
        
        logger.debug(f"TVAE: Detected metadata for {len(real_data.columns)} columns")
        
        # Create and fit synthesizer
        self.synthesizer = TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_seed=self.random_state,
            verbose=True
        )
        
        logger.info("TVAE: Starting training...")
        self.synthesizer.fit(real_data)
        
        self.is_fitted = True
        logger.info("TVAE: Training completed")


class SyntheticDataGenerationPipeline:
    
    def __init__(
        self,
        ctgan_epochs: int = DEFAULT_EPOCHS,
        tvae_epochs: int = DEFAULT_EPOCHS,
        random_state: int = RANDOM_SEED,
        output_dir: Optional[str] = None
    ):
        self.ctgan_generator = CTGANSyntheticDataGenerator(epochs=ctgan_epochs, random_state=random_state)
        self.tvae_generator = TVAESyntheticDataGenerator(epochs=tvae_epochs, random_state=random_state)
        self.output_dir = output_dir
        self.generated_data = {}
        
        logger.info(f"Initialized SyntheticDataGenerationPipeline (output_dir={output_dir})")
    
    def fit_models(self, real_data: pd.DataFrame):
        logger.info("Pipeline: Fitting CTGAN and TVAE models...")
        logger.info(f"Pipeline: Real data shape: {real_data.shape}")
        
        self.ctgan_generator.fit(real_data)
        self.tvae_generator.fit(real_data)
        
        logger.info("Pipeline: Both models fitted successfully")
    
    def generate_synthetic_data(
        self,
        sizes: list = None,
        save_to_file: bool = True
    ) -> Dict[str, Dict[int, pd.DataFrame]]:

        if sizes is None:
            sizes = SYNTHETIC_DATA_SIZES
        
        logger.info(f"Pipeline: Generating synthetic data at sizes: {sizes}")
        
        # Generate using CTGAN
        logger.info("Pipeline: Generating data with CTGAN...")
        ctgan_data = self.ctgan_generator.generate_multiple_sizes(sizes)
        self.generated_data['CTGAN'] = ctgan_data
        
        # Generate using TVAE
        logger.info("Pipeline: Generating data with TVAE...")
        tvae_data = self.tvae_generator.generate_multiple_sizes(sizes)
        self.generated_data['TVAE'] = tvae_data
        
        # Save to file if requested
        if save_to_file and self.output_dir:
            self._save_generated_data(sizes)
        
        logger.info("Pipeline: Synthetic data generation completed")
        
        return self.generated_data
    
    def _save_generated_data(self, sizes: list):
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline: Saving synthetic data to {output_path}")
        
        for model_name, model_data in self.generated_data.items():
            for size, data in model_data.items():
                filename = f"synthetic_data_{model_name.lower()}_{size}rows.csv"
                filepath = output_path / filename
                
                data.to_csv(filepath, index=False)
                logger.info(f"Pipeline: Saved {filename} ({len(data)} rows)")
    
    def get_summary_statistics(self) -> Dict:
        summary = {}
        
        for model_name, model_data in self.generated_data.items():
            summary[model_name] = {}
            
            for size, data in model_data.items():
                summary[model_name][size] = {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'memory_mb': data.memory_usage(deep=True).sum() / 1024**2
                }
        
        return summary


def generate_synthetic_data_comparison(
    real_data: pd.DataFrame,
    sizes: list = None,
    output_dir: Optional[str] = None,
    ctgan_epochs: int = DEFAULT_EPOCHS,
    tvae_epochs: int = DEFAULT_EPOCHS,
    random_state: int = RANDOM_SEED
) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], Dict]:

    if sizes is None:
        sizes = SYNTHETIC_DATA_SIZES
    
    # Create and run pipeline
    pipeline = SyntheticDataGenerationPipeline(
        ctgan_epochs=ctgan_epochs,
        tvae_epochs=tvae_epochs,
        random_state=random_state,
        output_dir=output_dir
    )
    
    # Fit models
    pipeline.fit_models(real_data)
    
    # Generate synthetic data
    generated_data = pipeline.generate_synthetic_data(sizes, save_to_file=True)
    
    # Get summary
    summary = pipeline.get_summary_statistics()
    
    return generated_data, summary