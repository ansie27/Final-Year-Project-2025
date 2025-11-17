"""
Comparison module for CTGAN vs TVAE synthetic data generation.
Evaluates statistical similarity and ML utility for supplier and commodity data.
"""

import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import json

class SyntheticDataComparison:
    """
    Compare CTGAN and TVAE for generating synthetic supplier and commodity data.
    """
    
    def __init__(self, real_data: pd.DataFrame, data_type: str = 'supplier'):
        """
        Parameters:
        -----------
        real_data : pd.DataFrame
            Original dataset (supplier or commodity)
        data_type : str
            'supplier' or 'commodity' for labeling
        """
        self.real_data = real_data
        self.data_type = data_type
        self.metadata = None
        self.results = {}
        
        # Auto-detect metadata
        self._setup_metadata()
    
    def _setup_metadata(self):
        """Automatically detect and setup metadata."""
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.real_data)
    
    def generate_synthetic_data(
        self, 
        generator_type: str = 'ctgan', 
        epochs: int = 300
    ) -> Tuple[pd.DataFrame, object]:
        """Generate synthetic data using specified generator."""
        
        print(f"\n{'='*60}")
        print(f"Training {generator_type.upper()} on {self.data_type} data...")
        print(f"{'='*60}")
        
        if generator_type.lower() == 'ctgan':
            synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                verbose=True
            )
        elif generator_type.lower() == 'tvae':
            synthesizer = TVAESynthesizer(
                metadata=self.metadata,
                epochs=epochs,
                verbose=True
            )
        else:
            raise ValueError("generator_type must be 'ctgan' or 'tvae'")
        
        # Train
        synthesizer.fit(self.real_data)
        
        # Generate
        synthetic_data = synthesizer.sample(num_rows=len(self.real_data))
        
        return synthetic_data, synthesizer
    
    def calculate_js_divergence(
        self, 
        real_col: pd.Series, 
        syn_col: pd.Series, 
        bins: int = 30
    ) -> float:
        """Calculate Jensen-Shannon divergence for numerical columns."""
        
        # Create histograms
        hist_real, bin_edges = np.histogram(
            real_col.dropna(), 
            bins=bins, 
            density=True
        )
        hist_syn, _ = np.histogram(
            syn_col.dropna(), 
            bins=bin_edges, 
            density=True
        )
        
        # Normalize
        hist_real = hist_real / (hist_real.sum() + 1e-10)
        hist_syn = hist_syn / (hist_syn.sum() + 1e-10)
        
        # Calculate JS divergence
        js_div = jensenshannon(hist_real, hist_syn)
        
        return js_div
    
    def evaluate_statistical_similarity(
        self, 
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        generator_name: str
    ) -> Dict:
        """Evaluate statistical similarity between real and synthetic data."""
        
        print(f"\nEvaluating statistical similarity for {generator_name}...")
        
        metrics = {
            'generator': generator_name,
            'data_type': self.data_type,
            'column_metrics': {}
        }
        
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        categorical_cols = real.select_dtypes(include=['object', 'category']).columns
        
        # Numerical columns
        js_scores = []
        for col in numeric_cols:
            try:
                # KS test
                ks_stat, ks_pval = ks_2samp(
                    real[col].dropna(), 
                    synthetic[col].dropna()
                )
                
                # JS divergence
                js_div = self.calculate_js_divergence(real[col], synthetic[col])
                js_scores.append(js_div)
                
                metrics['column_metrics'][col] = {
                    'type': 'numerical',
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pval),
                    'js_divergence': float(js_div)
                }
            except Exception as e:
                print(f"  Warning: Could not evaluate {col}: {str(e)}")
        
        # Categorical columns
        for col in categorical_cols:
            try:
                real_counts = real[col].value_counts()
                syn_counts = synthetic[col].value_counts()
                
                all_categories = set(real_counts.index) | set(syn_counts.index)
                real_aligned = [real_counts.get(cat, 0) for cat in all_categories]
                syn_aligned = [syn_counts.get(cat, 0) for cat in all_categories]
                
                contingency = np.array([real_aligned, syn_aligned])
                chi2, pval, _, _ = chi2_contingency(contingency)
                
                metrics['column_metrics'][col] = {
                    'type': 'categorical',
                    'chi2_statistic': float(chi2),
                    'chi2_pvalue': float(pval)
                }
            except Exception as e:
                print(f"  Warning: Could not evaluate {col}: {str(e)}")
        
        # Aggregate metrics
        metrics['avg_js_divergence'] = float(np.mean(js_scores)) if js_scores else None
        
        print(f"  ✓ Average JS Divergence: {metrics['avg_js_divergence']:.4f}")
        
        return metrics
    
    def evaluate_ml_utility(
        self, 
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        generator_name: str,
        target_col: str
    ) -> Dict:
        """Evaluate ML utility: Train on Synthetic, Test on Real (TSTR)."""
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
        
        print(f"\nEvaluating ML utility for {generator_name}...")
        print(f"  Target column: {target_col}")
        
        # Check if classification or regression
        is_classification = (
            real[target_col].dtype in ['object', 'category'] or 
            real[target_col].nunique() < 10
        )
        
        task = 'classification' if is_classification else 'regression'
        print(f"  Task type: {task}")
        
        # Prepare data
        feature_cols = [col for col in real.columns if col != target_col]
        
        # Encode categorical features
        real_processed = real.copy()
        synthetic_processed = synthetic.copy()
        
        for col in feature_cols:
            if real[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                combined = pd.concat([real[col], synthetic[col]]).astype(str)
                le.fit(combined)
                real_processed[col] = le.transform(real[col].astype(str))
                synthetic_processed[col] = le.transform(synthetic[col].astype(str))
        
        # Encode target if classification
        if is_classification:
            le_target = LabelEncoder()
            le_target.fit(real[target_col].astype(str))
            real_processed[target_col] = le_target.transform(real[target_col].astype(str))
            synthetic_processed[target_col] = le_target.transform(synthetic[target_col].astype(str))
        
        # Train on synthetic, test on real
        X_train = synthetic_processed[feature_cols]
        y_train = synthetic_processed[target_col]
        X_test = real_processed[feature_cols]
        y_test = real_processed[target_col]
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  ✓ Accuracy: {acc:.4f}")
            print(f"  ✓ F1-Score: {f1:.4f}")
            
            return {
                'generator': generator_name,
                'task': 'classification',
                'accuracy': float(acc),
                'f1_score': float(f1)
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  ✓ R² Score: {r2:.4f}")
            print(f"  ✓ MAE: {mae:.4f}")
            
            return {
                'generator': generator_name,
                'task': 'regression',
                'r2_score': float(r2),
                'mae': float(mae)
            }
    
    def visualize_distributions(
        self, 
        synthetic_ctgan: pd.DataFrame, 
        synthetic_tvae: pd.DataFrame,
        num_cols: int = 4,
        save_path: Optional[str] = None
    ):
        """Create comparative distribution plots."""
        
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns[:num_cols]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize.")
            return
        
        fig, axes = plt.subplots(len(numeric_cols), 3, figsize=(15, 4*len(numeric_cols)))
        
        if len(numeric_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(numeric_cols):
            # Real
            axes[idx, 0].hist(
                self.real_data[col].dropna(), 
                bins=30, 
                alpha=0.7, 
                color='blue', 
                edgecolor='black'
            )
            axes[idx, 0].set_title(f'Real - {col}', fontsize=12, fontweight='bold')
            axes[idx, 0].set_ylabel('Frequency')
            
            # CTGAN
            axes[idx, 1].hist(
                synthetic_ctgan[col].dropna(), 
                bins=30, 
                alpha=0.7, 
                color='green', 
                edgecolor='black'
            )
            axes[idx, 1].set_title(f'CTGAN - {col}', fontsize=12, fontweight='bold')
            
            # TVAE
            axes[idx, 2].hist(
                synthetic_tvae[col].dropna(), 
                bins=30, 
                alpha=0.7, 
                color='red', 
                edgecolor='black'
            )
            axes[idx, 2].set_title(f'TVAE - {col}', fontsize=12, fontweight='bold')
        
        plt.suptitle(
            f'Distribution Comparison - {self.data_type.upper()}', 
            fontsize=16, 
            fontweight='bold'
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Distribution plot saved: {save_path}")
        
        plt.show()
    
    def compare_generators(
        self, 
        target_col: Optional[str] = None, 
        epochs: int = 300,
        save_outputs: bool = True
    ) -> Dict:
        """Main comparison pipeline."""
        
        print(f"\n{'='*70}")
        print(f" SYNTHETIC DATA GENERATION COMPARISON - {self.data_type.upper()} ".center(70))
        print(f"{'='*70}\n")
        
        # Generate synthetic data
        print("PHASE 1: Generating Synthetic Data")
        print("-" * 70)
        syn_ctgan, _ = self.generate_synthetic_data('ctgan', epochs)
        syn_tvae, _ = self.generate_synthetic_data('tvae', epochs)
        
        # Statistical similarity
        print(f"\n{'='*70}")
        print("PHASE 2: Statistical Similarity Evaluation")
        print("-" * 70)
        stats_ctgan = self.evaluate_statistical_similarity(self.real_data, syn_ctgan, 'CTGAN')
        stats_tvae = self.evaluate_statistical_similarity(self.real_data, syn_tvae, 'TVAE')
        
        self.results['statistical_similarity'] = {
            'CTGAN': stats_ctgan,
            'TVAE': stats_tvae
        }
        
        # ML utility
        if target_col:
            print(f"\n{'='*70}")
            print("PHASE 3: ML Utility Evaluation")
            print("-" * 70)
            ml_ctgan = self.evaluate_ml_utility(self.real_data, syn_ctgan, 'CTGAN', target_col)
            ml_tvae = self.evaluate_ml_utility(self.real_data, syn_tvae, 'TVAE', target_col)
            
            self.results['ml_utility'] = {
                'CTGAN': ml_ctgan,
                'TVAE': ml_tvae
            }
        
        # Visualization
        print(f"\n{'='*70}")
        print("PHASE 4: Visualization")
        print("-" * 70)
        
        viz_path = None
        if save_outputs:
            viz_path = f'outputs/visualizations/{self.data_type}_distribution_comparison.png'
        
        self.visualize_distributions(syn_ctgan, syn_tvae, save_path=viz_path)
        
        # Store synthetic data
        self.results['synthetic_data'] = {
            'CTGAN': syn_ctgan,
            'TVAE': syn_tvae
        }
        
        # Print summary
        self._print_summary()
        
        # Save results
        if save_outputs:
            self._save_results()
        
        return self.results
    
    def _print_summary(self):
        """Print comparison summary."""
        
        print(f"\n{'='*70}")
        print(f" COMPARISON SUMMARY - {self.data_type.upper()} ".center(70))
        print(f"{'='*70}\n")
        
        # Statistical
        print("1. STATISTICAL SIMILARITY (Jensen-Shannon Divergence)")
        print("-" * 70)
        ctgan_js = self.results['statistical_similarity']['CTGAN']['avg_js_divergence']
        tvae_js = self.results['statistical_similarity']['TVAE']['avg_js_divergence']
        
        print(f"  CTGAN: {ctgan_js:.6f}")
        print(f"  TVAE:  {tvae_js:.6f}")
        
        stat_winner = 'CTGAN' if ctgan_js < tvae_js else 'TVAE'
        print(f"  → Winner: {stat_winner} (lower is better)")
        
        # ML Utility
        if 'ml_utility' in self.results:
            print("\n2. MACHINE LEARNING UTILITY")
            print("-" * 70)
            
            ctgan_ml = self.results['ml_utility']['CTGAN']
            tvae_ml = self.results['ml_utility']['TVAE']
            
            if ctgan_ml['task'] == 'classification':
                print(f"  CTGAN - Accuracy: {ctgan_ml['accuracy']:.4f}, F1: {ctgan_ml['f1_score']:.4f}")
                print(f"  TVAE  - Accuracy: {tvae_ml['accuracy']:.4f}, F1: {tvae_ml['f1_score']:.4f}")
                ml_winner = 'CTGAN' if ctgan_ml['f1_score'] > tvae_ml['f1_score'] else 'TVAE'
                print(f"  → Winner: {ml_winner} (by F1-score)")
            else:
                print(f"  CTGAN - R²: {ctgan_ml['r2_score']:.4f}, MAE: {ctgan_ml['mae']:.4f}")
                print(f"  TVAE  - R²: {tvae_ml['r2_score']:.4f}, MAE: {tvae_ml['mae']:.4f}")
                ml_winner = 'CTGAN' if ctgan_ml['r2_score'] > tvae_ml['r2_score'] else 'TVAE'
                print(f"  → Winner: {ml_winner} (by R²)")
        
        print(f"\n{'='*70}\n")
    
    def _save_results(self):
        """Save comparison results to JSON."""
        
        # Prepare JSON-serializable results
        output = {
            'data_type': self.data_type,
            'statistical_similarity': self.results['statistical_similarity'],
            'ml_utility': self.results.get('ml_utility', {}),
            'recommendation': self._determine_winner()
        }
        
        output_path = f'outputs/results/{self.data_type}_comparison_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        print(f"✓ Results saved: {output_path}")
    
    def _determine_winner(self) -> str:
        """Determine overall winner with weighted scoring."""
        
        # Statistical score (70% weight)
        ctgan_js = self.results['statistical_similarity']['CTGAN']['avg_js_divergence']
        tvae_js = self.results['statistical_similarity']['TVAE']['avg_js_divergence']
        
        stat_score_ctgan = 1 / (1 + ctgan_js)
        stat_score_tvae = 1 / (1 + tvae_js)
        
        # ML score (30% weight)
        ml_score_ctgan = 0
        ml_score_tvae = 0
        
        if 'ml_utility' in self.results:
            ctgan_ml = self.results['ml_utility']['CTGAN']
            tvae_ml = self.results['ml_utility']['TVAE']
            
            if ctgan_ml['task'] == 'classification':
                ml_score_ctgan = ctgan_ml['f1_score']
                ml_score_tvae = tvae_ml['f1_score']
            else:
                ml_score_ctgan = max(0, ctgan_ml['r2_score'])
                ml_score_tvae = max(0, tvae_ml['r2_score'])
        
        # Final score
        final_ctgan = 0.7 * stat_score_ctgan + 0.3 * ml_score_ctgan
        final_tvae = 0.7 * stat_score_tvae + 0.3 * ml_score_tvae
        
        return 'CTGAN' if final_ctgan > final_tvae else 'TVAE'