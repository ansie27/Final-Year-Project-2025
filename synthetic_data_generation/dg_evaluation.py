# Evaluation of CTGAN and TVAE
# Metrics:
# Statistical metrics: KS test, Chi-Square test, Jensen-Shannon divergence
# Correlation metrics: Pearson and Spearman correlations
# Distribution metrics: Wasserstein distance, MSE
# Machine learning utilities to assess synthtic data quality
# - Train Synthetic Test Real (TSTR)
# - Train Real Test Synthetic (TRTS)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import json
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency, spearmanr, pearsonr, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Evaluation configuration
EVALUATION_SIZES = [500, 1000, 5000, 10000]
RANDOM_SEED = 42


class SyntheticDataEvaluator:
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.evaluation_results = {}
        
        logger.info("Initialized SyntheticDataEvaluator")
    
    # =====================================================================
    # STATISTICAL SIMILARITY METRICS
    # =====================================================================
    
    def calculate_ks_statistic(self, real_col: pd.Series, syn_col: pd.Series) -> Dict:
        try:
            ks_stat, p_val = ks_2samp(real_col.dropna(), syn_col.dropna())
            return {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(p_val)
            }
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return {'ks_statistic': np.nan, 'ks_pvalue': np.nan}
    
    def calculate_chi_square(self, real_col: pd.Series, syn_col: pd.Series) -> Dict:

        try:
            real_counts = real_col.value_counts()
            syn_counts = syn_col.value_counts()
            
            # Align categories
            all_categories = set(real_counts.index) | set(syn_counts.index)
            real_aligned = np.array([real_counts.get(cat, 0) for cat in all_categories])
            syn_aligned = np.array([syn_counts.get(cat, 0) for cat in all_categories])
            
            # Chi-square test
            contingency = np.array([real_aligned, syn_aligned])
            chi2, p_val, _, _ = chi2_contingency(contingency)
            
            return {
                'chi2_statistic': float(chi2),
                'chi2_pvalue': float(p_val)
            }
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return {'chi2_statistic': np.nan, 'chi2_pvalue': np.nan}
    
    def calculate_jensen_shannon(self, real_col: pd.Series, syn_col: pd.Series, bins: int = 30) -> float:
        try:
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
            
            js_div = jensenshannon(hist_real, hist_syn)
            return float(js_div)
        except Exception as e:
            logger.warning(f"Jensen-Shannon calculation failed: {e}")
            return np.nan
    
    def calculate_wasserstein_distance(self, real_col: pd.Series, syn_col: pd.Series) -> float:

        try:
            real_sorted = np.sort(real_col.dropna())
            syn_sorted = np.sort(syn_col.dropna())
            
            # Normalize to same length
            min_len = min(len(real_sorted), len(syn_sorted))
            real_sorted = real_sorted[:min_len]
            syn_sorted = syn_sorted[:min_len]
            
            wasserstein = wasserstein_distance(real_sorted, syn_sorted)
            return float(wasserstein)
        except Exception as e:
            logger.warning(f"Wasserstein distance calculation failed: {e}")
            return np.nan
    
    def calculate_mse(self, real_col: pd.Series, syn_col: pd.Series) -> float:

        try:
            # Ensure same length
            min_len = min(len(real_col), len(syn_col))
            real_sample = real_col.dropna().iloc[:min_len].values
            syn_sample = syn_col.dropna().iloc[:min_len].values
            
            mse = mean_squared_error(real_sample, syn_sample)
            return float(mse)
        except Exception as e:
            logger.warning(f"MSE calculation failed: {e}")
            return np.nan
    
    # =====================================================================
    # CORRELATION METRICS
    # =====================================================================
    
    def calculate_correlation_similarity(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame
    ) -> Dict:

        numeric_cols = real.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'correlation_similarity': np.nan, 'method': 'N/A'}
        
        try:
            # Calculate correlation matrices
            real_corr = real[numeric_cols].corr(method='pearson')
            syn_corr = synthetic[numeric_cols].corr(method='pearson')
            
            # Get upper triangle (avoid duplication)
            mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
            real_corr_vals = real_corr.values[mask]
            syn_corr_vals = syn_corr.values[mask]
            
            # Calculate correlation of correlations (Pearson)
            pearson_corr, _ = pearsonr(real_corr_vals, syn_corr_vals)
            
            # Also calculate Spearman for robustness
            spearman_corr, _ = spearmanr(real_corr_vals, syn_corr_vals)
            
            # Choose the better one (higher is better, closer to 1)
            best_method = 'Pearson' if pearson_corr >= spearman_corr else 'Spearman'
            best_corr = max(pearson_corr, spearman_corr)
            
            return {
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr),
                'best_correlation': float(best_corr),
                'best_method': best_method
            }
        except Exception as e:
            logger.warning(f"Correlation similarity calculation failed: {e}")
            return {
                'pearson_correlation': np.nan,
                'spearman_correlation': np.nan,
                'best_correlation': np.nan,
                'best_method': 'N/A'
            }
    
    # =====================================================================
    # ML UTILITY METRICS
    # =====================================================================
    
    def evaluate_tstr(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str
    ) -> Dict:
        return self._train_and_evaluate(
            train_data=synthetic,
            test_data=real,
            target_col=target_col,
            metric_name='TSTR'
        )
    
    def evaluate_trts(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str
    ) -> Dict:
        return self._train_and_evaluate(
            train_data=real,
            test_data=synthetic,
            target_col=target_col,
            metric_name='TRTS'
        )
    
    def _train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_col: str,
        metric_name: str
    ) -> Dict:
        
        try:
            # Check if classification or regression
            is_classification = (
                train_data[target_col].dtype in ['object', 'category'] or
                train_data[target_col].nunique() < 10
            )
            
            # Prepare features
            feature_cols = [col for col in train_data.columns if col != target_col]
            
            # Encode data
            train_processed = self._encode_data(train_data, feature_cols, target_col, is_classification)
            test_processed = self._encode_data(test_data, feature_cols, target_col, is_classification, train_processed)
            
            X_train = train_processed['X']
            y_train = train_processed['y']
            X_test = test_processed['X']
            y_test = test_processed['y']
            
            # Train and evaluate
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                return {
                    metric_name: 'Classification',
                    'accuracy': float(acc),
                    'f1_score': float(f1)
                }
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                return {
                    metric_name: 'Regression',
                    'r2_score': float(r2),
                    'mae': float(mae)
                }
        except Exception as e:
            logger.warning(f"{metric_name} evaluation failed: {e}")
            return {metric_name: 'Failed', 'error': str(e)}
    
    def _encode_data(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        is_classification: bool,
        reference_data: Optional[Dict] = None
    ) -> Dict:
        """Encode categorical features and target."""
        data_copy = data.copy()
        
        # Encode features
        X_encoded = data_copy[feature_cols].copy()
        for col in feature_cols:
            if data_copy[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                if reference_data is None:
                    le.fit(pd.concat([data_copy[col]]).astype(str))
                    X_encoded[col] = le.transform(data_copy[col].astype(str))
                else:
                    # Use reference encoders
                    X_encoded[col] = le.transform(data_copy[col].astype(str))
        
        # Encode target
        y_encoded = data_copy[target_col].copy()
        if is_classification:
            le_target = LabelEncoder()
            if reference_data is None:
                le_target.fit(pd.concat([data_copy[target_col]]).astype(str))
            y_encoded = le_target.transform(data_copy[target_col].astype(str))
        
        return {'X': X_encoded.values, 'y': y_encoded.values}
    
    # =====================================================================
    # COMPREHENSIVE EVALUATION
    # =====================================================================
    
    def evaluate_single_dataset(
        self,
        real_data: pd.DataFrame,
        synthetic_data: Dict[int, pd.DataFrame],
        dataset_name: str,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:

        results = []
        
        for size, syn_data in sorted(synthetic_data.items()):
            logger.info(f"Evaluating {dataset_name} dataset with {size} synthetic rows...")
            
            # Statistical metrics
            ks_stats = []
            chi2_stats = []
            js_divs = []
            wasserstein_dists = []
            mses = []
            
            numeric_cols = real_data.select_dtypes(include=[np.number]).columns
            categorical_cols = real_data.select_dtypes(include=['object', 'category']).columns
            
            # Numerical columns
            for col in numeric_cols:
                if col in syn_data.columns:
                    ks_result = self.calculate_ks_statistic(real_data[col], syn_data[col])
                    ks_stats.append(ks_result['ks_statistic'])
                    
                    js_div = self.calculate_jensen_shannon(real_data[col], syn_data[col])
                    js_divs.append(js_div)
                    
                    wasserstein = self.calculate_wasserstein_distance(real_data[col], syn_data[col])
                    wasserstein_dists.append(wasserstein)
                    
                    mse = self.calculate_mse(real_data[col], syn_data[col])
                    mses.append(mse)
            
            # Categorical columns
            for col in categorical_cols:
                if col in syn_data.columns:
                    chi2_result = self.calculate_chi_square(real_data[col], syn_data[col])
                    chi2_stats.append(chi2_result['chi2_statistic'])
            
            # Correlation similarity
            corr_sim = self.calculate_correlation_similarity(real_data, syn_data)
            
            # ML Utility
            tstr_result = {}
            trts_result = {}
            if target_col and target_col in real_data.columns and target_col in syn_data.columns:
                tstr_result = self.evaluate_tstr(real_data, syn_data, target_col)
                trts_result = self.evaluate_trts(real_data, syn_data, target_col)
            
            # Aggregate results
            result_row = {
                'Dataset': dataset_name,
                'Synthetic_Rows': size,
                'Avg_KS_Statistic': float(np.mean(ks_stats)) if ks_stats else np.nan,
                'Avg_Chi2_Statistic': float(np.mean(chi2_stats)) if chi2_stats else np.nan,
                'Avg_Jensen_Shannon': float(np.mean(js_divs)) if js_divs else np.nan,
                'Avg_Wasserstein_Distance': float(np.mean(wasserstein_dists)) if wasserstein_dists else np.nan,
                'Avg_MSE': float(np.mean(mses)) if mses else np.nan,
                'Best_Correlation_Method': corr_sim['best_method'],
                'Correlation_Similarity': corr_sim['best_correlation'],
                **tstr_result,
                **trts_result
            }
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def generate_comparison_summary(
        self,
        supplier_results: pd.DataFrame,
        commodity_results: pd.DataFrame
    ) -> pd.DataFrame:

        combined = pd.concat([supplier_results, commodity_results], ignore_index=True)
        
        # Reorder columns for better readability
        col_order = [
            'Dataset', 'Synthetic_Rows',
            'Avg_KS_Statistic', 'Avg_Chi2_Statistic', 'Avg_Jensen_Shannon',
            'Avg_Wasserstein_Distance', 'Avg_MSE',
            'Correlation_Similarity', 'Best_Correlation_Method'
        ]
        
        # Add TSTR and TRTS columns if they exist
        if any('TSTR' in col for col in combined.columns):
            col_order.extend([col for col in combined.columns if 'TSTR' in col])
        if any('TRTS' in col for col in combined.columns):
            col_order.extend([col for col in combined.columns if 'TRTS' in col])
        
        available_cols = [col for col in col_order if col in combined.columns]
        
        return combined[available_cols]
    
    def export_results(
        self,
        results_df: pd.DataFrame,
        output_path: str
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
    
    def visualize_results(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None
    ):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Synthetic Data Generation Evaluation Results', fontsize=16, fontweight='bold')
        
        metrics = [
            'Avg_KS_Statistic',
            'Avg_Chi2_Statistic',
            'Avg_Jensen_Shannon',
            'Avg_Wasserstein_Distance',
            'Avg_MSE',
            'Correlation_Similarity'
        ]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            if metric in results_df.columns:
                for dataset in results_df['Dataset'].unique():
                    data = results_df[results_df['Dataset'] == dataset]
                    ax.plot(data['Synthetic_Rows'], data[metric], marker='o', label=dataset)
                
                ax.set_xlabel('Synthetic Rows')
                ax.set_ylabel(metric.replace('_', ' '))
                ax.set_title(metric.replace('_', ' '))
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_path}")
        
        plt.show()

def evaluate_synthetic_data_models(
    supplier_real: pd.DataFrame,
    supplier_synthetic: Dict[int, pd.DataFrame],
    commodity_real: pd.DataFrame,
    commodity_synthetic: Dict[int, pd.DataFrame],
    supplier_target: Optional[str] = None,
    commodity_target: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    evaluator = SyntheticDataEvaluator()
    
    # Evaluate supplier dataset
    supplier_results = evaluator.evaluate_single_dataset(
        supplier_real,
        supplier_synthetic,
        'Supplier',
        supplier_target
    )
    
    # Evaluate commodity dataset
    commodity_results = evaluator.evaluate_single_dataset(
        commodity_real,
        commodity_synthetic,
        'Commodity',
        commodity_target
    )
    
    # Combined summary
    combined_summary = evaluator.generate_comparison_summary(
        supplier_results,
        commodity_results
    )
    
    # Export results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.export_results(supplier_results, output_dir / 'supplier_evaluation.csv')
        evaluator.export_results(commodity_results, output_dir / 'commodity_evaluation.csv')
        evaluator.export_results(combined_summary, output_dir / 'combined_evaluation_summary.csv')
        
        evaluator.visualize_results(combined_summary, output_dir / 'evaluation_visualization.png')
    
    return supplier_results, commodity_results, combined_summary