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
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency, spearmanr, pearsonr, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import warnings
import logging
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

EVALUATION_SIZES = [500, 1000, 5000, 10000]
RANDOM_SEED = 42


class SyntheticDataEvaluator:
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.evaluation_results = {}
        logger.info("Initialized SyntheticDataEvaluator")
    
    # Statistical metrics (unchanged - these are correct)
    def calculate_ks_statistic(self, real_col: pd.Series, syn_col: pd.Series) -> Dict:
        try:
            ks_stat, p_val = ks_2samp(real_col.dropna(), syn_col.dropna())
            return {'ks_statistic': float(ks_stat), 'ks_pvalue': float(p_val)}
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return {'ks_statistic': np.nan, 'ks_pvalue': np.nan}
    
    def calculate_chi_square(self, real_col: pd.Series, syn_col: pd.Series) -> Dict:
        try:
            real_counts = real_col.value_counts()
            syn_counts = syn_col.value_counts()
            
            all_categories = set(real_counts.index) | set(syn_counts.index)
            real_aligned = np.array([real_counts.get(cat, 0) for cat in all_categories])
            syn_aligned = np.array([syn_counts.get(cat, 0) for cat in all_categories])
            
            contingency = np.array([real_aligned, syn_aligned])
            chi2, p_val, _, _ = chi2_contingency(contingency)
            
            return {'chi2_statistic': float(chi2), 'chi2_pvalue': float(p_val)}
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return {'chi2_statistic': np.nan, 'chi2_pvalue': np.nan}
    
    def calculate_jensen_shannon(self, real_col: pd.Series, syn_col: pd.Series, bins: int = 30) -> float:
        try:
            hist_real, bin_edges = np.histogram(real_col.dropna(), bins=bins, density=True)
            hist_syn, _ = np.histogram(syn_col.dropna(), bins=bin_edges, density=True)
            
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
            min_len = min(len(real_col), len(syn_col))
            real_sample = real_col.dropna().iloc[:min_len].values
            syn_sample = syn_col.dropna().iloc[:min_len].values
            
            mse = mean_squared_error(real_sample, syn_sample)
            return float(mse)
        except Exception as e:
            logger.warning(f"MSE calculation failed: {e}")
            return np.nan
    
    def calculate_correlation_similarity(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'pearson_correlation': np.nan, 'spearman_correlation': np.nan, 
                    'best_correlation': np.nan, 'best_method': 'N/A'}
        
        try:
            real_corr = real[numeric_cols].corr(method='pearson')
            syn_corr = synthetic[numeric_cols].corr(method='pearson')
            
            mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
            real_corr_vals = real_corr.values[mask]
            syn_corr_vals = syn_corr.values[mask]
            
            pearson_corr, _ = pearsonr(real_corr_vals, syn_corr_vals)
            spearman_corr, _ = spearmanr(real_corr_vals, syn_corr_vals)
            
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
            return {'pearson_correlation': np.nan, 'spearman_correlation': np.nan,
                    'best_correlation': np.nan, 'best_method': 'N/A'}
    
    # FIXED ML Utility Metrics
    def evaluate_tstr(self, real: pd.DataFrame, synthetic: pd.DataFrame, target_col: str) -> Dict:
        """Train on Synthetic, Test on Real"""
        return self._train_and_evaluate(
            train_data=synthetic,
            test_data=real,
            target_col=target_col,
            metric_name='TSTR'
        )
    
    def evaluate_trts(self, real: pd.DataFrame, synthetic: pd.DataFrame, target_col: str) -> Dict:
        """Train on Real, Test on Synthetic"""
        return self._train_and_evaluate(
            train_data=real,
            test_data=synthetic,
            target_col=target_col,
            metric_name='TRTS'
        )
    
    def _train_and_evaluate(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                           target_col: str, metric_name: str) -> Dict:
        try:
            # Check if target exists
            if target_col not in train_data.columns or target_col not in test_data.columns:
                logger.warning(f"Target column '{target_col}' not found")
                return {metric_name: 'Failed', 'error': 'Target column not found'}
            
            # Determine task type
            is_classification = (
                train_data[target_col].dtype in ['object', 'category'] or
                train_data[target_col].nunique() < 10
            )
            
            # Prepare features
            feature_cols = [col for col in train_data.columns if col != target_col]
            
            # Only keep common columns
            common_features = [col for col in feature_cols if col in test_data.columns]
            if len(common_features) == 0:
                return {metric_name: 'Failed', 'error': 'No common features'}
            
            # Process data
            X_train, y_train, encoders = self._prepare_features(
                train_data[common_features + [target_col]], 
                common_features, 
                target_col, 
                is_classification
            )
            
            X_test, y_test, _ = self._prepare_features(
                test_data[common_features + [target_col]], 
                common_features, 
                target_col, 
                is_classification,
                encoders=encoders  # Use same encoders
            )
            
            # Train and evaluate
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, max_depth=10)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                return {
                    f'{metric_name}_task': 'Classification',
                    f'{metric_name}_accuracy': float(acc),
                    f'{metric_name}_f1_score': float(f1)
                }
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, max_depth=10)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                return {
                    f'{metric_name}_task': 'Regression',
                    f'{metric_name}_r2_score': float(r2),
                    f'{metric_name}_mae': float(mae)
                }
        except Exception as e:
            logger.warning(f"{metric_name} evaluation failed: {e}")
            return {f'{metric_name}_task': 'Failed', 'error': str(e)}
    
    def _prepare_features(self, data: pd.DataFrame, feature_cols: List[str], 
                         target_col: str, is_classification: bool, 
                         encoders: Optional[Dict] = None) -> Tuple:
        """
        FIXED: Properly handle categorical encoding with shared encoders
        """
        data_copy = data.copy()
        
        if encoders is None:
            encoders = {}
        
        # Encode features
        X_encoded = pd.DataFrame(index=data_copy.index)
        
        for col in feature_cols:
            if data_copy[col].dtype in ['object', 'category']:
                # Categorical feature
                if col not in encoders:
                    # Create new encoder
                    le = LabelEncoder()
                    le.fit(data_copy[col].astype(str))
                    encoders[col] = le
                else:
                    le = encoders[col]
                
                # Transform, handling unseen labels
                try:
                    X_encoded[col] = le.transform(data_copy[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    X_encoded[col] = data_copy[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            else:
                # Numeric feature
                X_encoded[col] = data_copy[col].fillna(data_copy[col].median())
        
        # Encode target
        if is_classification:
            if 'target' not in encoders:
                le_target = LabelEncoder()
                le_target.fit(data_copy[target_col].astype(str))
                encoders['target'] = le_target
            else:
                le_target = encoders['target']
            
            try:
                y_encoded = le_target.transform(data_copy[target_col].astype(str))
            except ValueError:
                # Handle unseen target categories
                y_encoded = data_copy[target_col].astype(str).apply(
                    lambda x: le_target.transform([x])[0] if x in le_target.classes_ else -1
                )
                y_encoded = y_encoded.values
        else:
            y_encoded = data_copy[target_col].fillna(data_copy[target_col].median()).values
        
        return X_encoded.values, y_encoded, encoders
    
    # Comprehensive evaluation (unchanged)
    def evaluate_single_dataset(self, real_data: pd.DataFrame, 
                               synthetic_data: Dict[int, pd.DataFrame],
                               dataset_name: str, target_col: Optional[str] = None) -> pd.DataFrame:
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
                'Avg_KS_Statistic': float(np.nanmean(ks_stats)) if ks_stats else np.nan,
                'Avg_Chi2_Statistic': float(np.nanmean(chi2_stats)) if chi2_stats else np.nan,
                'Avg_Jensen_Shannon': float(np.nanmean(js_divs)) if js_divs else np.nan,
                'Avg_Wasserstein_Distance': float(np.nanmean(wasserstein_dists)) if wasserstein_dists else np.nan,
                'Avg_MSE': float(np.nanmean(mses)) if mses else np.nan,
                'Best_Correlation_Method': corr_sim['best_method'],
                'Correlation_Similarity': corr_sim['best_correlation'],
                **tstr_result,
                **trts_result
            }
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def export_results(self, results_df: pd.DataFrame, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")