# Evaluation of Synthetic Data Quality
# Techniques:
# 1. Kolmogorov-Smirnov Test (KS)
# 2. Chi-Square Test
# 3. Jensen-Shannon Divergence (JSD)
# 4. Pearson Correlation
# 5. Wasserstein Distance

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class SyntheticDataEvaluator:
    def __init__(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        self.real_df = real_df.copy()
        self.synthetic_df = synthetic_df.copy()
        
        # Ensure same columns
        common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
        self.real_df = self.real_df[common_cols]
        self.synthetic_df = self.synthetic_df[common_cols]
        
        logger.info("Initialized evaluator:")
        logger.info("  Real data: %d rows, %d columns", len(real_df), len(real_df.columns))
        logger.info("  Synthetic data: %d rows, %d columns", len(synthetic_df), len(synthetic_df.columns))
    
    def identify_column_types(self) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical columns."""
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.real_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return numerical_cols, categorical_cols
    
    def kolmogorov_smirnov_test(self, numerical_cols: List[str]) -> Dict[str, Dict]:
        """
        Perform Kolmogorov-Smirnov test for numerical columns.
        
        The KS test measures the maximum distance between cumulative distributions.
        - Statistic closer to 0 = more similar distributions
        - p-value > 0.05 = cannot reject null hypothesis (distributions are similar)
        
        Returns:
        --------
        Dict with KS statistic and p-value for each column
        """
        results = {}
        
        for col in numerical_cols:
            real_data = self.real_df[col].dropna()
            syn_data = self.synthetic_df[col].dropna()
            
            if len(real_data) == 0 or len(syn_data) == 0:
                continue
            
            statistic, p_value = ks_2samp(real_data, syn_data)
            
            results[col] = {
                'statistic': statistic,
                'p_value': p_value,
                'similar': p_value > 0.05,  # Cannot reject H0
            }
        
        return results
    
    def chi_square_test(self, categorical_cols: List[str]) -> Dict[str, Dict]:
        """
        Perform Chi-Square test for categorical columns.
        
        Tests independence between real and synthetic data distributions.
        - p-value > 0.05 = distributions are similar
        
        Returns:
        --------
        Dict with chi2 statistic and p-value for each column
        """
        results = {}
        
        for col in categorical_cols:
            real_counts = self.real_df[col].value_counts()
            syn_counts = self.synthetic_df[col].value_counts()
            
            # Align categories
            all_categories = sorted(set(real_counts.index) | set(syn_counts.index))
            real_freq = [real_counts.get(cat, 0) for cat in all_categories]
            syn_freq = [syn_counts.get(cat, 0) for cat in all_categories]
            
            # Skip if too few categories
            if len(all_categories) < 2:
                continue
            
            # Create contingency table
            contingency_table = np.array([real_freq, syn_freq])
            
            # Perform test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results[col] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'similar': p_value > 0.05,
            }
        
        return results

    # Jensen-Shannon Divergence  
    # If JSD < 0.1 then it indicates similarity
    # 0 = identical distributions, 1 = completely different distributions
    def jensen_shannon_divergence(self, categorical_cols: List[str]) -> Dict[str, float]:
        results = {}
        
        for col in categorical_cols:
            real_counts = self.real_df[col].value_counts(normalize=True)
            syn_counts = self.synthetic_df[col].value_counts(normalize=True)
            
            # Align categories
            all_categories = sorted(set(real_counts.index) | set(syn_counts.index))
            real_prob = np.array([real_counts.get(cat, 0) for cat in all_categories])
            syn_prob = np.array([syn_counts.get(cat, 0) for cat in all_categories])
            
            # Calculate JSD
            jsd = jensenshannon(real_prob, syn_prob)
            
            results[col] = {
                'jsd': jsd,
                'similar': jsd < 0.1,
            }
        
        return results

    # Pearson correlation
    def pearson_correlation(self, numerical_cols: List[str]) -> Dict[str, float]:
        """
        Calculate Pearson correlation coefficient between real and synthetic data.
        
        Measures linear correlation between variables.
        - Range: [-1, 1]
        - 1 = perfect positive correlation
        - 0 = no correlation
        - -1 = perfect negative correlation
        
        Returns:
        --------
        Dict with correlation coefficient for each column pair
        """
        results = {}
        
        if len(numerical_cols) < 2:
            logger.warning("Need at least 2 numerical columns for correlation")
            return results
        
        real_corr = self.real_df[numerical_cols].corr()
        syn_corr = self.synthetic_df[numerical_cols].corr()
        
        # Calculate correlation between correlation matrices
        real_corr_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
        syn_corr_flat = syn_corr.values[np.triu_indices_from(syn_corr.values, k=1)]
        
        if len(real_corr_flat) > 0:
            pearson_r, p_value = stats.pearsonr(real_corr_flat, syn_corr_flat)
            
            results['overall_correlation'] = {
                'pearson_r': pearson_r,
                'p_value': p_value,
                'similar': pearson_r > 0.8,  # High correlation
            }
        
        return results

    def wasserstein_distance_test(self, numerical_cols: List[str]) -> Dict[str, float]:
        """
        Calculate Wasserstein Distance (Earth Mover's Distance) for numerical columns.
        
        Measures the minimum cost to transform one distribution into another.
        - Lower values = more similar distributions
        - 0 = identical distributions
        
        Returns:
        --------
        Dict with Wasserstein distance for each column
        """
        results = {}
        
        for col in numerical_cols:
            real_data = self.real_df[col].dropna().values
            syn_data = self.synthetic_df[col].dropna().values
            
            if len(real_data) == 0 or len(syn_data) == 0:
                continue
            
            distance = wasserstein_distance(real_data, syn_data)
            
            # Normalize by data range
            data_range = max(real_data.max(), syn_data.max()) - min(real_data.min(), syn_data.min())
            normalized_distance = distance / data_range if data_range > 0 else 0
            
            results[col] = {
                'distance': distance,
                'normalized_distance': normalized_distance,
                'similar': normalized_distance < 0.1,  # Threshold
            }
        
        return results

    def evaluate_all(self) -> Dict:
        """Run all evaluation tests and return comprehensive results."""
        logger.info("Starting comprehensive evaluation...")
        
        numerical_cols, categorical_cols = self.identify_column_types()
        
        logger.info("Numerical columns: %d", len(numerical_cols))
        logger.info("Categorical columns: %d", len(categorical_cols))
        
        results = {
            'column_types': {
                'numerical': numerical_cols,
                'categorical': categorical_cols,
            },
            'ks_test': self.kolmogorov_smirnov_test(numerical_cols) if numerical_cols else {},
            'chi_square_test': self.chi_square_test(categorical_cols) if categorical_cols else {},
            'jsd': self.jensen_shannon_divergence(categorical_cols) if categorical_cols else {},
            'pearson_correlation': self.pearson_correlation(numerical_cols) if numerical_cols else {},
            'wasserstein_distance': self.wasserstein_distance_test(numerical_cols) if numerical_cols else {},
        }
        
        logger.info("Evaluation complete!")
        return results

    def display_results(self, results: Dict) -> None:
        """Display results in formatted tables."""
        
        print("\n" + "="*80)
        print("SYNTHETIC DATA QUALITY EVALUATION")
        print("="*80 + "\n")
        
        # KS Test Results
        if results['ks_test']:
            print("KOLMOGOROV-SMIRNOV TEST (Numerical Columns)")
            print("-" * 80)
            table = PrettyTable()
            table.field_names = ["Column", "KS Statistic", "p-value", "Similar?"]
            
            for col, res in results['ks_test'].items():
                table.add_row([
                    col[:30],
                    f"{res['statistic']:.4f}",
                    f"{res['p_value']:.4f}",
                    "✓" if res['similar'] else "✗"
                ])
            
            print(table)
            print()
        
        # Chi-Square Test Results
        if results['chi_square_test']:
            print("CHI-SQUARE TEST (Categorical Columns)")
            print("-" * 80)
            table = PrettyTable()
            table.field_names = ["Column", "Chi2 Statistic", "p-value", "Similar?"]
            
            for col, res in results['chi_square_test'].items():
                table.add_row([
                    col[:30],
                    f"{res['chi2_statistic']:.4f}",
                    f"{res['p_value']:.4f}",
                    "✓" if res['similar'] else "✗"
                ])
            
            print(table)
            print()
        
        # JSD Results
        if results['jsd']:
            print("JENSEN-SHANNON DIVERGENCE (Categorical Columns)")
            print("-" * 80)
            table = PrettyTable()
            table.field_names = ["Column", "JSD", "Similar?"]
            
            for col, res in results['jsd'].items():
                table.add_row([
                    col[:30],
                    f"{res['jsd']:.4f}",
                    "✓" if res['similar'] else "✗"
                ])
            
            print(table)
            print()
        
        # Pearson Correlation Results
        if results['pearson_correlation']:
            print("PEARSON CORRELATION (Correlation Structure)")
            print("-" * 80)
            table = PrettyTable()
            table.field_names = ["Metric", "Value", "Similar?"]
            
            for key, res in results['pearson_correlation'].items():
                table.add_row([
                    key,
                    f"{res['pearson_r']:.4f}",
                    "✓" if res['similar'] else "✗"
                ])
            
            print(table)
            print()
        
        # Wasserstein Distance Results
        if results['wasserstein_distance']:
            print("WASSERSTEIN DISTANCE (Numerical Columns)")
            print("-" * 80)
            table = PrettyTable()
            table.field_names = ["Column", "Distance", "Normalized", "Similar?"]
            
            for col, res in results['wasserstein_distance'].items():
                table.add_row([
                    col[:30],
                    f"{res['distance']:.4f}",
                    f"{res['normalized_distance']:.4f}",
                    "✓" if res['similar'] else "✗"
                ])
            
            print(table)
            print()
        
        # Summary
        self._print_summary(results)

    def _print_summary(self, results: Dict) -> None:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        
        total_tests = 0
        passed_tests = 0
        
        # Count KS tests
        if results['ks_test']:
            for res in results['ks_test'].values():
                total_tests += 1
                if res['similar']:
                    passed_tests += 1
        
        # Count Chi-Square tests
        if results['chi_square_test']:
            for res in results['chi_square_test'].values():
                total_tests += 1
                if res['similar']:
                    passed_tests += 1
        
        # Count JSD tests
        if results['jsd']:
            for res in results['jsd'].values():
                total_tests += 1
                if res['similar']:
                    passed_tests += 1
        
        # Count Correlation tests
        if results['pearson_correlation']:
            for res in results['pearson_correlation'].values():
                total_tests += 1
                if res['similar']:
                    passed_tests += 1
        
        # Count Wasserstein tests
        if results['wasserstein_distance']:
            for res in results['wasserstein_distance'].values():
                total_tests += 1
                if res['similar']:
                    passed_tests += 1
        
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print("="*80 + "\n")
        
        if pass_rate >= 80:
            print("EXCELLENT: Synthetic data closely resembles real data")
        elif pass_rate >= 60:
            print("GOOD: Synthetic data is reasonably similar to real data")
        elif pass_rate >= 40:
            print("FAIR: Synthetic data has noticeable differences from real data")
        else:
            print("POOR: Synthetic data significantly differs from real data")
        
        print()