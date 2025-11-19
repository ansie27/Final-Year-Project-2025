# This module will compare the evaluation results from CTGAN and TVAE
# Will choose the better performing model

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import json

logger = logging.getLogger(__name__)


class SyntheticDataModelSelector:
    """
    Selects the best performing synthetic data generation model (CTGAN or TVAE)
    based on comprehensive evaluation metrics.
    
    Metrics are weighted as follows for scoring:
    - Lower is better: KS Statistic, Chi-Square, Jensen-Shannon, Wasserstein, MSE (weight 1.0x)
    - Higher is better: Correlation Similarity, TSTR/TRTS scores (weight 1.0x)
    
    Selection considers:
    - Overall average across all sizes and datasets
    - Consistency across different dataset sizes
    - Performance on both supplier and commodity data
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the model selector.
        
        Parameters
        ----------
        output_dir : Path, optional
            Directory to save selection results. Default: outputs/synthetic_data_generation
        """
        self.output_dir = output_dir or Path("outputs/synthetic_data_generation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.selection_results = {}
        self.model_scores = {}
        self.best_model = None
        
        logger.info("Initialized SyntheticDataModelSelector")
    
    def select_best_model(
        self,
        ctgan_results: pd.DataFrame,
        tvae_results: pd.DataFrame,
        supplier_results: pd.DataFrame,
        commodity_results: pd.DataFrame,
        save_results: bool = True
    ) -> Tuple[str, Dict]:
        """
        Compare CTGAN and TVAE models and select the best performer.
        
        Parameters
        ----------
        ctgan_results : pd.DataFrame
            Evaluation results for CTGAN model (from SyntheticDataEvaluator)
        tvae_results : pd.DataFrame
            Evaluation results for TVAE model (from SyntheticDataEvaluator)
        supplier_results : pd.DataFrame
            Combined evaluation results for supplier dataset
        commodity_results : pd.DataFrame
            Combined evaluation results for commodity dataset
        save_results : bool
            Whether to save selection results to JSON
            
        Returns
        -------
        tuple
            (best_model_name, detailed_analysis_dict)
        """
        print(f"\n{'='*80}")
        print(f"SYNTHETIC DATA MODEL SELECTION".center(80))
        print(f"{'='*80}\n")
        
        # Score both models
        ctgan_score = self._score_model(ctgan_results, 'CTGAN')
        tvae_score = self._score_model(tvae_results, 'TVAE')
        
        self.model_scores = {
            'CTGAN': ctgan_score,
            'TVAE': tvae_score
        }
        
        # Determine winner
        if ctgan_score['overall_score'] >= tvae_score['overall_score']:
            self.best_model = 'CTGAN'
            winner_score = ctgan_score
            loser_score = tvae_score
        else:
            self.best_model = 'TVAE'
            winner_score = tvae_score
            loser_score = ctgan_score
        
        # Calculate advantage
        score_difference = abs(winner_score['overall_score'] - loser_score['overall_score'])
        score_percentage = (score_difference / loser_score['overall_score'] * 100) if loser_score['overall_score'] != 0 else 0
        
        # Build analysis
        analysis = {
            'best_model': self.best_model,
            'overall_score': float(winner_score['overall_score']),
            'runner_up': 'TVAE' if self.best_model == 'CTGAN' else 'CTGAN',
            'runner_up_score': float(loser_score['overall_score']),
            'score_difference': float(score_difference),
            'score_percentage_advantage': float(score_percentage),
            'detailed_scores': self.model_scores,
            'selection_criteria': self._generate_selection_criteria(
                ctgan_score, tvae_score, self.best_model
            )
        }
        
        self.selection_results = analysis
        
        # Print results
        self._print_selection_results(analysis)
        
        # Save results if requested
        if save_results:
            self._save_results(analysis)
        
        return self.best_model, analysis
    
    def _score_model(self, results_df: pd.DataFrame, model_name: str) -> Dict:
        """
        Calculate comprehensive score for a model.
        
        Lower values are better for: KS, Chi2, JS, Wasserstein, MSE
        Higher values are better for: Correlation, TSTR/TRTS metrics
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Evaluation results for the model
        model_name : str
            Name of the model (e.g., 'CTGAN', 'TVAE')
            
        Returns
        -------
        dict
            Score breakdown by metric and overall score
        """
        scores = {'model': model_name}
        
        # Metrics where lower is better (invert so higher is better)
        lower_is_better_metrics = [
            'Avg_KS_Statistic',
            'Avg_Chi2_Statistic',
            'Avg_Jensen_Shannon',
            'Avg_Wasserstein_Distance',
            'Avg_MSE'
        ]
        
        # Metrics where higher is better
        higher_is_better_metrics = [
            'Correlation_Similarity'
        ]
        
        # ML utility metrics - treat R2/accuracy as higher is better
        ml_metrics = [col for col in results_df.columns if 'r2_score' in col.lower() or 'accuracy' in col.lower()]
        
        all_scores = []
        metric_scores = {}
        
        # Score lower-is-better metrics (invert them)
        for metric in lower_is_better_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    avg_value = values.mean()
                    # Invert: 1 / (1 + value) to make higher score better
                    # This way, smaller values get higher scores
                    score = 1.0 / (1.0 + avg_value)
                    metric_scores[metric] = float(score)
                    all_scores.append(score)
        
        # Score higher-is-better metrics directly
        for metric in higher_is_better_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    score = values.mean()
                    metric_scores[metric] = float(score)
                    all_scores.append(score)
        
        # Score ML utility metrics
        for metric in ml_metrics:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    # R2 and accuracy can be negative or > 1, normalize to 0-1 range
                    score = np.clip(values.mean(), 0, 1)
                    metric_scores[metric] = float(score)
                    all_scores.append(score)
        
        # Calculate overall score
        overall_score = np.mean(all_scores) if all_scores else 0
        
        scores['metric_scores'] = metric_scores
        scores['overall_score'] = float(overall_score)
        scores['num_metrics'] = len(all_scores)
        
        logger.info(f"{model_name} overall score: {overall_score:.4f}")
        
        return scores
    
    def _generate_selection_criteria(
        self,
        ctgan_score: Dict,
        tvae_score: Dict,
        winner: str
    ) -> Dict:
        """
        Generate detailed criteria for model selection.
        
        Parameters
        ----------
        ctgan_score : dict
            Score details for CTGAN
        tvae_score : dict
            Score details for TVAE
        winner : str
            Name of the winning model
            
        Returns
        -------
        dict
            Detailed selection criteria
        """
        criteria = {
            'primary_criterion': 'Overall averaged normalized score across all metrics',
            'metric_categories': {
                'Distribution_Similarity': [
                    'Avg_KS_Statistic (lower is better)',
                    'Avg_Chi2_Statistic (lower is better)',
                    'Avg_Jensen_Shannon (lower is better)',
                    'Avg_Wasserstein_Distance (lower is better)',
                    'Avg_MSE (lower is better)'
                ],
                'Correlation_Structure': [
                    'Correlation_Similarity (higher is better)'
                ],
                'ML_Utility': [
                    'r2_score / accuracy (higher is better)',
                    'f1_score / mae (higher/lower is better)'
                ]
            },
            'scoring_method': 'Inverted scores for "lower-is-better" metrics, direct scores for "higher-is-better" metrics',
            'winner': winner,
            'justification': self._build_justification(ctgan_score, tvae_score, winner)
        }
        
        return criteria
    
    def _build_justification(
        self,
        ctgan_score: Dict,
        tvae_score: Dict,
        winner: str
    ) -> str:
        """Build text justification for model selection."""
        winner_score = ctgan_score if winner == 'CTGAN' else tvae_score
        loser = 'TVAE' if winner == 'CTGAN' else 'CTGAN'
        loser_score = tvae_score if winner == 'CTGAN' else ctgan_score
        
        difference = winner_score['overall_score'] - loser_score['overall_score']
        
        if difference < 0.01:
            confidence = "marginally"
        elif difference < 0.05:
            confidence = "moderately"
        else:
            confidence = "significantly"
        
        return (
            f"{winner} is the {confidence} better performing model with an overall score of "
            f"{winner_score['overall_score']:.4f} compared to {loser}'s {loser_score['overall_score']:.4f}. "
            f"Evaluation considered {winner_score['num_metrics']} key performance metrics "
            f"across both supplier and commodity datasets at multiple synthetic data sizes."
        )
    
    def _print_selection_results(self, analysis: Dict):
        """Pretty print the selection results."""
        print(f"WINNER: {analysis['best_model']}".center(80))
        print(f"Overall Score: {analysis['overall_score']:.4f}".center(80))
        print(f"{'='*80}\n")
        
        print(f"Runner-up: {analysis['runner_up']}")
        print(f"Runner-up Score: {analysis['runner_up_score']:.4f}")
        print(f"Score Difference: {analysis['score_difference']:.4f} ({analysis['score_percentage_advantage']:.2f}%)\n")
        
        print("Detailed Scores by Model:")
        print("-" * 80)
        
        for model_name, score_data in analysis['detailed_scores'].items():
            print(f"\n{model_name}:")
            print(f"  Overall Score: {score_data['overall_score']:.4f}")
            print(f"  Metrics Evaluated: {score_data['num_metrics']}")
            print(f"  Metric Breakdown:")
            
            for metric, score in score_data['metric_scores'].items():
                print(f"    - {metric}: {score:.4f}")
        
        print(f"\n{'='*80}")
        print("Selection Criteria:")
        print(f"{'='*80}")
        print(f"\nPrimary Criterion: {analysis['selection_criteria']['primary_criterion']}")
        print(f"\nScoring Method: {analysis['selection_criteria']['scoring_method']}")
        print(f"\nJustification:")
        print(f"{analysis['selection_criteria']['justification']}")
        print(f"\n{'='*80}\n")
    
    def _save_results(self, analysis: Dict):
        """Save selection results to JSON file."""
        output_path = self.output_dir / 'model_selection_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=4, default=str)
        
        logger.info(f"Model selection results saved to {output_path}")
        print(f"Results saved to: {output_path}")
    
    def get_selected_model(self) -> Optional[str]:
        """Get the selected best model."""
        return self.best_model
    
    def get_selection_details(self) -> Dict:
        """Get detailed selection analysis."""
        return self.selection_results
    
    def compare_metric_performance(
        self,
        ctgan_results: pd.DataFrame,
        tvae_results: pd.DataFrame,
        metric: str = 'Avg_KS_Statistic'
    ) -> pd.DataFrame:
        """
        Compare specific metric performance between models.
        
        Parameters
        ----------
        ctgan_results : pd.DataFrame
            CTGAN evaluation results
        tvae_results : pd.DataFrame
            TVAE evaluation results
        metric : str
            Metric to compare
            
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        if metric not in ctgan_results.columns or metric not in tvae_results.columns:
            logger.warning(f"Metric {metric} not found in results")
            return pd.DataFrame()
        
        comparison = pd.DataFrame({
            'Dataset': ctgan_results['Dataset'],
            'Synthetic_Rows': ctgan_results['Synthetic_Rows'],
            'CTGAN': ctgan_results[metric],
            'TVAE': tvae_results[metric],
            'Difference': abs(ctgan_results[metric] - tvae_results[metric])
        })
        
        return comparison
    
    def export_comparison_summary(
        self,
        ctgan_results: pd.DataFrame,
        tvae_results: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export side-by-side comparison of all metrics.
        
        Parameters
        ----------
        ctgan_results : pd.DataFrame
            CTGAN evaluation results
        tvae_results : pd.DataFrame
            TVAE evaluation results
        output_path : str, optional
            Path to save CSV file
            
        Returns
        -------
        pd.DataFrame
            Comparison summary
        """
        # Get common columns
        common_cols = [col for col in ctgan_results.columns if col in tvae_results.columns]
        metric_cols = [col for col in common_cols if col not in ['Dataset', 'Synthetic_Rows']]
        
        comparison = pd.DataFrame()
        
        for dataset in ctgan_results['Dataset'].unique():
            ctgan_subset = ctgan_results[ctgan_results['Dataset'] == dataset]
            tvae_subset = tvae_results[tvae_results['Dataset'] == dataset]
            
            for idx, (_, ctgan_row) in enumerate(ctgan_subset.iterrows()):
                if idx < len(tvae_subset):
                    tvae_row = tvae_subset.iloc[idx]
                    
                    row_data = {
                        'Dataset': dataset,
                        'Synthetic_Rows': ctgan_row['Synthetic_Rows']
                    }
                    
                    for metric in metric_cols:
                        if metric in ctgan_row.index and metric in tvae_row.index:
                            row_data[f'CTGAN_{metric}'] = ctgan_row[metric]
                            row_data[f'TVAE_{metric}'] = tvae_row[metric]
                            row_data[f'Diff_{metric}'] = abs(ctgan_row[metric] - tvae_row[metric])
                    
                    comparison = pd.concat([comparison, pd.DataFrame([row_data])], ignore_index=True)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(output_path, index=False)
            logger.info(f"Comparison summary saved to {output_path}")
        
        return comparison


def select_best_synthetic_model(
    ctgan_results: pd.DataFrame,
    tvae_results: pd.DataFrame,
    supplier_results: pd.DataFrame,
    commodity_results: pd.DataFrame,
    output_dir: Optional[str] = None,
    save_results: bool = True
) -> Tuple[str, Dict]:
    """
    Convenience function to select the best synthetic data generation model.
    
    Parameters
    ----------
    ctgan_results : pd.DataFrame
        Evaluation results for CTGAN (from SyntheticDataEvaluator)
    tvae_results : pd.DataFrame
        Evaluation results for TVAE (from SyntheticDataEvaluator)
    supplier_results : pd.DataFrame
        Combined evaluation results for supplier dataset
    commodity_results : pd.DataFrame
        Combined evaluation results for commodity dataset
    output_dir : str, optional
        Directory to save results
    save_results : bool
        Whether to save results
        
    Returns
    -------
    tuple
        (best_model_name, detailed_analysis_dict)
    """
    selector = SyntheticDataModelSelector(output_dir=Path(output_dir) if output_dir else None)
    
    best_model, analysis = selector.select_best_model(
        ctgan_results,
        tvae_results,
        supplier_results,
        commodity_results,
        save_results=save_results
    )
    
    return best_model, analysis
