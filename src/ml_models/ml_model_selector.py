"""
Machine Learning Model Selector for Risk Prediction

This module selects the most suitable model for risk prediction in the supply chain
risk management study based on comprehensive model comparison.

Selection Criteria:
- Primary metric: R² Score (higher is better)
- Secondary metrics: RMSE, MAE
- Interpretability: Model explainability for decision-making
- Computational efficiency: Training and prediction speed
- Robustness: Cross-validation performance

For this study, the selected model will be:
1. Compared across Random Forest, XGBoost, and Gradient Boosting
2. Evaluated using R² Score as primary metric
3. Used for final risk prediction in the pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import logging
import json

from .ml_comparison import ModelComparison, compare_regression_models

logger = logging.getLogger(__name__)


class MLModelSelector:
    """
    Selects the most suitable ML model for risk prediction based on comprehensive evaluation.
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        Initialize model selector.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        test_size : float
            Proportion of data for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        self.comparator = ModelComparison(random_state=random_state, test_size=test_size)
        self.selected_model = None
        self.selected_model_name = None
        self.comparison_results = None
        self.selection_criteria = self._get_selection_criteria()
        
        logger.info("Initialized MLModelSelector for risk prediction")
    
    def _get_selection_criteria(self) -> Dict[str, str]:
        """
        Get the selection criteria for model evaluation.
        
        Returns
        -------
        dict
            Criteria and their descriptions
        """
        return {
            'r2_score': 'Primary metric - proportion of variance explained (0-1)',
            'rmse': 'Root Mean Squared Error - lower is better',
            'mae': 'Mean Absolute Error - lower is better',
            'interpretability': 'Model explainability for business decision-making',
            'computational_efficiency': 'Training and prediction speed',
            'robustness': 'Model stability and generalization capability'
        }
    
    def select_best_model(
        self,
        data: pd.DataFrame,
        target_column: str = 'Overall_Risk_Score',
        exclude_columns: Optional[list] = None
    ) -> Tuple[str, object, Dict]:
        """
        Select the best ML model for risk prediction.
        
        Performs model comparison and selects the best performer based on:
        1. R² Score (primary metric)
        2. RMSE and MAE (secondary metrics)
        3. Model characteristics (interpretability, efficiency)
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset with features and target variable
        target_column : str
            Name of target variable column
        exclude_columns : list, optional
            Columns to exclude from feature selection
            
        Returns
        -------
        tuple
            (selected_model_name, selected_model, comparison_results)
        """
        logger.info("Starting model selection process...")
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Target column: {target_column}")
        
        # Run model comparison
        logger.info("\nPhase 1: Model Comparison")
        logger.info("-" * 80)
        self.comparison_results = self.comparator.compare_models(
            data,
            target_column=target_column,
            exclude_columns=exclude_columns
        )
        
        # Get best model from comparison
        self.selected_model_name, best_r2 = self.comparator.identify_best_model()
        self.selected_model = self.comparator.get_best_model()
        
        logger.info(f"\nPhase 2: Selection Result")
        logger.info("-" * 80)
        logger.info(f"Selected Model: {self.selected_model_name}")
        logger.info(f"R² Score: {best_r2:.4f}")
        
        # Print detailed report
        self._print_selection_report()
        
        return self.selected_model_name, self.selected_model, self.comparison_results
    
    def _print_selection_report(self):
        """Print detailed model selection report."""
        print("\n" + "="*80)
        print("MACHINE LEARNING MODEL SELECTION FOR RISK PREDICTION")
        print("="*80 + "\n")
        
        # Print selection criteria
        print("SELECTION CRITERIA:")
        print("-" * 80)
        for criterion, description in self.selection_criteria.items():
            print(f"  {criterion}: {description}")
        
        print("\n\nMODEL COMPARISON RESULTS:")
        print("-" * 80)
        
        # Print metrics for each model
        summary = self.comparator.get_comparison_summary()
        print(summary.to_string(index=False))
        
        # Print analysis
        print("\n\nANALYSIS:")
        print("-" * 80)
        
        if self.comparison_results:
            # Calculate performance delta
            sorted_models = sorted(
                [(name, result['metrics']['r2_score']) 
                 for name, result in self.comparison_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            best_r2 = sorted_models[0][1]
            
            for i, (model_name, r2) in enumerate(sorted_models, 1):
                delta = ((best_r2 - r2) / best_r2 * 100) if best_r2 > 0 else 0
                marker = " <-- SELECTED" if model_name == self.selected_model_name else ""
                print(f"  {i}. {model_name:<20} R²: {r2:.4f} ({delta:+.1f}% from best){marker}")
        
        print("\n\nRECOMMENDATION:")
        print("-" * 80)
        print(f"Selected Model: {self.selected_model_name}")
        print(f"\nRationale:")
        print(f"  - Highest R² Score: {self.comparator.best_score:.4f}")
        
        # Model-specific rationale
        if self.selected_model_name == "Random Forest":
            print(f"  - Excellent interpretability through feature importance")
            print(f"  - Robust to outliers and non-linear relationships")
            print(f"  - Handles mixed data types effectively")
            print(f"  - Fast prediction for real-time risk assessment")
        
        elif self.selected_model_name == "XGBoost":
            print(f"  - Superior gradient boosting performance")
            print(f"  - Handles complex feature interactions")
            print(f"  - Feature importance for model interpretability")
            print(f"  - Built-in regularization prevents overfitting")
        
        elif self.selected_model_name == "Gradient Boosting":
            print(f"  - Strong sequential learning approach")
            print(f"  - Excellent generalization capability")
            print(f"  - Reduced variance through multiple learners")
            print(f"  - Interpretable through feature importance")
        
        print("\n" + "="*80 + "\n")
    
    def get_selected_model(self) -> object:
        """
        Get the selected model.
        
        Returns
        -------
        object
            The selected ML model
        """
        if self.selected_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        return self.selected_model
    
    def get_selected_model_name(self) -> str:
        """
        Get the name of the selected model.
        
        Returns
        -------
        str
            Name of selected model
        """
        if self.selected_model_name is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        return self.selected_model_name
    
    def get_model_metrics(self, model_name: Optional[str] = None) -> Dict:
        """
        Get performance metrics for a specific model.
        
        Parameters
        ----------
        model_name : str, optional
            Name of model. If None, returns metrics for selected model.
            
        Returns
        -------
        dict
            Performance metrics
        """
        if self.comparison_results is None:
            raise ValueError("No comparison results available. Call select_best_model() first.")
        
        if model_name is None:
            model_name = self.selected_model_name
        
        if model_name not in self.comparison_results:
            raise ValueError(f"Model '{model_name}' not found in results")
        
        return self.comparison_results[model_name]['metrics']
    
    def export_selection_results(
        self,
        output_path: Union[str, Path]
    ):
        """
        Export model selection results to JSON file.
        
        Parameters
        ----------
        output_path : str or Path
            Path to save results
        """
        if self.comparison_results is None:
            raise ValueError("No results to export. Call select_best_model() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'selection_date': pd.Timestamp.now().isoformat(),
            'selected_model': self.selected_model_name,
            'selection_criteria': self.selection_criteria,
            'comparison_results': {},
            'best_metrics': self.get_model_metrics(self.selected_model_name)
        }
        
        # Add all model metrics
        for model_name, result in self.comparison_results.items():
            export_data['comparison_results'][model_name] = result['metrics']
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        logger.info(f"Selection results exported to {output_path}")
    
    def predict_risk(
        self,
        X_new: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Use selected model to predict risk scores for new data.
        
        Parameters
        ----------
        X_new : pd.DataFrame
            New features for prediction
        **kwargs : dict
            Additional arguments passed to model's predict method
            
        Returns
        -------
        np.ndarray
            Predicted risk scores
        """
        if self.selected_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        
        logger.info(f"Predicting risk using {self.selected_model_name}...")
        predictions = self.selected_model.predict(X_new, **kwargs)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from selected model.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            Feature importance DataFrame with columns ['feature', 'importance']
        """
        if self.selected_model is None:
            raise ValueError("No model selected. Call select_best_model() first.")
        
        # Different models have different methods for feature importance
        importance_df = None
        
        try:
            # Try to get feature importances_ attribute (sklearn models)
            if hasattr(self.selected_model, 'feature_importances_'):
                importances = self.selected_model.feature_importances_
                
                # Get feature names if available
                if hasattr(self.selected_model, 'feature_names_in_'):
                    feature_names = self.selected_model.feature_names_in_
                else:
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            # Try custom get_feature_importance method (custom predictor classes)
            elif hasattr(self.selected_model, 'get_feature_importance'):
                importance_df = self.selected_model.get_feature_importance()
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        if importance_df is None:
            logger.warning(f"Feature importance not available for {self.selected_model_name}")
            return pd.DataFrame()
        
        return importance_df.head(top_n)


def select_best_model_for_risk_prediction(
    data: pd.DataFrame,
    target_column: str = 'Overall_Risk_Score',
    output_dir: Optional[Union[str, Path]] = None,
    random_state: int = 42
) -> Tuple[str, object]:
    """
    Convenience function to select best model in one call.
    
    This function:
    1. Compares Random Forest, XGBoost, and Gradient Boosting models
    2. Evaluates using R² Score as primary metric
    3. Returns the best performing model
    4. Optionally exports results
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Target variable column
    output_dir : str or Path, optional
        Directory to save results
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (best_model_name, best_model)
    """
    selector = MLModelSelector(random_state=random_state)
    model_name, model, results = selector.select_best_model(data, target_column)
    
    if output_dir:
        output_path = Path(output_dir) / "model_selection_results.json"
        selector.export_selection_results(output_path)
    
    return model_name, model
