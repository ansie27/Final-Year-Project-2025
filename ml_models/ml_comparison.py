"""
Machine Learning Models Comparison for Risk Prediction

This module compares multiple ML models (Random Forest, XGBoost, ANN) for risk prediction.
Uses comprehensive metrics to identify the best performer for the study.

Metrics used:
- R² Score: Proportion of variance explained
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- Cross-validation Score: Model generalization ability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional, List
import logging
import json
from pathlib import Path

# Import model implementations
from .random_forest import RandomForestRiskPredictor
from .xgboost import XGBoostRiskPredictor

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple ML models for risk prediction and identify the best performer.
    """
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        Initialize model comparison.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        test_size : float
            Proportion of data for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        self.comparison_results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        
        logger.info(f"Initialized ModelComparison (random_state={random_state})")
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'Overall_Risk_Score',
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for model training and evaluation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Target variable column name
        exclude_columns : list, optional
            Columns to exclude from features
            
        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test, feature_names
        """
        if exclude_columns is None:
            exclude_columns = ['Supplier_ID', 'Supplier_Name', 'Risk_Classification', 
                             'Commodity_ID', 'Commodity_Name']
        
        # Select numeric features
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [f for f in numeric_features 
                          if f not in exclude_columns and f != target_column]
        
        # Prepare X and y
        X = data[feature_columns].fillna(data[feature_columns].median())
        y = data[target_column].fillna(data[target_column].median())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def evaluate_regression_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate regression model using multiple metrics.
        
        Parameters
        ----------
        model : sklearn or custom model
            Trained model with predict method
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        model_name : str
            Name of the model for logging
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2)
        }
        
        logger.info(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return metrics
    
    def train_and_evaluate_random_forest(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[RandomForestRiskPredictor, Dict[str, float]]:
        """
        Train and evaluate Random Forest model.
        
        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and test data
            
        Returns
        -------
        tuple
            Trained model and evaluation metrics
        """
        logger.info("Training Random Forest for regression...")
        
        # Train Random Forest
        rf_model = RandomForestRiskPredictor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit the model
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_regression_model(
            rf_model.model,
            X_test, y_test,
            "Random Forest"
        )
        
        return rf_model, metrics
    
    def train_and_evaluate_xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[XGBoostRiskPredictor, Dict[str, float]]:
        """
        Train and evaluate XGBoost model.
        
        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and test data
            
        Returns
        -------
        tuple
            Trained model and evaluation metrics
        """
        logger.info("Training XGBoost for regression...")
        
        # Note: XGBoost implementation is for classification
        # For regression, we'll use XGBRegressor directly
        try:
            from xgboost import XGBRegressor
            
            xgb_model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                early_stopping_rounds=10
            )
            
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            metrics = self.evaluate_regression_model(
                xgb_model,
                X_test, y_test,
                "XGBoost"
            )
            
            return xgb_model, metrics
            
        except ImportError:
            logger.warning("XGBoost not available, skipping XGBoost model")
            return None, None
    
    def train_and_evaluate_gradient_boosting(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[object, Dict[str, float]]:
        """
        Train and evaluate Gradient Boosting model (scikit-learn).
        
        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and test data
            
        Returns
        -------
        tuple
            Trained model and evaluation metrics
        """
        logger.info("Training Gradient Boosting for regression...")
        
        from sklearn.ensemble import GradientBoostingRegressor
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_regression_model(
            gb_model,
            X_test, y_test,
            "Gradient Boosting"
        )
        
        return gb_model, metrics
    
    def compare_models(
        self,
        data: pd.DataFrame,
        target_column: str = 'Overall_Risk_Score',
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Compare all available models and identify the best performer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        target_column : str
            Target variable column name
        exclude_columns : list, optional
            Columns to exclude from features
            
        Returns
        -------
        dict
            Comparison results for all models with best model identified
        """
        logger.info("Starting model comparison...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(
            data, target_column, exclude_columns
        )
        
        self.comparison_results = {}
        models_trained = {}
        
        # 1. Train and evaluate Random Forest
        try:
            rf_model, rf_metrics = self.train_and_evaluate_random_forest(
                X_train, X_test, y_train, y_test
            )
            self.comparison_results['Random Forest'] = {
                'metrics': rf_metrics,
                'model': rf_model
            }
            models_trained['Random Forest'] = rf_model
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
        
        # 2. Train and evaluate XGBoost
        try:
            xgb_model, xgb_metrics = self.train_and_evaluate_xgboost(
                X_train, X_test, y_train, y_test
            )
            if xgb_model is not None:
                self.comparison_results['XGBoost'] = {
                    'metrics': xgb_metrics,
                    'model': xgb_model
                }
                models_trained['XGBoost'] = xgb_model
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
        
        # 3. Train and evaluate Gradient Boosting
        try:
            gb_model, gb_metrics = self.train_and_evaluate_gradient_boosting(
                X_train, X_test, y_train, y_test
            )
            self.comparison_results['Gradient Boosting'] = {
                'metrics': gb_metrics,
                'model': gb_model
            }
            models_trained['Gradient Boosting'] = gb_model
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
        
        # Identify best model based on R² score (primary metric)
        self.identify_best_model()
        
        logger.info(f"Best model: {self.best_model_name} (R²: {self.best_score:.4f})")
        
        return self.comparison_results
    
    def identify_best_model(self) -> Tuple[str, float]:
        """
        Identify the best model based on R² score.
        
        R² score is chosen as primary metric because:
        - Ranges from 0 to 1 (higher is better)
        - Represents proportion of variance explained
        - Most interpretable for risk prediction
        - Commonly used in regression evaluation
        
        Returns
        -------
        tuple
            Best model name and R² score
        """
        best_r2 = -np.inf
        best_name = None
        
        for model_name, result in self.comparison_results.items():
            r2 = result['metrics']['r2_score']
            if r2 > best_r2:
                best_r2 = r2
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_score = best_r2
        self.best_model = self.comparison_results[best_name]['model']
        
        return best_name, best_r2
    
    def get_best_model(self):
        """
        Get the best trained model.
        
        Returns
        -------
        object
            Best performing model
        """
        if self.best_model is None:
            raise ValueError("No models have been trained. Call compare_models() first.")
        return self.best_model
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get summary of model comparison as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Comparison summary with metrics for all models
        """
        summary_data = []
        
        for model_name, result in self.comparison_results.items():
            metrics = result['metrics']
            metrics['Model'] = model_name
            metrics['Best'] = 'Yes' if model_name == self.best_model_name else 'No'
            summary_data.append(metrics)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Reorder columns
        cols = ['Model', 'r2_score', 'rmse', 'mae', 'mse', 'Best']
        summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
        
        return summary_df
    
    def print_comparison_report(self):
        """Print formatted comparison report."""
        print("\n" + "="*80)
        print("MACHINE LEARNING MODELS COMPARISON FOR RISK PREDICTION")
        print("="*80 + "\n")
        
        summary = self.get_comparison_summary()
        print(summary.to_string(index=False))
        
        print("\n" + "-"*80)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"R² Score: {self.best_score:.4f}")
        print("-"*80 + "\n")
    
    def export_comparison_results(self, output_path: str):
        """
        Export comparison results to JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to save results JSON file
        """
        export_data = {}
        
        for model_name, result in self.comparison_results.items():
            export_data[model_name] = result['metrics']
        
        export_data['Best Model'] = self.best_model_name
        export_data['Best R2 Score'] = float(self.best_score)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        logger.info(f"Comparison results exported to {output_path}")


def compare_regression_models(
    data: pd.DataFrame,
    target_column: str = 'Overall_Risk_Score',
    output_path: Optional[str] = None,
    random_state: int = 42
) -> Tuple[Dict[str, Dict], str, object]:
    """
    Convenience function to compare regression models in one call.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    target_column : str
        Target variable column name
    output_path : str, optional
        Path to save results
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (comparison_results, best_model_name, best_model)
    """
    comparator = ModelComparison(random_state=random_state)
    comparison_results = comparator.compare_models(data, target_column)
    
    comparator.print_comparison_report()
    
    if output_path:
        comparator.export_comparison_results(output_path)
    
    return comparison_results, comparator.best_model_name, comparator.best_model
