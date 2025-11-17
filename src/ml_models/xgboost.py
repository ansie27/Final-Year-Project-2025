import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from .base_risk_predictor import BaseRiskPredictor
import logging

logger = logging.getLogger(__name__)

class XGBoostRiskPredictor(BaseRiskPredictor):    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(model_name="XGBoost", random_state=random_state)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        
        # Build model
        self.model = self._build_model()
        
        # Store training history
        self.evals_result = {}
    
    def _build_model(self):
        """Build XGBClassifier with specified hyperparameters."""
        logger.debug(f"Building XGBoost with {self.n_estimators} estimators")
        
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss',  # Multi-class log loss
            early_stopping_rounds=10,
            verbosity=0
        )
    
    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **fit_params
    ):
        """Train XGBoost model with early stopping."""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Store evaluation results
        self.evals_result = self.model.evals_result()
        
        # Log best iteration
        logger.info(f"Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score:.4f}")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from XGBoost.
        
        Parameters
        ----------
        importance_type : str, default='gain'
            Type of importance:
            - 'gain': average gain across all splits
            - 'weight': number of times feature appears
            - 'cover': average coverage of splits
        
        Returns
        -------
        importance_df : pd.DataFrame
            Columns: ['feature', 'importance']
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # XGBoost provides importance scores directly
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map feature names
        importance_df = pd.DataFrame([
            {'feature': self.feature_names[int(k.replace('f', ''))], 
             'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def plot_training_history(self):
        """
        Plot training and validation loss curves.
        
        Useful for diagnosing overfitting/underfitting.
        """
        import matplotlib.pyplot as plt
        
        results = self.evals_result
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
        ax.legend()
        ax.set_ylabel('Log Loss')
        ax.set_xlabel('Boosting Round')
        ax.set_title(f'{self.model_name} Training History')
        plt.show()
        
        return fig