"""
Random Forest implementation for risk prediction.

Implements Section 3.1.1 of documentation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_risk_predictor import BaseRiskPredictor
import logging

logger = logging.getLogger(__name__)


class RandomForestRiskPredictor(BaseRiskPredictor):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        class_weight: str = 'balanced',
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(model_name="Random Forest", random_state=random_state)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build RandomForestClassifier with specified hyperparameters."""
        logger.debug(f"Building Random Forest with {self.n_estimators} trees")
        
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,  # Out-of-bag score for internal validation
            verbose=0
        )
    
    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **fit_params
    ):
        """Train Random Forest model."""
        self.model.fit(X_train, y_train)
        
        # Log OOB score
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"OOB Score: {self.model.oob_score_:.4f}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        
        Uses Gini importance (mean decrease in impurity).
        
        Returns
        -------
        importance_df : pd.DataFrame
            Columns: ['feature', 'importance']
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def get_tree_depth_stats(self) -> dict:
        """
        Get statistics about tree depths in the forest.
        
        Returns
        -------
        stats : dict
            Contains min, max, mean, median tree depths
        """
        depths = [tree.get_depth() for tree in self.model.estimators_]
        
        return {
            'min_depth': np.min(depths),
            'max_depth': np.max(depths),
            'mean_depth': np.mean(depths),
            'median_depth': np.median(depths)
        }