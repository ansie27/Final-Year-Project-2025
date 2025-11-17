# Abstract class that implements common preprocessing, evaluation, and reporting
# This is to ensure fairness between the evaluated models

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


class BaseRiskPredictor(ABC):
    """
    Abstract base class for supplier/commodity risk prediction models.
    
    All models (RF, XGBoost, ANN) inherit from this to ensure:
    - Consistent data preprocessing
    - Standardized training/evaluation
    - Uniform prediction interface
    - Fair performance comparison
    
    Attributes
    ----------
    model : object
        The underlying model instance (RandomForestClassifier, XGBClassifier, etc.)
    model_name : str
        Human-readable model name for reporting
    is_fitted : bool
        Whether model has been trained
    feature_names : list of str
        Names of input features
    classes : np.ndarray
        Unique class labels (e.g., ['Low Risk', 'Medium Risk', 'High Risk'])
    performance_metrics : dict
        Evaluation metrics from most recent test
    """
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize base risk predictor.
        
        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "Random Forest", "XGBoost", "ANN")
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.classes = None
        self.performance_metrics = {}
        
        logger.info(f"Initialized {self.model_name} predictor")
    
    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build the underlying model instance.
        
        Must be implemented by each subclass to create:
        - RandomForestClassifier for RF
        - XGBClassifier for XGBoost
        - Sequential model for ANN
        
        Parameters
        ----------
        **kwargs : dict
            Model-specific hyperparameters
        
        Returns
        -------
        model : object
            Built model instance
        """
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        **fit_params
    ) -> 'BaseRiskPredictor':
        """
        Train the model on risk data.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix (supplier/commodity attributes)
        y : pd.Series, shape (n_samples,)
            Target risk labels (e.g., 0=Low, 1=Medium, 2=High)
        validation_split : float, default=0.2
            Proportion of data to use for validation
        **fit_params : dict
            Additional parameters for model training
            (e.g., verbose=False, callbacks=[...])
        
        Returns
        -------
        self : BaseRiskPredictor
            Fitted model instance
        
        Notes
        -----
        Automatically handles train/validation split and stores metrics.
        """
        logger.info(f"Training {self.model_name}...")
        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.classes = np.unique(y)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.debug(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Model-specific training
        self._fit_model(X_train, y_train, X_val, y_val, **fit_params)
        
        self.is_fitted = True
        
        # Evaluate on validation set
        self._evaluate_on_validation(X_val, y_val)
        
        logger.info(f"{self.model_name} training completed")
        return self
    
    @abstractmethod
    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **fit_params
    ):
        """
        Model-specific training logic.
        
        Must be implemented by subclasses.
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk labels for new data.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted risk class labels
        
        Raises
        ------
        RuntimeError
            If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            Feature matrix
        
        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Class probability estimates
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on test set.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            True test labels
        average : str, default='weighted'
            Averaging strategy for multi-class metrics
            Options: 'micro', 'macro', 'weighted'
        
        Returns
        -------
        metrics : dict
            Dictionary containing:
            - accuracy
            - precision
            - recall
            - f1_score
            - roc_auc (if binary or with OvR)
            - confusion_matrix
        """
        logger.info(f"Evaluating {self.model_name} on test set...")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC-AUC for multi-class (One-vs-Rest)
        try:
            if len(self.classes) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multi-class (OvR)
                metrics['roc_auc'] = roc_auc_score(
                    y_test, y_pred_proba,
                    multi_class='ovr',
                    average=average
                )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
        self.performance_metrics = metrics
        
        # Log summary
        logger.info(f"{self.model_name} Test Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def _evaluate_on_validation(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Internal validation set evaluation during training."""
        val_metrics = self.evaluate(X_val, y_val)
        logger.debug(f"Validation metrics stored: {val_metrics}")
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, np.ndarray]:
        """
        Perform k-fold cross-validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        cv : int, default=5
            Number of folds
        scoring : str, default='f1_weighted'
            Scoring metric to use
        
        Returns
        -------
        cv_results : dict
            Contains 'scores' array and 'mean'/'std' statistics
        """
        logger.info(f"Performing {cv}-fold cross-validation for {self.model_name}...")
        
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        logger.info(f"CV {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Must be implemented by subclasses.
        Different models have different importance metrics:
        - RF/XGBoost: built-in feature_importances_
        - ANN: use SHAP or permutation importance
        
        Returns
        -------
        importance_df : pd.DataFrame
            Columns: ['feature', 'importance']
            Sorted by importance descending
        """
        pass
    
    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """
        Generate detailed classification report.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            True labels
        
        Returns
        -------
        report : str
            Formatted classification report
        """
        y_pred = self.predict(X_test)
        
        # Map numeric labels to risk level names if available
        target_names = [f"Risk_Level_{c}" for c in self.classes]
        
        return classification_report(
            y_test, y_pred,
            target_names=target_names,
            digits=4
        )
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save model (e.g., 'outputs/models/rf_model.pkl')
        """
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"{self.model_name} saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'BaseRiskPredictor':
        """
        Load trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved model
        
        Returns
        -------
        model : BaseRiskPredictor
            Loaded model instance
        """
        import joblib
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name} ({status})"