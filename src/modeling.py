"""
Machine Learning Modeling Module for Green Supply Chain Risk Management

This module provides functions for building and training machine learning models
for risk prediction, classification, and clustering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
import warnings
warnings.filterwarnings('ignore')

# Missing XGBoost and ANN + selecting the most suitable model


def prepare_features(data, target_column='Overall_Risk_Score', exclude_columns=None):
    """
    Prepare features for machine learning models.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target variable column
    exclude_columns : list, optional
        Columns to exclude from features
        
    Returns:
    --------
    tuple
        X (features), y (target), feature_names
    """
    if exclude_columns is None:
        exclude_columns = ['Supplier_ID', 'Supplier_Name', 'Risk_Classification']
    
    # Select numeric features
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [f for f in numeric_features 
                      if f not in exclude_columns and f != target_column]
    
    X = data[feature_columns].fillna(data[feature_columns].median())
    y = data[target_column].fillna(data[target_column].median())
    
    return X, y, feature_columns


def train_risk_prediction_model(data, target_column='Overall_Risk_Score', 
                                model_type='random_forest', test_size=0.2, random_state=42):
    """
    Train a model to predict risk scores.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target variable column
    model_type : str
        Type of model ('random_forest', 'gradient_boosting', 'linear')
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Model, predictions, and evaluation metrics
    """
    # Prepare data
    X, y, feature_names = prepare_features(data, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        X_train_final = X_train
        X_test_final = X_test
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        X_train_final = X_train
        X_test_final = X_test
    elif model_type == 'linear':
        model = LinearRegression()
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_final)
    y_test_pred = model.predict(X_test_final)
    
    # Evaluation metrics
    train_metrics = {
        'mse': mean_squared_error(y_train, y_train_pred),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_test_pred),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
    }
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = None
    
    results = {
        'model': model,
        'scaler': scaler if model_type == 'linear' else None,
        'feature_names': feature_names,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance
    }
    
    return results


def train_risk_classification_model(data, target_column='Risk_Classification',
                                   model_type='random_forest', test_size=0.2, random_state=42):
    """
    Train a model to classify risk levels.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target classification column
    model_type : str
        Type of model ('random_forest', 'logistic')
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Model, predictions, and evaluation metrics
    """
    # Prepare data
    exclude_cols = ['Supplier_ID', 'Supplier_Name', 'Overall_Risk_Score']
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [f for f in numeric_features if f not in exclude_cols]
    
    X = data[feature_columns].fillna(data[feature_columns].median())
    y = data[target_column]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        X_train_final = X_train
        X_test_final = X_test
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_final)
    y_test_pred = model.predict(X_test_final)
    
    # Evaluation metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_test_pred, 
                                        target_names=le.classes_, zero_division=0)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = None
    
    results = {
        'model': model,
        'label_encoder': le,
        'scaler': scaler,
        'feature_names': feature_columns,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'feature_importance': feature_importance
    }
    
    return results


def perform_clustering(data, n_clusters=4, method='kmeans', features=None):
    """
    Perform clustering analysis on suppliers.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    n_clusters : int
        Number of clusters
    method : str
        Clustering method ('kmeans', 'dbscan')
    features : list, optional
        Features to use for clustering
        
    Returns:
    --------
    dict
        Clustering results
    """
    # Select features
    if features is None:
        features = [
            'Environmental_Risk_Score', 'Compliance_Risk_Score',
            'Operational_Risk_Score', 'Financial_Risk_Score'
        ]
    
    available_features = [f for f in features if f in data.columns]
    X = data[available_features].fillna(data[available_features].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = clusterer.fit_predict(X_scaled)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        clusters = clusterer.fit_predict(X_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Calculate silhouette score
    if len(np.unique(clusters)) > 1:
        silhouette = silhouette_score(X_scaled, clusters)
    else:
        silhouette = -1
    
    # Add clusters to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # Cluster statistics
    cluster_stats = data_with_clusters.groupby('Cluster')[available_features].mean()
    
    results = {
        'clusterer': clusterer,
        'scaler': scaler,
        'clusters': clusters,
        'silhouette_score': silhouette,
        'data_with_clusters': data_with_clusters,
        'cluster_stats': cluster_stats,
        'n_clusters': len(np.unique(clusters))
    }
    
    return results


def perform_cross_validation(data, target_column='Overall_Risk_Score', 
                            model_type='random_forest', cv=5, random_state=42):
    """
    Perform cross-validation for model evaluation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    target_column : str
        Target variable column
    model_type : str
        Type of model
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    # Prepare data
    X, y, feature_names = prepare_features(data, target_column)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        X_final = X
    elif model_type == 'linear':
        model = LinearRegression()
        X_final = X_scaled
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_final, y, cv=cv, 
                               scoring='r2', n_jobs=-1)
    
    results = {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'model_type': model_type
    }
    
    return results


if __name__ == "__main__":
    print("Modeling Module")
    print("Import this module to use ML modeling functions in your pipeline.")

