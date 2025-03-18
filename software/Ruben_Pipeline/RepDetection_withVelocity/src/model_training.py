# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:18:41 2025

@author: rkfurst
"""

# src/model_training.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Any
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTrainer")

class ModelTrainer:
    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize the ModelTrainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_training_data(self, features_df: pd.DataFrame, true_velocities: Dict[str, float]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare the training data by mapping true velocities to features.
        
        Args:
            features_df: DataFrame containing extracted features
            true_velocities: Dictionary mapping 'file_name_repN' to true velocity values
            
        Returns:
            Tuple of (feature DataFrame, target array)
        """
        # Create target vector
        y = []
        filtered_indices = []
        
        for i, row in features_df.iterrows():
            file_name = row['file_name']
            rep_number = row['rep_number']
            key = f"{file_name}_rep{rep_number}"
            
            if key in true_velocities:
                y.append(true_velocities[key])
                filtered_indices.append(i)
            else:
                logger.warning(f"No true velocity found for {key}")
        
        # Filter features to only include those with true velocities
        X = features_df.iloc[filtered_indices].copy()
        
        # Remove identifier columns
        X = X.drop(columns=['file_name', 'rep_number'])
        
        logger.info(f"Prepared training data with {len(X)} samples")
        return X, np.array(y)
    
    def train_velocity_model(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Train a model to predict velocity from features.
        
        Args:
            X: Feature DataFrame
            y: Target velocity array
            
        Returns:
            Tuple of (trained model pipeline, performance metrics)
        """
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = pipeline.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        logger.info(f"Model trained with metrics: {metrics}")
        return pipeline, metrics
    
    def perform_cross_validation(self, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation to assess model stability.
        
        Args:
            X: Feature DataFrame
            y: Target velocity array
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of CV scores
        """
        # Create the pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Set up cross-validation
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Calculate cross-validation scores
        cv_scores = {
            'mse': -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_squared_error'),
            'mae': -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error'),
            'r2': cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        }
        
        # Add RMSE
        cv_scores['rmse'] = np.sqrt(cv_scores['mse'])
        
        # Calculate mean and std for all metrics
        cv_summary = {
            metric: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            for metric, scores in cv_scores.items()
        }
        
        logger.info(f"Cross-validation results: {cv_summary}")
        return cv_summary
    
    def train_and_save_model(self, features_df: pd.DataFrame, true_velocities: Dict[str, float]) -> str:
        """
        Train and save the model.
        
        Args:
            features_df: DataFrame containing extracted features
            true_velocities: Dictionary mapping 'file_name_repN' to true velocity values
            
        Returns:
            Path to the saved model
        """
        # Prepare training data
        X, y = self.prepare_training_data(features_df, true_velocities)
        
        # Perform cross-validation
        cv_results = self.perform_cross_validation(X, y)
        
        # Train the final model on all data
        model, metrics = self.train_velocity_model(X, y)
        
        # Get feature importance
        feature_importance = None
        if hasattr(model['regressor'], 'feature_importances_'):
            feature_importance = {
                feature: float(importance)
                for feature, importance in zip(X.columns, model['regressor'].feature_importances_)
            }
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Save the model
        model_path = os.path.join(self.models_dir, "velocity_model.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'metrics': metrics,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'feature_names': list(X.columns),
            'n_samples': len(X)
        }
        
        metadata_path = os.path.join(self.models_dir, "velocity_model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path