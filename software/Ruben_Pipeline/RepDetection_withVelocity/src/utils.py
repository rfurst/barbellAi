# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:21:23 2025

@author: rkfurst
"""

# src/utils.py
import os
import logging
import json
from typing import Dict, Any

def setup_logging(log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, logs to console only)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("DeadliftVelocity")
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_true_velocities_template(processed_dir: str = "data/processed", 
                                    output_path: str = "data/true_velocities.json"):
    """
    Create a template JSON file for entering true velocities.
    
    Args:
        processed_dir: Directory containing processed data
        output_path: Path to save the template JSON
    """
    # Load repetition metadata
    metadata_path = os.path.join(processed_dir, 'repetition_metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        rep_meta = json.load(f)
    
    # Create template with empty values
    template = {}
    for rep in rep_meta:
        file_name = rep['file_name']
        rep_number = rep['rep_number']
        key = f"{file_name}_rep{rep_number}"
        template[key] = 0.0  # Placeholder value
    
    # Save template
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"True velocities template created at {output_path}")
    print("Please fill in the actual velocity values before training the model.")

def calculate_model_performance(true_values: Dict[str, float], predicted_values: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate model performance metrics.
    
    Args:
        true_values: Dictionary of true velocity values
        predicted_values: Dictionary of predicted velocity values
        
    Returns:
        Dictionary of performance metrics
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Extract matching keys
    common_keys = set(true_values.keys()) & set(predicted_values.keys())
    
    if not common_keys:
        return {
            'error': 'No matching keys between true and predicted values'
        }
    
    # Create arrays
    y_true = np.array([true_values[k] for k in common_keys])
    y_pred = np.array([predicted_values[k] for k in common_keys])
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate percentage error
    percentage_errors = np.abs(y_true - y_pred) / y_true * 100
    mean_percentage_error = np.mean(percentage_errors)
    max_percentage_error = np.max(percentage_errors)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mean_percentage_error': mean_percentage_error,
        'max_percentage_error': max_percentage_error,
        'n_samples': len(common_keys)
    }