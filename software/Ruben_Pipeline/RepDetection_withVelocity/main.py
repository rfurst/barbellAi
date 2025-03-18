# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:20:16 2025

@author: rkfurst
"""

# main.py
import os
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_processing import DataProcessor, load_processed_data
from src.feature_extraction import FeatureExtractor
from src.model_training import ModelTrainer
from src.velocity_calculator import EnhancedVelocityCalculator
from src.utils import setup_logging

# Setup logging
logger = setup_logging()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Deadlift Velocity Calculator')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, required=True, choices=['preprocess', 'extract', 'train', 'calculate', 'all'],
                        help='Operation mode')
    
    # Data directories
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing raw data files')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory for processed data')
    parser.add_argument('--models-dir', type=str, default='data/models',
                        help='Directory for trained models')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for calculation outputs')
    
    # Optional arguments
    parser.add_argument('--use-ml', action='store_true', default=True,
                        help='Use ML model for velocity calculation')
    parser.add_argument('--adjustment-factor', type=float, default=0.85,
                        help='Biomechanical adjustment factor for deadlift velocity')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations')
    parser.add_argument('--true-velocities', type=str, default=None,
                        help='Path to JSON file with true velocities for training')
    
    return parser.parse_args()

def load_true_velocities(file_path: str) -> Dict[str, float]:
    """
    Load true velocity values from a JSON file.
    
    Expected format:
    {
        "file1_rep1": 1.28,
        "file1_rep2": 1.13,
        ...
    }
    
    Args:
        file_path: Path to the JSON file with true velocities
        
    Returns:
        Dictionary mapping 'file_name_repN' to true velocity values
    """
    if not file_path or not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r') as f:
            velocities = json.load(f)
        return velocities
    except Exception as e:
        logger.error(f"Error loading true velocities: {e}")
        return {}

def preprocess_data(args):
    """Preprocess all data files."""
    processor = DataProcessor(args.data_dir, args.processed_dir)
    repetitions, file_repetitions = processor.preprocess_all_data()
    logger.info(f"Preprocessed {len(repetitions)} repetitions")
    return repetitions

def extract_features(args, repetitions=None):
    """Extract features from preprocessed data."""
    if repetitions is None:
        repetitions = load_processed_data(args.processed_dir)
    
    extractor = FeatureExtractor()
    features_df = extractor.process_all_repetitions(repetitions)
    
    # Save features
    os.makedirs(args.processed_dir, exist_ok=True)
    features_path = os.path.join(args.processed_dir, 'features.csv')
    features_df.to_csv(features_path, index=False)
    
    logger.info(f"Extracted features saved to {features_path}")
    return features_df

def train_model(args, features_df=None):
    """Train the velocity prediction model."""
    if features_df is None:
        features_path = os.path.join(args.processed_dir, 'features.csv')
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
        else:
            logger.error(f"Features file not found at {features_path}")
            return None
    
    # Load true velocities
    true_velocities = load_true_velocities(args.true_velocities)
    if not true_velocities:
        logger.error("No true velocities available for training")
        return None
    
    trainer = ModelTrainer(args.models_dir)
    model_path = trainer.train_and_save_model(features_df, true_velocities)
    
    logger.info(f"Model trained and saved to {model_path}")
    return model_path

def calculate_velocities(args):
    """Calculate velocities using the enhanced model."""
    # Load processed data
    repetitions = load_processed_data(args.processed_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the calculator
    model_path = os.path.join(args.models_dir, "velocity_model.pkl")
    calculator = EnhancedVelocityCalculator(
        model_path=model_path,
        adjustment_factor=args.adjustment_factor,
        use_ml_model=args.use_ml
    )
    
    # Calculate velocities for all repetitions
    all_results = []
    for rep in repetitions:
        logger.info(f"Processing {rep['file_name']} rep {rep['rep_number']}")
        result = calculator.calculate_velocity(rep)
        all_results.append(result)
    
    # Export results
    export_results(all_results, args)
    
    logger.info(f"Calculated velocities for {len(all_results)} repetitions")
    return all_results

def export_results(all_results, args):
    """Export calculation results."""
    # Prepare directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Export JSON summary
    summary = {
        'calculation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'use_ml_model': args.use_ml,
        'adjustment_factor': args.adjustment_factor,
        'repetitions': []
    }
    
    for result in all_results:
        summary['repetitions'].append({
            'file_name': result['rep_number'],
            'rep_number': result['rep_number'],
            'metrics': result['metrics']
        })
    
    # Save summary JSON
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_results(all_results, output_dir)
    
    logger.info(f"Results exported to {output_dir}")

def visualize_results(all_results, output_dir):
    """Generate visualizations of the results."""
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract data for plotting
    rep_numbers = [result['rep_number'] for result in all_results]
    peak_velocities = [result['metrics']['peak_velocity'] for result in all_results]
    mean_velocities = [result['metrics']['mean_velocity'] for result in all_results]
    
    # 1. Peak Velocities Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(rep_numbers, peak_velocities, color='blue', alpha=0.7)
    plt.axhline(y=np.mean(peak_velocities), color='r', linestyle='--', 
                label=f'Mean: {np.mean(peak_velocities):.3f} m/s')
    plt.ylabel('Peak Velocity (m/s)')
    plt.title('Peak Velocities by Repetition')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rep_numbers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'peak_velocities.png'), dpi=300)
    plt.close()
    
    # 2. Mean Velocities Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(rep_numbers, mean_velocities, color='green', alpha=0.7)
    plt.axhline(y=np.mean(mean_velocities), color='r', linestyle='--', 
                label=f'Mean: {np.mean(mean_velocities):.3f} m/s')
    plt.ylabel('Mean Velocity (m/s)')
    plt.title('Mean Velocities by Repetition')
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rep_numbers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mean_velocities.png'), dpi=300)
    plt.close()
    
    # 3. Individual Repetition Plots
    for result in all_results:
        rep_number = result['rep_number']
        time = result['time']
        velocity = result['velocity']
        acceleration = result['acceleration']
        
        plt.figure(figsize=(12, 8))
        
        # Velocity plot
        plt.subplot(2, 1, 1)
        plt.plot(time, velocity, 'b-', label='Velocity')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Mark peak velocity
        peak_idx = result['metrics']['peak_velocity_index']
        plt.plot(time[peak_idx], velocity[peak_idx], 'ro', 
                 label=f'Peak: {velocity[peak_idx]:.3f} m/s')
        
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Repetition {rep_number} - Velocity Profile')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Acceleration plot
        plt.subplot(2, 1, 2)
        plt.plot(time, acceleration, 'r-', label='Acceleration')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Acceleration (m/s²)')
        plt.xlabel('Time (s)')
        plt.title(f'Repetition {rep_number} - Acceleration Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'rep_{rep_number}_profile.png'), dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {viz_dir}")

def run_pipeline(args):
    """Run the complete pipeline."""
    # Step 1: Preprocess data
    repetitions = preprocess_data(args)
    
    # Step 2: Extract features
    features_df = extract_features(args, repetitions)
    
    # Step 3: Train model (if true velocities provided)
    if args.true_velocities:
        model_path = train_model(args, features_df)
    else:
        logger.warning("No true velocities provided, skipping model training")
    
    # Step 4: Calculate velocities
    all_results = calculate_velocities(args)
    
    logger.info("Pipeline completed successfully")
    return all_results

def main():
    """Main function."""
    args = parse_arguments()
    
    # Execute the requested mode
    if args.mode == 'preprocess':
        preprocess_data(args)
    elif args.mode == 'extract':
        extract_features(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'calculate':
        calculate_velocities(args)
    elif args.mode == 'all':
        run_pipeline(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()