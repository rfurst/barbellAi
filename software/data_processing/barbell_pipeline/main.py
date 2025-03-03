"""
Main script demonstrating how to use the barbell pipeline.
"""
import numpy as np
from pathlib import Path
from .core.interfaces import Pipeline
from .physics.deadlift_processor import DeadliftPhysicsProcessor
from .physics.velocity_calculator import TrapzVelocityCalculator
from .ml.deadlift_detector import DeadliftRepDetector
from .data.data_loader import DataLoader

def train_pipeline(data_dir: str, model_save_dir: str):
    """
    Train the pipeline using data from the specified directory.
    
    Args:
        data_dir: Directory containing training data
        model_save_dir: Directory to save trained models
    """
    # Initialize components with deadlift-specific parameters
    physics_processor = DeadliftPhysicsProcessor(
        gravity_sign=-1.0,  # Adjust based on your sensor orientation
        sensor_offset=(0.0, 0.0, 0.05),  # Adjust based on your sensor placement
        stillness_acc_std=0.05,
        stillness_gyro_std=0.04,
        cf_still_alpha=0.7,
        cf_pull_alpha=0.95
    )
    
    rep_detector = DeadliftRepDetector(
        start_acc_threshold=1.3,
        concentric_min_vel=0.15,
        peak_vel_threshold=0.05,
        min_rep_duration=15,
        max_rep_duration=250
    )
    
    velocity_calculator = TrapzVelocityCalculator(drift_correction=True)
    
    # Create pipeline
    pipeline = Pipeline(
        physics_processor=physics_processor,
        rep_detector=rep_detector,
        velocity_calculator=velocity_calculator
    )
    
    # Load training data
    data_loader = DataLoader(data_dir)
    imu_data_list, labels_list = data_loader.load_training_data()
    
    # Process raw data
    processed_data_list = []
    for imu_data in imu_data_list:
        processed_data = physics_processor.process_orientation(imu_data)
        processed_data_list.append(processed_data)
    
    # Train rep detector
    rep_detector.train(processed_data_list, labels_list)
    
    # Save trained model
    save_path = Path(model_save_dir) / 'rep_detector.joblib'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rep_detector.save(str(save_path))
    print(f"Trained model saved to {save_path}")
    
def process_new_data(imu_data_path: str, model_dir: str) -> list:
    """
    Process new IMU data using trained models.
    
    Args:
        imu_data_path: Path to IMU data file
        model_dir: Directory containing trained models
        
    Returns:
        List of RepData objects with detected reps and velocities
    """
    # Load models with deadlift-specific parameters
    physics_processor = DeadliftPhysicsProcessor(
        gravity_sign=-1.0,
        sensor_offset=(0.0, 0.0, 0.05),
        stillness_acc_std=0.05,
        stillness_gyro_std=0.04,
        cf_still_alpha=0.7,
        cf_pull_alpha=0.95
    )
    
    rep_detector = DeadliftRepDetector.load(str(Path(model_dir) / 'rep_detector.joblib'))
    velocity_calculator = TrapzVelocityCalculator(drift_correction=True)
    
    # Create pipeline
    pipeline = Pipeline(
        physics_processor=physics_processor,
        rep_detector=rep_detector,
        velocity_calculator=velocity_calculator
    )
    
    # Load and process data
    data_loader = DataLoader(str(Path(imu_data_path).parent))
    imu_data = data_loader.load_imu_data(Path(imu_data_path).name)
    
    # Process through pipeline
    reps = pipeline.process(imu_data)
    
    return reps
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Barbell Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the pipeline')
    train_parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    train_parser.add_argument('--model-dir', required=True, help='Directory to save models')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process new data')
    process_parser.add_argument('--input', required=True, help='Input IMU data file')
    process_parser.add_argument('--model-dir', required=True, help='Directory containing trained models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_pipeline(args.data_dir, args.model_dir)
    elif args.command == 'process':
        reps = process_new_data(args.input, args.model_dir)
        print(f"\nDetected {len(reps)} reps:")
        for i, rep in enumerate(reps, 1):
            print(f"Rep {i}: Peak velocity = {rep.peak_velocity:.2f} m/s") 