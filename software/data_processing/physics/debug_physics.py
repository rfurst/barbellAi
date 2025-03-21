"""
Debug script for testing and visualizing physics processing components.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from core.interfaces import ImuData, ProcessedData
from physics.complementary_filter import ComplementaryFilterProcessor
from physics.velocity_calculator import TrapzVelocityCalculator
from physics.visualization import (
    plot_orientation_tracking,
    plot_3d_orientation,
    plot_velocity_calculation
)

def load_test_data(data_path: str) -> ImuData:
    """Load test data from CSV file."""
    df = pd.read_csv(data_path)
    
    # Map columns if needed
    column_mapping = {
        'timestamp': 'timestamp',
        'accel_x': 'ax',
        'accel_y': 'ay',
        'accel_z': 'az',
        'gyro_x': 'gx',
        'gyro_y': 'gy',
        'gyro_z': 'gz'
    }
    
    df = df.rename(columns=column_mapping)
    
    return ImuData(
        timestamp=df['timestamp'].values,
        accel=df[['ax', 'ay', 'az']].values,
        gyro=df[['gx', 'gy', 'gz']].values
    )

def test_physics_processing(data_path: str, output_dir: str):
    """
    Test physics processing components and generate visualizations.
    
    Args:
        data_path: Path to IMU data file
        output_dir: Directory to save visualizations
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"Loading data from {data_path}...")
    imu_data = load_test_data(data_path)
    
    # Process orientation
    print("Processing orientation...")
    physics_processor = ComplementaryFilterProcessor()
    processed_data = physics_processor.process_orientation(imu_data)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Plot orientation tracking
    plot_orientation_tracking(
        imu_data,
        processed_data,
        save_path=str(output_dir / 'orientation_tracking.png')
    )
    
    # Plot 3D orientation
    plot_3d_orientation(
        processed_data,
        save_path=str(output_dir / '3d_orientation.png')
    )
    
    # Test velocity calculation
    print("Testing velocity calculation...")
    velocity_calculator = TrapzVelocityCalculator()
    
    # Create a test rep (you can modify these indices)
    test_rep = {
        'start_idx': 100,
        'end_idx': 200
    }
    
    # Calculate velocity
    velocity = velocity_calculator.calculate_velocity(processed_data, test_rep)
    print(f"Calculated peak velocity: {velocity:.2f} m/s")
    
    # Plot velocity calculation
    plot_velocity_calculation(
        processed_data,
        test_rep,
        save_path=str(output_dir / 'velocity_calculation.png')
    )
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug physics processing')
    parser.add_argument('--data-path', required=True, help='Path to IMU data file')
    parser.add_argument('--output-dir', required=True, help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    test_physics_processing(args.data_path, args.output_dir) 