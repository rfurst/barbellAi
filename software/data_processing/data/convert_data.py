"""
Script to prepare data for the pipeline by copying and renaming files without modification.
"""
import pandas as pd
import json
from pathlib import Path
import shutil
import re

def prepare_dataset(imu_csv_path: Path, velocity_xlsx_path: Path, output_dir: Path):
    """Prepare a single dataset for the pipeline by copying and renaming files."""
    # Copy IMU data without modification
    output_name = imu_csv_path.stem
    shutil.copy2(imu_csv_path, output_dir / f"{output_name}_imu.csv")
    
    # Load velocity labels to create the label file
    velocity_data = pd.read_excel(velocity_xlsx_path)
    
    # Create label file with just the velocity data
    # We'll let the physics processor handle the rep boundaries
    reps = []
    for rep_num, peak_velocity in zip(velocity_data['Rep Number'], 
                                    velocity_data['Measured Peak Velocity (m/s)']):
        reps.append({
            'peak_velocity': float(peak_velocity)
        })
    
    # Save labels
    with open(output_dir / f"{output_name}_labels.json", 'w') as f:
        json.dump({'reps': reps}, f, indent=2)

def prepare_all_data(data_dir: Path, output_dir: Path):
    """Prepare all datasets for the pipeline."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all IMU data files
    imu_files = []
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            imu_files.extend(subdir.glob('imu_data_*.csv'))
    
    for imu_file in imu_files:
        # Find corresponding velocity file
        base_name = re.match(r'imu_data_(.*?)\.csv', imu_file.name).group(1)
        velocity_file = next(imu_file.parent.glob(f'velocity_validated_imu_data_{base_name}.xlsx'), None)
        
        if velocity_file:
            print(f"Preparing {imu_file.name}...")
            try:
                prepare_dataset(imu_file, velocity_file, output_dir)
                print(f"Successfully prepared {imu_file.name}")
            except Exception as e:
                print(f"Error preparing {imu_file.name}: {str(e)}")
        else:
            print(f"No velocity data found for {imu_file.name}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for pipeline')
    parser.add_argument('--data-dir', required=True, help='Root directory containing raw data')
    parser.add_argument('--output-dir', required=True, help='Directory to save prepared data')
    
    args = parser.parse_args()
    
    prepare_all_data(Path(args.data_dir), Path(args.output_dir)) 