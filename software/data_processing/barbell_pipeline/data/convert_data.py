"""
Script to organize raw data into the pipeline's required format without modifying the raw data.
"""
import pandas as pd
import json
from pathlib import Path
import shutil
import re

def convert_dataset(imu_csv_path: Path, velocity_xlsx_path: Path, output_dir: Path):
    """
    Organize a single dataset into the pipeline format.
    Preserves raw data exactly as is and creates corresponding label file.
    """
    # Load velocity labels
    velocity_data = pd.read_excel(velocity_xlsx_path)
    
    # Create labels from velocity data
    reps = []
    for _, row in velocity_data.iterrows():
        reps.append({
            'rep_number': int(row['Rep Number']),
            'peak_velocity': float(row['Measured Peak Velocity (m/s)'])
        })
    
    # Copy raw IMU data file (preserving exactly as is)
    output_name = imu_csv_path.stem
    shutil.copy2(imu_csv_path, output_dir / f"{output_name}_imu.csv")
    
    # Save labels
    with open(output_dir / f"{output_name}_labels.json", 'w') as f:
        json.dump({
            'reps': reps,
            'source_imu_file': str(imu_csv_path.name),
            'source_velocity_file': str(velocity_xlsx_path.name)
        }, f, indent=2)

def convert_all_data(data_dir: Path, output_dir: Path):
    """Convert all datasets in the data directory."""
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
            print(f"Processing {imu_file.name}...")
            try:
                convert_dataset(imu_file, velocity_file, output_dir)
                print(f"Successfully processed {imu_file.name}")
            except Exception as e:
                print(f"Error processing {imu_file.name}: {str(e)}")
        else:
            print(f"No velocity data found for {imu_file.name}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize data into pipeline format')
    parser.add_argument('--data-dir', required=True, help='Root directory containing raw data')
    parser.add_argument('--output-dir', required=True, help='Directory to save organized data')
    
    args = parser.parse_args()
    
    convert_all_data(Path(args.data_dir), Path(args.output_dir)) 