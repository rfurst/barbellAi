"""
Data loader for IMU data and labels.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from ..core.interfaces import ImuData

class DataLoader:
    def __init__(self, data_dir: str):
        """Initialize data loader with data directory."""
        self.data_dir = Path(data_dir)
        
    def load_imu_data(self, filename: str) -> ImuData:
        """Load IMU data from CSV file."""
        # Read CSV file
        df = pd.read_csv(self.data_dir / filename)
        
        # Extract timestamps and convert to seconds if in milliseconds
        timestamps = df['tMillis'].values
        if timestamps[1] - timestamps[0] > 1000:  # Probably milliseconds
            timestamps = timestamps / 1000.0
            
        return ImuData(
            timestamp=timestamps,
            accel_x=df['ax'].values,
            accel_y=df['ay'].values,
            accel_z=df['az'].values,
            gyro_x=df['gx'].values,
            gyro_y=df['gy'].values,
            gyro_z=df['gz'].values
        )
        
    def load_labels(self, imu_filename: str) -> list:
        """Load labels for an IMU data file."""
        # Convert IMU filename to label filename
        label_file = self.data_dir / imu_filename.replace('_imu.csv', '_labels.json')
        
        if not label_file.exists():
            print(f"Warning: No labels found for {imu_filename}")
            return []
            
        with open(label_file, 'r') as f:
            labels = json.load(f)
            
        return labels
        
    def load_training_data(self) -> tuple[list[ImuData], list]:
        """Load all training data and labels."""
        imu_files = list(self.data_dir.glob('*_imu.csv'))
        
        imu_data_list = []
        labels_list = []
        
        for imu_file in imu_files:
            # Load IMU data
            imu_data = self.load_imu_data(imu_file.name)
            imu_data_list.append(imu_data)
            
            # Load corresponding labels
            labels = self.load_labels(imu_file.name)
            labels_list.append(labels)
            
        return imu_data_list, labels_list 