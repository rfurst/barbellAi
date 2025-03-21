"""
Data loader for managing training data and ground truth labels.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict
from ..core.interfaces import ImuData, RepData

class DataLoader:
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to directory containing raw data and labels
        """
        self.data_dir = Path(data_dir)
        
    def load_imu_data(self, filename: str) -> ImuData:
        """
        Load IMU data from a CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            ImuData container with loaded data
        """
        df = pd.read_csv(self.data_dir / filename)
        
        # Assuming CSV has columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        timestamp = df['timestamp'].values
        accel = df[['accel_x', 'accel_y', 'accel_z']].values
        gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
        
        return ImuData(
            timestamp=timestamp,
            accel=accel,
            gyro=gyro
        )
        
    def load_labels(self, filename: str) -> List[RepData]:
        """
        Load rep labels from a JSON file.
        
        Args:
            filename: Name of the JSON file
            
        Returns:
            List of RepData objects
        """
        with open(self.data_dir / filename, 'r') as f:
            label_data = json.load(f)
            
        reps = []
        for rep in label_data['reps']:
            reps.append(RepData(
                start_idx=rep['start_idx'],
                end_idx=rep['end_idx'],
                peak_velocity=rep['peak_velocity']
            ))
            
        return reps
        
    def save_labels(self, filename: str, reps: List[RepData]):
        """
        Save rep labels to a JSON file.
        
        Args:
            filename: Name of the JSON file
            reps: List of RepData objects to save
        """
        label_data = {
            'reps': [
                {
                    'start_idx': rep.start_idx,
                    'end_idx': rep.end_idx,
                    'peak_velocity': rep.peak_velocity
                }
                for rep in reps
            ]
        }
        
        with open(self.data_dir / filename, 'w') as f:
            json.dump(label_data, f, indent=2)
            
    def load_training_data(self) -> Tuple[List[ImuData], List[List[RepData]]]:
        """
        Load all available training data and labels.
        
        Returns:
            Tuple of (list of ImuData, list of rep labels)
        """
        imu_files = list(self.data_dir.glob('*_imu.csv'))
        imu_data_list = []
        labels_list = []
        
        for imu_file in imu_files:
            # Load IMU data
            imu_data = self.load_imu_data(imu_file.name)
            imu_data_list.append(imu_data)
            
            # Load corresponding labels
            label_file = imu_file.stem.replace('_imu', '_labels.json')
            try:
                labels = self.load_labels(label_file)
                labels_list.append(labels)
            except FileNotFoundError:
                print(f"Warning: No labels found for {imu_file.name}")
                labels_list.append([])
                
        return imu_data_list, labels_list 