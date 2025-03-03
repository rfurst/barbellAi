"""
Machine learning based rep detector using peak detection and validation.
"""
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from ..core.interfaces import RepDetector, ProcessedData, RepData
from typing import List, Optional, Tuple
import joblib

class MLRepDetector(RepDetector):
    def __init__(
        self,
        window_size: int = 25,
        prominence: float = 0.5,
        distance: int = 50
    ):
        """
        Initialize the ML-based rep detector.
        
        Args:
            window_size: Size of the smoothing window
            prominence: Minimum prominence for peak detection
            distance: Minimum distance between peaks
        """
        self.window_size = window_size
        self.prominence = prominence
        self.distance = distance
        self.scaler = StandardScaler()
        self.trained = False
        
    def preprocess_signal(self, accel: np.ndarray) -> np.ndarray:
        """Preprocess the acceleration signal."""
        # Use vertical acceleration
        vert_accel = accel[:, 2]
        
        # Smooth the signal
        if len(vert_accel) > self.window_size:
            vert_accel = savgol_filter(vert_accel, self.window_size, 3)
            
        return vert_accel
        
    def train(self, processed_data_list: List[ProcessedData], 
              labeled_reps_list: List[List[RepData]]):
        """
        Train the detector using labeled data.
        
        Args:
            processed_data_list: List of processed acceleration data
            labeled_reps_list: List of labeled reps for each dataset
        """
        # Collect features for scaling
        all_features = []
        for processed_data in processed_data_list:
            vert_accel = self.preprocess_signal(processed_data.world_accel)
            all_features.append(vert_accel)
            
        # Fit scaler
        self.scaler.fit(np.concatenate(all_features).reshape(-1, 1))
        
        # Analyze labeled reps to determine optimal parameters
        prominences = []
        distances = []
        
        for processed_data, labeled_reps in zip(processed_data_list, labeled_reps_list):
            vert_accel = self.preprocess_signal(processed_data.world_accel)
            scaled_accel = self.scaler.transform(vert_accel.reshape(-1, 1)).ravel()
            
            for rep in labeled_reps:
                # Find actual peak in rep region
                rep_region = scaled_accel[rep.start_idx:rep.end_idx]
                peak_idx = rep.start_idx + np.argmax(rep_region)
                
                if peak_idx > 0:
                    # Calculate prominence
                    left_min = np.min(scaled_accel[rep.start_idx:peak_idx])
                    right_min = np.min(scaled_accel[peak_idx:rep.end_idx])
                    prominence = scaled_accel[peak_idx] - max(left_min, right_min)
                    prominences.append(prominence)
                    
                    # Calculate typical distance between reps
                    if len(labeled_reps) > 1:
                        distances.extend(np.diff([r.start_idx for r in labeled_reps]))
        
        # Update parameters based on training data
        if prominences:
            self.prominence = np.percentile(prominences, 25) * 0.8  # Be slightly more lenient
        if distances:
            self.distance = int(np.median(distances) * 0.8)  # Be slightly more lenient
            
        self.trained = True
        
    def detect_reps(self, processed_data: ProcessedData) -> List[RepData]:
        """
        Detect reps in the processed data.
        
        Args:
            processed_data: Processed IMU data
            
        Returns:
            List of detected reps
        """
        # Preprocess the signal
        vert_accel = self.preprocess_signal(processed_data.world_accel)
        
        # Scale the signal
        if not self.trained:
            # If not trained, fit scaler on this data
            self.scaler.fit(vert_accel.reshape(-1, 1))
            
        scaled_accel = self.scaler.transform(vert_accel.reshape(-1, 1)).ravel()
        
        # Find peaks
        peaks, properties = find_peaks(
            scaled_accel,
            prominence=self.prominence,
            distance=self.distance
        )
        
        # Convert peaks to rep data
        reps = []
        for peak_idx, prominence in zip(peaks, properties["prominences"]):
            # Find rep boundaries
            left_base = int(properties["left_bases"][len(reps)])
            right_base = int(properties["right_bases"][len(reps)])
            
            rep = RepData(
                start_idx=left_base,
                end_idx=right_base,
                peak_velocity=0.0  # Will be filled in by velocity calculator
            )
            reps.append(rep)
            
        return reps
        
    def save(self, path: str):
        """Save the trained model."""
        joblib.dump({
            'scaler': self.scaler,
            'window_size': self.window_size,
            'prominence': self.prominence,
            'distance': self.distance,
            'trained': self.trained
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'MLRepDetector':
        """Load a trained model."""
        data = joblib.load(path)
        detector = cls(
            window_size=data['window_size'],
            prominence=data['prominence'],
            distance=data['distance']
        )
        detector.scaler = data['scaler']
        detector.trained = data['trained']
        return detector 