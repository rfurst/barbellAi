"""
Core interfaces and data classes for the barbell pipeline.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import joblib

@dataclass
class ImuData:
    """Raw IMU data."""
    timestamp: np.ndarray
    accel_x: np.ndarray
    accel_y: np.ndarray
    accel_z: np.ndarray
    gyro_x: np.ndarray
    gyro_y: np.ndarray
    gyro_z: np.ndarray

@dataclass
class ProcessedData:
    """Processed IMU data with world frame acceleration and orientation."""
    timestamp: np.ndarray
    world_accel: np.ndarray  # Shape: (n_samples, 3)
    orientation: np.ndarray   # Shape: (n_samples, 4) quaternions

@dataclass
class RepData:
    """Detected repetition data."""
    start_idx: int
    end_idx: int
    peak_velocity: float

class PhysicsProcessor(ABC):
    """Abstract base class for physics processing."""
    @abstractmethod
    def process_orientation(self, imu_data: ImuData) -> ProcessedData:
        """Process IMU data to get world frame acceleration."""
        pass

class RepDetector(ABC):
    """Abstract base class for repetition detection."""
    @abstractmethod
    def detect_reps(self, processed_data: ProcessedData) -> list[RepData]:
        """Detect repetitions in processed data."""
        pass
        
    def train(self, processed_data_list: list[ProcessedData], labels_list: list) -> None:
        """Train the detector using labeled data."""
        pass
        
    def save(self, path: str) -> None:
        """Save the trained model."""
        joblib.dump(self, path)
        
    @classmethod
    def load(cls, path: str) -> 'RepDetector':
        """Load a trained model."""
        return joblib.load(path)

class Pipeline:
    """Main pipeline combining physics processing and rep detection."""
    def __init__(
        self,
        physics_processor: PhysicsProcessor,
        rep_detector: RepDetector,
        velocity_calculator=None  # Optional
    ):
        self.physics_processor = physics_processor
        self.rep_detector = rep_detector
        self.velocity_calculator = velocity_calculator
        
    def process(self, imu_data: ImuData) -> list[RepData]:
        """Process IMU data to detect reps."""
        # Physics processing
        processed_data = self.physics_processor.process_orientation(imu_data)
        
        # Rep detection
        reps = self.rep_detector.detect_reps(processed_data)
        
        return reps 