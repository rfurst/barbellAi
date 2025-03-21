"""
Core interfaces for the barbell pipeline components.
"""
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ImuData:
    """Container for raw IMU data."""
    timestamp: np.ndarray  # Timestamps in seconds
    accel: np.ndarray     # Raw accelerometer data (N x 3)
    gyro: np.ndarray      # Raw gyroscope data (N x 3)
    
@dataclass
class ProcessedData:
    """Container for processed IMU data in world frame."""
    timestamp: np.ndarray
    world_accel: np.ndarray  # World frame acceleration
    orientation: np.ndarray   # Orientation quaternions
    
@dataclass
class RepData:
    """Container for detected rep information."""
    start_idx: int
    end_idx: int
    peak_velocity: float
    
class PhysicsProcessor(ABC):
    """Interface for physics-based processing components."""
    
    @abstractmethod
    def process_orientation(self, imu_data: ImuData) -> ProcessedData:
        """Convert raw IMU data to world frame data."""
        pass
    
class RepDetector(ABC):
    """Interface for rep detection components."""
    
    @abstractmethod
    def detect_reps(self, processed_data: ProcessedData) -> List[RepData]:
        """Detect individual reps from processed data."""
        pass
    
class VelocityCalculator(ABC):
    """Interface for velocity calculation components."""
    
    @abstractmethod
    def calculate_velocity(self, processed_data: ProcessedData, rep: RepData) -> float:
        """Calculate peak velocity for a given rep."""
        pass

class Pipeline:
    """Main pipeline that orchestrates the processing components."""
    
    def __init__(
        self,
        physics_processor: PhysicsProcessor,
        rep_detector: RepDetector,
        velocity_calculator: VelocityCalculator
    ):
        self.physics_processor = physics_processor
        self.rep_detector = rep_detector
        self.velocity_calculator = velocity_calculator
    
    def process(self, imu_data: ImuData) -> List[RepData]:
        """Process raw IMU data and return detected reps with velocities."""
        # Convert to world frame
        processed_data = self.physics_processor.process_orientation(imu_data)
        
        # Detect reps
        reps = self.rep_detector.detect_reps(processed_data)
        
        # Calculate velocities
        for rep in reps:
            rep.peak_velocity = self.velocity_calculator.calculate_velocity(
                processed_data, rep
            )
        
        return reps 