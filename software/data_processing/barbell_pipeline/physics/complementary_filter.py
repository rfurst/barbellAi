"""
Complementary filter implementation for orientation tracking.
"""
import numpy as np
from ..core.interfaces import PhysicsProcessor, ImuData, ProcessedData
from scipy.spatial.transform import Rotation

class ComplementaryFilterProcessor(PhysicsProcessor):
    def __init__(self, alpha=0.96, dt=1/100):
        """
        Initialize the complementary filter.
        
        Args:
            alpha: Weight for gyroscope integration (0.96 is typically good)
            dt: Time step between measurements in seconds
        """
        self.alpha = alpha
        self.dt = dt
        
    def _initialize_orientation(self, initial_accel):
        """Initialize orientation from initial accelerometer reading."""
        # Normalize acceleration
        norm_accel = initial_accel / np.linalg.norm(initial_accel)
        
        # Find rotation that aligns [0, 0, 1] with gravity
        v = np.cross([0, 0, 1], norm_accel)
        s = np.linalg.norm(v)
        c = np.dot([0, 0, 1], norm_accel)
        
        if s < 1e-10:  # Vectors are parallel
            return Rotation.from_quat([0, 0, 0, 1])
            
        v_skew = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * (1 - c) / (s * s)
        return Rotation.from_matrix(R)
    
    def process_orientation(self, imu_data: ImuData) -> ProcessedData:
        """
        Process IMU data using complementary filter.
        
        Args:
            imu_data: Raw IMU data container
            
        Returns:
            ProcessedData containing world frame acceleration and orientation
        """
        n_samples = len(imu_data.timestamp)
        orientations = np.zeros((n_samples, 4))  # Quaternions
        world_accel = np.zeros((n_samples, 3))
        
        # Initialize orientation from first accelerometer reading
        current_orientation = self._initialize_orientation(imu_data.accel[0])
        orientations[0] = current_orientation.as_quat()
        
        for i in range(1, n_samples):
            # Gyroscope integration
            gyro = imu_data.gyro[i]
            delta_q = Rotation.from_rotvec(gyro * self.dt)
            gyro_orientation = current_orientation * delta_q
            
            # Accelerometer orientation
            accel = imu_data.accel[i]
            accel_orientation = self._initialize_orientation(accel)
            
            # Complementary filter
            gyro_q = gyro_orientation.as_quat()
            accel_q = accel_orientation.as_quat()
            
            # Ensure quaternions are in the same hemisphere
            if np.dot(gyro_q, accel_q) < 0:
                accel_q = -accel_q
                
            # Weighted average
            current_q = self.alpha * gyro_q + (1 - self.alpha) * accel_q
            current_orientation = Rotation.from_quat(current_q / np.linalg.norm(current_q))
            
            # Store results
            orientations[i] = current_orientation.as_quat()
            
            # Transform acceleration to world frame
            world_accel[i] = current_orientation.apply(accel) - np.array([0, 0, 9.81])
            
        return ProcessedData(
            timestamp=imu_data.timestamp,
            world_accel=world_accel,
            orientation=orientations
        ) 