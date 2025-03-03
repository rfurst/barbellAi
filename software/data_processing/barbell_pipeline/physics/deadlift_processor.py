"""
Enhanced physics processor specifically optimized for deadlift tracking.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
from ..core.interfaces import PhysicsProcessor, ImuData, ProcessedData

class DeadliftPhysicsProcessor(PhysicsProcessor):
    def __init__(
        self,
        # Sensor calibration parameters
        n_calib: int = 50,
        gravity_sign: float = -1.0,
        sensor_offset: tuple = (0.0, 0.0, 0.05),
        
        # Stillness detection parameters
        stillness_acc_std: float = 0.05,
        stillness_gyro_std: float = 0.04,
        stillness_window_size: int = 15,
        min_stillness_duration: int = 20,
        reinit_cooldown: int = 50,
        stillness_gravity_error: float = 0.15,
        
        # Complementary filter parameters
        cf_still_alpha: float = 0.7,
        cf_pull_alpha: float = 0.95,
        cf_general_alpha: float = 0.85,
        
        # Signal processing
        filter_cutoff: float = 5.0,
        sampling_freq: float = 104
    ):
        """Initialize the deadlift-specific physics processor."""
        self.n_calib = n_calib
        self.gravity_sign = gravity_sign
        self.sensor_offset = np.array(sensor_offset)
        
        # Stillness detection
        self.stillness_acc_std = stillness_acc_std
        self.stillness_gyro_std = stillness_gyro_std
        self.stillness_window_size = stillness_window_size
        self.min_stillness_duration = min_stillness_duration
        self.reinit_cooldown = reinit_cooldown
        self.stillness_gravity_error = stillness_gravity_error
        
        # Complementary filter
        self.cf_still_alpha = cf_still_alpha
        self.cf_pull_alpha = cf_pull_alpha
        self.cf_general_alpha = cf_general_alpha
        
        # Signal processing
        self.filter_cutoff = filter_cutoff
        self.sampling_freq = sampling_freq
        
        # Initialize buffers
        self.acc_buffer = deque(maxlen=stillness_window_size)
        self.gyro_buffer = deque(maxlen=stillness_window_size)
        self.stillness_duration = 0
        self.last_reinit = 0
        
        # Initialize state
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # Initial orientation quaternion
        self.gyro_bias = np.zeros(3)
        
    def butter_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter to data."""
        nyq = 0.5 * self.sampling_freq
        normal_cutoff = self.filter_cutoff / nyq
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
        
    def is_still(self, acc: np.ndarray, gyro: np.ndarray) -> bool:
        """Detect if the sensor is stationary."""
        self.acc_buffer.append(acc)
        self.gyro_buffer.append(gyro)
        
        if len(self.acc_buffer) < self.stillness_window_size:
            return False
            
        acc_std = np.std(self.acc_buffer, axis=0)
        gyro_std = np.std(self.gyro_buffer, axis=0)
        
        acc_still = np.all(acc_std < self.stillness_acc_std)
        gyro_still = np.all(gyro_std < self.stillness_gyro_std)
        
        # Check gravity magnitude
        acc_mean = np.mean(self.acc_buffer, axis=0)
        gravity_error = abs(np.linalg.norm(acc_mean) - 9.81) / 9.81
        
        return acc_still and gyro_still and gravity_error < self.stillness_gravity_error
        
    def update_orientation(self, acc: np.ndarray, gyro: np.ndarray, dt: float, is_still: bool) -> None:
        """Update orientation using complementary filter."""
        # Remove estimated bias
        gyro_corrected = gyro - self.gyro_bias
        
        # Choose alpha based on motion state
        if is_still:
            alpha = self.cf_still_alpha
            # Update gyro bias during stillness
            if self.stillness_duration > self.min_stillness_duration:
                self.gyro_bias = 0.98 * self.gyro_bias + 0.02 * gyro
        else:
            # Use more gyro-heavy filter during dynamic motion
            alpha = self.cf_pull_alpha if np.linalg.norm(acc) > 12 else self.cf_general_alpha
            
        # Predict orientation from gyro
        w = gyro_corrected
        q_dot = 0.5 * np.array([
            -self.q[1]*w[0] - self.q[2]*w[1] - self.q[3]*w[2],
            self.q[0]*w[0] - self.q[3]*w[1] + self.q[2]*w[2],
            self.q[3]*w[0] + self.q[0]*w[1] - self.q[1]*w[2],
            -self.q[2]*w[0] + self.q[1]*w[1] + self.q[0]*w[2]
        ])
        q_gyro = self.q + q_dot * dt
        q_gyro = q_gyro / np.linalg.norm(q_gyro)
        
        # Calculate orientation from accelerometer
        acc_norm = acc / np.linalg.norm(acc)
        q_acc = self.initialize_quaternion_from_acc(acc_norm[0], acc_norm[1], acc_norm[2])
        
        # Complementary filter update
        if np.dot(q_gyro, q_acc) < 0:  # Ensure shortest path
            q_acc = -q_acc
        self.q = alpha * q_gyro + (1 - alpha) * q_acc
        self.q = self.q / np.linalg.norm(self.q)
        
    def initialize_quaternion_from_acc(self, ax: float, ay: float, az: float) -> np.ndarray:
        """Initialize quaternion from accelerometer reading."""
        # Find rotation that aligns [0, 0, 1] with gravity
        v = np.cross([0, 0, 1], [ax, ay, az])
        s = np.linalg.norm(v)
        c = np.dot([0, 0, 1], [ax, ay, az])
        
        if s < 1e-10:  # Vectors are parallel
            return np.array([1, 0, 0, 0])
            
        v = v / s
        angle = np.arctan2(s, c)
        sin_a = np.sin(angle/2)
        cos_a = np.cos(angle/2)
        
        return np.array([cos_a, v[0]*sin_a, v[1]*sin_a, v[2]*sin_a])
        
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by the current orientation quaternion."""
        q = self.q
        qv = np.array([0, v[0], v[1], v[2]])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        rotated = self.quat_multiply(self.quat_multiply(q, qv), q_conj)
        return rotated[1:]
        
    def quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
    def process_orientation(self, imu_data: ImuData) -> ProcessedData:
        """Process IMU data to get world frame acceleration."""
        # Extract data
        accel = np.column_stack([imu_data.accel_x, imu_data.accel_y, imu_data.accel_z])
        gyro = np.column_stack([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])
        
        # Calculate time steps
        timestamps = imu_data.timestamp
        dt = np.diff(timestamps, prepend=timestamps[0])
        
        # Initialize from first N_CALIB samples
        if len(accel) >= self.n_calib:
            acc_init = np.mean(accel[:self.n_calib], axis=0)
            self.q = self.initialize_quaternion_from_acc(acc_init[0], acc_init[1], acc_init[2])
            self.gyro_bias = np.mean(gyro[:self.n_calib], axis=0)
        
        # Process each sample
        n_samples = len(timestamps)
        world_accel = np.zeros((n_samples, 3))
        orientations = np.zeros((n_samples, 4))
        
        for i in range(n_samples):
            # Check stillness
            is_still = self.is_still(accel[i], gyro[i])
            if is_still:
                self.stillness_duration += 1
            else:
                self.stillness_duration = 0
                
            # Update orientation
            self.update_orientation(accel[i], gyro[i], dt[i], is_still)
            
            # Store orientation
            orientations[i] = self.q
            
            # Transform acceleration to world frame
            acc_world = self.rotate_vector(accel[i])
            
            # Apply gravity compensation
            acc_world[2] -= 9.81 * self.gravity_sign
            
            # Store world frame acceleration
            world_accel[i] = acc_world
            
        # Apply low-pass filter to world acceleration
        world_accel = self.butter_lowpass_filter(world_accel)
        
        return ProcessedData(
            timestamp=timestamps,
            world_accel=world_accel,
            orientation=orientations
        ) 