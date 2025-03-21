"""
Visualization tools for debugging physics processing components.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.interfaces import ImuData, ProcessedData
from scipy.spatial.transform import Rotation

def plot_orientation_tracking(imu_data: ImuData, processed_data: ProcessedData, save_path: str = None):
    """
    Plot raw IMU data and processed orientation data.
    
    Args:
        imu_data: Raw IMU data
        processed_data: Processed data with orientation
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Raw acceleration
    ax1 = fig.add_subplot(221)
    ax1.plot(imu_data.timestamp, imu_data.accel[:, 0], label='X')
    ax1.plot(imu_data.timestamp, imu_data.accel[:, 1], label='Y')
    ax1.plot(imu_data.timestamp, imu_data.accel[:, 2], label='Z')
    ax1.set_title('Raw Acceleration')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True)
    
    # Raw angular velocity
    ax2 = fig.add_subplot(222)
    ax2.plot(imu_data.timestamp, imu_data.gyro[:, 0], label='X')
    ax2.plot(imu_data.timestamp, imu_data.gyro[:, 1], label='Y')
    ax2.plot(imu_data.timestamp, imu_data.gyro[:, 2], label='Z')
    ax2.set_title('Raw Angular Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.legend()
    ax2.grid(True)
    
    # World frame acceleration
    ax3 = fig.add_subplot(223)
    ax3.plot(processed_data.timestamp, processed_data.world_accel[:, 0], label='X')
    ax3.plot(processed_data.timestamp, processed_data.world_accel[:, 1], label='Y')
    ax3.plot(processed_data.timestamp, processed_data.world_accel[:, 2], label='Z')
    ax3.set_title('World Frame Acceleration')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.legend()
    ax3.grid(True)
    
    # Orientation (quaternions)
    ax4 = fig.add_subplot(224)
    ax4.plot(processed_data.timestamp, processed_data.orientation[:, 0], label='w')
    ax4.plot(processed_data.timestamp, processed_data.orientation[:, 1], label='x')
    ax4.plot(processed_data.timestamp, processed_data.orientation[:, 2], label='y')
    ax4.plot(processed_data.timestamp, processed_data.orientation[:, 3], label='z')
    ax4.set_title('Orientation (Quaternions)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Quaternion Components')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_3d_orientation(processed_data: ProcessedData, save_path: str = None):
    """
    Plot 3D visualization of orientation changes.
    
    Args:
        processed_data: Processed data with orientation
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to rotation matrices
    rotations = Rotation.from_quat(processed_data.orientation)
    
    # Plot coordinate axes at different points in time
    times = np.linspace(0, len(processed_data.timestamp)-1, 20, dtype=int)
    scale = 0.1  # Scale factor for axes
    
    for t in times:
        R = rotations[t].as_matrix()
        origin = np.array([processed_data.timestamp[t], 0, 0])
        
        # Plot X, Y, Z axes
        for i, color in enumerate(['r', 'g', 'b']):
            axis = R[:, i] * scale
            ax.quiver(origin[0], origin[1], origin[2],
                     axis[0], axis[1], axis[2],
                     color=color, alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Orientation Visualization')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_velocity_calculation(processed_data: ProcessedData, rep: dict, save_path: str = None):
    """
    Plot velocity calculation details for a specific rep.
    
    Args:
        processed_data: Processed data with world frame acceleration
        rep: Dictionary containing rep information
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Vertical acceleration during rep
    ax1 = fig.add_subplot(221)
    ax1.plot(processed_data.timestamp[rep['start_idx']:rep['end_idx']],
             processed_data.world_accel[rep['start_idx']:rep['end_idx'], 2])
    ax1.set_title('Vertical Acceleration During Rep')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.grid(True)
    
    # Cumulative velocity
    time = processed_data.timestamp[rep['start_idx']:rep['end_idx']]
    accel = processed_data.world_accel[rep['start_idx']:rep['end_idx'], 2]
    velocity = np.cumsum(accel * np.diff(time, prepend=time[0]))
    
    ax2 = fig.add_subplot(222)
    ax2.plot(time, velocity)
    ax2.set_title('Cumulative Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    
    # Acceleration vs Velocity
    ax3 = fig.add_subplot(223)
    ax3.scatter(accel, velocity, c=time, cmap='viridis')
    ax3.set_title('Acceleration vs Velocity')
    ax3.set_xlabel('Acceleration (m/s²)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.grid(True)
    
    # Phase plot
    ax4 = fig.add_subplot(224)
    ax4.plot(velocity, accel)
    ax4.set_title('Phase Plot')
    ax4.set_xlabel('Velocity (m/s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show() 