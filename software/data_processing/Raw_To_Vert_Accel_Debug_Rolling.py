#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Barbell IMU Processing for Deadlift Tracking
-----------------------------------------------------
Specifically optimized for deadlift exercise velocity tracking.
"""

import numpy as np
import pandas as pd
import math
from collections import deque
from scipy.signal import savgol_filter, butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

##############################################################################
# ------------------------ USER TUNABLES -------------------------------------
##############################################################################

INPUT_CSV  = "imu_data_Farm1.csv"
VERT_CSV   = "vertical_output.csv"
DEBUG_CSV  = "debug_output.csv"

# Sensor calibration parameters
N_CALIB = 50
GRAVITY_SIGN = -1.0         # Set to 1.0 if gravity is positive in your system
SENSOR_OFFSET = (0.0, 0.0, 0.05)  # Sensor position relative to bar center in meters

# Enhanced stillness detection
STILLNESS_ACC_STD = 0.05     # Standard deviation threshold for acceleration (m/s²)
STILLNESS_GYRO_STD = 0.04    # Standard deviation threshold for gyroscope (rad/s)
STILLNESS_WINDOW_SIZE = 15   # Window size for stillness detection
MIN_STILLNESS_DURATION = 20  # Minimum samples needed to confirm stillness
REINIT_COOLDOWN = 50         # Samples between reinitializations
STILLNESS_GRAVITY_ERROR = 0.15  # Gravity magnitude error tolerance for quality stillness

# Deadlift-specific parameters
DEADLIFT_START_ACC_THRESHOLD = 1.3  # Vertical acceleration to detect rep start (m/s²)
DEADLIFT_CONCENTRIC_MIN_VEL = 0.15  # Minimum velocity during concentric phase (m/s)
DEADLIFT_PEAK_VEL_THRESHOLD = 0.05  # Velocity threshold for detecting peak (m/s)
DEADLIFT_MIN_REP_DURATION = 15      # Minimum samples for valid deadlift rep
DEADLIFT_MAX_REP_DURATION = 250     # Maximum samples for valid deadlift rep

# Complementary filter parameters
CF_STILL_ALPHA = 0.7         # Trust in gyro during stillness (0-1)
CF_PULL_ALPHA = 0.95         # Trust in gyro during pull phase (0-1)
CF_GENERAL_ALPHA = 0.85      # Default trust in gyro (0-1)

# Signal processing
FILTER_CUTOFF = 5.0          # Cutoff frequency for low-pass filter (Hz)
SAMPLING_FREQ = 104          # IMU sampling frequency (Hz)

# Final processing
FORCE_ZUPT_AT_END = True
SMOOTH_VELOCITY = True
SMOOTH_WINDOW = 21
SMOOTH_POLY = 3

##############################################################################
# Rep State Machine - Deadlift Specific
##############################################################################

# Deadlift phase states
DL_STATE_IDLE = 0       # Between reps, bar relatively still
DL_STATE_SETUP = 1      # Initial slight movements before pull
DL_STATE_PULL = 2       # Initial acceleration (breaking the floor)
DL_STATE_CONCENTRIC = 3 # Main lifting phase (upward movement)
DL_STATE_LOCKOUT = 4    # Brief transition at top of movement
DL_STATE_ECCENTRIC = 5  # Lowering phase
DL_STATE_FINISH = 6     # Rep finishing, returning to baseline

class DeadliftDetector:
    """Detect and validate deadlift reps based on vertical acceleration and velocity."""
    
    def __init__(self):
        self.state = DL_STATE_IDLE
        self.rep_start_time = 0
        self.pull_start_time = 0
        self.concentric_start_time = 0
        self.lockout_start_time = 0
        self.eccentric_start_time = 0
        self.rep_end_time = 0
        
        self.current_rep_duration = 0
        self.concentric_vel_samples = []
        self.eccentric_vel_samples = []
        self.rep_count = 0
        
        # Rep metrics tracking
        self.peak_velocity = 0
        self.concentric_duration = 0
        self.eccentric_duration = 0
        self.rep_rom = 0  # Range of motion
        
        # Buffer for peak detection
        self.vel_buffer = deque(maxlen=5)
        
        # Rep quality tracking
        self.rep_quality = 0  # 0-100% quality score
        
    def process_sample(self, vertical_acc, vertical_vel, time_idx, is_still):
        """Process a new sample and update deadlift detection state."""
        prev_state = self.state
        
        # Buffer for peak detection
        self.vel_buffer.append(vertical_vel)
        
        # State machine transitions for deadlift
        if self.state == DL_STATE_IDLE:
            # Reset metrics when idle
            self.peak_velocity = 0
            self.rep_rom = 0
            
            # Transition to setup if we detect movement or acceleration
            if abs(vertical_acc) > DEADLIFT_START_ACC_THRESHOLD/2 and not is_still:
                self.state = DL_STATE_SETUP
                self.rep_start_time = time_idx
                self.current_rep_duration = 0
                self.concentric_vel_samples = []
                self.eccentric_vel_samples = []
        
        elif self.state == DL_STATE_SETUP:
            # Wait for the initial pull (significant upward acceleration)
            if vertical_acc > DEADLIFT_START_ACC_THRESHOLD:
                self.state = DL_STATE_PULL
                self.pull_start_time = time_idx
            # Timeout if setup takes too long
            elif time_idx - self.rep_start_time > 60:
                self.state = DL_STATE_IDLE
        
        elif self.state == DL_STATE_PULL:
            # Transition to concentric phase when velocity becomes significant
            if vertical_vel > DEADLIFT_CONCENTRIC_MIN_VEL:
                self.state = DL_STATE_CONCENTRIC
                self.concentric_start_time = time_idx
            # Reset if pull doesn't lead to movement within reasonable time
            elif time_idx - self.pull_start_time > 30:
                self.state = DL_STATE_IDLE
        
        elif self.state == DL_STATE_CONCENTRIC:
            # Track velocity during concentric phase
            self.concentric_vel_samples.append(vertical_vel)
            
            # Update peak velocity
            if vertical_vel > self.peak_velocity:
                self.peak_velocity = vertical_vel
            
            # Range of motion estimation (integrate velocity)
            self.rep_rom += vertical_vel * (1/SAMPLING_FREQ)
            
            # Detect lockout when velocity approaches zero after having a peak
            if (len(self.vel_buffer) == self.vel_buffer.maxlen and
                max(self.vel_buffer) < DEADLIFT_PEAK_VEL_THRESHOLD and
                self.peak_velocity > DEADLIFT_CONCENTRIC_MIN_VEL * 2):
                
                self.state = DL_STATE_LOCKOUT
                self.lockout_start_time = time_idx
                self.concentric_duration = time_idx - self.concentric_start_time
        
        elif self.state == DL_STATE_LOCKOUT:
            # Check for beginning of descent (negative velocity)
            if vertical_vel < -DEADLIFT_PEAK_VEL_THRESHOLD:
                self.state = DL_STATE_ECCENTRIC
                self.eccentric_start_time = time_idx
            # Timeout if lockout lasts too long
            elif time_idx - self.lockout_start_time > 50:
                # This is still a valid rep, just unusually long lockout
                self.state = DL_STATE_FINISH
                self.rep_end_time = time_idx
        
        elif self.state == DL_STATE_ECCENTRIC:
            # Track velocity during eccentric phase
            self.eccentric_vel_samples.append(vertical_vel)
            
            # End rep when velocity approaches zero and acceleration is low
            if abs(vertical_vel) < DEADLIFT_PEAK_VEL_THRESHOLD and abs(vertical_acc) < DEADLIFT_START_ACC_THRESHOLD/2:
                self.state = DL_STATE_FINISH
                self.rep_end_time = time_idx
                self.eccentric_duration = time_idx - self.eccentric_start_time
        
        elif self.state == DL_STATE_FINISH:
            # Validate the rep
            self.current_rep_duration = time_idx - self.rep_start_time
            
            valid_rep = False
            if (self.current_rep_duration >= DEADLIFT_MIN_REP_DURATION and 
                self.current_rep_duration <= DEADLIFT_MAX_REP_DURATION and
                len(self.concentric_vel_samples) > 5 and
                len(self.eccentric_vel_samples) > 5 and
                self.peak_velocity > DEADLIFT_CONCENTRIC_MIN_VEL):
                
                # Calculate rep quality
                ideal_ratio = 0.6  # Ideal concentric:eccentric time ratio for deadlift
                actual_ratio = self.concentric_duration / max(1, self.eccentric_duration)
                time_score = 100 * (1 - min(1, abs(actual_ratio - ideal_ratio) / ideal_ratio))
                
                vel_consistency = 100 * (1 - np.std(self.concentric_vel_samples) / 
                                       max(0.01, self.peak_velocity))
                
                self.rep_quality = int(0.7 * time_score + 0.3 * vel_consistency)
                
                # Valid rep!
                valid_rep = True
                self.rep_count += 1
                print(f"Rep {self.rep_count} detected: Duration={self.current_rep_duration}, "
                      f"Peak vel={self.peak_velocity:.2f} m/s, Quality={self.rep_quality}%")
            
            # Return to idle state
            self.state = DL_STATE_IDLE
            
            # Recheck after validation
            if not valid_rep:
                print("Invalid rep detected - ignoring")
                self.rep_count = max(0, self.rep_count - 1)  # Ensure no negative counts
        
        # Increment duration counter
        self.current_rep_duration = time_idx - self.rep_start_time
        
        # Timeout safety - if a rep takes too long, reset
        if self.state != DL_STATE_IDLE and self.current_rep_duration > DEADLIFT_MAX_REP_DURATION:
            self.state = DL_STATE_IDLE
            
        # Return if state changed and current state
        return self.state != prev_state, self.state

    def get_rep_info(self):
        """Return information about the current/last rep"""
        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "peak_velocity": self.peak_velocity,
            "concentric_duration": self.concentric_duration,
            "eccentric_duration": self.eccentric_duration,
            "rep_quality": self.rep_quality,
            "rep_rom": self.rep_rom
        }

def generate_debug_plots(df_dbg, output_dir="./"):
    """
    Generate focused debug plots for state detection analysis.
    Just call this at the end of your main function.
    """
    # Extract data from debug DataFrame
    times = df_dbg["tMillis"].values
    roll_deg = df_dbg["roll_deg"].values
    pitch_deg = df_dbg["pitch_deg"].values
    yaw_deg = df_dbg["yaw_deg"].values
    
    is_still = df_dbg["is_still"].values.astype(float) if "is_still" in df_dbg.columns else df_dbg["skip_impact"].values.astype(float)
    gravity_update = df_dbg["gravity_update"].values.astype(float)
    deadlift_state = df_dbg["deadlift_state"].values if "deadlift_state" in df_dbg.columns else np.zeros_like(times)
    
    acc_world_x = df_dbg["acc_world_x"].values
    acc_world_y = df_dbg["acc_world_y"].values
    acc_world_z = df_dbg["acc_world_z"].values
    
    vertical_acc = df_dbg["vertical_acc"].values
    
    # Compute accelerometer magnitudes
    acc_mag = np.sqrt(acc_world_x**2 + acc_world_y**2 + acc_world_z**2)
    horizontal_acc = np.sqrt(acc_world_x**2 + acc_world_y**2)
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle("Barbell State Detection Analysis", fontsize=16)
    
    # 1. Orientation
    ax1 = axes[0]
    ax1.plot(times, roll_deg, label="Roll (deg)", color='#1f77b4')
    ax1.plot(times, pitch_deg, label="Pitch (deg)", color='#ff7f0e')
    ax1.plot(times, yaw_deg, label="Yaw (deg)", color='#2ca02c')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Barbell Orientation")
    
    # 2. State Flags
    ax2 = axes[1]
    ax2.plot(times, is_still, label="Is Still", color='#1f77b4', drawstyle='steps-post')
    ax2.plot(times, gravity_update, label="Gravity Update", color='#ff7f0e', drawstyle='steps-post', alpha=0.7)
    
    # Add vertical markers for state changes
    prev_state = is_still[0]
    for i in range(1, len(is_still)):
        if is_still[i] != prev_state:
            ax2.axvline(x=times[i], color='r', linestyle='--', alpha=0.5)
            prev_state = is_still[i]
    
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylabel("State (0/1)")
    ax2.set_title("Stillness Detection")
    
    # 3. World-frame Acceleration
    ax3 = axes[2]
    ax3.plot(times, acc_world_x, label="World X", color='#1f77b4')
    ax3.plot(times, acc_world_y, label="World Y", color='#ff7f0e')
    ax3.plot(times, acc_world_z, label="World Z", color='#2ca02c')
    ax3.plot(times, np.zeros_like(times), 'k--', alpha=0.3)  # Zero reference
    
    # Shade the background during stillness for context
    for i in range(len(is_still)-1):
        if is_still[i] > 0.5:
            ax3.axvspan(times[i], times[i+1], color='green', alpha=0.1)
    
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.set_title("World-Frame Acceleration Components")
    
    # 4. Directional Acceleration Components
    ax4 = axes[3]
    ax4.plot(times, vertical_acc, label="Vertical Acc", color='#2ca02c', linewidth=2)
    ax4.plot(times, horizontal_acc, label="Horizontal Acc", color='#d62728', linewidth=2)
    ax4.plot(times, acc_mag, label="Total Mag", color='#9467bd', alpha=0.5)
    ax4.plot(times, np.zeros_like(times), 'k--', alpha=0.3)  # Zero reference
    
    # Add threshold reference line
    if "DEADLIFT_START_ACC_THRESHOLD" in globals():
        threshold = DEADLIFT_START_ACC_THRESHOLD
    else:
        threshold = 1.3  # Default value if not defined
    
    ax4.axhline(y=threshold, color='#2ca02c', linestyle='--', 
                label=f"Vertical Threshold ({threshold} m/s²)", alpha=0.7)
    
    # Shade the background during stillness for context
    for i in range(len(is_still)-1):
        if is_still[i] > 0.5:
            ax4.axvspan(times[i], times[i+1], color='green', alpha=0.1)
    
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylabel("Acceleration (m/s²)")
    ax4.set_title("Acceleration Components (Vertical vs Horizontal)")
    
    # 5. Vertical/Horizontal Ratio - Key for Distinguishing Rolling from Deadlift
    ax5 = axes[4]
    
    # Calculate vertical/horizontal ratio (with safety for division by zero)
    v_h_ratio = np.zeros_like(vertical_acc)
    for i in range(len(vertical_acc)):
        if abs(horizontal_acc[i]) > 0.2:  # Only meaningful when horizontal acc exists
            v_h_ratio[i] = abs(vertical_acc[i]) / abs(horizontal_acc[i])
        else:
            v_h_ratio[i] = 0  # Set to zero when horizontal is near zero
    
    ax5.plot(times, v_h_ratio, label="Vertical/Horizontal Ratio", color='#ff7f0e')
    ax5.axhline(y=1.0, color='r', linestyle='--', label="Equal Ratio", alpha=0.7)
    
    # Highlight regions of deadlift vs roll
    for i in range(len(v_h_ratio)):
        if abs(acc_mag[i]) > 3.0:  # Only for significant motion
            if v_h_ratio[i] > 1.5:  # Vertical dominant - likely deadlift
                ax5.scatter(times[i], v_h_ratio[i], color='green', alpha=0.5, marker='o', s=20)
            elif v_h_ratio[i] < 0.7:  # Horizontal dominant - likely rolling
                ax5.scatter(times[i], v_h_ratio[i], color='red', alpha=0.5, marker='o', s=20)
    
    # Add annotations for key events
    for i in range(1, len(vertical_acc)):
        # Find significant vertical acceleration spikes
        if (abs(vertical_acc[i]) > threshold and 
            abs(vertical_acc[i-1]) <= threshold):
            ax5.annotate(f"{vertical_acc[i]:.1f} m/s²", 
                         xy=(times[i], min(5, v_h_ratio[i])), 
                         xytext=(times[i], min(5, v_h_ratio[i]) + 1),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=8, ha='center')
    
    ax5.set_ylim([0, 5])  # Cap the ratio for better visualization
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylabel("Ratio")
    ax5.set_xlabel("Time (ms)")
    ax5.set_title("Vertical/Horizontal Acceleration Ratio (>1 = Vertical Dominant)")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{output_dir}/state_detection_analysis.png", dpi=150)
    print(f"Saved state detection analysis to {output_dir}/state_detection_analysis.png")
    
    # Create a second focused plot just for the key metrics
    plt.figure(figsize=(14, 10))
    
    # Top: Vertical vs Horizontal acceleration
    plt.subplot(2, 1, 1)
    plt.plot(times, vertical_acc, label="Vertical Acc", color='#2ca02c', linewidth=2)
    plt.plot(times, horizontal_acc, label="Horizontal Acc", color='#d62728', linewidth=2)
    plt.axhline(y=threshold, color='#2ca02c', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight still periods
    for i in range(len(is_still)-1):
        if is_still[i] > 0.5:
            plt.axvspan(times[i], times[i+1], color='green', alpha=0.1)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Vertical vs Horizontal Acceleration")
    
    # Bottom: V/H Ratio with clearer annotations
    plt.subplot(2, 1, 2)
    plt.plot(times, v_h_ratio, label="V/H Ratio", color='#ff7f0e', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label="Equal Ratio", alpha=0.7)
    
    # Add text labels for motion types
    motion_segments = []
    current_type = "unknown"
    start_idx = 0
    
    for i in range(len(v_h_ratio)):
        if abs(acc_mag[i]) > 2.0:  # Significant motion
            motion_type = "vert" if v_h_ratio[i] > 1.2 else "horiz" if v_h_ratio[i] < 0.8 else "mixed"
            
            if motion_type != current_type:
                if current_type != "unknown" and i - start_idx > 10:  # Minimum segment length
                    motion_segments.append((start_idx, i-1, current_type))
                start_idx = i
                current_type = motion_type
    
    # Add the last segment
    if current_type != "unknown" and len(v_h_ratio) - start_idx > 10:
        motion_segments.append((start_idx, len(v_h_ratio)-1, current_type))
    
    # Annotate motion segments
    for start, end, motion_type in motion_segments:
        mid = (start + end) // 2
        if motion_type == "vert":
            color = 'green'
            label = "VERTICAL DOMINANT"
        elif motion_type == "horiz":
            color = 'red'
            label = "HORIZONTAL DOMINANT"
        else:
            color = 'orange'
            label = "MIXED MOTION"
        
        if end - start > 20:  # Only label significant segments
            plt.annotate(label, xy=(times[mid], 2.5), fontsize=9, 
                         ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.3))
    
    plt.ylim([0, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel("Ratio")
    plt.xlabel("Time (ms)")
    plt.title("Vertical/Horizontal Ratio - Motion Type Indicator")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/motion_type_analysis.png", dpi=150)
    print(f"Saved motion type analysis to {output_dir}/motion_type_analysis.png")


##############################################################################
# Signal Processing and Filtering Functions
##############################################################################

def butter_lowpass_filter(data, cutoff, fs, order=2):
    """
    Low-pass filter for sensor data
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def filter_signals(signals, window_size=5):
    """
    Apply filtering to a batch of signals
    """
    # For continuous online processing, use a simple moving average
    result = []
    for signal in signals:
        # Convert to numpy array if it's a deque
        if isinstance(signal, deque):
            signal = np.array(signal)
            
        filtered = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        # Pad the beginning to maintain same length
        padding = np.full(window_size-1, filtered[0])
        result.append(np.concatenate([padding, filtered]))
    
    return result

def validate_sensor_reading(acc, gyro, prev_acc, prev_gyro, dt):
    """
    Check for sensor reading validity, replace erroneous readings
    """
    # Maximum physically possible changes
    MAX_ACC_CHANGE = 100  # m/s²/s * dt
    MAX_GYRO_CHANGE = 50  # rad/s²/s * dt
    
    # Check for large jumps in accelerometer
    acc_diff = np.linalg.norm(acc - prev_acc) / dt
    if acc_diff > MAX_ACC_CHANGE:
        # Use previous value
        acc = prev_acc
    
    # Check for large jumps in gyroscope
    gyro_diff = np.linalg.norm(gyro - prev_gyro) / dt
    if gyro_diff > MAX_GYRO_CHANGE:
        # Use previous value
        gyro = prev_gyro
    
    return acc, gyro

##############################################################################
# Quaternion Math and Orientation Functions
##############################################################################

def complementary_filter_update(q, accel, gyro, alpha, dt):
    """
    Updates orientation using complementary filter approach
    
    Args:
        q: Current quaternion [w,x,y,z]
        accel: Accelerometer reading [x,y,z]
        gyro: Gyroscope reading [x,y,z]
        alpha: Weight for gyro integration (0-1)
        dt: Time step
    
    Returns:
        Updated quaternion
    """
    # Gyro integration part (predictions)
    q_gyro = integrate_quaternion(q, gyro, dt)
    
    # Accelerometer correction part (if not in high dynamic motion)
    a_mag = np.linalg.norm(accel)
    if 9.81 * (1 - STILLNESS_GRAVITY_ERROR) < a_mag < 9.81 * (1 + STILLNESS_GRAVITY_ERROR):
        # Accelerometer is close to gravity - can be used for correction
        q_accel = initialize_quaternion_from_acc(accel[0], accel[1], accel[2])
        
        # Spherical linear interpolation between gyro and accel quaternions
        q_updated = slerp_quaternion(q_gyro, q_accel, 1-alpha)
    else:
        # During high dynamics, trust gyro more
        q_updated = q_gyro
    
    return quat_normalize(q_updated)

def slerp_quaternion(q1, q2, t):
    """
    Spherical linear interpolation between quaternions
    """
    # Compute the cosine of the angle between quaternions
    dot = np.sum(q1 * q2)
    
    # If negative dot, negate one quaternion to take shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Set default interpolation
    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t*(q2 - q1)
        return quat_normalize(result)
    
    # Calculate angle and sin
    theta = np.arccos(min(dot, 1.0))
    sin_theta = np.sin(theta)
    
    # SLERP formula
    s1 = np.sin((1-t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta
    
    result = s1 * q1 + s2 * q2
    return quat_normalize(result)

def update_gyro_bias(b_g, gyro_readings, stillness_duration, is_still):
    """
    Updates gyro bias when device is detected to be still
    """
    if not is_still or stillness_duration < 10:
        return b_g
    
    # Calculate mean of gyro readings during stillness
    gyro_mean = np.mean(gyro_readings, axis=0)
    
    # Apply exponential filter for smooth updates (more weight on longer stillness)
    alpha = min(0.1, 0.005 * stillness_duration)  # Cap at 0.1
    b_g_new = b_g * (1 - alpha) + gyro_mean * alpha
    
    return b_g_new

def skew(v):
    """Create skew-symmetric matrix from vector."""
    x,y,z = v
    return np.array([
        [   0, -z,  y],
        [  z,   0, -x],
        [ -y,  x,   0]
    ])

def integrate_quaternion(q, gyro, dt):
    """Integrate orientation given gyro in rad/s."""
    gq = np.array([0., gyro[0], gyro[1], gyro[2]])
    dq = quat_multiply(q, gq) * 0.5 * dt
    return quat_normalize(q + dq)

def quat_multiply(q, r):
    """Multiply two quaternions."""
    w1,x1,y1,z1 = q
    w2,x2,y2,z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q):
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.,0.,0.,0.])
    return q/n

def quat_conjugate(q):
    """Conjugate of quaternion."""
    return q * np.array([1., -1., -1., -1.])

def rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    qv = np.hstack(([0.], v))
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]

def initialize_quaternion_from_acc(ax, ay, az):
    """
    Initialize orientation so that sensor's measured accel
    aligns with [0,0,GRAVITY_SIGN*9.81] in world frame.
    """
    a = np.array([ax, ay, az])
    norm_a = np.linalg.norm(a)
    if norm_a < 1e-6:
        return np.array([1.,0.,0.,0.])
    a_norm = a/norm_a
    
    g_vec = np.array([0.,0., GRAVITY_SIGN*1.0])  # Unit vector
    cross_ = np.cross(a_norm, g_vec)
    s = np.linalg.norm(cross_)
    c = np.dot(a_norm, g_vec)
    
    if s < 1e-12:
        # Means a_norm is basically parallel to g_vec
        if c > 0:
            return np.array([1.,0.,0.,0.])  # Identity
        else:
            # 180° rotation around x-axis
            return np.array([0.,1.,0.,0.])
    
    axis = cross_/s
    angle = math.atan2(s, c)
    
    half = angle * 0.5
    w = math.cos(half)
    xyz = axis * math.sin(half)
    
    return quat_normalize(np.hstack(([w], xyz)))

def quat_to_euler(q):
    """Convert quaternion to (roll, pitch, yaw) in degrees."""
    w,x,y,z = q
    # roll (x-axis rotation)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

##############################################################################
# Main Function
##############################################################################

def main():
    #------------------------------------------------------------------
    # 0) Read data & calibrate
    #------------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)
    if len(df) < N_CALIB:
        print(f"ERROR: Not enough data for calibration! Need {N_CALIB} rows.")
        return

    for c in ["tMillis","ax","ay","az","gx","gy","gz"]:
        if c not in df.columns:
            print(f"ERROR: Missing {c} in CSV.")
            return

    t_ms  = df["tMillis"].values.astype(float)
    ax_raw = df["ax"].values
    ay_raw = df["ay"].values
    az_raw = df["az"].values
    gx_raw = df["gx"].values
    gy_raw = df["gy"].values
    gz_raw = df["gz"].values

    # Calculate average timestep for when we filter
    dt_array = np.diff(t_ms)/1000.
    dt_array = np.concatenate(([dt_array[0]], dt_array))  # Duplicate first dt
    avg_dt = np.mean(dt_array)

    # Calibration offsets from initial stillness
    axb = np.mean(ax_raw[:N_CALIB])
    ayb = np.mean(ay_raw[:N_CALIB])
    azb = np.mean(az_raw[:N_CALIB])
    gxb = np.mean(gx_raw[:N_CALIB])
    gyb = np.mean(gy_raw[:N_CALIB])
    gzb = np.mean(gz_raw[:N_CALIB])

    # Apply calibration
    ax0 = ax_raw - axb
    ay0 = ay_raw - ayb
    az0 = az_raw - azb
    gx0 = gx_raw - gxb
    gy0 = gy_raw - gyb
    gz0 = gz_raw - gzb

    # Convert to physical units
    ax0 *= 9.81  # to m/s²
    ay0 *= 9.81
    az0 *= 9.81
    gx0 *= math.pi/180.0  # to rad/s
    gy0 *= math.pi/180.0
    gz0 *= math.pi/180.0

    # Apply low-pass filter to all signals
    window_size = int(SAMPLING_FREQ / FILTER_CUTOFF * 2)  # Filter window based on cutoff
    window_size = max(3, min(window_size, 15))  # Constrain between 3-15 samples
    
    ax_filtered = butter_lowpass_filter(ax0, FILTER_CUTOFF, SAMPLING_FREQ)
    ay_filtered = butter_lowpass_filter(ay0, FILTER_CUTOFF, SAMPLING_FREQ)
    az_filtered = butter_lowpass_filter(az0, FILTER_CUTOFF, SAMPLING_FREQ)
    gx_filtered = butter_lowpass_filter(gx0, FILTER_CUTOFF, SAMPLING_FREQ)
    gy_filtered = butter_lowpass_filter(gy0, FILTER_CUTOFF, SAMPLING_FREQ)
    gz_filtered = butter_lowpass_filter(gz0, FILTER_CUTOFF, SAMPLING_FREQ)

    #------------------------------------------------------------------
    # 1) Initialize orientation and variables
    #------------------------------------------------------------------
    # Initialize orientation from first N_CALIB samples
    ax_init = np.mean(ax_filtered[:N_CALIB])
    ay_init = np.mean(ay_filtered[:N_CALIB])
    az_init = np.mean(az_filtered[:N_CALIB])
    q = initialize_quaternion_from_acc(ax_init, ay_init, az_init)

    # Initial variables
    b_g = np.zeros(3)  # gyro bias
    v = np.zeros(3)    # velocity
    
    # Complementary filter alpha (gyro trust factor)
    comp_filter_alpha = CF_STILL_ALPHA

    # Lever-arm variables
    r_body = np.array(SENSOR_OFFSET)
    old_omega_world = np.zeros(3)
    v_corrected = np.zeros(3)

    # Rolling buffers for stillness detection
    recent_ax = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_ay = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_az = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gx = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gy = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gz = deque(maxlen=STILLNESS_WINDOW_SIZE)

    # Deadlift detector initialization
    deadlift_detector = DeadliftDetector()

    # Variables for stillness tracking
    was_still_previous = False
    stillness_duration = 0
    last_reinit_time = -REINIT_COOLDOWN
    prev_acc = np.array([ax_filtered[0], ay_filtered[0], az_filtered[0]])
    prev_gyro = np.array([gx_filtered[0], gy_filtered[0], gz_filtered[0]])

    final_data = []
    debug_data = []

    #----------------------------------------------
    # Main Loop
    #----------------------------------------------
    for i in range(len(t_ms)):
        tstamp = t_ms[i]
        dt = dt_array[i]

        # Get filtered readings
        a_sens = np.array([ax_filtered[i], ay_filtered[i], az_filtered[i]])
        g_sens = np.array([gx_filtered[i], gy_filtered[i], gz_filtered[i]])
        
        # Validate sensor readings against previous values
        a_sens, g_sens = validate_sensor_reading(a_sens, g_sens, prev_acc, prev_gyro, dt)
        prev_acc, prev_gyro = a_sens.copy(), g_sens.copy()

        # Initialize gating flags
        do_full_reinit = False
        do_gravity_update = False

        # STILLNESS detection with enhanced quality assessment
        recent_ax.append(a_sens[0])
        recent_ay.append(a_sens[1])
        recent_az.append(a_sens[2])
        recent_gx.append(g_sens[0])
        recent_gy.append(g_sens[1])
        recent_gz.append(g_sens[2])

        is_still = False
        stillness_quality = 0
        
        if len(recent_ax) == STILLNESS_WINDOW_SIZE:
            # Calculate standard deviations
            std_ax = np.std(recent_ax)
            std_ay = np.std(recent_ay)
            std_az = np.std(recent_az)
            std_gx = np.std(recent_gx)
            std_gy = np.std(recent_gy)
            std_gz = np.std(recent_gz)
            
            # Check if all below thresholds
            if (std_ax < STILLNESS_ACC_STD and
                std_ay < STILLNESS_ACC_STD and
                std_az < STILLNESS_ACC_STD and
                std_gx < STILLNESS_GYRO_STD and
                std_gy < STILLNESS_GYRO_STD and
                std_gz < STILLNESS_GYRO_STD):
                
                is_still = True
                stillness_duration += 1
                
                # Calculate stillness quality
                a_mag = np.linalg.norm(a_sens)
                g_mag = np.linalg.norm(g_sens - b_g)
                
                # Gravity magnitude error
                grav_error = abs(a_mag - 9.81) / 9.81
                
                # Z-axis dominance (for barbell level)
                z_dominance = abs(a_sens[2]) / (a_mag + 0.001)
                
                # Quality increases with low gyro, correct gravity, and Z dominance
                gyro_quality = max(0, 1 - g_mag/0.1)  # 0.1 rad/s threshold
                grav_quality = max(0, 1 - grav_error/0.1)  # 10% gravity error threshold
                
                stillness_quality = (gyro_quality * 0.3 + 
                                    grav_quality * 0.4 + 
                                    z_dominance * 0.3)
            else:
                stillness_duration = 0
        
        # Enhanced gating logic with cooldown and quality check
        if is_still and (not was_still_previous):
            # Just became still - do regular gravity update
            do_gravity_update = True
        elif is_still and stillness_duration >= MIN_STILLNESS_DURATION:
            if (i - last_reinit_time > REINIT_COOLDOWN and stillness_quality > 0.7):
                # If high quality stillness and enough time since last reinit, do full reinit
                do_full_reinit = True
                last_reinit_time = i
            else:
                # Otherwise do regular gravity update
                do_gravity_update = True
        elif is_still:
            # Still but not long enough duration - do gravity update
            do_gravity_update = True
            
        # Update gyro bias during stillness
        if is_still:
            # Create array from deques
            recent_gyros = np.column_stack([
                np.array(list(recent_gx)),
                np.array(list(recent_gy)),
                np.array(list(recent_gz))
            ])
            b_g = update_gyro_bias(b_g, recent_gyros, stillness_duration, is_still)

        # Apply complementary filter for orientation update
        # Adjust alpha based on deadlift state
        dl_state = deadlift_detector.state
        if dl_state == DL_STATE_IDLE or dl_state == DL_STATE_SETUP:
            comp_filter_alpha = CF_STILL_ALPHA
        elif dl_state == DL_STATE_PULL or dl_state == DL_STATE_CONCENTRIC:
            comp_filter_alpha = CF_PULL_ALPHA
        else:
            comp_filter_alpha = CF_GENERAL_ALPHA
            
        # Full reinitialization from high-quality stillness
        if do_full_reinit:
            # Calculate mean accelerometer during stillness
            ax_mean = np.mean(recent_ax)
            ay_mean = np.mean(recent_ay)
            az_mean = np.mean(recent_az)
            
            # Create new quaternion
            q = initialize_quaternion_from_acc(ax_mean, ay_mean, az_mean)
            
            # Reset velocity but keep gyro bias
            v = np.zeros(3)
            
            print(f"Reinitializing orientation at t={tstamp}ms (quality={stillness_quality:.2f})")
        
        # Regular orientation update
        elif i > 0:
            gyro_corr = g_sens - b_g
            q = complementary_filter_update(q, a_sens, gyro_corr, comp_filter_alpha, dt)
        
        # World-frame calculations with lever-arm compensation
        gyro_corr = g_sens - b_g
        omega_world = rotate_vector(q, gyro_corr)
        
        # Estimate angular acceleration
        if i == 0:
            alpha_world_est = np.zeros(3)
        else:
            alpha_world_est = (omega_world - old_omega_world) / dt
        old_omega_world = omega_world.copy()
        
        # Apply lever-arm compensation
        g_val = 9.81
        a_world = rotate_vector(q, a_sens) - np.array([0., 0., GRAVITY_SIGN*g_val])
        
        r_world = rotate_vector(q, r_body)
        aC = -np.cross(omega_world, np.cross(omega_world, r_world))  # Centripetal
        aT = np.cross(alpha_world_est, r_world)  # Tangential
        a_corrected = a_world - (aC + aT)
        
        # Special handling for vertical component
        v_corrected += a_corrected * dt

        horizontal_acc_mag = np.sqrt(a_corrected[0]**2 + a_corrected[1]**2)
    
        # Calculate vertical/horizontal ratio for motion type analysis
        v_h_ratio = 0
        if horizontal_acc_mag > 0.2:  # Only meaningful when there's horizontal motion
            v_h_ratio = abs(vertical_acc) / horizontal_acc_mag
        
        # Zero small velocities during stillness to prevent drift
        if is_still and stillness_quality > 0.6:
            # Apply exponential decay to velocity
            v_corrected *= 0.9
            
            # Hard reset if velocity is small
            if np.linalg.norm(v_corrected) < 0.05:
                v_corrected = np.zeros(3)
        
        # Extract vertical components
        vertical_acc = a_corrected[2]
        vertical_vel = v_corrected[2]
        
        # Update deadlift detector
        state_changed, dl_state = deadlift_detector.process_sample(
            vertical_acc, vertical_vel, i, is_still)
        
        rep_info = deadlift_detector.get_rep_info()
        
        # Store results
        final_data.append((tstamp, vertical_acc, vertical_vel))
        
        # Keep track of previous stillness state
        was_still_previous = is_still
        
        # Store debug info
        rollDeg, pitchDeg, yawDeg = quat_to_euler(q)
        a_mag_g = np.linalg.norm(a_sens) / 9.81
        
        # IMPORTANT: The order must match debug_cols exactly
        debug_data.append((
            tstamp,
            rollDeg, pitchDeg, yawDeg,
            is_still,                  # is_still
            do_full_reinit,            # full_reinit
            do_gravity_update,         # gravity_update
            dl_state,                  # deadlift_state
            rep_info["rep_count"],     # rep_count
            a_mag_g,                   # acc_mag_g
            np.linalg.norm(gyro_corr), # gyro_mag
            np.linalg.norm(v_corrected), # total_speed
            a_corrected[0], a_corrected[1], a_corrected[2],
            v_corrected[0], v_corrected[1], v_corrected[2],
            vertical_acc, vertical_vel,
            stillness_quality,         # stillness_quality
            rep_info["peak_velocity"], # peak_velocity
            rep_info["rep_quality"],   # rep_quality 
            rep_info["rep_rom"],       # rep_rom (range of motion)
            comp_filter_alpha,          # comp_filter_alpha
            horizontal_acc_mag,
            v_h_ratio
        ))

    #------------------------------------------------------------------
    # Post-processing
    #------------------------------------------------------------------
    if FORCE_ZUPT_AT_END and len(final_data) > 1:
        n = len(final_data)
        final_vel_end = final_data[-1][2]
        new_final = []
        for i, row in enumerate(final_data):
            frac = i/(n-1)
            corrected_vel = row[2] - final_vel_end*frac
            new_final.append((row[0], row[1], corrected_vel))
        final_data = new_final

    if SMOOTH_VELOCITY:
        arr_vel = np.array([r[2] for r in final_data])
        if len(arr_vel) >= SMOOTH_WINDOW:
            arr_vel_smooth = savgol_filter(arr_vel, SMOOTH_WINDOW, SMOOTH_POLY)
            new_final = []
            for i, row in enumerate(final_data):
                new_final.append((row[0], row[1], arr_vel_smooth[i]))
            final_data = new_final

    #------------------------------------------------------------------
    # Write out CSVs
    #------------------------------------------------------------------
    df_vert = pd.DataFrame(final_data, columns=["tMillis","vertical_acc","vertical_vel"])
    df_vert.to_csv(VERT_CSV, index=False)
    print(f"Saved final vertical results to {VERT_CSV}.")

    # EXACT matching order for debug_data
    debug_cols = [
        "tMillis",
        "roll_deg","pitch_deg","yaw_deg",
        "is_still","full_reinit","gravity_update","deadlift_state","rep_count",
        "acc_mag_g","gyro_mag","total_speed",
        "acc_world_x","acc_world_y","acc_world_z",
        "vel_world_x","vel_world_y","vel_world_z",
        "vertical_acc","vertical_vel","horizontal_acc_mag","v_h_ratio",
        "stillness_quality","peak_velocity","rep_quality","rep_rom","comp_filter_alpha"
    ]
    df_dbg = pd.DataFrame(debug_data, columns=debug_cols)
    df_dbg.to_csv(DEBUG_CSV, index=False)
    print(f"Saved debug info to {DEBUG_CSV}.")

    plot_results(final_data, df_dbg)
    generate_debug_plots(df_dbg, "./")
    generate_motion_analysis(df_dbg)


##############################################################################
# Visualization Functions
##############################################################################

def plot_results(final_data, df_dbg):
    """Generate detailed visualization for deadlift analysis."""
    times   = np.array([row[0] for row in final_data])
    vertAcc = np.array([row[1] for row in final_data])
    vertVel = np.array([row[2] for row in final_data])

    # Retrieve debug info
    roll_deg = df_dbg["roll_deg"].values
    pitch_deg = df_dbg["pitch_deg"].values
    yaw_deg = df_dbg["yaw_deg"].values

    is_still = df_dbg["is_still"].values.astype(float)
    full_reinit = df_dbg["full_reinit"].values.astype(float)
    gravity_update = df_dbg["gravity_update"].values.astype(float)
    deadlift_state = df_dbg["deadlift_state"].values
    rep_count = df_dbg["rep_count"].values

    acc_world_x = df_dbg["acc_world_x"].values
    acc_world_y = df_dbg["acc_world_y"].values
    acc_world_z = df_dbg["acc_world_z"].values

    vel_world_x = df_dbg["vel_world_x"].values
    vel_world_y = df_dbg["vel_world_y"].values
    vel_world_z = df_dbg["vel_world_z"].values

    acc_mag_g = df_dbg["acc_mag_g"].values
    stillness_quality = df_dbg["stillness_quality"].values
    
    peak_velocity = df_dbg["peak_velocity"].values
    rep_quality = df_dbg["rep_quality"].values
    rep_rom = df_dbg["rep_rom"].values
    comp_filter_alpha = df_dbg["comp_filter_alpha"].values

    # We'll create detailed plots
    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    fig.suptitle("Deadlift Barbell Motion Analysis", fontsize=16)

    # (1) Orientation
    ax1 = axes[0]
    ax1.plot(times, roll_deg,  label="Roll (deg)", color='#1f77b4')
    ax1.plot(times, pitch_deg, label="Pitch (deg)", color='#ff7f0e')
    ax1.plot(times, yaw_deg,   label="Yaw (deg)", color='#2ca02c')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Barbell Orientation")

    # (2) Deadlift State Machine
    ax2 = axes[1]
    
    # Create colored regions for different states
    state_names = ["Idle", "Setup", "Pull", "Concentric", "Lockout", "Eccentric", "Finish"]
    state_colors = ['#f8f9fa', '#f8e5e5', '#fed8b1', '#d1f0c2', '#c2e9f0', '#d5c2f0', '#f0c2e9']
    
    # Find contiguous regions of the same state
    state_segments = []
    current_state = deadlift_state[0]
    start_idx = 0
    
    for i in range(1, len(deadlift_state)):
        if deadlift_state[i] != current_state:
            state_segments.append((start_idx, i-1, current_state))
            current_state = deadlift_state[i]
            start_idx = i
    
    # Add the last segment
    state_segments.append((start_idx, len(deadlift_state)-1, current_state))
    
    # Draw rectangles for each state segment
    for start, end, state in state_segments:
        if start == end:
            continue
        color = state_colors[state]
        ax2.add_patch(Rectangle((times[start], -0.1), 
                               times[end] - times[start], 
                               1.2, 
                               color=color,
                               alpha=0.5))
    
    # Plot still and reinit flags
    ax2.step(times, is_still, label="IsStill", where="post", color='#1f77b4', alpha=0.7)
    ax2.step(times, full_reinit, label="FullReinit", where="post", color='#d62728', alpha=0.7)
    ax2.step(times, stillness_quality, label="StillnessQuality", where="post", color='#9467bd', alpha=0.7)
    
    # Add rep counter as text annotations
    last_rep = 0
    for i in range(len(rep_count)):
        if rep_count[i] > last_rep:
            ax2.text(times[i], 0.5, f"Rep {int(rep_count[i])}", 
                     fontweight='bold', fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            last_rep = rep_count[i]
    
    ax2.set_ylim([-0.1, 1.1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylabel("State Values")
    ax2.set_title("Deadlift State Machine & Stillness Detection")
    
    # Add state legend
    legend_elements = []
    for i, name in enumerate(state_names):
        legend_elements.append(Patch(facecolor=state_colors[i], alpha=0.5, label=name))
    ax2.legend(handles=legend_elements, loc='upper left', ncol=4)

    # (3) 3D Acceleration (world frame)
    ax3 = axes[2]
    ax3.plot(times, acc_world_x, label="World X (m/s²)", color='#1f77b4')
    ax3.plot(times, acc_world_y, label="World Y (m/s²)", color='#ff7f0e')
    ax3.plot(times, acc_world_z, label="World Z (m/s²)", color='#2ca02c')
    
    # Highlight regions with high vertical acceleration
    for i in range(len(vertAcc)):
        if abs(vertAcc[i]) > DEADLIFT_START_ACC_THRESHOLD:
            ax3.axvspan(times[i]-5, times[i]+5, color='#2ca02c', alpha=0.2)
    
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylabel("m/s²")
    ax3.set_title("World-Frame Acceleration (Lever-Arm Corrected)")

    # (4) 3D Velocity (world frame)
    ax4 = axes[3]
    ax4.plot(times, vel_world_x, label="World X (m/s)", color='#1f77b4')
    ax4.plot(times, vel_world_y, label="World Y (m/s)", color='#ff7f0e')
    ax4.plot(times, vel_world_z, label="World Z (m/s)", color='#2ca02c')
    
    # Add horizontal line at deadlift concentric threshold
    ax4.axhline(y=DEADLIFT_CONCENTRIC_MIN_VEL, color='#2ca02c', linestyle='--', 
                alpha=0.5, label=f"Concentric Threshold ({DEADLIFT_CONCENTRIC_MIN_VEL} m/s)")
    
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylabel("m/s")
    ax4.set_title("World-Frame Velocity (Integrated)")

    # (5) Final Vertical Acc & Vel with complementary filter alpha
    ax5 = axes[4]
    ax5.plot(times, vertAcc, label="Vertical Acc (m/s²)", color='blue')
    ax5.plot(times, vertVel, label="Vertical Vel (m/s)", color='red')
    
    # Plot filter alpha on secondary y-axis
    ax5_2 = ax5.twinx()
    ax5_2.plot(times, comp_filter_alpha, label="Filter Alpha", color='green', alpha=0.5)
    ax5_2.set_ylim([0, 1.1])
    ax5_2.set_ylabel("Comp Filter Alpha")
    
    # Add detected peak velocities as markers
    for i in range(1, len(peak_velocity)):
        if peak_velocity[i] > 0 and peak_velocity[i-1] == 0:
            ax5.plot(times[i], peak_velocity[i], 'ro', markersize=8)
            ax5.text(times[i], peak_velocity[i] + 0.1, 
                   f"{peak_velocity[i]:.2f} m/s", fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left')
    ax5_2.legend(loc='upper right')
    ax5.set_ylabel("m/s² / m/s")
    ax5.set_title("Final Vertical Acceleration & Velocity")

    # (6) Rep Quality Metrics
    ax6 = axes[5]
    
    # Create a scatter plot for rep quality
    non_zero_indices = np.where(rep_quality > 0)[0]
    if len(non_zero_indices) > 0:
        ax6.scatter(times[non_zero_indices], rep_quality[non_zero_indices], 
                   label="Rep Quality (%)", color='purple', s=100, marker='*')
        
        # Add text annotations for quality
        for idx in non_zero_indices:
            ax6.text(times[idx], rep_quality[idx] + 5, 
                   f"{int(rep_quality[idx])}%", fontsize=9, ha='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='purple'))
    
    # Plot ROM for each rep
    non_zero_rom = np.where(rep_rom > 0)[0]
    if len(non_zero_rom) > 0:
        ax6_2 = ax6.twinx()
        ax6_2.scatter(times[non_zero_rom], rep_rom[non_zero_rom], 
                    label="Rep ROM (m)", color='orange', s=80, marker='D')
        
        # Add text annotations for ROM
        for idx in non_zero_rom:
            if rep_rom[idx] > 0.01:  # Only annotate significant ROM values
                ax6_2.text(times[idx], rep_rom[idx] + 0.02, 
                        f"{rep_rom[idx]:.2f}m", fontsize=9, ha='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))
        
        ax6_2.set_ylim([0, max(1.0, np.max(rep_rom) * 1.2)])
        ax6_2.set_ylabel("Range of Motion (m)")
        ax6_2.legend(loc='upper right')
    
    # Plot net acceleration for reference
    ax6.plot(times, acc_mag_g, label="Net Acc (G)", color='gray', alpha=0.3)
    
    ax6.set_ylim([0, 110])
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left')
    ax6.set_ylabel("Quality (%)")
    ax6.set_xlabel("Time (ms)")
    ax6.set_title("Rep Quality Metrics")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("deadlift_analysis.png", dpi=150)
    print("Saved visualization to deadlift_analysis.png")
    plt.show()

def generate_motion_analysis(df_dbg):
    """Generate analysis of vertical vs horizontal motion patterns"""
    times = df_dbg["tMillis"].values
    
    # Extract accelerations
    acc_x = df_dbg["acc_world_x"].values
    acc_y = df_dbg["acc_world_y"].values
    acc_z = df_dbg["acc_world_z"].values
    
    # Calculate components
    vertical_acc = acc_z
    horizontal_acc = np.sqrt(acc_x**2 + acc_y**2)
    
    # Stillness flag (use what's available in your data)
    is_still = df_dbg["is_still"].values if "is_still" in df_dbg.columns else df_dbg["skip_impact"].values.astype(float)
    
    # Create figure with just the essential plots
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Vertical vs Horizontal acceleration
    plt.subplot(2, 1, 1)
    plt.plot(times, vertical_acc, label="Vertical Acc", color='green', linewidth=2)
    plt.plot(times, horizontal_acc, label="Horizontal Acc", color='red', linewidth=2)
    
    # Add threshold line (adjust to your actual threshold)
    threshold = 1.3  # Your deadlift detection threshold
    plt.axhline(y=threshold, color='green', linestyle='--', label=f"Detection Threshold ({threshold})")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Highlight stillness periods
    for i in range(len(is_still)-1):
        if is_still[i] > 0.5:
            plt.axvspan(times[i], times[i+1], color='green', alpha=0.1)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Vertical vs Horizontal Acceleration")
    
    # Plot 2: Vertical-to-Horizontal Ratio
    plt.subplot(2, 1, 2)
    
    # Calculate ratio with safety for division by zero
    v_h_ratio = np.zeros_like(vertical_acc)
    for i in range(len(vertical_acc)):
        if abs(horizontal_acc[i]) > 0.2:
            v_h_ratio[i] = abs(vertical_acc[i]) / horizontal_acc[i]
    
    plt.plot(times, v_h_ratio, label="Vertical/Horizontal Ratio", color='blue', linewidth=2)
    plt.axhline(y=1.0, color='red', linestyle='--', label="Equal Ratio (1.0)")
    
    # Add helpful annotations
    plt.annotate("Vertical Dominant (Deadlift)", xy=(0.05, 0.85), xycoords='axes fraction', 
                 fontsize=10, bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.3))
    plt.annotate("Horizontal Dominant (Rolling)", xy=(0.05, 0.25), xycoords='axes fraction', 
                 fontsize=10, bbox=dict(boxstyle="round", fc="lightcoral", alpha=0.3))
    
    plt.ylim([0, 5])  # Cap ratio for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel("Ratio")
    plt.xlabel("Time (ms)")
    plt.title("Vertical/Horizontal Ratio - Key for Distinguishing Rolling from Deadlift")
    
    plt.tight_layout()
    plt.savefig("motion_analysis.png", dpi=150)
    print("Saved motion analysis to motion_analysis.png")

##############################################################################

if __name__=="__main__":
    main()