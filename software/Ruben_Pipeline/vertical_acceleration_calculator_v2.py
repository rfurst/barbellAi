# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:11:39 2025

@author: rkfurst
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import os
import pywt  # For wavelet analysis


def verify_array_lengths(arrays_dict):
    """
    Verify that all arrays in the dictionary have the same length.
    
    Parameters:
    -----------
    arrays_dict : dict
        Dictionary of arrays to check
    
    Returns:
    --------
    tuple
        (is_valid: bool, error_message: str)
    """
    lengths = {name: len(arr) for name, arr in arrays_dict.items()}
    reference_length = next(iter(lengths.values()))
    
    mismatched = {name: length for name, length in lengths.items() 
                 if length != reference_length}
    
    if mismatched:
        error_msg = "Array length mismatch detected:\n"
        error_msg += f"  Expected length: {reference_length}\n"
        error_msg += "  Mismatched arrays:\n"
        for name, length in mismatched.items():
            error_msg += f"    - {name}: {length} (diff: {length - reference_length})\n"
        return False, error_msg
    
    return True, "All arrays have matching lengths."


def calculate_vertical_acceleration(csv_file, output_file=None, plot_results=True, rotation_correction=True):
    """
    Calculate vertical acceleration from raw IMU data using quaternion-based orientation,
    physics-based rotation correction, and wavelet signal processing.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing IMU data
    output_file : str, optional
        Path to save processed data. If None, no file is saved.
    plot_results : bool, optional
        Whether to generate plots of the results
    rotation_correction : bool, optional
        Whether to apply correction for rotation-induced acceleration
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing time and vertical acceleration
    """
    print(f"Processing {csv_file}...")
    
    # Step 1: Load and preprocess the data
    try:
        data = pd.read_csv(csv_file)
        
        # Check for required columns
        required_cols = ['tMillis', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        # Convert time to seconds for easier interpretation
        data['time_s'] = (data['tMillis'] - data['tMillis'].iloc[0]) / 1000.0
        
        # Calculate time differences for quaternion integration
        data['dt'] = np.append([0], np.diff(data['time_s']))
        
        print(f"  Loaded {len(data)} samples spanning {data['time_s'].max():.2f} seconds")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Store the original data length for verification later
    original_data_length = len(data)
    
    # Step 2: Implement quaternion-based orientation tracking
    # Initialize quaternion array (represents orientation)
    orientations = np.zeros((len(data), 4))
    orientations[0] = [1, 0, 0, 0]  # Identity quaternion (w, x, y, z)
    
    # Initialize gravity direction array
    gravity = np.zeros((len(data), 3))
    
    # Parameters for static detection (used for drift correction)
    static_threshold = 0.1  # rad/s - threshold for detecting "static" periods
    acc_magnitude = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
    gravity_magnitude = 1.0  # Expected magnitude of gravity (1g)
    acc_threshold = 0.05  # Threshold for detecting "clean" gravity (g)
    
    # Track calculated angular velocities for later use
    angular_velocity_mag = np.zeros(len(data))
    
    for i in range(1, len(data)):
        # Get angular velocity in rad/s (convert from deg/s)
        gx_rad = np.radians(data['gx'].iloc[i])
        gy_rad = np.radians(data['gy'].iloc[i])
        gz_rad = np.radians(data['gz'].iloc[i])
        
        # Get time difference
        dt = data['dt'].iloc[i]
        
        # Calculate magnitude of angular velocity for static detection
        angular_velocity_mag[i] = np.sqrt(gx_rad**2 + gy_rad**2 + gz_rad**2)
        
        # Create angular velocity quaternion (0, ωx, ωy, ωz)
        omega_quat = np.array([0, gx_rad, gy_rad, gz_rad])
        
        # Current orientation
        current_q = orientations[i-1]
        
        # Quaternion integration using first-order approximation
        # q_dot = 0.5 * q ⊗ ω
        q_dot = 0.5 * quaternion_multiply(current_q, omega_quat)
        
        # Update orientation: q_new = q_old + q_dot * dt
        next_q = current_q + q_dot * dt
        
        # Normalize the quaternion to prevent drift in magnitude
        next_q = next_q / np.linalg.norm(next_q)
        
        # Store the new orientation
        orientations[i] = next_q
        
        # Correct for drift during static periods
        is_static = angular_velocity_mag[i] < static_threshold
        is_clean_gravity = abs(acc_magnitude.iloc[i] - gravity_magnitude) < acc_threshold
        
        if is_static and is_clean_gravity:
            # Get current acceleration vector
            acc = np.array([data['ax'].iloc[i], data['ay'].iloc[i], data['az'].iloc[i]])
            acc_normalized = acc / np.linalg.norm(acc)
            
            # Use accelerometer to correct for drift
            # Finding rotation from current gravity estimate to measured gravity
            # Current gravity direction based on orientation
            current_gravity = rotate_vector_by_quaternion([0, 0, 1], next_q)
            
            # Find quaternion that rotates current_gravity to acc_normalized
            correction_q = find_rotation_quaternion(current_gravity, acc_normalized)
            
            # Apply small correction factor (adjust for sensitivity)
            correction_factor = 0.05
            slerped_q = quaternion_slerp(next_q, 
                                         quaternion_multiply(correction_q, next_q),
                                         correction_factor)
            
            # Update orientation with drift correction
            orientations[i] = slerped_q / np.linalg.norm(slerped_q)
        
        # Calculate gravity direction from orientation
        gravity[i] = rotate_vector_by_quaternion([0, 0, 1], orientations[i])
    
    # Step 3: Calculate vertical acceleration with physics-based rotation correction
    vertical_accel = np.zeros(len(data))
    centripetal_accel = np.zeros(len(data)) if rotation_correction else None
    
    for i in range(len(data)):
        # Get current accelerometer reading
        accel = np.array([data['ax'].iloc[i], data['ay'].iloc[i], data['az'].iloc[i]])
        
        # Project raw acceleration onto gravity direction
        raw_vertical = np.dot(accel, gravity[i])
        
        # Subtract 1g to get actual acceleration
        vertical_value = raw_vertical - 1.0
        
        # Apply physics-based rotation correction if enabled
        if rotation_correction and i > 0:
            # Convert angular velocity to rad/s
            omega = np.array([
                np.radians(data['gx'].iloc[i]),
                np.radians(data['gy'].iloc[i]),
                np.radians(data['gz'].iloc[i])
            ])
            
            # Compute centripetal acceleration (ω × (ω × r))
            # For simplicity, use gravity direction as position vector
            # This assumes rotation around the body center
            omega_cross_r = np.cross(omega, gravity[i])
            centripetal = np.cross(omega, omega_cross_r)
            
            # Project centripetal acceleration onto gravity direction
            centripetal_component = np.dot(centripetal, gravity[i])
            
            # Store for plotting
            centripetal_accel[i] = centripetal_component
            
            # Remove centripetal component from vertical acceleration
            vertical_value -= centripetal_component
        
        vertical_accel[i] = vertical_value
    
    # Step 4: Apply wavelet-based signal processing with robust fallback
    try:
        if len(data) > 64:  # Need enough samples for wavelet decomposition
            print("  Applying wavelet denoising...")
            vertical_accel_filtered = wavelet_denoise(vertical_accel, wavelet='db4')
            
            # Verify length - critical check
            if len(vertical_accel_filtered) != len(vertical_accel):
                print(f"  WARNING: Wavelet output length mismatch: {len(vertical_accel_filtered)} vs {len(vertical_accel)}")
                # Force length correction
                vertical_accel_filtered = fix_array_length(vertical_accel_filtered, len(vertical_accel))
        else:
            # Fallback to traditional filtering for small datasets
            print("  Dataset too small for wavelets, using traditional filter...")
            vertical_accel_filtered = apply_traditional_filter(vertical_accel, data['time_s'])
    except Exception as e:
        print(f"  WARNING: Wavelet processing failed ({str(e)}), falling back to traditional filter")
        # Fallback to traditional filtering when wavelet fails
        vertical_accel_filtered = apply_traditional_filter(vertical_accel, data['time_s'])
    
    # Step 5: Create output DataFrame with length verification
    # Prepare data dictionary for DataFrame
    result_data = {
        'time_s': data['time_s'],
        'tMillis': data['tMillis'],
        'raw_vertical_accel': vertical_accel,
        'vertical_accel': vertical_accel_filtered,
        'gravity_x': gravity[:, 0],
        'gravity_y': gravity[:, 1],
        'gravity_z': gravity[:, 2],
        'quat_w': orientations[:, 0],
        'quat_x': orientations[:, 1],
        'quat_y': orientations[:, 2],
        'quat_z': orientations[:, 3]
    }
    
    # Add centripetal acceleration if rotation correction was enabled
    if rotation_correction:
        result_data['centripetal_accel'] = centripetal_accel
    
    # Verify array lengths before creating DataFrame
    is_valid, error_message = verify_array_lengths(result_data)
    if not is_valid:
        print("ERROR: Cannot create DataFrame due to array length mismatch.")
        print(error_message)
        
        # Attempt to fix array lengths
        print("Attempting to fix array length mismatches...")
        reference_length = len(data['time_s'])
        
        # Fix each array to match the reference length
        for name, array in result_data.items():
            if len(array) != reference_length:
                if len(array) > reference_length:
                    print(f"  Trimming {name} from {len(array)} to {reference_length}")
                    result_data[name] = array[:reference_length]
                else:
                    print(f"  Padding {name} from {len(array)} to {reference_length}")
                    pad_amount = reference_length - len(array)
                    result_data[name] = np.append(array, np.repeat(array[-1], pad_amount))
        
        # Verify again after fixing
        is_valid, error_message = verify_array_lengths(result_data)
        if not is_valid:
            print("ERROR: Failed to fix array length mismatches.")
            print(error_message)
            return None
        else:
            print("Successfully fixed array length mismatches.")
    
    # Create DataFrame after verification
    result = pd.DataFrame(result_data)
    
    # Step 6: Generate statistics and summary
    print(f"  Vertical acceleration statistics:")
    print(f"    Mean: {result['vertical_accel'].mean():.4f} g")
    print(f"    Min: {result['vertical_accel'].min():.4f} g")
    print(f"    Max: {result['vertical_accel'].max():.4f} g")
    print(f"    Range: {result['vertical_accel'].max() - result['vertical_accel'].min():.4f} g")
    
    # Step 7: Save results if requested
    if output_file:
        # Save full detailed results
        result.to_csv(output_file, index=False)
        print(f"  Full results saved to {output_file}")
        
        # Save simplified results with only timestamp and vertical acceleration
        simple_output_file = output_file.replace('.csv', '_simple.csv')
        simple_result = pd.DataFrame({
            'time_s': result['time_s'],
            'tMillis': result['tMillis'],
            'vertical_accel': result['vertical_accel']
        })
        simple_result.to_csv(simple_output_file, index=False)
        print(f"  Simplified results (timestamp and vertical acceleration only) saved to {simple_output_file}")
    
    # Step 8: Plot results if requested
    if plot_results:
        plt.figure(figsize=(12, 15))  # Increased height for additional plots
        
        # Plot 1: Raw acceleration data
        plt.subplot(5, 1, 1)
        plt.plot(result['time_s'], data['ax'], 'r-', alpha=0.7, label='ax')
        plt.plot(result['time_s'], data['ay'], 'g-', alpha=0.7, label='ay')
        plt.plot(result['time_s'], data['az'], 'b-', alpha=0.7, label='az')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Raw Acceleration')
        plt.ylabel('Acceleration (g)')
        
        # Plot 2: Angular velocity
        plt.subplot(5, 1, 2)
        plt.plot(result['time_s'], data['gx'], 'r-', alpha=0.7, label='gx')
        plt.plot(result['time_s'], data['gy'], 'g-', alpha=0.7, label='gy')
        plt.plot(result['time_s'], data['gz'], 'b-', alpha=0.7, label='gz')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Angular Velocity')
        plt.ylabel('Degrees/Second')
        
        # Plot 3: Quaternion components
        plt.subplot(5, 1, 3)
        plt.plot(result['time_s'], result['quat_w'], 'k-', label='quat_w')
        plt.plot(result['time_s'], result['quat_x'], 'r-', label='quat_x')
        plt.plot(result['time_s'], result['quat_y'], 'g-', label='quat_y')
        plt.plot(result['time_s'], result['quat_z'], 'b-', label='quat_z')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Quaternion Orientation')
        plt.ylabel('Component Value')
        
        # Plot 4: Gravity direction
        plt.subplot(5, 1, 4)
        plt.plot(result['time_s'], result['gravity_x'], 'r-', label='gravity_x')
        plt.plot(result['time_s'], result['gravity_y'], 'g-', label='gravity_y')
        plt.plot(result['time_s'], result['gravity_z'], 'b-', label='gravity_z')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Gravity Direction (Quaternion-based)')
        plt.ylabel('Component')
        
        # Plot 5: Vertical acceleration with components
        plt.subplot(5, 1, 5)
        plt.plot(result['time_s'], result['raw_vertical_accel'], 'k-', alpha=0.4, label='Unfiltered')
        plt.plot(result['time_s'], result['vertical_accel'], 'r-', label='Filtered (Wavelet)')
        
        if rotation_correction:
            plt.plot(result['time_s'], result['centripetal_accel'], 'g--', alpha=0.5, label='Centripetal Component')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Vertical Acceleration with Physics-Based Corrections')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        
        plt.tight_layout()
        
        # Save the plot if output file is specified
        if output_file:
            plot_file = output_file.replace('.csv', '.png')
            plt.savefig(plot_file, dpi=300)  # Higher DPI for better quality plots
            print(f"  Plot saved to {plot_file}")
        
        plt.show()
    
    return result


# ===== Array Length Fixing Helpers =====

def fix_array_length(array, target_length):
    """
    Fixes the length of an array to match the target length.
    
    Parameters:
    -----------
    array : numpy.ndarray
        Array to fix
    target_length : int
        Desired length
        
    Returns:
    --------
    numpy.ndarray
        Array with corrected length
    """
    current_length = len(array)
    
    if current_length == target_length:
        return array
    
    if current_length > target_length:
        # Trim the array
        return array[:target_length]
    else:
        # Pad the array by repeating the last value
        pad_amount = target_length - current_length
        return np.append(array, np.repeat(array[-1], pad_amount))


def apply_traditional_filter(signal_data, time_data):
    """
    Apply a traditional Butterworth filter to the signal.
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        Signal to filter
    time_data : numpy.ndarray
        Time data for calculating sample rate
        
    Returns:
    --------
    numpy.ndarray
        Filtered signal
    """
    # Calculate sample rate
    fs = 1.0 / np.mean(np.diff(time_data))
    
    # Design Butterworth high-pass filter
    # This removes slow drifts while keeping faster movements
    b, a = signal.butter(2, 0.1, 'highpass', fs=fs)
    
    # Apply filter (zero-phase to avoid phase distortion)
    filtered = signal.filtfilt(b, a, signal_data)
    
    return filtered


def wavelet_denoise(signal, wavelet='db4', level=None, threshold_mode='soft'):
    """
    Apply wavelet denoising to a signal with length preservation.
    
    Parameters:
    -----------
    signal : array-like
        Input signal to denoise
    wavelet : str, optional
        Wavelet to use (default: 'db4')
    level : int, optional
        Decomposition level. If None, auto-determined.
    threshold_mode : str, optional
        Threshold mode: 'soft' or 'hard'
        
    Returns:
    --------
    array-like
        Denoised signal with guaranteed same length as input
    """
    # Store original length for later verification
    original_length = len(signal)
    
    # Determine decomposition level if not specified
    if level is None:
        # Calculate maximum possible level
        max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        # Use a reasonable level (not too deep, not too shallow)
        level = min(max_level, 4)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Estimate noise level using median absolute deviation of finest detail coefficients
    finest_detail = coeffs[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    
    # Calculate threshold (using VisuShrink / universal threshold)
    # This could be optimized or adapted based on specific noise characteristics
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply thresholding to detail coefficients (keep approximation intact)
    new_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged
    
    for i in range(1, len(coeffs)):
        # Use different scaling for different levels
        # The finest levels typically have more noise
        level_scale = 1.0 - (0.1 * (len(coeffs) - i - 1))
        level_threshold = threshold * level_scale
        
        # Apply threshold
        new_coeffs.append(pywt.threshold(coeffs[i], level_threshold, mode=threshold_mode))
    
    # Reconstruct signal
    reconstructed_signal = pywt.waverec(new_coeffs, wavelet)
    
    # Fix length mismatch if any (wavelet transforms sometimes can change array length)
    # This is a critical step to ensure the returned array has exactly the same length
    # as the input array, preventing the "All arrays must be of the same length" error
    if len(reconstructed_signal) != original_length:
        print(f"  Fixing wavelet output length: {len(reconstructed_signal)} → {original_length}")
        reconstructed_signal = fix_array_length(reconstructed_signal, original_length)
    
    return reconstructed_signal


# ===== Quaternion Operations =====

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Parameters:
    q1, q2: Quaternions in format [w, x, y, z]
    
    Returns:
    Quaternion product
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    
    Parameters:
    q: Quaternion in format [w, x, y, z]
    
    Returns:
    Conjugate quaternion [w, -x, -y, -z]
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector using a quaternion rotation.
    
    Parameters:
    v: 3D vector [x, y, z]
    q: Quaternion in format [w, x, y, z]
    
    Returns:
    Rotated 3D vector
    """
    # Convert vector to quaternion format (0, x, y, z)
    v_quat = np.array([0, v[0], v[1], v[2]])
    
    # Perform rotation: q * v * q^(-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    # Extract vector part (x, y, z)
    return rotated[1:]

def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation between quaternions.
    
    Parameters:
    q1, q2: Quaternions to interpolate between
    t: Interpolation factor (0 to 1)
    
    Returns:
    Interpolated quaternion
    """
    # Ensure unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute the cosine of the angle between quaternions
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, negate q2
    # This is to ensure we take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot to valid range
    dot = min(1.0, max(-1.0, dot))
    
    # Compute the interpolation factors
    angle = np.arccos(dot)
    
    # Check if quaternions are very close
    if angle < 1e-10:
        return q1
    
    sin_angle = np.sin(angle)
    factor1 = np.sin((1.0 - t) * angle) / sin_angle
    factor2 = np.sin(t * angle) / sin_angle
    
    # Perform the interpolation
    result = factor1 * q1 + factor2 * q2
    
    # Normalize the result
    return result / np.linalg.norm(result)

def find_rotation_quaternion(v1, v2):
    """
    Find quaternion that rotates vector v1 to vector v2.
    
    Parameters:
    v1, v2: 3D unit vectors
    
    Returns:
    Quaternion representing the rotation from v1 to v2
    """
    # Ensure unit vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Compute the cross product (rotation axis)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    # Check if vectors are parallel or anti-parallel
    if axis_norm < 1e-10:
        # Vectors are parallel
        if np.dot(v1, v2) > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Vectors are anti-parallel, find perpendicular axis
        # Find a vector perpendicular to v1
        if abs(v1[0]) < abs(v1[1]):
            if abs(v1[0]) < abs(v1[2]):
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 0.0, 1.0])
        else:
            if abs(v1[1]) < abs(v1[2]):
                perp = np.array([0.0, 1.0, 0.0])
            else:
                perp = np.array([0.0, 0.0, 1.0])
                
        axis = np.cross(v1, perp)
        axis = axis / np.linalg.norm(axis)
        
        # 180-degree rotation around perpendicular axis
        return np.array([0.0, axis[0], axis[1], axis[2]])
    
    # Normalize the axis
    axis = axis / axis_norm
    
    # Compute the dot product (cosine of angle)
    cos_angle = np.dot(v1, v2)
    
    # Compute the half angle
    half_angle = np.arccos(cos_angle) / 2
    
    # Compute quaternion components
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    
    # Return quaternion [w, x, y, z]
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def process_multiple_files(input_dir, output_dir=None, rotation_correction=True):
    """
    Process multiple CSV files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files
    output_dir : str, optional
        Directory to save results. If None, results are saved alongside inputs.
    rotation_correction : bool, optional
        Whether to apply correction for rotation-induced acceleration
    """
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        
        if output_dir:
            output_file = os.path.join(output_dir, f"vertical_{csv_file}")
        else:
            output_file = os.path.join(input_dir, f"vertical_{csv_file}")
        
        # Process the file
        calculate_vertical_acceleration(input_path, output_file, plot_results=True, rotation_correction=rotation_correction)


if __name__ == "__main__":
    import argparse
    import sys
    
    # Check for required packages first
    required_packages = ['pandas', 'numpy', 'scipy', 'matplotlib', 'pywt']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("ERROR: Missing required Python packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install these packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr with conda:")
        print(f"conda install {' '.join(missing_packages)}")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Calculate vertical acceleration from IMU data")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--no-rotation-correction", action="store_true", 
                        help="Disable correction for rotation-induced acceleration")
    parser.add_argument("--use-traditional-filter", action="store_true",
                        help="Use traditional Butterworth filter instead of wavelet denoising")
    
    args = parser.parse_args()
    
    # Check if input is a directory or file
    if os.path.isdir(args.input):
        process_multiple_files(args.input, args.output, not args.no_rotation_correction)
    else:
        calculate_vertical_acceleration(args.input, args.output, not args.no_plot, 
                                      not args.no_rotation_correction)