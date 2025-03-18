"""
Enhanced Auto-Adaptive Deadlift Detector with Two-Pass Orientation Validation and Velocity Calculation Support
This script automatically detects the concentric (lifting) phase of deadlift repetitions
from IMU sensor data with robust orientation detection to prevent signal inversion issues.
It processes the signal in both orientations and selects the one producing more plausible results.
It also exports the processed data in formats suitable for velocity calculation.
The script now provides comprehensive data export features designed specifically for velocity
calculation in a separate script. Two output formats are supported:

Comprehensive CSV - Contains all processed data with additional columns for phase identification,
including rep_number, phase type, phase_progress, and other metadata needed for integration
JSON Summary - Contains key metrics and timestamps for each detected concentric phase with
all necessary metadata for accurate velocity calculation

These outputs provide all the necessary data for numerical integration from acceleration to velocity
in a follow-up script, including precise timestamps, properly oriented acceleration values, and
complete phase boundary information.
Usage:
python enhanced_deadlift_detector.py input_file.csv --target-rep-count 6 --export-format both
Output Files:
- {input_filename_base}_processed_data.csv: Comprehensive CSV with all data points
- {input_filename_base}_rep_summary.json: Concentric phase summary in JSON format
Author: Claude
Date: March 8, 2025
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import os
import warnings
import json
import datetime
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")

class DatasetProfiler:
    """
    Analyzes dataset characteristics to determine optimal processing parameters.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.profile = {}
    
    def profile_dataset(self, data):
        """
        Generate a comprehensive profile of the dataset to guide parameter selection.
        
        Args:
            data: DataFrame containing the dataset
            
        Returns:
            Dictionary containing dataset profile information
        """
        # Identify available columns and their types
        self.profile["columns"] = self._classify_columns(data)
        
        # Analyze sampling characteristics
        self.profile["sampling"] = self._analyze_sampling(data)
        
        # Analyze signal characteristics
        self.profile["signal"] = self._analyze_signal_characteristics(data)
        
        # Analyze orientation information
        self.profile["orientation"] = self._analyze_orientation_data(data)
        
        # Determine recommended processing parameters
        self.profile["recommended_params"] = self._determine_optimal_parameters()
        
        return self.profile
    
    def _classify_columns(self, data):
        """Classify columns into categories based on names and characteristics."""
        columns = {
            "time": [],
            "acceleration": [],
            "vertical_acceleration": [],
            "orientation": [],
            "angular_velocity": [],
            "gravity": [],
            "quaternion": [],
            "other": []
        }
        
        # Time column patterns
        time_patterns = ["time", "t_", "timestamp", "tmillis"]
        
        # Acceleration patterns
        accel_patterns = ["acc", "accel", "ax", "ay", "az", "a_x", "a_y", "a_z"]
        
        # Vertical acceleration patterns
        vert_patterns = ["vertical", "vert_acc", "v_acc", "raw_vertical"]
        
        # Orientation patterns
        orient_patterns = ["quat", "orientation", "rot"]
        
        # Angular velocity patterns
        gyro_patterns = ["gyro", "angular", "gx", "gy", "gz", "omega"]
        
        # Gravity patterns
        grav_patterns = ["grav", "g_x", "g_y", "g_z"]
        
        # Quaternion patterns
        quat_patterns = ["quat", "q_"]
        
        # Classify each column
        for col in data.columns:
            col_lower = col.lower()
            
            # Check for time columns
            if any(pattern in col_lower for pattern in time_patterns):
                columns["time"].append(col)
            
            # Check for vertical acceleration columns
            elif any(pattern in col_lower for pattern in vert_patterns):
                columns["vertical_acceleration"].append(col)
            
            # Check for acceleration columns
            elif any(pattern in col_lower for pattern in accel_patterns):
                columns["acceleration"].append(col)
            
            # Check for quaternion columns
            elif any(pattern in col_lower for pattern in quat_patterns):
                columns["quaternion"].append(col)
                columns["orientation"].append(col)
            
            # Check for gravity columns
            elif any(pattern in col_lower for pattern in grav_patterns):
                columns["gravity"].append(col)
                columns["orientation"].append(col)
            
            # Check for angular velocity columns
            elif any(pattern in col_lower for pattern in gyro_patterns):
                columns["angular_velocity"].append(col)
            
            # Check for other orientation columns
            elif any(pattern in col_lower for pattern in orient_patterns):
                columns["orientation"].append(col)
            
            # Other columns
            else:
                columns["other"].append(col)
        
        # Validate and refine classification with statistical analysis
        self._validate_column_classification(data, columns)
        
        return columns
    
    def _validate_column_classification(self, data, columns):
        """
        Validate column classification using statistical properties.
        """
        # Validate time columns - should be monotonically increasing
        for col in list(columns["time"]):
            if col in data.columns and not data[col].is_monotonic_increasing:
                # Move to other category
                columns["time"].remove(col)
                columns["other"].append(col)
        
        # Validate acceleration columns - should have significant variance
        for category in ["acceleration", "vertical_acceleration"]:
            for col in list(columns[category]):
                if col in data.columns and data[col].var() < 0.00001:
                    if self.verbose:
                        print(f"Warning: Column {col} has very low variance, might not be acceleration data")
    
    def _analyze_sampling(self, data):
        """
        Analyze sampling characteristics of the dataset.
        """
        sampling_info = {}
        
        # Find primary time column
        time_col = self._get_primary_time_column(data, self.profile["columns"]["time"])
        
        if time_col:
            # Calculate sampling rate
            time_diffs = np.diff(data[time_col])
            sampling_info["mean_interval"] = np.mean(time_diffs)
            sampling_info["sampling_rate"] = 1.0 / sampling_info["mean_interval"]
            sampling_info["interval_std"] = np.std(time_diffs)
            sampling_info["is_uniform"] = sampling_info["interval_std"] < 0.1 * sampling_info["mean_interval"]
        
        return sampling_info
    
    def _get_primary_time_column(self, data, time_columns):
        """
        Determine the primary time column to use.
        """
        if not time_columns:
            # If no time columns classified, look for "time_s" or "tMillis" directly
            if "time_s" in data.columns:
                return "time_s"
            elif "tMillis" in data.columns:
                return "tMillis"
            return None
        
        # Prefer "time_s" if available
        if "time_s" in time_columns:
            return "time_s"
        
        # Otherwise use the first available time column
        return time_columns[0]
    
    def _analyze_signal_characteristics(self, data):
        """
        Analyze signal characteristics to guide filter selection.
        """
        signal_info = {}
        
        # Analyze vertical acceleration columns if available
        vert_acc_cols = self.profile["columns"]["vertical_acceleration"]
        if vert_acc_cols:
            for col in vert_acc_cols:
                if col in data.columns:
                    signal = data[col].values
                    signal_info[col] = self._compute_signal_metrics(signal)
        
        # Analyze regular acceleration columns
        accel_cols = self.profile["columns"]["acceleration"]
        if accel_cols:
            for col in accel_cols:
                if col in data.columns:
                    signal = data[col].values
                    signal_info[col] = self._compute_signal_metrics(signal)
        
        return signal_info
    
    def _analyze_orientation_data(self, data):
        """
        Analyze orientation data (quaternions, gravity vectors) to determine sensor positioning.
        """
        orientation_info = {}
        
        # Check for quaternion data
        quat_cols = [col for col in data.columns if "quat" in col.lower()]
        has_quaternions = len(quat_cols) >= 4  # Need all 4 components
        
        if has_quaternions:
            orientation_info["has_quaternions"] = True
            # Identify the component names
            quat_w_col = next((col for col in quat_cols if "w" in col.lower()), None)
            quat_x_col = next((col for col in quat_cols if "x" in col.lower()), None)
            quat_y_col = next((col for col in quat_cols if "y" in col.lower()), None)
            quat_z_col = next((col for col in quat_cols if "z" in col.lower()), None)
            
            if quat_w_col and quat_x_col and quat_y_col and quat_z_col:
                orientation_info["quaternion_columns"] = {
                    "w": quat_w_col,
                    "x": quat_x_col,
                    "y": quat_y_col,
                    "z": quat_z_col
                }
                
                # Calculate mean quaternion to get average orientation
                mean_qw = data[quat_w_col].mean()
                mean_qx = data[quat_x_col].mean()
                mean_qy = data[quat_y_col].mean()
                mean_qz = data[quat_z_col].mean()
                
                # Normalize
                norm = np.sqrt(mean_qw**2 + mean_qx**2 + mean_qy**2 + mean_qz**2)
                if norm > 0:
                    mean_qw /= norm
                    mean_qx /= norm
                    mean_qy /= norm
                    mean_qz /= norm
                
                    orientation_info["mean_quaternion"] = {
                        "w": mean_qw,
                        "x": mean_qx,
                        "y": mean_qy,
                        "z": mean_qz
                    }
        else:
            orientation_info["has_quaternions"] = False
        
        # Check for gravity vector data
        grav_cols = [col for col in data.columns if "gravity" in col.lower() or "grav_" in col.lower()]
        has_gravity = len(grav_cols) >= 3  # Need all 3 components
        
        if has_gravity:
            orientation_info["has_gravity"] = True
            # Identify the component names
            grav_x_col = next((col for col in grav_cols if "x" in col.lower()), None)
            grav_y_col = next((col for col in grav_cols if "y" in col.lower()), None)
            grav_z_col = next((col for col in grav_cols if "z" in col.lower()), None)
            
            if grav_x_col and grav_y_col and grav_z_col:
                orientation_info["gravity_columns"] = {
                    "x": grav_x_col,
                    "y": grav_y_col,
                    "z": grav_z_col
                }
                
                # Calculate mean gravity vector to get average orientation
                mean_gx = data[grav_x_col].mean()
                mean_gy = data[grav_y_col].mean()
                mean_gz = data[grav_z_col].mean()
                
                # Calculate norm of gravity
                norm = np.sqrt(mean_gx**2 + mean_gy**2 + mean_gz**2)
                
                # Store normalized gravity components
                if norm > 0.5:  # Should be close to 1g total
                    orientation_info["mean_gravity"] = {
                        "x": mean_gx / norm,
                        "y": mean_gy / norm,
                        "z": mean_gz / norm
                    }
                    
                    # Determine primary gravity axis (closest to -1)
                    grav_components = {"x": mean_gx / norm, "y": mean_gy / norm, "z": mean_gz / norm}
                    primary_axis = min(grav_components.items(), key=lambda x: x[1])[0]
                    orientation_info["primary_gravity_axis"] = primary_axis
        else:
            orientation_info["has_gravity"] = False
        
        return orientation_info
    
    def _compute_signal_metrics(self, signal):
        """
        Compute metrics to characterize a signal.
        """
        metrics = {}
        
        # Filter out NaNs for computation
        signal = signal[~np.isnan(signal)]
        if len(signal) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "range": 0, 
                    "mad": 0, "estimated_snr": 0, "peak_count": 0, "skewness": 0}
        
        # Basic statistics
        metrics["mean"] = np.mean(signal)
        metrics["std"] = np.std(signal)
        metrics["min"] = np.min(signal)
        metrics["max"] = np.max(signal)
        metrics["range"] = metrics["max"] - metrics["min"]
        
        # Calculate skewness and kurtosis for distribution shape
        if len(signal) > 2:
            # Skewness gives us information about asymmetry
            # Positive skew means tail on right, negative means tail on left
            diff = signal - metrics["mean"]
            skewness = np.mean(diff**3) / (np.mean(diff**2)**1.5) if np.mean(diff**2) > 0 else 0
            metrics["skewness"] = skewness
            
            # Kurtosis tells us about the "tailedness" of the distribution
            kurtosis = np.mean(diff**4) / (np.mean(diff**2)**2) - 3 if np.mean(diff**2) > 0 else 0
            metrics["kurtosis"] = kurtosis
        else:
            metrics["skewness"] = 0
            metrics["kurtosis"] = 0
        
        # Noise estimation (using median absolute deviation)
        median = np.median(signal)
        metrics["mad"] = np.median(np.abs(signal - median))
        
        # Approximate signal-to-noise ratio
        # Using 95th percentile as signal level and MAD as noise level
        p95 = np.percentile(np.abs(signal), 95)
        metrics["estimated_snr"] = p95 / metrics["mad"] if metrics["mad"] > 0 else 10
        
        # Characterize peak distribution
        # Find peaks with minimal parameters
        try:
            peaks, _ = signal.find_peaks(signal, height=metrics["mean"] + metrics["std"] * 0.5)
            if len(peaks) > 0:
                metrics["peak_count"] = len(peaks)
                peak_heights = signal[peaks]
                metrics["mean_peak_height"] = np.mean(peak_heights)
                metrics["peak_height_std"] = np.std(peak_heights)
            else:
                metrics["peak_count"] = 0
        except:
            metrics["peak_count"] = 0
        
        # Calculate positive and negative signal percentages
        pos_signal = signal[signal > 0]
        neg_signal = signal[signal < 0]
        metrics["pos_ratio"] = len(pos_signal) / len(signal) if len(signal) > 0 else 0.5
        metrics["neg_ratio"] = len(neg_signal) / len(signal) if len(signal) > 0 else 0.5
        metrics["pos_mean"] = np.mean(pos_signal) if len(pos_signal) > 0 else 0
        metrics["neg_mean"] = np.mean(neg_signal) if len(neg_signal) > 0 else 0
        
        # Analyze frequency content if signal is long enough
        if len(signal) > 20:
            try:
                # Use FFT to analyze frequency components
                fft_values = np.abs(np.fft.rfft(signal))
                freqs = np.fft.rfftfreq(len(signal))
                
                # Find dominant frequencies
                if len(fft_values) > 0:
                    dominant_idx = np.argmax(fft_values[1:]) + 1  # Skip DC component
                    metrics["dominant_freq"] = freqs[dominant_idx]
                    metrics["dominant_freq_magnitude"] = fft_values[dominant_idx]
                    
                    # Calculate spectral energy distribution
                    total_energy = np.sum(fft_values**2)
                    if total_energy > 0:
                        # Low frequency energy ratio (relevant for human movement)
                        low_freq_mask = freqs < 0.1  # Frequencies below ~10% of Nyquist
                        low_freq_energy = np.sum(fft_values[low_freq_mask]**2)
                        metrics["low_freq_energy_ratio"] = low_freq_energy / total_energy
            except:
                # If FFT analysis fails, continue without frequency metrics
                pass
        
        return metrics
    
    def _determine_optimal_parameters(self):
        """
        Determine optimal processing parameters based on dataset profile.
        """
        params = {}
        
        # Default parameters
        params["filter_type"] = "butterworth"
        params["filter_order"] = 3
        params["cutoff_freq"] = 5.0
        params["min_duration"] = 0.2  # Reduced default minimum duration
        params["max_duration"] = 2.5
        params["smoothing_window"] = 7
        
        # Adjust based on sampling rate if available
        if "sampling" in self.profile and "sampling_rate" in self.profile["sampling"]:
            sampling_rate = self.profile["sampling"]["sampling_rate"]
            # Adjust cutoff frequency based on sampling rate
            params["cutoff_freq"] = min(sampling_rate / 8, 5.0)
            
            # Adjust smoothing window based on sampling rate
            params["smoothing_window"] = max(3, int(sampling_rate / 15))
            
            # For higher sampling rates, we can have shorter durations
            if sampling_rate > 40:  # Higher sampling rate datasets
                params["min_duration"] = 0.15  # Even shorter minimum duration for high sample rate data
        
        # Adjust based on signal characteristics
        if "signal" in self.profile:
            signal_info = self.profile["signal"]
            
            # Find column with best SNR
            best_snr = 0
            best_col = None
            for col, metrics in signal_info.items():
                if "estimated_snr" in metrics and metrics["estimated_snr"] > best_snr:
                    best_snr = metrics["estimated_snr"]
                    best_col = col
            
            if best_col and "peak_count" in signal_info[best_col]:
                peak_count = signal_info[best_col]["peak_count"]
                
                # Adjust parameters based on signal quality
                if best_snr < 3:  # Noisy signal
                    params["filter_type"] = "butterworth"
                    params["filter_order"] = 4
                    params["cutoff_freq"] = min(params["cutoff_freq"], 3.0)
                elif best_snr > 10:  # Clean signal
                    params["filter_type"] = "savgol"
                    params["filter_order"] = 2
                
                # Adjust duration parameters based on detected peaks
                if peak_count > 0 and "sampling" in self.profile and "sampling_rate" in self.profile["sampling"]:
                    sampling_rate = self.profile["sampling"]["sampling_rate"]
                    data_duration = 1
                    signal_length = 0
                    try:
                        signal_length = len(signal_info[best_col])
                    except:
                        signal_length = peak_count * 10  # Fallback estimate
                        
                    if signal_length > 0 and sampling_rate > 0:
                        data_duration = signal_length / sampling_rate
                        avg_peak_interval = data_duration / max(1, peak_count)
                        
                        # Less restrictive minimum duration to catch more repetitions
                        params["min_duration"] = min(0.2, max(0.1, avg_peak_interval * 0.1))
                        params["max_duration"] = min(3.0, max(1.0, avg_peak_interval * 0.8))
            
            # Analyze frequency content if available to further refine filter parameters
            for col, metrics in signal_info.items():
                if "dominant_freq" in metrics and "low_freq_energy_ratio" in metrics:
                    # If there's strong low-frequency content (typical for deadlifts)
                    if metrics["low_freq_energy_ratio"] > 0.7:
                        # Adjust cutoff frequency to focus on the movement frequency band
                        dominant_freq = metrics["dominant_freq"]
                        if dominant_freq > 0:
                            # Set cutoff frequency to 2-3x the dominant frequency to capture
                            # the main movement plus some harmonics
                            suggested_cutoff = min(10.0, dominant_freq * 3.0)
                            params["cutoff_freq"] = min(params["cutoff_freq"], suggested_cutoff)
        
        # Add orientation analysis to parameters to guide signal processing
        if "orientation" in self.profile:
            orientation_info = self.profile["orientation"]
            
            # If we have gravity vector information, use it to inform processing
            if orientation_info.get("has_gravity", False) and "mean_gravity" in orientation_info:
                # Calculate vertical component dominance - indicates if acceleration is primarily vertical
                grav = orientation_info["mean_gravity"]
                max_component = max(abs(grav["x"]), abs(grav["y"]), abs(grav["z"]))
                vertical_dominance = max_component / np.sqrt(grav["x"]**2 + grav["y"]**2 + grav["z"]**2)
                
                params["vertical_dominance"] = vertical_dominance
                
                # If there's strong alignment with gravity axis, we can be more confident
                # in our signal processing parameters
                if vertical_dominance > 0.9:  # Strong vertical alignment
                    # For strongly vertical signals, we can use more aggressive filtering
                    params["filter_order"] = min(4, params["filter_order"] + 1)
                else:
                    # For less aligned signals, be more conservative
                    params["min_duration"] = max(0.15, params["min_duration"])  # Slightly longer min duration
        
        return params

class TwoPassOrientationDetector:
    """
    Enhanced orientation detector that processes the signal in both orientations
    and selects the one that produces more plausible deadlift patterns.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def determine_best_orientation(self, data, profile, detector_instance):
        """
        Determine the best orientation by testing both possibilities and selecting
        the one that produces more plausible deadlift patterns.
        
        Args:
            data: DataFrame containing the data
            profile: Dataset profile
            detector_instance: The deadlift detector instance (for access to detection methods)
            
        Returns:
            Tuple of (selected_column, should_negate, orientation_info)
        """
        if self.verbose:
            print("Starting two-pass orientation detection...")
        
        # Step 1: Select the most likely vertical acceleration column
        selected_column = self._select_vertical_accel_column(data, profile)
        
        if not selected_column:
            raise ValueError("Could not find suitable vertical acceleration source")
        
        if self.verbose:
            print(f"Selected vertical acceleration column: {selected_column}")
        
        # Step 2: Perform frequency domain analysis on the signal
        freq_analysis = self._analyze_signal_frequency(data, selected_column)
        
        # Step 3: Process the signal in both orientations
        # First pass: Original orientation
        orientation1_results = self._test_orientation(data, selected_column, False, profile, detector_instance)
        
        # Second pass: Inverted orientation
        orientation2_results = self._test_orientation(data, selected_column, True, profile, detector_instance)
        
        # Step 4: Enhanced comparison - use both biomechanical metrics and frequency analysis
        should_negate, orientation_info = self._compare_orientation_results(
            orientation1_results, orientation2_results, freq_analysis)
        
        if self.verbose:
            print(f"Selected orientation for {selected_column}: " + 
                 f"{'inverted' if should_negate else 'original'} " +
                 f"(confidence: {orientation_info['confidence']:.2f})")
        
        return selected_column, should_negate, orientation_info
    
    def _analyze_signal_frequency(self, data, column):
        """
        Perform frequency domain analysis to help determine correct orientation.
        Deadlifts typically have characteristic frequency distributions.
        
        Args:
            data: DataFrame with the data
            column: Column to analyze
            
        Returns:
            Dictionary with frequency analysis results
        """
        results = {}
        
        # Get the signal in both orientations
        signal_orig = data[column].values
        signal_inv = -signal_orig
        
        # Remove NaNs
        signal_orig = signal_orig[~np.isnan(signal_orig)]
        signal_inv = signal_inv[~np.isnan(signal_inv)]
        
        if len(signal_orig) < 10:
            return results
        
        try:
            # Analyze frequency content of both orientations
            for name, signal in [("original", signal_orig), ("inverted", signal_inv)]:
                # Apply window function to reduce spectral leakage
                window = np.hanning(len(signal))
                windowed_signal = signal * window
                
                # Compute FFT
                fft_values = np.abs(np.fft.rfft(windowed_signal))
                freqs = np.fft.rfftfreq(len(signal))
                
                if len(fft_values) > 1:
                    # Find dominant frequencies (excluding DC component)
                    sorted_idx = np.argsort(fft_values[1:])[::-1] + 1
                    top_freqs = freqs[sorted_idx[:5]]  # Top 5 frequencies
                    top_magnitudes = fft_values[sorted_idx[:5]]
                    
                    # Calculate spectral energy distribution
                    total_energy = np.sum(fft_values**2)
                    
                    # Low frequency energy ratio (relevant for human movement)
                    if total_energy > 0:
                        # Frequencies below ~5% of Nyquist (typical for human movement)
                        low_freq_mask = freqs < 0.05
                        low_freq_energy = np.sum(fft_values[low_freq_mask]**2)
                        low_freq_ratio = low_freq_energy / total_energy
                        
                        # Calculate spectral shape metrics
                        spectral_centroid = np.sum(freqs * fft_values**2) / total_energy if total_energy > 0 else 0
                        
                        # Store results
                        results[name] = {
                            "top_frequencies": top_freqs,
                            "top_magnitudes": top_magnitudes,
                            "low_freq_energy_ratio": low_freq_ratio,
                            "spectral_centroid": spectral_centroid
                        }
                        
                        # Calculate spectral kurtosis (peakedness of frequency distribution)
                        # Higher values suggest more distinctive frequency patterns
                        if spectral_centroid > 0:
                            spectral_kurtosis = np.sum(((freqs - spectral_centroid) / spectral_centroid)**4 * 
                                                      fft_values**2) / total_energy
                            results[name]["spectral_kurtosis"] = spectral_kurtosis
            
            # Compare frequency characteristics of both orientations
            if "original" in results and "inverted" in results:
                # Human movement (like deadlifts) typically has concentrated energy in lower frequencies
                # The correct orientation often has higher low-frequency energy ratio
                orig_low_ratio = results["original"].get("low_freq_energy_ratio", 0)
                inv_low_ratio = results["inverted"].get("low_freq_energy_ratio", 0)
                
                results["low_freq_ratio_comparison"] = {
                    "original": orig_low_ratio,
                    "inverted": inv_low_ratio,
                    "difference": orig_low_ratio - inv_low_ratio
                }
                
                # Calculate periodicity score based on spectral kurtosis
                # Higher kurtosis suggests more defined frequency peaks (more periodic)
                orig_kurtosis = results["original"].get("spectral_kurtosis", 0)
                inv_kurtosis = results["inverted"].get("spectral_kurtosis", 0)
                
                results["periodicity_comparison"] = {
                    "original": orig_kurtosis,
                    "inverted": inv_kurtosis,
                    "difference": orig_kurtosis - inv_kurtosis
                }
        except Exception as e:
            if self.verbose:
                print(f"Frequency analysis error: {e}")
        
        return results
    
    def _select_vertical_accel_column(self, data, profile):
        """
        Select the most likely vertical acceleration column from the dataset.
        """
        # First, prefer pre-calculated vertical acceleration columns
        vert_cols = profile["columns"]["vertical_acceleration"]
        if vert_cols:
            for col in vert_cols:
                if col in data.columns and data[col].var() > 0.00001:
                    return col
        
        # If no vertical acceleration, use the acceleration column with highest variance
        accel_cols = profile["columns"]["acceleration"]
        if accel_cols:
            best_var = 0
            best_col = None
            for col in accel_cols:
                if col in data.columns:
                    var = data[col].var()
                    if var > best_var:
                        best_var = var
                        best_col = col
            
            if best_col:
                return best_col
        
        # Check for gravity-aligned acceleration
        if "orientation" in profile and profile["orientation"].get("has_gravity", False):
            grav_info = profile["orientation"].get("mean_gravity", {})
            
            # Find primary gravity axis (closest to -1 or 1)
            primary_axis = None
            max_value = 0
            
            for axis, value in grav_info.items():
                if abs(value) > abs(max_value):
                    max_value = value
                    primary_axis = axis
            
            if primary_axis:
                # Look for acceleration along this axis
                axis_pattern = primary_axis.lower()  # x, y, or z
                
                # Find acceleration columns that match the primary gravity axis
                axis_cols = [col for col in accel_cols if axis_pattern in col.lower()]
                
                if axis_cols and axis_cols[0] in data.columns:
                    return axis_cols[0]
        
        # Last resort: use any numeric column with sufficient variance
        best_var = 0
        best_col = None
        for col in data.columns:
            if data[col].dtype.kind in 'fc':  # float or complex
                var = data[col].var()
                if var > best_var:
                    best_var = var
                    best_col = col
        
        return best_col
    
    def _test_orientation(self, data, column, negate, profile, detector_instance):
        """
        Test an orientation by processing the signal and evaluating the results.
        """
        # Create a processor instance for this test
        processor = AdaptiveSignalProcessor(verbose=False)
        
        # Process the signal with the given orientation
        processed_data = processor.process_signal(data, column, profile, negate)
        
        # Set up parameters for phase detection
        threshold_manager = AdaptiveThresholdManager(verbose=False)
        thresholds = threshold_manager.determine_thresholds(processed_data, profile)
        peak_threshold = thresholds['peak_threshold']
        baseline_threshold = thresholds['baseline_threshold']
        
        # Get duration parameters
        min_duration = profile['recommended_params']['min_duration']
        max_duration = profile['recommended_params']['max_duration']
        
        # Find peaks
        peaks = detector_instance._find_peaks_internal(processed_data, peak_threshold, 
                                                     baseline_threshold, min_duration, processor.sampling_rate)
        
        # Detect phases
        phases = []
        for peak in peaks:
            phase = detector_instance._find_concentric_phase_internal(
                processed_data, peak["index"], peak_threshold, baseline_threshold,
                min_duration, max_duration, processor.sampling_rate
            )
            if phase:
                phases.append(phase)
        
        # Filter overlapping phases
        phases = detector_instance._filter_overlapping_phases(phases)
        
        # Calculate biomechanical scores
        biomech_scores = self._calculate_biomechanical_scores(processed_data, phases)
        
        # Calculate advanced metrics for better orientation detection
        advanced_metrics = self._calculate_advanced_metrics(processed_data, phases, processor.sampling_rate)
        
        # Return results for comparison
        return {
            "column": column,
            "negate": negate,
            "processed_data": processed_data,
            "peaks": peaks,
            "phases": phases,
            "peak_threshold": peak_threshold,
            "baseline_threshold": baseline_threshold,
            "biomech_scores": biomech_scores,
            "advanced_metrics": advanced_metrics
        }
    
    def _calculate_biomechanical_scores(self, processed_data, phases):
        """
        Calculate biomechanical plausibility scores for the detected phases.
        """
        if not phases:
            return {
                "overall_score": 0,
                "phase_count": 0,
                "phase_consistency": 0,
                "acceleration_profile": 0,
                "timing_regularity": 0,
                "positive_dominance": 0
            }
        
        scores = {}
        
        # Score 1: Number of detected phases (more is generally better)
        scores["phase_count"] = min(1.0, len(phases) / 5)  # Cap at 5 phases for full score
        
        # Score 2: Consistency of phase durations (should be similar)
        durations = [phase["duration"] for phase in phases]
        duration_std = np.std(durations) if len(durations) > 1 else 0
        duration_mean = np.mean(durations) if durations else 0
        if duration_mean > 0:
            cv = duration_std / duration_mean  # Coefficient of variation
            scores["phase_consistency"] = max(0, 1.0 - cv)  # Lower variability is better
        else:
            scores["phase_consistency"] = 0
        
        # Score 3: Acceleration profile (positive during concentric phase)
        positive_ratios = []
        for phase in phases:
            start_idx = phase["startIdx"]
            end_idx = phase["endIdx"]
            if start_idx < end_idx:
                phase_values = processed_data["smoothed"].iloc[start_idx:end_idx+1].values
                pos_ratio = sum(1 for v in phase_values if v > 0) / len(phase_values) if len(phase_values) > 0 else 0
                positive_ratios.append(pos_ratio)
        
        avg_positive_ratio = np.mean(positive_ratios) if positive_ratios else 0
        scores["acceleration_profile"] = avg_positive_ratio  # Higher ratio of positive values is better
        
        # Score 4: Timing regularity (regular intervals between phases)
        if len(phases) > 1:
            start_times = [phase["startTime"] for phase in phases]
            intervals = np.diff(start_times)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            if interval_mean > 0:
                interval_cv = interval_std / interval_mean
                scores["timing_regularity"] = max(0, 1.0 - min(interval_cv, 1.0))  # Lower variability is better
            else:
                scores["timing_regularity"] = 0
        else:
            scores["timing_regularity"] = 0.5  # Neutral score for single phase
        
        # Score 5: Overall signal positive dominance during phases
        all_phase_values = []
        for phase in phases:
            start_idx = phase["startIdx"]
            end_idx = phase["endIdx"]
            if start_idx < end_idx:
                phase_values = processed_data["smoothed"].iloc[start_idx:end_idx+1].values
                all_phase_values.extend(phase_values)
        
        overall_pos_ratio = sum(1 for v in all_phase_values if v > 0) / len(all_phase_values) if all_phase_values else 0
        scores["positive_dominance"] = overall_pos_ratio
        
        # Score 6: Peak distribution - examine how acceleration peaks are distributed within phases
        if len(phases) > 0 and "smoothed" in processed_data.columns:
            peak_positions = []
            for phase in phases:
                start_idx = phase["startIdx"]
                end_idx = phase["endIdx"]
                peak_idx = phase["peakIdx"]
                
                if end_idx > start_idx:
                    # Calculate relative position of peak within phase (0-1)
                    rel_position = (peak_idx - start_idx) / (end_idx - start_idx)
                    peak_positions.append(rel_position)
            
            if peak_positions:
                # Calculate peak position mean and variance
                peak_pos_mean = np.mean(peak_positions)
                peak_pos_std = np.std(peak_positions)
                
                # Score based on peak position consistency and alignment with expected patterns
                # Deadlift peaks typically occur in first half of concentric phase
                pos_score = 1.0 - min(1.0, peak_pos_std * 2)  # Lower variation is better
                alignment_score = 1.0 - min(1.0, abs(peak_pos_mean - 0.3) * 2)  # Closer to 0.3 is better
                
                scores["peak_position"] = (pos_score + alignment_score) / 2
            else:
                scores["peak_position"] = 0
        else:
            scores["peak_position"] = 0
        
        # Calculate overall score (weighted average)
        scores["overall_score"] = (
            scores["phase_count"] * 0.15 +
            scores["phase_consistency"] * 0.15 +
            scores["acceleration_profile"] * 0.25 +  # Most important criterion
            scores["timing_regularity"] * 0.15 +
            scores["positive_dominance"] * 0.2 +
            scores.get("peak_position", 0) * 0.1
        )
        
        return scores
    
    def _calculate_advanced_metrics(self, processed_data, phases, sampling_rate):
        """
        Calculate advanced metrics for orientation detection.
        
        Args:
            processed_data: DataFrame with processed signal data
            phases: List of detected phases
            sampling_rate: Sampling rate of the signal
            
        Returns:
            Dictionary with advanced metrics
        """
        metrics = {}
        
        # Skip if no phases detected
        if not phases or "smoothed" not in processed_data.columns:
            return metrics
        
        try:
            # 1. Calculate smoothness metrics
            smoothness_scores = []
            derivative_variations = []
            
            for phase in phases:
                start_idx = phase["startIdx"]
                end_idx = phase["endIdx"]
                
                if end_idx > start_idx:
                    # Get phase signal
                    phase_signal = processed_data["smoothed"].iloc[start_idx:end_idx+1].values
                    
                    if len(phase_signal) > 3:
                        # Calculate normalized jerk (measure of smoothness)
                        # Lower values indicate smoother movement
                        if "smoothed_derivative" in processed_data.columns:
                            # Use pre-calculated derivative if available
                            phase_deriv = processed_data["smoothed_derivative"].iloc[start_idx:end_idx+1].values
                            
                            # Calculate second derivative (jerk)
                            phase_jerk = np.diff(phase_deriv)
                            
                            if len(phase_jerk) > 0:
                                # Normalize by phase duration and peak acceleration
                                phase_duration = phase["duration"]
                                peak_accel = phase["peakAcceleration"]
                                
                                if phase_duration > 0 and peak_accel > 0:
                                    norm_factor = phase_duration**3 / peak_accel
                                    norm_jerk = np.sum(phase_jerk**2) * norm_factor
                                    
                                    # Convert to score (lower jerk = higher score)
                                    smoothness = np.exp(-norm_jerk / 10)  # Exponential scaling
                                    smoothness_scores.append(smoothness)
                                    
                                    # Also calculate derivative variation
                                    if len(phase_deriv) > 1:
                                        deriv_cv = np.std(phase_deriv) / np.mean(np.abs(phase_deriv)) if np.mean(np.abs(phase_deriv)) > 0 else 0
                                        derivative_variations.append(deriv_cv)
            
            # Calculate mean smoothness if we have scores
            if smoothness_scores:
                metrics["mean_smoothness"] = np.mean(smoothness_scores)
                metrics["smoothness_std"] = np.std(smoothness_scores)
            
            if derivative_variations:
                metrics["mean_derivative_variation"] = np.mean(derivative_variations)
            
            # 2. Calculate movement pattern consistency
            if len(phases) >= 2:
                # Extract fixed-length normalized signals for each phase
                norm_length = 50  # Standard length for comparison
                normalized_signals = []
                
                for phase in phases:
                    start_idx = phase["startIdx"]
                    end_idx = phase["endIdx"]
                    
                    if end_idx > start_idx:
                        # Get phase signal
                        phase_signal = processed_data["smoothed"].iloc[start_idx:end_idx+1].values
                        
                        if len(phase_signal) > 2:
                            # Normalize amplitude
                            if max(phase_signal) - min(phase_signal) > 0:
                                norm_signal = (phase_signal - min(phase_signal)) / (max(phase_signal) - min(phase_signal))
                            else:
                                norm_signal = np.zeros_like(phase_signal)
                            
                            # Resample to standard length using linear interpolation
                            indices = np.linspace(0, len(norm_signal)-1, norm_length)
                            resampled = np.interp(indices, np.arange(len(norm_signal)), norm_signal)
                            
                            normalized_signals.append(resampled)
                
                # Calculate cross-correlations between pairs of normalized signals
                if len(normalized_signals) >= 2:
                    correlations = []
                    
                    for i in range(len(normalized_signals)):
                        for j in range(i+1, len(normalized_signals)):
                            # Calculate cross-correlation
                            xcorr = np.correlate(normalized_signals[i], normalized_signals[j], mode='full')
                            
                            # Get maximum correlation value
                            max_corr = np.max(xcorr) / norm_length  # Normalize by length
                            correlations.append(max_corr)
                    
                    if correlations:
                        metrics["mean_pattern_correlation"] = np.mean(correlations)
                        metrics["min_pattern_correlation"] = np.min(correlations)
            
            # 3. Energy distribution analysis
            if "smoothed" in processed_data.columns:
                # Get the full signal
                signal = processed_data["smoothed"].values
                
                # Calculate energy in different regions
                phase_indices = np.zeros(len(signal), dtype=bool)
                
                # Mark all phase regions
                for phase in phases:
                    start_idx = phase["startIdx"]
                    end_idx = phase["endIdx"]
                    
                    if start_idx < end_idx and end_idx < len(phase_indices):
                        phase_indices[start_idx:end_idx+1] = True
                
                # Calculate energy in phase vs. non-phase regions
                phase_energy = np.sum(signal[phase_indices]**2)
                non_phase_energy = np.sum(signal[~phase_indices]**2)
                
                total_energy = phase_energy + non_phase_energy
                
                if total_energy > 0:
                    metrics["phase_energy_ratio"] = phase_energy / total_energy
                    
                    # For correct orientation, energy should be more concentrated in phases
                    # This is especially true for deadlifts, which have distinct concentric phases
        except Exception as e:
            # In case of any errors, continue without these metrics
            if self.verbose:
                print(f"Error calculating advanced metrics: {e}")
        
        return metrics
    
    def _compare_orientation_results(self, orientation1_results, orientation2_results, freq_analysis=None):
        """
        Enhanced comparison of results from both orientations to select the better one.
        
        Args:
            orientation1_results: Results from original orientation
            orientation2_results: Results from inverted orientation
            freq_analysis: Results from frequency domain analysis
            
        Returns:
            Tuple of (should_negate, orientation_info)
        """
        # Get biomechanical scores for both orientations
        scores1 = orientation1_results["biomech_scores"]["overall_score"]
        scores2 = orientation2_results["biomech_scores"]["overall_score"]
        
        # Get advanced metrics if available
        adv_metrics1 = orientation1_results.get("advanced_metrics", {})
        adv_metrics2 = orientation2_results.get("advanced_metrics", {})
        
        # Initialize decision with default values
        should_negate = False
        confidence = 0.5
        decision_factors = []
        
        # Create a scoring system with multiple factors
        factors = {}
        
        # Factor 1: Biomechanical scores comparison
        bio_diff = scores2 - scores1
        if abs(bio_diff) > 0.1:  # Significant difference
            factors["biomechanical"] = {
                "favors_inverted": bio_diff > 0,
                "weight": 0.4,  # High weight for biomechanical plausibility
                "confidence": min(0.9, abs(bio_diff) * 2) 
            }
        
        # Factor 2: Number of detected phases
        phases1 = len(orientation1_results["phases"])
        phases2 = len(orientation2_results["phases"])
        
        if max(phases1, phases2) > 0:
            phase_ratio = abs(phases2 - phases1) / max(1, max(phases1, phases2))
            
            if phase_ratio > 0.2:  # Significant difference in phase count
                factors["phase_count"] = {
                    "favors_inverted": phases2 > phases1,
                    "weight": 0.2,
                    "confidence": min(0.8, phase_ratio * 2)
                }
        
        # Factor 3: Movement smoothness (if available)
        if "mean_smoothness" in adv_metrics1 and "mean_smoothness" in adv_metrics2:
            smooth1 = adv_metrics1["mean_smoothness"]
            smooth2 = adv_metrics2["mean_smoothness"]
            
            smooth_diff = smooth2 - smooth1
            if abs(smooth_diff) > 0.1:  # Significant difference in smoothness
                factors["smoothness"] = {
                    "favors_inverted": smooth_diff > 0,
                    "weight": 0.15,
                    "confidence": min(0.7, abs(smooth_diff) * 2)
                }
        
        # Factor 4: Pattern consistency (if available)
        if "mean_pattern_correlation" in adv_metrics1 and "mean_pattern_correlation" in adv_metrics2:
            corr1 = adv_metrics1["mean_pattern_correlation"]
            corr2 = adv_metrics2["mean_pattern_correlation"]
            
            corr_diff = corr2 - corr1
            if abs(corr_diff) > 0.1:  # Significant difference in pattern correlation
                factors["pattern_consistency"] = {
                    "favors_inverted": corr_diff > 0,
                    "weight": 0.15,
                    "confidence": min(0.7, abs(corr_diff) * 2)
                }
        
        # Factor 5: Frequency analysis (if available)
        if freq_analysis and "low_freq_ratio_comparison" in freq_analysis:
            freq_diff = freq_analysis["low_freq_ratio_comparison"]["difference"]
            
            # For deadlifts, correct orientation typically has higher low-frequency energy
            if abs(freq_diff) > 0.1:  # Significant difference
                factors["frequency_content"] = {
                    "favors_inverted": freq_diff < 0,  # Negative diff means inverted has higher low-freq ratio
                    "weight": 0.15,
                    "confidence": min(0.7, abs(freq_diff) * 2)
                }
        
        # Factor 6: Energy concentration (if available)
        if "phase_energy_ratio" in adv_metrics1 and "phase_energy_ratio" in adv_metrics2:
            energy1 = adv_metrics1["phase_energy_ratio"]
            energy2 = adv_metrics2["phase_energy_ratio"]
            
            energy_diff = energy2 - energy1
            if abs(energy_diff) > 0.1:  # Significant difference
                factors["energy_concentration"] = {
                    "favors_inverted": energy_diff > 0,
                    "weight": 0.15,
                    "confidence": min(0.7, abs(energy_diff) * 2)
                }
        
        # Factor 7: Positive acceleration dominance
        pos_dom1 = orientation1_results["biomech_scores"]["positive_dominance"]
        pos_dom2 = orientation2_results["biomech_scores"]["positive_dominance"]
        
        dom_diff = pos_dom2 - pos_dom1
        if abs(dom_diff) > 0.15:  # Significant difference
            factors["positive_dominance"] = {
                "favors_inverted": dom_diff > 0,
                "weight": 0.25,
                "confidence": min(0.8, abs(dom_diff) * 2)
            }
        
        # Special case: If one orientation has no phases but the other does, use the one with phases
        if phases1 == 0 and phases2 > 0:
            should_negate = True
            confidence = 0.8
            decision_factors = ["no_phases_in_orientation1"]
            
            # Create orientation info
            orientation_info = {
                "confidence": confidence,
                "decision_factors": decision_factors,
                "original_score": float(scores1),
                "inverted_score": float(scores2),
                "original_phases": phases1,
                "inverted_phases": phases2,
                "orientation1_scores": orientation1_results["biomech_scores"],
                "orientation2_scores": orientation2_results["biomech_scores"]
            }
            
            return should_negate, orientation_info
            
        elif phases2 == 0 and phases1 > 0:
            should_negate = False
            confidence = 0.8
            decision_factors = ["no_phases_in_orientation2"]
            
            # Create orientation info
            orientation_info = {
                "confidence": confidence,
                "decision_factors": decision_factors,
                "original_score": float(scores1),
                "inverted_score": float(scores2),
                "original_phases": phases1,
                "inverted_phases": phases2,
                "orientation1_scores": orientation1_results["biomech_scores"],
                "orientation2_scores": orientation2_results["biomech_scores"]
            }
            
            return should_negate, orientation_info
        
        # Calculate weighted decision if we have factors
        if factors:
            # Sum of evidence for each orientation
            inverted_evidence = 0
            original_evidence = 0
            total_weight = 0
            
            for name, factor in factors.items():
                weight = factor["weight"]
                evidence = weight * factor["confidence"]
                total_weight += weight
                
                if factor["favors_inverted"]:
                    inverted_evidence += evidence
                else:
                    original_evidence += evidence
                
                # Add to decision factors
                decision_factors.append(f"{name}:{factor['favors_inverted']}")
            
            # Normalize by total weight
            if total_weight > 0:
                inverted_evidence /= total_weight
                original_evidence /= total_weight
            
            # Determine orientation
            should_negate = inverted_evidence > original_evidence
            
            # Calculate confidence (difference between evidence)
            evidence_diff = abs(inverted_evidence - original_evidence)
            confidence = 0.5 + min(0.45, evidence_diff)  # Max 0.95 confidence
        else:
            # Simple fallback if no factors available
            should_negate = scores2 > scores1
            confidence = 0.5 + 0.4 * (abs(scores2 - scores1) / max(1.0, max(scores1, scores2)))
            decision_factors = ["simple_score_comparison"]
        
        # Create orientation info dictionary with detailed decision information
        orientation_info = {
            "confidence": confidence,
            "decision_factors": decision_factors,
            "original_score": float(scores1),
            "inverted_score": float(scores2),
            "original_phases": phases1,
            "inverted_phases": phases2,
            "factors_considered": list(factors.keys()),
            "orientation1_scores": orientation1_results["biomech_scores"],
            "orientation2_scores": orientation2_results["biomech_scores"]
        }
        
        return should_negate, orientation_info

class AdaptiveSignalProcessor:
    """
    Processes acceleration signals using adaptive filter selection.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.sampling_rate = None
    
    def process_signal(self, data, column, profile, should_negate=False):
        """
        Process acceleration signal using parameters optimized for the dataset.
        
        Args:
            data: DataFrame containing the data
            column: Column name containing acceleration data
            profile: Dataset profile from DatasetProfiler
            should_negate: Whether to negate the acceleration values
            
        Returns:
            DataFrame with processed signal in 'value' and 'smoothed' columns
        """
        working_data = data.copy()
        
        # Add value column with proper orientation
        if should_negate:
            working_data['value'] = -data[column]
        else:
            working_data['value'] = data[column]
        
        # Store original values for reference
        working_data['original_value'] = data[column]
        
        # Get sampling rate from profile
        if 'sampling' in profile and 'sampling_rate' in profile['sampling']:
            self.sampling_rate = profile['sampling']['sampling_rate']
        else:
            # Estimate sampling rate from time column
            time_cols = profile['columns']['time']
            if time_cols and time_cols[0] in data.columns:
                time_col = time_cols[0]
                self.sampling_rate = 1.0 / np.median(np.diff(data[time_col]))
            else:
                # Default sampling rate if we can't determine it
                self.sampling_rate = 100.0
        
        if self.verbose:
            print(f"Signal processing using sampling rate: {self.sampling_rate:.2f} Hz")
        
        # Get optimal parameters from profile
        params = profile['recommended_params']
        
        # Apply preprocessing steps
        working_data = self._preprocess_signal(working_data)
        
        # Apply selected filter based on parameters
        filter_type = params.get('filter_type', 'butterworth')
        
        if filter_type == 'butterworth':
            working_data = self._apply_butterworth_filter(
                working_data, 
                cutoff_freq=params.get('cutoff_freq', 5.0),
                filter_order=params.get('filter_order', 3)
            )
        elif filter_type == 'savgol':
            working_data = self._apply_savgol_filter(
                working_data,
                window_length=params.get('smoothing_window', 7),
                poly_order=params.get('filter_order', 2)
            )
        else:  # Default to moving average
            working_data = self._apply_moving_average(
                working_data,
                window_size=params.get('smoothing_window', 7)
            )
        
        # Calculate derivatives for phase detection
        time_col = self._get_time_column(data, profile)
        
        if time_col and time_col in working_data.columns:
            working_data['derivative'] = np.gradient(working_data['value'], working_data[time_col])
            working_data['smoothed_derivative'] = np.gradient(working_data['smoothed'], working_data[time_col])
        
        return working_data
    
    def _get_time_column(self, data, profile):
        """Get primary time column from data."""
        time_cols = profile['columns']['time']
        
        # Check for known time columns first
        if 'time_s' in data.columns:
            return 'time_s'
        elif time_cols and time_cols[0] in data.columns:
            return time_cols[0]
        
        return None
    
    def _preprocess_signal(self, data):
        """
        Apply preprocessing steps to the signal.
        """
        processed = data.copy()
        
        # Handle missing values
        processed['value'] = processed['value'].interpolate(method='linear')
        
        # Remove outliers using rolling median method
        window = max(5, int(self.sampling_rate / 10))
        median = processed['value'].rolling(window=window, center=True).median()
        std = processed['value'].rolling(window=window, center=True).std()
        
        # Replace extreme outliers with median values
        threshold = 4.0  # 4 standard deviations
        outlier_mask = (processed['value'] - median).abs() > threshold * std
        processed.loc[outlier_mask, 'value'] = median[outlier_mask]
        
        # Fill any NaNs created by the rolling operations
        processed['value'] = processed['value'].interpolate(method='linear')
        processed['value'] = processed['value'].fillna(method='bfill').fillna(method='ffill')
        
        return processed
    
    def _apply_butterworth_filter(self, data, cutoff_freq, filter_order):
        """
        Apply Butterworth filter to the signal.
        """
        # Design Butterworth filter
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff = min(0.95, cutoff_freq / nyquist)  # Ensure it's below Nyquist frequency
        
        if self.verbose:
            print(f"Applying Butterworth filter: order={filter_order}, cutoff={cutoff_freq:.2f} Hz (normalized: {normalized_cutoff:.2f})")
        
        try:
            b, a = signal.butter(filter_order, normalized_cutoff, btype='low')
            
            # Apply zero-phase filtering
            smoothed = signal.filtfilt(b, a, data['value'].values)
            
            # Store result
            result = data.copy()
            result['smoothed'] = smoothed
            
            return result
        except Exception as e:
            # Fall back to moving average if filter design fails
            if self.verbose:
                print(f"Butterworth filter failed: {e}. Falling back to moving average.")
            return self._apply_moving_average(data, window_size=7)
    
    def _apply_savgol_filter(self, data, window_length, poly_order):
        """
        Apply Savitzky-Golay filter to the signal.
        """
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure window length isn't too large for the data
        window_length = min(window_length, len(data) - 1 if len(data) % 2 == 0 else len(data) - 2)
        
        # Ensure window length is at least 3
        window_length = max(3, window_length)
        
        # Ensure polynomial order is less than window length
        poly_order = min(poly_order, window_length - 1)
        
        if self.verbose:
            print(f"Applying Savitzky-Golay filter: window={window_length}, order={poly_order}")
        
        try:
            # Apply filter
            smoothed = signal.savgol_filter(
                data['value'].values, 
                window_length=window_length,
                polyorder=poly_order
            )
            
            # Store result
            result = data.copy()
            result['smoothed'] = smoothed
            
            return result
        except Exception as e:
            # Fall back to moving average if filter fails
            if self.verbose:
                print(f"Savitzky-Golay filter failed: {e}. Falling back to moving average.")
            return self._apply_moving_average(data, window_size=7)
    
    def _apply_moving_average(self, data, window_size):
        """
        Apply moving average filter to the signal.
        """
        if self.verbose:
            print(f"Applying moving average filter: window={window_size}")
        
        # Apply filter
        result = data.copy()
        result['smoothed'] = result['value'].rolling(window=window_size, center=True).mean()
        
        # Handle edge effects
        result['smoothed'] = result['smoothed'].fillna(method='bfill').fillna(method='ffill')
        
        return result

class AdaptiveThresholdManager:
    """
    Determines optimal thresholds for repetition detection based on signal characteristics.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.peak_threshold = 0.3
        self.baseline_threshold = 0.05
    
    def determine_thresholds(self, data, profile):
        """
        Calculate optimal thresholds for the processed signal.
        
        Args:
            data: DataFrame with processed signal
            profile: Dataset profile
            
        Returns:
            Dict containing threshold values
        """
        # Analyze the smoothed signal to determine thresholds
        smoothed_values = data['smoothed'].abs().values
        smoothed_values = smoothed_values[~np.isnan(smoothed_values)]
        
        if len(smoothed_values) == 0:
            # Default thresholds if no valid data
            return {
                'peak_threshold': 0.3,
                'baseline_threshold': 0.05
            }
        
        # Calculate percentiles for threshold determination
        p50 = np.percentile(smoothed_values, 50)  # Median
        p75 = np.percentile(smoothed_values, 75)
        p90 = np.percentile(smoothed_values, 90)
        p95 = np.percentile(smoothed_values, 95)
        
        # Analyze noise level using lower percentiles
        noise_estimate = p50
        
        # Calculate signal-to-noise ratio
        snr = p95 / noise_estimate if noise_estimate > 0 else 10
        
        if self.verbose:
            print(f"Signal percentiles - p50: {p50:.4f}, p75: {p75:.4f}, p90: {p90:.4f}, p95: {p95:.4f}")
            print(f"Estimated SNR: {snr:.2f}")
        
        # Advanced detection of signal characteristics to better determine thresholds
        # Get signal metrics from profile
        signal_metrics = {}
        if "signal" in profile:
            for col, metrics in profile["signal"].items():
                if "skewness" in metrics and "kurtosis" in metrics:
                    signal_metrics = metrics
                    break  # Use the first column with complete metrics
        
        # Adjust threshold determination based on signal distribution
        skewness = signal_metrics.get("skewness", 0)
        kurtosis = signal_metrics.get("kurtosis", 0)
        
        # Positive skewness suggests more small values with occasional large peaks
        # Common in well-defined lift signals - can use lower threshold
        threshold_adjustment = 0
        if skewness > 1.0:
            threshold_adjustment -= 0.1  # Lower threshold for positively skewed data
        elif skewness < -1.0:
            threshold_adjustment += 0.1  # Raise threshold for negatively skewed data
        
        # High kurtosis suggests more outliers/peaks - can use higher threshold
        if kurtosis > 2.0:
            threshold_adjustment += 0.05  # Slightly raise threshold for high-kurtosis data
        
        # Determine peak threshold based on SNR, percentiles, and distribution
        if snr > 10:  # Very clean signal
            self.peak_threshold = max(0.1, p90 * 0.6) + threshold_adjustment
        elif snr > 5:  # Good signal
            self.peak_threshold = max(0.15, p90 * 0.7) + threshold_adjustment
        else:  # Noisy signal
            self.peak_threshold = max(0.2, p95 * 0.7) + threshold_adjustment
        
        # Ensure peak threshold isn't too restrictive
        self.peak_threshold = min(self.peak_threshold, 0.8)
        
        # Set baseline threshold based on noise estimate and distribution
        baseline_adjustment = 0
        if skewness > 1.0:
            # For positively skewed data, use lower baseline threshold
            baseline_adjustment -= 0.01
        
        self.baseline_threshold = max(0.01, noise_estimate * 1.5) + baseline_adjustment
        
        if self.verbose:
            print(f"Determined thresholds - Peak: {self.peak_threshold:.4f}, Baseline: {self.baseline_threshold:.4f}")
        
        # Return thresholds
        return {
            'peak_threshold': self.peak_threshold,
            'baseline_threshold': self.baseline_threshold
        }
    
    def refine_thresholds(self, data, peaks, target_rep_count):
        """
        Refine thresholds based on detected peaks and target repetition count.
        
        Args:
            data: DataFrame with processed signal
            peaks: List of initially detected peaks
            target_rep_count: Target number of repetitions
            
        Returns:
            Dict containing refined threshold values
        """
        # If we don't have enough peaks, try lowering the threshold
        if len(peaks) < target_rep_count:
            if self.verbose:
                print(f"Not enough peaks detected ({len(peaks)} < {target_rep_count}), adjusting thresholds")
            
            # Try to find a threshold that gives the target number of reps
            smoothed_values = data['smoothed'].abs().values
            smoothed_values = smoothed_values[~np.isnan(smoothed_values)]
            
            if len(smoothed_values) == 0:
                return {
                    'peak_threshold': self.peak_threshold,
                    'baseline_threshold': self.baseline_threshold
                }
            
            # Calculate potential new thresholds using lower percentiles to be more generous
            p75 = np.percentile(smoothed_values, 75)
            p70 = np.percentile(smoothed_values, 70)
            p60 = np.percentile(smoothed_values, 60)
            
            # Use signal characteristics to guide threshold adjustment
            signal_dist = {}
            try:
                # Calculate distribution moments to guide adjustment
                mean = np.mean(smoothed_values)
                std = np.std(smoothed_values)
                
                # Calculate skewness
                diff = smoothed_values - mean
                skewness = np.mean(diff**3) / (np.mean(diff**2)**1.5) if np.mean(diff**2) > 0 else 0
                
                # For positively skewed data (typical for deadlifts), we can be more aggressive
                # in lowering thresholds
                adjustment_factor = 0.6
                if skewness > 1.0:
                    adjustment_factor = 0.5  # More aggressive for highly skewed data
                
                signal_dist["skewness"] = skewness
                signal_dist["adjustment_factor"] = adjustment_factor
            except:
                adjustment_factor = 0.6  # Default if calculation fails
            
            # More adaptive threshold lowering based on distribution
            new_threshold = max(0.05, p60 * adjustment_factor)
            
            if new_threshold < self.peak_threshold:
                self.peak_threshold = new_threshold
                # Also reduce the baseline threshold to help identify shorter phases
                self.baseline_threshold = max(0.01, self.baseline_threshold * 0.7)
                if self.verbose:
                    print(f"Lowered peak threshold to {self.peak_threshold:.4f}")
                    print(f"Lowered baseline threshold to {self.baseline_threshold:.4f}")
        
        # If we have too many peaks, try increasing the threshold
        elif len(peaks) > target_rep_count * 2:
            if self.verbose:
                print(f"Too many peaks detected ({len(peaks)} > {target_rep_count*2}), adjusting thresholds")
            
            # Sort peaks by value
            peak_values = sorted([p['value'] for p in peaks], reverse=True)
            
            # Set threshold to just below the Nth highest peak
            if len(peak_values) > target_rep_count:
                # Use target_rep_count + buffer to ensure we have some margin
                index = min(len(peak_values) - 1, int(target_rep_count * 1.2))
                new_threshold = peak_values[index] * 0.95
                
                if new_threshold > self.peak_threshold:
                    self.peak_threshold = new_threshold
                    if self.verbose:
                        print(f"Raised peak threshold to {self.peak_threshold:.4f}")
        
        # Return refined thresholds
        return {
            'peak_threshold': self.peak_threshold,
            'baseline_threshold': self.baseline_threshold
        }

class EnhancedDeadliftDetector:
    """
    Enhanced deadlift concentric phase detector with robust orientation detection
    and improved signal processing.
    """
    def __init__(self, target_rep_count=None, verbose=False, visualize=True, force_orientation=None):
        """
        Initialize the enhanced deadlift detector.
        
        Args:
            target_rep_count: Target number of repetitions to detect (optional)
            verbose: Whether to print detailed information
            visualize: Whether to generate visualizations
            force_orientation: Optional dict to force specific orientation settings
                e.g., {'column': 'vertical_accel', 'negate': True}
        """
        self.target_rep_count = target_rep_count
        self.verbose = verbose
        self.visualize = visualize
        self.force_orientation = force_orientation
        
        # Component initialization
        self.profiler = DatasetProfiler(verbose=verbose)
        self.orientation_detector = TwoPassOrientationDetector(verbose=verbose)
        self.signal_processor = AdaptiveSignalProcessor(verbose=verbose)
        self.threshold_manager = AdaptiveThresholdManager(verbose=verbose)
        
        # Results storage
        self.profile = None
        self.data_processed = None
        self.raw_data = None
        self.peaks = None
        self.phases = None
        self.selected_column = None
        self.should_negate = False
        self.orientation_info = None
        
        # Detection parameters
        self.peak_threshold = None
        self.baseline_threshold = None
        self.min_duration = None
        self.max_duration = None
    
    def detect_phases(self, data):
        """
        Detect deadlift concentric phases with automatic parameter adaptation.
        
        Args:
            data: DataFrame containing IMU data
            
        Returns:
            List of dictionaries with detected phase information
        """
        # Store raw data
        self.raw_data = data.copy()
        
        # Ensure we have a primary time column
        data = self._ensure_time_seconds(data)
        
        if self.verbose:
            print("\n--- Starting automatic deadlift phase detection ---")
        
        # Step 1: Profile the dataset
        self.profile = self.profiler.profile_dataset(data)
        
        # Step 2: Determine optimal orientation
        if self.force_orientation:
            self.selected_column = self.force_orientation.get('column')
            self.should_negate = self.force_orientation.get('negate', False)
            self.orientation_info = {"forced": True}
            
            if self.verbose:
                print(f"Using forced orientation: {self.selected_column}" + 
                      f"{' (negated)' if self.should_negate else ''}")
        else:
            # Use two-pass orientation detection
            self.selected_column, self.should_negate, self.orientation_info = (
                self.orientation_detector.determine_best_orientation(data, self.profile, self))
            
            if self.verbose:
                print(f"Selected data source: {self.selected_column}" + 
                      f"{' (negated)' if self.should_negate else ''}")
        
        # Step 3: Process the signal
        processed_data = self.signal_processor.process_signal(
            data, self.selected_column, self.profile, self.should_negate)
        self.data_processed = processed_data
        
        # Step 4: Determine optimal thresholds
        thresholds = self.threshold_manager.determine_thresholds(
            processed_data, self.profile)
        
        self.peak_threshold = thresholds['peak_threshold']
        self.baseline_threshold = thresholds['baseline_threshold']
        
        # Step 5: Set duration parameters from profile
        self.min_duration = self.profile['recommended_params']['min_duration']
        self.max_duration = self.profile['recommended_params']['max_duration']
        
        if self.verbose:
            print(f"Duration parameters - Min: {self.min_duration:.2f}s, Max: {self.max_duration:.2f}s")
        
        # Step 6: Find peaks
        peaks = self._find_peaks(processed_data)
        self.peaks = peaks
        
        if self.verbose:
            print(f"Found {len(peaks)} acceleration peaks above {self.peak_threshold:.4f}g")
        
        # Step 7: Refine thresholds if needed based on target_rep_count
        if self.target_rep_count is not None and self.target_rep_count > 0:
            refined_thresholds = self.threshold_manager.refine_thresholds(
                processed_data, peaks, self.target_rep_count)
            
            # If thresholds changed, re-detect peaks
            if (refined_thresholds['peak_threshold'] != self.peak_threshold or
                refined_thresholds['baseline_threshold'] != self.baseline_threshold):
                
                self.peak_threshold = refined_thresholds['peak_threshold']
                self.baseline_threshold = refined_thresholds['baseline_threshold']
                
                # Re-detect peaks with new thresholds
                peaks = self._find_peaks(processed_data)
                self.peaks = peaks
                
                if self.verbose:
                    print(f"After threshold refinement: Found {len(peaks)} acceleration peaks")
        
        # Step 8: Select best peaks if too many
        rep_peaks = peaks
        if self.target_rep_count is not None and len(peaks) > self.target_rep_count * 2:
            # Sort peaks by value (highest first)
            sorted_peaks = sorted(peaks, key=lambda x: x["value"], reverse=True)
            # Take the top N peaks with some margin
            rep_peaks = sorted_peaks[:min(len(sorted_peaks), int(self.target_rep_count * 2))]
            # Re-sort by time for sequential processing
            rep_peaks.sort(key=lambda x: x["time"])
            
            if self.verbose:
                print(f"Selected {len(rep_peaks)} strongest peaks for processing")
        
        # Step 9: Find concentric phases for the selected peaks
        concentric_phases = []
        for peak in rep_peaks:
            phase = self._find_concentric_phase(processed_data, peak["index"])
            if phase:
                concentric_phases.append(phase)
        
        # Step 10: Filter out overlapping phases
        concentric_phases = self._filter_overlapping_phases(concentric_phases)
        
        # Multi-pass approach: if we didn't find enough phases, relax constraints and try again
        attempts = 0
        original_min_duration = self.min_duration
        while (self.target_rep_count is not None and 
               len(concentric_phases) < self.target_rep_count and 
               attempts < 3):
            
            attempts += 1
            
            # Progressively relax the minimum duration constraint
            self.min_duration = original_min_duration * (0.7 ** attempts)
            
            if self.verbose:
                print(f"Not enough phases found. Attempt {attempts}: relaxing minimum duration to {self.min_duration:.2f}s")
            
            # Try again with relaxed constraints
            additional_phases = []
            for peak in rep_peaks:
                # Skip peaks that already have a phase
                if any(p["peakIdx"] == peak["index"] for p in concentric_phases):
                    continue
                
                phase = self._find_concentric_phase(processed_data, peak["index"])
                if phase:
                    additional_phases.append(phase)
            
            # Add new phases and filter overlapping ones
            if additional_phases:
                concentric_phases.extend(additional_phases)
                concentric_phases = self._filter_overlapping_phases(concentric_phases)
                
                if self.verbose:
                    print(f"Found {len(additional_phases)} additional phases")
        
        # If still not enough phases, try a more aggressive peak detection approach
        if (self.target_rep_count is not None and 
            len(concentric_phases) < self.target_rep_count):
            
            if self.verbose:
                print(f"Still not enough phases. Using fallback detection method.")
            
            # Lower threshold dramatically for a final attempt
            original_peak_threshold = self.peak_threshold
            self.peak_threshold = self.peak_threshold * 0.5
            
            # Try to find more peaks
            fallback_peaks = self._find_peaks(processed_data)
            new_peaks = [p for p in fallback_peaks if not any(existing["index"] == p["index"] for existing in peaks)]
            
            if new_peaks:
                if self.verbose:
                    print(f"Found {len(new_peaks)} additional peaks with lower threshold")
                
                # Process new peaks
                for peak in new_peaks:
                    # Skip peaks that already have a phase
                    if any(abs(p["peakTime"] - peak["time"]) < 0.5 for p in concentric_phases):
                        continue
                    
                    phase = self._find_concentric_phase(processed_data, peak["index"])
                    if phase:
                        concentric_phases.append(phase)
                
                # Filter overlapping phases again
                concentric_phases = self._filter_overlapping_phases(concentric_phases)
            
            # Restore original threshold
            self.peak_threshold = original_peak_threshold
        
        # Apply temporal consistency check to identify and remove outliers
        if len(concentric_phases) > 2 and self.target_rep_count is not None:
            concentric_phases = self._apply_temporal_consistency(concentric_phases, processed_data)
        
        # Final fallback: If we still don't have enough repetitions and target_rep_count is set,
        # create phases directly from the highest peaks that don't yet have phases
        if (self.target_rep_count is not None and 
            len(concentric_phases) < self.target_rep_count and
            len(peaks) >= self.target_rep_count):
            
            if self.verbose:
                print(f"Using direct peak-to-phase conversion to reach target rep count")
            
            # Sort all peaks by value
            sorted_all_peaks = sorted(peaks, key=lambda x: x["value"], reverse=True)
            
            # For each peak, starting with the highest
            for peak in sorted_all_peaks:
                # Skip if we've reached the target
                if len(concentric_phases) >= self.target_rep_count:
                    break
                
                # Skip peaks that already have a phase
                if any(abs(p["peakTime"] - peak["time"]) < 0.5 for p in concentric_phases):
                    continue
                
                # Skip peaks that are likely impacts
                if self._is_likely_impact(processed_data, peak["index"]):
                    continue
                
                # Create a basic phase directly from the peak
                peak_idx = peak["index"]
                peak_time = peak["time"]
                peak_value = peak["value"]
                
                # Use fixed offsets for start and end
                samples_per_tenth = max(1, int(self.signal_processor.sampling_rate * 0.1))
                start_idx = max(0, peak_idx - samples_per_tenth)
                end_idx = min(len(processed_data) - 1, peak_idx + samples_per_tenth)
                
                start_time = processed_data["time_s"].iloc[start_idx]
                end_time = processed_data["time_s"].iloc[end_idx]
                duration = end_time - start_time
                
                # Create the phase
                phase = {
                    "startTime": start_time,
                    "endTime": end_time,
                    "peakTime": peak_time,
                    "duration": duration,
                    "peakAcceleration": peak_value,
                    "startIdx": start_idx,
                    "endIdx": end_idx,
                    "peakIdx": peak_idx,
                    "isFallback": True  # Mark as fallback phase
                }
                
                concentric_phases.append(phase)
            
            # Sort by time again
            concentric_phases.sort(key=lambda x: x["startTime"])
        
        # Final selection if target_rep_count is specified and we have too many phases
        if self.target_rep_count is not None and len(concentric_phases) > self.target_rep_count:
            # Use a combination of peak height and temporal consistency for selection
            selected_phases = self._select_best_phases(concentric_phases, self.target_rep_count)
            
            if self.verbose:
                print(f"Filtered to {len(selected_phases)} best phases")
            
            concentric_phases = selected_phases
        
        self.phases = concentric_phases
        
        if self.verbose:
            print(f"Found {len(concentric_phases)} concentric phases")
        
        # Step 12: Generate visualizations if enabled
        if self.visualize:
            self._visualize_detection(processed_data, peaks, concentric_phases)
        
        return concentric_phases
    
    def export_data_for_velocity_calculation(self, output_dir=None, base_filename=None):
        """
        Export processed data and phase information for velocity calculation.
        
        Args:
            output_dir: Directory to save output files (default: same as input file)
            base_filename: Base filename for output files (default: derived from input file)
            
        Returns:
            Dictionary with paths to created output files
        """
        if self.data_processed is None or self.phases is None:
            raise ValueError("No detected phases available. Run detect_phases() first.")
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Default base filename if not provided
        if base_filename is None:
            base_filename = "deadlift_data"
        
        # Create output filenames
        csv_filename = os.path.join(output_dir, f"{base_filename}_processed_data.csv")
        json_filename = os.path.join(output_dir, f"{base_filename}_rep_summary.json")
        
        # 1. Export comprehensive CSV with phase markings
        self._export_comprehensive_csv(csv_filename)
        
        # 2. Export concentric phase summary as JSON
        self._export_phase_summary_json(json_filename)
        
        return {
            "csv": csv_filename,
            "json": json_filename
        }
    
    def _export_comprehensive_csv(self, filename):
        """
        Export a comprehensive CSV file with all processed data and phase markings.
        
        Args:
            filename: Path to save the CSV file
        """
        if self.verbose:
            print(f"Exporting comprehensive CSV data to {filename}")
        
        # Create a copy of the processed data
        export_data = self.data_processed.copy()
        
        # Add columns for phase information
        export_data["rep_number"] = 0
        export_data["phase"] = "rest"
        export_data["phase_progress"] = 0.0
        export_data["phase_start_time"] = np.nan
        export_data["phase_end_time"] = np.nan
        export_data["phase_duration"] = np.nan
        export_data["peak_acceleration"] = np.nan
        export_data["peak_time"] = np.nan
        
        # Sort phases by start time
        sorted_phases = sorted(self.phases, key=lambda x: x["startTime"])
        
        # Fill in phase information
        for i, phase in enumerate(sorted_phases):
            rep_number = i + 1
            start_idx = phase["startIdx"]
            end_idx = phase["endIdx"]
            peak_idx = phase["peakIdx"]
            duration = phase["duration"]
            
            # Set rep number for all data points in this phase
            export_data.loc[start_idx:end_idx, "rep_number"] = rep_number
            
            # Mark as concentric phase
            export_data.loc[start_idx:end_idx, "phase"] = "concentric"
            
            # Add phase metadata to all points in the phase
            export_data.loc[start_idx:end_idx, "phase_start_time"] = phase["startTime"]
            export_data.loc[start_idx:end_idx, "phase_end_time"] = phase["endTime"]
            export_data.loc[start_idx:end_idx, "phase_duration"] = duration
            export_data.loc[start_idx:end_idx, "peak_acceleration"] = phase["peakAcceleration"]
            export_data.loc[start_idx:end_idx, "peak_time"] = phase["peakTime"]
            
            # Calculate progress through the phase for each point
            # This is useful for visualizing and analyzing the progress through each rep
            for j in range(start_idx, end_idx + 1):
                if duration > 0:
                    progress = (export_data.loc[j, "time_s"] - phase["startTime"]) / duration
                    export_data.loc[j, "phase_progress"] = max(0.0, min(1.0, progress))
        
        # Add delta_time column for numerical integration
        export_data["delta_time"] = np.nan
        for i in range(1, len(export_data)):
            export_data.loc[i, "delta_time"] = export_data.loc[i, "time_s"] - export_data.loc[i-1, "time_s"]
        
        # Export to CSV
        export_data.to_csv(filename, index=False, float_format="%.6f")
        
        if self.verbose:
            print(f"Exported comprehensive data to {filename} ({len(export_data)} rows)")
    
    def _export_phase_summary_json(self, filename):
        """
        Export a JSON summary of the detected concentric phases.
        
        Args:
            filename: Path to save the JSON file
        """
        if self.verbose:
            print(f"Exporting concentric phase summary to {filename}")
        
        try:
            # Create metadata section - ENSURE ALL VALUES ARE SERIALIZABLE TYPES
            metadata = {
                "source_file": str(getattr(self, "source_filename", "unknown")),
                "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sampling_rate": float(self.signal_processor.sampling_rate),
                "acceleration_column": str(self.selected_column),
                "signal_inverted": bool(self.should_negate),  # Explicitly convert to boolean
                "filter_type": str(self.profile["recommended_params"].get("filter_type", "butterworth")),
                "cutoff_frequency": float(self.profile["recommended_params"].get("cutoff_freq", 5.0)),
                "filter_order": int(self.profile["recommended_params"].get("filter_order", 3)),
                "detection_threshold": float(self.peak_threshold),
                "baseline_threshold": float(self.baseline_threshold),
                "min_duration": float(self.min_duration),
                "max_duration": float(self.max_duration)
            }
            
            # Add orientation confidence if available
            if hasattr(self, 'orientation_info') and self.orientation_info and "confidence" in self.orientation_info:
                metadata["orientation_confidence"] = float(self.orientation_info["confidence"])
            
            # Create repetitions list with explicit type conversions
            repetitions = []
            sorted_phases = sorted(self.phases, key=lambda x: x["startTime"])
            
            for i, phase in enumerate(sorted_phases):
                rep_data = {
                    "rep_number": i + 1,
                    "concentric_phase": {
                        "start_time": float(phase["startTime"]),
                        "end_time": float(phase["endTime"]),
                        "peak_time": float(phase["peakTime"]),
                        "duration": float(phase["duration"]),
                        "peak_acceleration": float(phase["peakAcceleration"]),
                        "start_index": int(phase["startIdx"]),
                        "end_index": int(phase["endIdx"]),
                        "peak_index": int(phase["peakIdx"])
                    },
                    "data_indices": {
                        "start": int(phase["startIdx"]),
                        "end": int(phase["endIdx"])
                    }
                }
                
                # Add a flag for fallback phases
                if "isFallback" in phase and phase["isFallback"]:
                    rep_data["is_fallback_detection"] = bool(phase["isFallback"])
                
                repetitions.append(rep_data)
            
            # Combine metadata and repetitions into final JSON structure
            summary_data = {
                "metadata": metadata,
                "repetitions": repetitions
            }
            
            # Write to JSON file with proper error handling
            with open(filename, 'w') as f:
                json.dump(summary_data, f, indent=2)
                f.flush()  # Ensure data is written to disk
            
            if self.verbose:
                print(f"Exported phase summary to {filename} ({len(repetitions)} repetitions)")
                
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            # Create a minimal valid JSON file as fallback
            try:
                with open(filename, 'w') as f:
                    basic_data = {
                        "metadata": {
                            "error": f"Export error: {str(e)}",
                            "source_file": str(getattr(self, "source_filename", "unknown")),
                            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "acceleration_column": str(self.selected_column) if hasattr(self, "selected_column") else "unknown",
                            "signal_inverted": bool(self.should_negate) if hasattr(self, "should_negate") else False
                        },
                        "repetitions": []
                    }
                    json.dump(basic_data, f, indent=2)
            except Exception as fallback_error:
                print(f"Failed to create fallback JSON file: {fallback_error}")
    
    def _find_peaks_internal(self, data, peak_threshold, baseline_threshold, min_duration, sampling_rate):
        """
        Internal method for finding peaks for orientation detection.
        """
        # Use scipy's find_peaks for robust detection
        try:
            peak_indices, properties = signal.find_peaks(
                data['smoothed'].values,
                height=peak_threshold,
                distance=int(min_duration * sampling_rate * 0.8),
                prominence=peak_threshold * 0.4
            )
            
            peaks = []
            for i, peak_idx in enumerate(peak_indices):
                peaks.append({
                    "index": peak_idx,
                    "time": data["time_s"].iloc[peak_idx],
                    "value": data["smoothed"].iloc[peak_idx],
                    "prominence": properties["prominences"][i]
                })
            
            return peaks
        except Exception as e:
            # Simpler fallback approach
            smoothed = data['smoothed'].values
            peaks = []
            for i in range(2, len(smoothed) - 2):
                if (smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i-2] and
                    smoothed[i] > smoothed[i+1] and smoothed[i] > smoothed[i+2] and
                    smoothed[i] > peak_threshold):
                    peaks.append({
                        "index": i,
                        "time": data["time_s"].iloc[i],
                        "value": smoothed[i],
                        "prominence": smoothed[i] - max(smoothed[i-2], smoothed[i+2])
                    })
            return peaks
    
    def _find_concentric_phase_internal(self, data, peak_index, peak_threshold, baseline_threshold,
                                         min_duration, max_duration, sampling_rate):
        """
        Internal method for finding concentric phases for orientation detection.
        """
        if peak_index >= len(data) or peak_index < 0:
            return None
        
        peak_value = data["smoothed"].iloc[peak_index]
        peak_time = data["time_s"].iloc[peak_index]
        
        # For high sampling rate datasets, use a smaller window
        small_window = int(0.5 * sampling_rate)
        small_start_idx = max(0, peak_index - small_window)
        small_end_idx = min(len(data) - 1, peak_index + small_window)
        
        # Find start of concentric phase (looking backward from peak)
        start_idx = peak_index
        
        for i in range(peak_index, small_start_idx, -1):
            curr_value = data["smoothed"].iloc[i]
            
            # Stop when we find a point that's below baseline
            if curr_value < baseline_threshold:
                start_idx = i + 1  # Use the point just after
                break
        
        # Find end of concentric phase (looking forward from peak)
        end_idx = peak_index
        
        for i in range(peak_index, small_end_idx):
            curr_value = data["smoothed"].iloc[i]
            
            # Stop when we find a point that's below baseline
            if curr_value < baseline_threshold:
                end_idx = i
                break
        
        # Get times for the detected points
        start_time = data["time_s"].iloc[start_idx]
        end_time = data["time_s"].iloc[end_idx]
        duration = end_time - start_time
        
        # Validate duration
        if min_duration <= duration <= max_duration:
            return {
                "startTime": start_time,
                "endTime": end_time,
                "peakTime": peak_time,
                "duration": duration,
                "peakAcceleration": peak_value,
                "startIdx": start_idx,
                "endIdx": end_idx,
                "peakIdx": peak_index
            }
        
        return None
    
    def _ensure_time_seconds(self, data):
        """Ensure we have a time column in seconds."""
        # Check if time_s exists
        if "time_s" in data.columns:
            return data
        
        # If tMillis exists, create time_s
        if "tMillis" in data.columns:
            data_copy = data.copy()
            t_millis = data_copy["tMillis"].values
            t_millis = t_millis - t_millis[0]  # Start from 0
            data_copy["time_s"] = t_millis / 1000.0  # Convert to seconds
            return data_copy
        
        # If no time columns, create artificial one
        time_cols = [col for col in data.columns if "time" in col.lower() or "t_" in col.lower()]
        if time_cols:
            # Use first available time column
            time_col = time_cols[0]
            data_copy = data.copy()
            data_copy["time_s"] = data_copy[time_col].values
            return data_copy
        
        # Create artificial time axis
        data_copy = data.copy()
        data_copy["time_s"] = np.arange(len(data)) / 100.0  # Assume 100 Hz
        if self.verbose:
            print("Warning: No time column found. Created artificial time axis at 100 Hz.")
        return data_copy
    
    def _find_peaks(self, data):
        """Find peaks in the processed signal using adaptive thresholds."""
        return self._find_peaks_internal(
            data, self.peak_threshold, self.baseline_threshold, 
            self.min_duration, self.signal_processor.sampling_rate)
    
    def _find_concentric_phase(self, data, peak_index):
        """Find the start and end of a concentric phase for a given peak."""
        if peak_index >= len(data) or peak_index < 0:
            return None
        
        peak_value = data["smoothed"].iloc[peak_index]
        peak_time = data["time_s"].iloc[peak_index]
        
        # Check for impact pattern - sharp peak with rapid oscillation afterward
        # This is characteristic of barbell drops or impacts rather than lifts
        if self._is_likely_impact(data, peak_index):
            if self.verbose:
                print(f"  Rejected phase at {peak_time:.2f}s - likely an impact/collision rather than a lift")
            return None
        
        # Determine search window
        window_size_samples = int(2.0 * self.signal_processor.sampling_rate)
        start_search_idx = max(0, peak_index - window_size_samples)
        end_search_idx = min(len(data) - 1, peak_index + window_size_samples)
        
        # For high sampling rate datasets, use a smaller initial window 
        # to avoid missing short concentric phases
        if self.signal_processor.sampling_rate > 40:  # High sampling rate dataset
            small_window = int(0.5 * self.signal_processor.sampling_rate)
            small_start_idx = max(0, peak_index - small_window)
            small_end_idx = min(len(data) - 1, peak_index + small_window)
            
            # Try with smaller window first
            phase = self._find_phase_in_window(data, peak_index, peak_time, peak_value, 
                                               small_start_idx, small_end_idx)
            if phase:
                # Verify this isn't an impact pattern
                if self._validate_lift_pattern(data, phase):
                    return phase
                elif self.verbose:
                    print(f"  Rejected phase at {peak_time:.2f}s - failed validation as genuine lift")
        
        # Find start of concentric phase (looking backward from peak)
        start_idx = peak_index
        min_start_idx = max(0, peak_index - int(0.5 * self.signal_processor.sampling_rate))
        
        for i in range(peak_index, start_search_idx, -1):
            curr_value = data["smoothed"].iloc[i]
            prev_value = data["smoothed"].iloc[max(0, i-1)]
            
            # Calculate slope (even if derivative isn't available)
            curr_deriv = 0
            if "smoothed_derivative" in data.columns:
                curr_deriv = data["smoothed_derivative"].iloc[i]
            else:
                # Approximate derivative
                time_diff = data["time_s"].iloc[i] - data["time_s"].iloc[max(0, i-1)]
                if time_diff > 0:
                    curr_deriv = (curr_value - prev_value) / time_diff
            
            # More permissive start detection:
            # Stop when we find a significant valley (local minimum) or the value drops below baseline
            is_valley = (i > 0 and i < len(data)-1 and 
                         curr_value < prev_value and 
                         curr_value < data["smoothed"].iloc[min(len(data)-1, i+1)])
            
            # Stop at a valley, or when value drops below baseline, or slope is strongly negative
            if is_valley or curr_value < self.baseline_threshold or curr_deriv < -2.0:
                start_idx = i  # Include the valley point
                # Don't go too far back - limit to reasonable range
                if start_idx < min_start_idx:
                    start_idx = min_start_idx
                break
        
        # Find end of concentric phase (looking forward from peak)
        end_idx = peak_index
        max_end_idx = min(len(data) - 1, peak_index + int(0.5 * self.signal_processor.sampling_rate))
        
        for i in range(peak_index, end_search_idx):
            curr_value = data["smoothed"].iloc[i]
            
            # More permissive end detection
            # Stop when we reach a value below baseline threshold
            if curr_value < self.baseline_threshold * 0.8:
                end_idx = i
                break
            
            # Don't go too far forward - limit to reasonable range
            if i >= max_end_idx:
                end_idx = max_end_idx
                break
        
        # Get times for the detected points
        start_time = data["time_s"].iloc[start_idx]
        end_time = data["time_s"].iloc[end_idx]
        duration = end_time - start_time
        
        # Handle edge case where start is after end (shouldn't happen but just in case)
        if duration <= 0:
            # Set minimum duration of 0.1s centered on peak
            half_min_dur = 0.05  # half of minimum duration
            start_time = peak_time - half_min_dur
            end_time = peak_time + half_min_dur
            duration = end_time - start_time
            
            # Find the closest indices to these times
            time_diffs_start = abs(data["time_s"] - start_time)
            time_diffs_end = abs(data["time_s"] - end_time)
            start_idx = time_diffs_start.argmin()
            end_idx = time_diffs_end.argmin()
        
        # Create the candidate phase
        phase = {
            "startTime": start_time,
            "endTime": end_time,
            "peakTime": peak_time,
            "duration": duration,
            "peakAcceleration": peak_value,
            "startIdx": start_idx,
            "endIdx": end_idx,
            "peakIdx": peak_index
        }
        
        # Validate that this is a genuine lift pattern and not an impact
        if not self._validate_lift_pattern(data, phase):
            if self.verbose:
                print(f"  Rejected phase at {peak_time:.2f}s - failed validation as genuine lift")
            return None
        
        # Validate duration
        if self.min_duration <= duration <= self.max_duration:
            return phase
        
        # If too short but close to min_duration, extend it slightly
        if duration < self.min_duration and duration > self.min_duration * 0.7:
            extension_needed = self.min_duration - duration
            # Distribute extension on both sides
            half_extension = extension_needed / 2
            
            # Adjust start and end times
            start_time -= half_extension
            end_time += half_extension
            duration = end_time - start_time
            
            # Find the closest indices to these times
            time_diffs_start = abs(data["time_s"] - start_time)
            time_diffs_end = abs(data["time_s"] - end_time)
            start_idx = time_diffs_start.argmin()
            end_idx = time_diffs_end.argmin()
            
            # Create the adjusted phase
            return {
                "startTime": start_time,
                "endTime": end_time,
                "peakTime": peak_time,
                "duration": duration,
                "peakAcceleration": peak_value,
                "startIdx": start_idx,
                "endIdx": end_idx,
                "peakIdx": peak_index
            }
        
        # Rejection reason for debugging
        if self.verbose:
            if duration < self.min_duration:
                print(f"  Rejected phase at {peak_time:.2f}s - too short: {duration:.2f}s < {self.min_duration:.2f}s")
            elif duration > self.max_duration:
                print(f"  Rejected phase at {peak_time:.2f}s - too long: {duration:.2f}s > {self.max_duration:.2f}s")
        
        return None
    
    def _find_phase_in_window(self, data, peak_index, peak_time, peak_value, start_search_idx, end_search_idx):
        """Helper method to find a phase within a specific window."""
        # Find start of concentric phase (looking backward from peak)
        start_idx = peak_index
        for i in range(peak_index, start_search_idx, -1):
            curr_value = data["smoothed"].iloc[i]
            
            # Stop when we find a point that's below baseline
            if curr_value < self.baseline_threshold:
                start_idx = i + 1  # Use the point just after
                break
        
        # Find end of concentric phase (looking forward from peak)
        end_idx = peak_index
        for i in range(peak_index, end_search_idx):
            curr_value = data["smoothed"].iloc[i]
            
            # Stop when we find a point that's below baseline
            if curr_value < self.baseline_threshold:
                end_idx = i
                break
        
        # Get times for the detected points
        start_time = data["time_s"].iloc[start_idx]
        end_time = data["time_s"].iloc[end_idx]
        duration = end_time - start_time
        
        # Validate duration
        if self.min_duration <= duration <= self.max_duration:
            return {
                "startTime": start_time,
                "endTime": end_time,
                "peakTime": peak_time,
                "duration": duration,
                "peakAcceleration": peak_value,
                "startIdx": start_idx,
                "endIdx": end_idx,
                "peakIdx": peak_index
            }
        
        return None
    
    def _is_likely_impact(self, data, peak_index):
        """
        Determine if a peak is likely an impact or collision rather than a lift.
        Impact patterns typically have:
        1. Very sharp peaks with rapid onset
        2. Fast oscillations after the peak
        3. Often a large negative peak immediately following the positive peak
        """
        if peak_index <= 1 or peak_index >= len(data) - 2:
            return False  # Too close to edge of data
        
        # Check for rapid oscillation after peak (characteristic of impacts)
        peak_value = data["smoothed"].iloc[peak_index]
        post_peak_window = min(int(0.2 * self.signal_processor.sampling_rate), 10)
        end_idx = min(len(data) - 1, peak_index + post_peak_window)
        
        # Look for sharp negative values following the peak
        post_peak_values = data["smoothed"].iloc[peak_index:end_idx+1].values
        if len(post_peak_values) > 2:
            # Check for strong negative acceleration immediately following the peak
            # (characteristic of an impact/collision)
            min_post_value = min(post_peak_values)
            
            # Impacts often have a large negative component after the peak
            if min_post_value < -0.3 * peak_value:
                return True
            
            # Check for oscillations (rapid sign changes) after the peak
            sign_changes = 0
            for i in range(1, len(post_peak_values)):
                if (post_peak_values[i] * post_peak_values[i-1]) < 0:
                    sign_changes += 1
            
            # Multiple sign changes in a short window suggest an impact
            if sign_changes >= 2:
                return True
        
        # Check for extremely sharp onset (characteristic of impacts)
        pre_peak_window = min(int(0.1 * self.signal_processor.sampling_rate), 5)
        start_idx = max(0, peak_index - pre_peak_window)
        
        # Calculate the average slope leading to the peak
        if start_idx < peak_index:
            time_delta = data["time_s"].iloc[peak_index] - data["time_s"].iloc[start_idx]
            if time_delta > 0:
                avg_slope = (peak_value - data["smoothed"].iloc[start_idx]) / time_delta
                
                # Very high slopes suggest impact rather than controlled human movement
                if avg_slope > 15:  # Extremely fast rise time
                    return True
        
        # Look for the pattern where there's a sharp rise followed by an immediate sharp fall
        # (classic impact pattern)
        if peak_index > 0 and peak_index < len(data) - 1:
            prev_value = data["smoothed"].iloc[peak_index - 1]
            next_value = data["smoothed"].iloc[peak_index + 1]
            
            # If the peak is surrounded by much lower values, it's a spike (likely impact)
            if (peak_value > 2 * prev_value and peak_value > 2 * next_value):
                # Extremely sharp peak
                return True
        
        return False
    
    def _validate_lift_pattern(self, data, phase, strict=False):
        """
        Validate that a detected phase is a genuine deadlift concentric phase 
        rather than an impact or noise.
        
        Args:
            data: Processed data containing time and acceleration
            phase: Phase to validate
            strict: Whether to apply stricter validation for suspicious phases
            
        Returns:
            Boolean indicating whether the phase is a valid lift
        """
        start_idx = phase["startIdx"]
        peak_idx = phase["peakIdx"]
        end_idx = phase["endIdx"]
        
        # Skip validation for phases that are manually expanded/fallback
        if "isFallback" in phase and phase["isFallback"]:
            return True
        
        # Calculate the ratio of the acceleration rise time to total phase duration
        if peak_idx > start_idx:
            rise_percentage = (peak_idx - start_idx) / (end_idx - start_idx) if end_idx > start_idx else 0
            
            # Genuine lifts usually don't have the peak right at the start or end
            if rise_percentage < 0.1 or rise_percentage > 0.9:
                return False
        
        # Check the signal consistency by looking at the standard deviation of the derivative
        # Impacts typically have much more erratic signals
        if "smoothed_derivative" in data.columns:
            phase_derivatives = data["smoothed_derivative"].iloc[start_idx:end_idx+1]
            if len(phase_derivatives) > 2:
                derivative_std = np.std(phase_derivatives)
                
                # Threshold depends on strictness
                threshold = 10 if strict else 15
                
                # Very high variability in the derivative suggests an impact
                if derivative_std > threshold:
                    return False
        
        # Verify this isn't part of a drop/impact sequence by checking if there's a large
        # negative acceleration before or after this phase
        check_window = int(0.3 * self.signal_processor.sampling_rate)
        
        # Check before the phase
        pre_start_idx = max(0, start_idx - check_window)
        if pre_start_idx < start_idx:
            pre_values = data["smoothed"].iloc[pre_start_idx:start_idx].values
            
            # Threshold depends on strictness
            threshold = -0.3 if strict else -0.5
            
            if len(pre_values) > 0 and min(pre_values) < threshold:
                # Strong negative acceleration just before this phase suggests 
                # this might be a rebound after a drop
                return False
        
        # Check for significant negative component during the phase 
        # Real deadlift concentric phases shouldn't have large negative components
        phase_values = data["smoothed"].iloc[start_idx:end_idx+1].values
        if len(phase_values) > 2:
            min_phase_value = min(phase_values)
            
            # Threshold depends on strictness
            threshold = -0.2 if strict else -0.3
            
            # If we see strong negative acceleration within the phase
            if min_phase_value < threshold:
                return False
            
            # Calculate the ratio of positive to total signal
            positive_samples = sum(1 for v in phase_values if v > 0)
            positive_ratio = positive_samples / len(phase_values)
            
            # Threshold depends on strictness
            threshold = 0.7 if strict else 0.6
            
            # Genuine concentric phases should be predominantly positive
            if positive_ratio < threshold:
                return False
        
        return True
    
    def _filter_overlapping_phases(self, phases):
        """
        Filter out overlapping concentric phases, keeping the one with higher peak.
        
        Args:
            phases: List of detected concentric phases
            
        Returns:
            Filtered list with non-overlapping phases
        """
        if not phases:
            return []
        
        # Sort phases by start time
        sorted_phases = sorted(phases, key=lambda x: x["startTime"])
        
        filtered_phases = []
        for current in sorted_phases:
            overlapping = False
            
            for i, existing in enumerate(filtered_phases):
                # Check for significant overlap
                current_duration = current["duration"]
                existing_duration = existing["duration"]
                
                overlap_start = max(current["startTime"], existing["startTime"])
                overlap_end = min(current["endTime"], existing["endTime"])
                overlap_duration = max(0, overlap_end - overlap_start)
                
                current_overlap_pct = overlap_duration / current_duration if current_duration > 0 else 0
                existing_overlap_pct = overlap_duration / existing_duration if existing_duration > 0 else 0
                
                if max(current_overlap_pct, existing_overlap_pct) > 0.3:
                    overlapping = True
                    
                    # If current phase has higher peak, replace existing
                    if current["peakAcceleration"] > existing["peakAcceleration"]:
                        filtered_phases[i] = current
                    
                    break
            
            if not overlapping:
                filtered_phases.append(current)
        
        return filtered_phases
    
    def _apply_temporal_consistency(self, phases, data):
        """
        Apply temporal consistency checks to identify and remove outliers.
        Real deadlift repetitions often follow a somewhat regular cadence.
        
        Args:
            phases: List of detected phases
            data: Processed data containing time and acceleration
            
        Returns:
            List of phases with outliers removed
        """
        if len(phases) <= 2:
            return phases
        
        # Sort phases by time
        sorted_phases = sorted(phases, key=lambda x: x["startTime"])
        
        # Calculate the intervals between repetitions
        intervals = []
        for i in range(1, len(sorted_phases)):
            interval = sorted_phases[i]["startTime"] - sorted_phases[i-1]["endTime"]
            intervals.append(interval)
        
        # Calculate the mean and standard deviation of intervals
        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Identify outlier intervals (unusually short intervals often indicate false positives)
            suspicious_indices = []
            for i, interval in enumerate(intervals):
                # If an interval is unusually short, mark the phase at the end of that interval
                # as suspicious (likely a false positive)
                if interval < max(1.0, mean_interval - 1.5 * std_interval):
                    suspicious_indices.append(i + 1)  # +1 because intervals[0] is between phases[0] and phases[1]
            
            if suspicious_indices and self.verbose:
                print(f"Identified {len(suspicious_indices)} phases with suspicious timing")
            
            # Check each suspicious phase in more detail
            filtered_phases = []
            for i, phase in enumerate(sorted_phases):
                # If this is a suspicious phase, do additional validation
                if i in suspicious_indices:
                    # Calculate the ratio of this phase's peak to the mean peak of other phases
                    other_peaks = [p["peakAcceleration"] for j, p in enumerate(sorted_phases) if j != i]
                    mean_other_peaks = np.mean(other_peaks) if other_peaks else 0
                    
                    # If this phase has a much lower peak than others, it's likely a false positive
                    if phase["peakAcceleration"] < 0.7 * mean_other_peaks:
                        if self.verbose:
                            print(f"Removing suspicious phase at {phase['startTime']:.2f}s (low peak to others ratio)")
                        continue
                    
                    # If this phase doesn't have clear lift characteristics, filter it out
                    if not self._validate_lift_pattern(data, phase, strict=True):
                        if self.verbose:
                            print(f"Removing suspicious phase at {phase['startTime']:.2f}s (failed strict validation)")
                        continue
                
                # Keep phases that pass all checks
                filtered_phases.append(phase)
            
            return filtered_phases
        
        return sorted_phases
    
    def _select_best_phases(self, phases, target_count):
        """
        Select the best phases based on a combination of peak height and temporal consistency.
        
        Args:
            phases: List of detected phases
            target_count: Target number of phases to select
            
        Returns:
            List of selected phases
        """
        if len(phases) <= target_count:
            return phases
        
        # Sort phases by time
        sorted_phases = sorted(phases, key=lambda x: x["startTime"])
        
        # Calculate a scoring function for each phase that considers:
        # 1. Peak acceleration (higher is better)
        # 2. Regular spacing (phases that fit a regular cadence are better)
        
        # First pass - just use peak acceleration for sorting
        for phase in sorted_phases:
            phase["score"] = phase["peakAcceleration"]
        
        # If we have enough phases, attempt to identify a regular cadence
        if len(sorted_phases) >= 3:
            # Calculate time between starts
            start_times = [phase["startTime"] for phase in sorted_phases]
            intervals = np.diff(start_times)
            
            # If intervals are reasonably consistent, boost score for phases that fit the pattern
            if len(intervals) >= 2:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # Only apply if the standard deviation is less than 50% of the mean
                # (indicates a somewhat regular cadence)
                if std_interval < 0.5 * mean_interval:
                    # Find expected start times based on a regular cadence
                    expected_starts = [start_times[0]]
                    for i in range(1, len(sorted_phases)):
                        expected_starts.append(expected_starts[0] + i * mean_interval)
                    
                    # Calculate deviation from expected start time
                    for i, phase in enumerate(sorted_phases):
                        # Lower deviation from expected start time gives higher score
                        deviation = abs(phase["startTime"] - expected_starts[i])
                        deviation_penalty = min(1.0, deviation / mean_interval)
                        
                        # Adjust score - keep peak acceleration but penalize deviation from pattern
                        phase["score"] = phase["peakAcceleration"] * (1.0 - deviation_penalty * 0.5)
        
        # Sort by score (highest first)
        sorted_phases.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top phases
        selected_phases = sorted_phases[:target_count]
        
        # Re-sort by time
        selected_phases.sort(key=lambda x: x["startTime"])
        
        return selected_phases
        
    def _visualize_detection(self, processed_data, peaks, phases):
        """
        Generate comprehensive visualizations of the detection process with
        improved orientation awareness.
        
        Args:
            processed_data: DataFrame with processed signals
            peaks: List of detected peaks
            phases: List of detected concentric phases
        """
        try:
            plt.figure(figsize=(12, 15))
            
            # Plot 1: Original vs. Oriented Signal (shows effect of orientation correction)
            plt.subplot(4, 1, 1)
            plt.plot(processed_data["time_s"], processed_data["original_value"], 'b-', alpha=0.4, label='Original')
            plt.plot(processed_data["time_s"], processed_data["value"], 'r-', label='Oriented')
            
            # Label the orientation decision
            orientation_desc = f"original orientation" if not self.should_negate else f"inverted"
            confidence = "unknown"
            if hasattr(self, 'orientation_info') and self.orientation_info and "confidence" in self.orientation_info:
                confidence = f"{self.orientation_info['confidence']:.2f}"
                
            plt.title(f"Signal Orientation: {self.selected_column} ({orientation_desc}, confidence: {confidence})")
            plt.ylabel('Acceleration (g)')
            plt.legend(loc='upper right')
            plt.grid(True)
            
            # Plot 2: Processed Signal with Thresholds and Peaks
            plt.subplot(4, 1, 2)
            plt.plot(processed_data["time_s"], processed_data["smoothed"], 'r-', label='Processed')
            
            # Draw thresholds
            plt.axhline(y=self.peak_threshold, color='g', linestyle='--', 
                       label=f'Peak Threshold ({self.peak_threshold:.3f}g)')
            plt.axhline(y=self.baseline_threshold, color='k', linestyle='--', 
                       label=f'Baseline ({self.baseline_threshold:.3f}g)')
            
            # Mark peaks
            peak_times = [p["time"] for p in peaks]
            peak_values = [p["value"] for p in peaks]
            plt.plot(peak_times, peak_values, 'go', label=f'Peaks ({len(peaks)})')
            
            plt.title('Signal Processing and Peak Detection')
            plt.ylabel('Acceleration (g)')
            plt.legend(loc='upper right')
            plt.grid(True)
            
            # Plot 3: Detected Concentric Phases
            plt.subplot(4, 1, 3)
            plt.plot(processed_data["time_s"], processed_data["smoothed"], 'r-', label='Processed Signal')
            
            # Highlight concentric phases
            for i, phase in enumerate(phases):
                start_time = phase["startTime"]
                end_time = phase["endTime"]
                peak_time = phase["peakTime"]
                
                # Highlight the phase region
                plt.axvspan(start_time, end_time, alpha=0.2, color='green')
                
                # Mark start, peak, and end
                plt.plot(start_time, processed_data["smoothed"].iloc[phase["startIdx"]], 'bs', markersize=6)
                plt.plot(peak_time, phase["peakAcceleration"], 'rs', markersize=8)
                plt.plot(end_time, processed_data["smoothed"].iloc[phase["endIdx"]], 'gs', markersize=6)
                
                # Add rep number
                plt.text(start_time, phase["peakAcceleration"], f" Rep {i+1}", 
                         fontsize=9, verticalalignment='bottom')
            
            plt.title(f'Detected Concentric Phases ({len(phases)} repetitions)')
            plt.ylabel('Acceleration (g)')
            plt.grid(True)
            
            # Plot 4: Derivative Analysis
            plt.subplot(4, 1, 4)
            if "smoothed_derivative" in processed_data.columns:
                plt.plot(processed_data["time_s"], processed_data["smoothed_derivative"], 'b-', 
                        label='Acceleration Derivative')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Highlight phases on this plot too
            for phase in phases:
                plt.axvspan(phase["startTime"], phase["endTime"], alpha=0.2, color='green')
            
            plt.title('Rate of Change in Acceleration')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Acceleration Derivative (g/s)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            if self.verbose:
                print(f"Visualization failed: {e}")
                
def main(file_path, target_rep_count=None, visualize=True, verbose=False,
output_dir=None, export_format='none', force_orientation=None):
    """
    Main function to run the enhanced deadlift detector.
    
    Args:
        file_path: Path to the CSV file with IMU data
        target_rep_count: Target number of repetitions to detect (optional)
        visualize: Whether to generate visualizations
        verbose: Whether to print detailed information
        output_dir: Directory to save output files (default: same as input file)
        export_format: Format of the exported data ('csv', 'json', 'both', or 'none')
        force_orientation: Optional dict to force specific orientation settings
        
    Returns:
        List of detected concentric phases
    """
    # Load data
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} data points")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    
    # Create detector
    detector = EnhancedDeadliftDetector(
        target_rep_count=target_rep_count,
        verbose=verbose,
        visualize=visualize,
        force_orientation=force_orientation
    )
    
    # Store source filename for export
    detector.source_filename = os.path.basename(file_path)
    
    # Detect deadlift concentric phases
    print("\nDetecting deadlift concentric phases...")
    concentric_phases = detector.detect_phases(data)
    
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(file_path) or os.getcwd()
    
    # Export data for velocity calculation if requested
    if export_format in ['csv', 'json', 'both']:
        try:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_files = detector.export_data_for_velocity_calculation(
                output_dir=output_dir,
                base_filename=base_filename
            )
            
            print("\nExported data for velocity calculation:")
            for file_type, file_path in output_files.items():
                if (export_format == 'both' or 
                    (export_format == 'csv' and file_type == 'csv') or
                    (export_format == 'json' and file_type == 'json')):
                    print(f"  {file_type.upper()}: {file_path}")
        except Exception as e:
            print(f"\nError exporting data: {e}")
    
    # Display results
    print("\nDetected concentric phases:")
    for i, phase in enumerate(concentric_phases):
        print(f"Rep {i+1}:")
        print(f"  Start time: {phase['startTime']:.2f}s")
        print(f"  End time: {phase['endTime']:.2f}s")
        print(f"  Peak time: {phase['peakTime']:.2f}s")
        print(f"  Duration: {phase['duration']:.2f}s")
        print(f"  Peak acceleration: {phase['peakAcceleration']:.4f}g")
    
    # Calculate aggregate statistics
    if concentric_phases:
        durations = [phase["duration"] for phase in concentric_phases]
        peak_accels = [phase["peakAcceleration"] for phase in concentric_phases]
        
        print("\nAggregate statistics:")
        print(f"  Duration - Mean: {np.mean(durations):.2f}s, " +
              f"Range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"  Peak acceleration - Mean: {np.mean(peak_accels):.4f}g, " +
              f"Range: {min(peak_accels):.4f}g - {max(peak_accels):.4f}g")
        
        # Calculate rest time between reps if more than one rep
        if len(concentric_phases) > 1:
            rest_times = []
            for i in range(1, len(concentric_phases)):
                rest_time = concentric_phases[i]["startTime"] - concentric_phases[i-1]["endTime"]
                rest_times.append(rest_time)
            
            print(f"  Rest time - Mean: {np.mean(rest_times):.2f}s, " +
                  f"Range: {min(rest_times):.2f}s - {max(rest_times):.2f}s")
    
    return concentric_phases

if __name__ == "__main__":
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced deadlift concentric phase detector with robust orientation detection and velocity calculation support'
    )
    
    # Required arguments
    parser.add_argument('file_path', type=str, 
                       help='Path to the CSV file with IMU data')
    
    # Detection options
    parser.add_argument('--target-rep-count', type=int, default=None,
                       help='Target number of repetitions to detect')
    parser.add_argument('--force-accel-column', type=str, default=None,
                       help='Force the use of a specific acceleration column')
    parser.add_argument('--force-invert', action='store_true',
                       help='Force signal inversion')
    
    # Visualization options
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization generation')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files (default: same as input file)')
    parser.add_argument('--export-format', type=str, choices=['csv', 'json', 'both', 'none'], default='none',
                       help='Format of the exported data for velocity calculation')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed debug output')
    
    args = parser.parse_args()
    
    # Set up force_orientation if requested
    force_orientation = None
    if args.force_accel_column:
        force_orientation = {
            'column': args.force_accel_column,
            'negate': args.force_invert
        }

    # Run the main function
    main(
        args.file_path,
        target_rep_count=args.target_rep_count,
        visualize=not args.no_visualize,
        verbose=args.debug,
        output_dir=args.output_dir,
        export_format=args.export_format,
        force_orientation=force_orientation
    )
    
    
    
    
    
    