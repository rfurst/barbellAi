# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:02:50 2025

@author: rkfurst
"""

# src/feature_extraction.py
import numpy as np
import pandas as pd
from scipy import signal, stats, fft
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureExtractor")

class FeatureExtractor:
    def __init__(self):
        """Initialize the FeatureExtractor."""
        pass
    
    def extract_features(self, rep: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a repetition.
        
        Args:
            rep: Dictionary containing repetition data
            
        Returns:
            Dictionary of features
        """
        rep_data = rep['data']
        acceleration = rep_data['smoothed'].values
        time = rep_data['time_s'].values
        
        # Calculate time-domain features
        time_features = self.extract_time_domain_features(acceleration, time)
        
        # Calculate frequency-domain features
        freq_features = self.extract_frequency_domain_features(acceleration, time)
        
        # Calculate shape features
        shape_features = self.extract_shape_features(acceleration, time)
        
        # Calculate context features
        context_features = self.extract_context_features(rep)
        
        # Combine all features
        features = {
            **time_features,
            **freq_features,
            **shape_features,
            **context_features
        }
        
        return features
    
    def extract_time_domain_features(self, acceleration: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from acceleration data.
        
        Args:
            acceleration: Array of acceleration values
            time: Array of time values
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Basic statistical features
        features['acc_mean'] = np.mean(acceleration)
        features['acc_std'] = np.std(acceleration)
        features['acc_min'] = np.min(acceleration)
        features['acc_max'] = np.max(acceleration)
        features['acc_range'] = np.max(acceleration) - np.min(acceleration)
        
        # Higher-order statistics
        features['acc_skewness'] = stats.skew(acceleration)
        features['acc_kurtosis'] = stats.kurtosis(acceleration)
        
        # Peak features
        peak_idx = np.argmax(acceleration)
        features['peak_acc'] = acceleration[peak_idx]
        features['time_to_peak'] = time[peak_idx] - time[0]
        features['peak_time_ratio'] = features['time_to_peak'] / (time[-1] - time[0])
        
        # Zero-crossing features
        zero_crossings = np.where(np.diff(np.signbit(acceleration)))[0]
        features['zero_crossing_count'] = len(zero_crossings)
        if len(zero_crossings) > 0:
            features['first_zero_crossing_time'] = time[zero_crossings[0]] - time[0]
        else:
            features['first_zero_crossing_time'] = -1
        
        # Derivative features
        acceleration_derivative = np.gradient(acceleration, time)
        features['jerk_mean'] = np.mean(acceleration_derivative)
        features['jerk_std'] = np.std(acceleration_derivative)
        features['jerk_max'] = np.max(acceleration_derivative)
        
        # Area under the curve (impulse)
        features['acc_impulse'] = np.trapz(acceleration, time)
        
        # Positive and negative phase info
        pos_acc = acceleration[acceleration > 0]
        neg_acc = acceleration[acceleration < 0]
        
        features['pos_acc_ratio'] = len(pos_acc) / len(acceleration) if len(acceleration) > 0 else 0
        features['pos_acc_mean'] = np.mean(pos_acc) if len(pos_acc) > 0 else 0
        features['neg_acc_mean'] = np.mean(neg_acc) if len(neg_acc) > 0 else 0
        
        return features
    
    def extract_frequency_domain_features(self, acceleration: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features from acceleration data.
        
        Args:
            acceleration: Array of acceleration values
            time: Array of time values
            
        Returns:
            Dictionary of frequency-domain features
        """
        features = {}
        
        # Check if we have enough data points
        if len(acceleration) < 4:
            features['dominant_freq'] = 0
            features['spectral_centroid'] = 0
            features['spectral_bandwidth'] = 0
            features['spectral_roll_off'] = 0
            return features
        
        # Calculate sampling rate
        sampling_rate = 1 / np.mean(np.diff(time))
        
        # Compute FFT
        acceleration_fft = fft.rfft(acceleration)
        fft_magnitude = np.abs(acceleration_fft)
        
        # Get frequency values
        freq = fft.rfftfreq(len(acceleration), 1 / sampling_rate)
        
        # Dominant frequency
        if len(freq) > 0 and np.max(fft_magnitude) > 0:
            dominant_idx = np.argmax(fft_magnitude)
            features['dominant_freq'] = freq[dominant_idx]
        else:
            features['dominant_freq'] = 0
        
        # Spectral centroid (weighted average of frequencies)
        if np.sum(fft_magnitude) > 0 and len(freq) > 0:
            features['spectral_centroid'] = np.sum(freq * fft_magnitude) / np.sum(fft_magnitude)
        else:
            features['spectral_centroid'] = 0
        
        # Spectral bandwidth (weighted standard deviation of frequencies)
        if np.sum(fft_magnitude) > 0 and len(freq) > 0:
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((freq - features['spectral_centroid']) ** 2) * fft_magnitude) / np.sum(fft_magnitude)
            )
        else:
            features['spectral_bandwidth'] = 0
        
        # Spectral roll-off (frequency below which 85% of the magnitude distribution is concentrated)
        if np.sum(fft_magnitude) > 0 and len(freq) > 0:
            cumsum = np.cumsum(fft_magnitude)
            rolloff_point = 0.85 * np.sum(fft_magnitude)
            rolloff_idx = np.argmax(cumsum >= rolloff_point)
            features['spectral_roll_off'] = freq[rolloff_idx] if rolloff_idx < len(freq) else freq[-1]
        else:
            features['spectral_roll_off'] = 0
        
        return features
    
    def extract_shape_features(self, acceleration: np.ndarray, time: np.ndarray) -> Dict[str, float]:
        """
        Extract shape-based features from acceleration data.
        
        Args:
            acceleration: Array of acceleration values
            time: Array of time values
            
        Returns:
            Dictionary of shape features
        """
        features = {}
        
        # Check if we have enough data points
        if len(acceleration) < 4:
            return {
                'curve_symmetry': 0,
                'first_half_energy': 0,
                'second_half_energy': 0,
                'energy_ratio': 0,
                'peak_width_ratio': 0
            }
        
        # Curve symmetry (correlation between first and second half)
        mid_point = len(acceleration) // 2
        first_half = acceleration[:mid_point]
        second_half = acceleration[mid_point:]
        
        # If one half is longer, truncate
        min_length = min(len(first_half), len(second_half))
        if min_length > 0:
            first_half = first_half[-min_length:]
            second_half_rev = second_half[:min_length][::-1]  # Reverse for symmetry comparison
            
            # Calculate correlation
            correlation = np.corrcoef(first_half, second_half_rev)[0, 1] if min_length > 1 else 0
            features['curve_symmetry'] = correlation if not np.isnan(correlation) else 0
        else:
            features['curve_symmetry'] = 0
        
        # Energy in first and second half
        features['first_half_energy'] = np.sum(acceleration[:mid_point]**2)
        features['second_half_energy'] = np.sum(acceleration[mid_point:]**2)
        
        # Ratio of energies
        total_energy = features['first_half_energy'] + features['second_half_energy']
        features['energy_ratio'] = (
            features['first_half_energy'] / total_energy if total_energy > 0 else 0.5
        )
        
        # Peak width ratio (width at half maximum relative to total duration)
        peak_idx = np.argmax(acceleration)
        peak_value = acceleration[peak_idx]
        half_max = peak_value / 2
        
        # Find indices where acceleration crosses half_max
        above_half_max = acceleration >= half_max
        if np.any(above_half_max):
            # Find first and last indices above half max
            indices = np.where(above_half_max)[0]
            first_idx = indices[0]
            last_idx = indices[-1]
            
            peak_width = time[last_idx] - time[first_idx]
            total_duration = time[-1] - time[0]
            
            features['peak_width_ratio'] = peak_width / total_duration if total_duration > 0 else 0
        else:
            features['peak_width_ratio'] = 0
        
        return features
    
    def extract_context_features(self, rep: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract context features from repetition metadata.
        
        Args:
            rep: Dictionary containing repetition information
            
        Returns:
            Dictionary of context features
        """
        features = {}
        
        # Repetition number (normalized)
        features['rep_number_norm'] = rep['rep_number'] / 10  # Assuming < 10 reps per set
        
        # Phase duration
        features['phase_duration'] = rep['concentric_phase']['duration']
        
        # Original peak acceleration from JSON
        features['original_peak_acc'] = rep['concentric_phase']['peak_acceleration']
        
        return features
    
    def process_all_repetitions(self, repetitions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract features from all repetitions and compile into a DataFrame.
        
        Args:
            repetitions: List of repetition dictionaries
            
        Returns:
            DataFrame containing features for all repetitions
        """
        all_features = []
        
        for rep in repetitions:
            try:
                features = self.extract_features(rep)
                
                # Add identifiers
                features['file_name'] = rep['file_name']
                features['rep_number'] = rep['rep_number']
                
                all_features.append(features)
                logger.debug(f"Extracted features for {rep['file_name']} rep {rep['rep_number']}")
            except Exception as e:
                logger.error(f"Error extracting features for {rep['file_name']} rep {rep['rep_number']}: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        logger.info(f"Extracted features for {len(all_features)} repetitions")
        return features_df