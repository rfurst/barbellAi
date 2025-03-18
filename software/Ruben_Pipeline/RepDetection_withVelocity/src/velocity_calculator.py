# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:19:39 2025

@author: rkfurst
"""

# src/velocity_calculator.py
import os
import numpy as np
import pandas as pd
import joblib
from scipy import signal
from typing import Dict, List, Tuple, Any
import logging
from src.feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedVelocityCalculator")

class EnhancedVelocityCalculator:
    def __init__(self, model_path: str = "data/models/velocity_model.pkl", 
                 adjustment_factor: float = 0.85, 
                 use_ml_model: bool = True):
        """
        Initialize the EnhancedVelocityCalculator.
        
        Args:
            model_path: Path to the trained ML model
            adjustment_factor: Default biomechanical adjustment factor (used as fallback)
            use_ml_model: Whether to use the ML model (if False, uses traditional calculation only)
        """
        self.adjustment_factor = adjustment_factor
        self.use_ml_model = use_ml_model
        self.feature_extractor = FeatureExtractor()
        
        # Load the ML model if available and requested
        self.model = None
        if use_ml_model and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None
        elif use_ml_model:
            logger.warning(f"ML model not found at {model_path}, falling back to traditional calculation")
    
    def calculate_velocity_traditional(self, rep_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate velocity using traditional integration method.
        
        Args:
            rep_data: DataFrame containing repetition data
            
        Returns:
            Tuple of (velocity array, time array, acceleration array)
        """
        # Extract data
        acceleration = rep_data['smoothed'].values
        time = rep_data['time_s'].values
        
        if 'delta_time' in rep_data.columns:
            delta_time = rep_data['delta_time'].values
        else:
            # Calculate delta time if not available
            delta_time = np.diff(time, prepend=time[0])
        
        # Apply signal processing
        # 1. Remove baseline drift
        acceleration_clean = acceleration - np.mean(acceleration[:3])
        
        # 2. Denoise the acceleration signal
        window_length = min(9, len(acceleration_clean) - 2)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            acceleration_denoised = signal.savgol_filter(acceleration_clean, window_length, 3)
        else:
            acceleration_denoised = acceleration_clean
        
        # Convert acceleration from g-units to m/s^2
        GRAVITY = 9.81
        acceleration_mps2 = acceleration_denoised * GRAVITY
        
        # Calculate velocity using trapezoidal integration
        velocity = np.zeros(len(acceleration_mps2))
        
        # First point has zero velocity (start of concentric phase)
        velocity[0] = 0.0
        
        # Integrate acceleration to get velocity
        for i in range(1, len(acceleration_mps2)):
            # Use actual delta_time for accurate integration
            delta_t = delta_time[i-1] if i-1 < len(delta_time) else time[i] - time[i-1]
            # Trapezoidal rule: v = v₀ + (a₁ + a₀)/2 * Δt
            velocity[i] = velocity[i-1] + (acceleration_mps2[i] + acceleration_mps2[i-1]) / 2 * delta_t
        
        # Apply enhanced physics-based drift correction
        corrected_velocity = self._apply_physics_correction(velocity, time, acceleration_mps2)
        
        # Apply the default adjustment factor
        adjusted_velocity = corrected_velocity * self.adjustment_factor
        
        return adjusted_velocity, time, acceleration_mps2
    
    def _apply_physics_correction(self, velocity: np.ndarray, time: np.ndarray, 
                                 acceleration: np.ndarray) -> np.ndarray:
        """
        Apply physics-based drift correction.
        
        Args:
            velocity: Array of velocity values
            time: Array of time values
            acceleration: Array of acceleration values
            
        Returns:
            Corrected velocity array
        """
        corrected_velocity = velocity.copy()
        
        # Step 1: Apply initial zero velocity constraint
        corrected_velocity[0] = 0.0
        
        # Step 2: Detect movement phases through acceleration profile analysis
        # Find peak acceleration
        peak_acc_idx = np.argmax(acceleration)
        
        # Find where acceleration crosses zero from positive to negative (deceleration starts)
        decel_start_idx = None
        for i in range(peak_acc_idx + 1, len(acceleration) - 1):
            if acceleration[i] < 0 and acceleration[i-1] >= 0:
                decel_start_idx = i
                break
        
        if decel_start_idx is None or decel_start_idx >= len(acceleration) - 2:
            decel_start_idx = len(acceleration) * 2 // 3  # Default to 2/3 through the movement
        
        # Step 3: Identify terminal phase of movement
        # Deadlifts typically end with deceleration to a stop at lockout
        terminal_phase_start = int(0.8 * len(acceleration))  # Last 20% of the movement
        
        # Step 4: Apply phase-specific corrections
        
        # Initial phase: Keep as calculated but ensure starting at zero
        initial_correction = velocity[0]
        for i in range(0, decel_start_idx):
            # Gradually reduce the initial correction effect
            phase_factor = (decel_start_idx - i) / decel_start_idx
            corrected_velocity[i] = velocity[i] - (initial_correction * phase_factor)
        
        # Middle phase: Apply gradual correction based on expected deceleration
        # Identify expected peak velocity - should occur around deceleration start
        peak_vel_idx = np.argmax(velocity[:decel_start_idx + 5])
        peak_vel = velocity[peak_vel_idx]
        
        # Terminal phase: Ensure velocity approaches a small positive value at the end
        end_target = 0.02 * peak_vel  # Target a very small fraction of peak velocity at end
        
        # Calculate correction factors for terminal phase
        for i in range(terminal_phase_start, len(velocity)):
            # How far are we through the terminal phase (0-1)
            phase_progress = (i - terminal_phase_start) / (len(velocity) - terminal_phase_start)
            
            # Current velocity should approach the end target
            target_velocity = peak_vel * (1 - phase_progress) + end_target * phase_progress
            
            # Apply smooth correction to approach target
            corrected_velocity[i] = (
                velocity[i] * (1 - phase_progress) + target_velocity * phase_progress
            )
        
        # Step 5: Ensure overall smoothness of the velocity curve
        # Apply a light smoothing to the corrected velocity curve
        window_length = min(5, len(corrected_velocity) - 2)
        if window_length >= 3 and window_length % 2 == 1:
            corrected_velocity = signal.savgol_filter(corrected_velocity, window_length, 2)
        
        return corrected_velocity
    
    def predict_velocity_ml(self, rep: Dict[str, Any]) -> float:
        """
        Predict peak velocity using the ML model.
        
        Args:
            rep: Dictionary containing repetition information
            
        Returns:
            Predicted peak velocity
        """
        if self.model is None:
            logger.warning("ML model not available, cannot predict velocity")
            return None
        
        # Extract features
        features = self.feature_extractor.extract_features(rep)
        
        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])
        
        # Remove identifier columns if present
        for col in ['file_name', 'rep_number']:
            if col in features_df.columns:
                features_df = features_df.drop(columns=[col])
        
        # Predict velocity
        try:
            predicted_velocity = self.model.predict(features_df)[0]
            return predicted_velocity
        except Exception as e:
            logger.error(f"Error predicting velocity: {e}")
            return None
    
    def calculate_velocity(self, rep: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate velocity using both traditional and ML methods, then combine results.
        
        Args:
            rep: Dictionary containing repetition information
            
        Returns:
            Dictionary with velocity calculation results
        """
        # Calculate traditional velocity
        velocity, time, acceleration = self.calculate_velocity_traditional(rep['data'])
        
        # Find peak velocity
        peak_idx = np.argmax(velocity)
        traditional_peak_velocity = velocity[peak_idx]
        
        # Calculate displacement
        displacement = np.zeros(len(velocity))
        for i in range(1, len(velocity)):
            delta_t = time[i] - time[i-1]
            displacement[i] = displacement[i-1] + (velocity[i] + velocity[i-1]) / 2 * delta_t
        
        # Use ML model to predict peak velocity if available
        ml_peak_velocity = None
        if self.use_ml_model and self.model is not None:
            ml_peak_velocity = self.predict_velocity_ml(rep)
        
        # Combine results (give preference to ML if available)
        final_peak_velocity = ml_peak_velocity if ml_peak_velocity is not None else traditional_peak_velocity
        
        # If ML model predicts velocity, adjust the velocity curve to match
        if ml_peak_velocity is not None:
            # Scale the velocity curve to match the ML prediction
            scaling_factor = ml_peak_velocity / traditional_peak_velocity if traditional_peak_velocity > 0 else 1.0
            velocity = velocity * scaling_factor
        
        # Prepare metrics
        peak_velocity_time = time[peak_idx]
        time_to_peak = peak_velocity_time - time[0]
        phase_duration = time[-1] - time[0]
        percent_to_peak = (time_to_peak / phase_duration * 100) if phase_duration > 0 else 0
        
        # Calculate mean velocity
        mean_velocity = np.mean(velocity)
        
        # Velocity at different percentages of the phase
        percentages = [25, 50, 75]
        velocity_at_percentages = {}
        
        for percent in percentages:
            # Find the time point at the specified percentage of the phase
            target_time = time[0] + phase_duration * percent / 100
            
            # Find the closest time point
            closest_idx = np.argmin(np.abs(time - target_time))
            
            # Store the velocity at this point
            velocity_at_percentages[f"velocity_at_{percent}_percent"] = float(velocity[closest_idx])
        
        # Calculate additional metrics
        if peak_idx > 0 and time_to_peak > 0:
            rvd = final_peak_velocity / time_to_peak  # Rate of velocity development
        else:
            rvd = 0.0
        
        # Velocity consistency (variation of velocity during the middle part of the lift)
        middle_start_idx = len(velocity) // 4
        middle_end_idx = 3 * len(velocity) // 4
        
        if middle_end_idx > middle_start_idx:
            middle_velocity = velocity[middle_start_idx:middle_end_idx]
            velocity_consistency = np.std(middle_velocity) / np.mean(middle_velocity) if np.mean(middle_velocity) > 0 else 0.0
        else:
            velocity_consistency = 0.0
        
        # Create result dictionary
        result = {
            'rep_number': rep['rep_number'],
            'velocity': velocity,
            'time': time,
            'acceleration': acceleration,
            'displacement': displacement,
            'metrics': {
                'peak_velocity': float(final_peak_velocity),
                'peak_velocity_time': float(peak_velocity_time),
                'peak_velocity_index': int(peak_idx),
                'mean_velocity': float(mean_velocity),
                'time_to_peak_velocity': float(time_to_peak),
                'percent_of_phase_to_peak': float(percent_to_peak),
                'traditional_peak_velocity': float(traditional_peak_velocity),
                'ml_peak_velocity': float(ml_peak_velocity) if ml_peak_velocity is not None else None,
                'rate_of_velocity_development': float(rvd),
                'velocity_consistency': float(velocity_consistency),
                **velocity_at_percentages
            },
            'concentric_phase': rep['concentric_phase'],
            'start_idx': rep['concentric_phase']['start_index'],
            'end_idx': rep['concentric_phase']['end_index']
        }
        
        return result