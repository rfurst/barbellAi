"""
Velocity calculation with drift correction.
"""
import numpy as np
from scipy.signal import savgol_filter
from ..core.interfaces import ProcessedData, RepData

class TrapzVelocityCalculator:
    def __init__(self, drift_correction: bool = True):
        """Initialize velocity calculator."""
        self.drift_correction = drift_correction
        
    def calculate_velocity(self, processed_data: ProcessedData, rep: RepData) -> float:
        """Calculate peak velocity for a given rep using trapezoidal integration."""
        # Extract vertical acceleration
        vert_acc = processed_data.world_accel[rep.start_idx:rep.end_idx, 2]
        dt = np.diff(processed_data.timestamp[rep.start_idx:rep.end_idx])
        dt = np.append(dt[0], dt)  # Add first dt to match array size
        
        # Apply drift correction if enabled
        if self.drift_correction:
            # Estimate and remove linear drift
            t = processed_data.timestamp[rep.start_idx:rep.end_idx]
            t = t - t[0]  # Make relative to start
            p = np.polyfit(t, vert_acc, 1)
            drift = p[0] * t + p[1]
            vert_acc = vert_acc - drift
        
        # Calculate velocity through integration
        velocity = np.cumsum(vert_acc * dt)
        
        # Smooth velocity
        if len(velocity) > 15:
            velocity = savgol_filter(velocity, 15, 3)
        
        # Return peak velocity
        return np.max(velocity) 