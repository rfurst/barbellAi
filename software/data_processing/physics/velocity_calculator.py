"""
Velocity calculator that integrates acceleration during the concentric phase of each rep.
"""
import numpy as np
from scipy.integrate import cumtrapz
from core.interfaces import VelocityCalculator, ProcessedData, RepData

class TrapzVelocityCalculator(VelocityCalculator):
    def __init__(self, drift_correction=True):
        """
        Initialize the velocity calculator.
        
        Args:
            drift_correction: Whether to apply drift correction
        """
        self.drift_correction = drift_correction
        
    def _find_concentric_phase(self, accel: np.ndarray, start_idx: int, end_idx: int) -> tuple:
        """
        Find the concentric phase of the rep (when the barbell is moving up).
        
        Args:
            accel: Vertical acceleration data
            start_idx: Start index of the rep
            end_idx: End index of the rep
            
        Returns:
            Tuple of (start_idx, end_idx) for the concentric phase
        """
        # The concentric phase is when acceleration is positive
        # We'll look for the main positive acceleration region
        rep_accel = accel[start_idx:end_idx]
        
        # Find where acceleration becomes positive
        pos_regions = rep_accel > 0
        
        if not np.any(pos_regions):
            return start_idx, end_idx
            
        # Find the longest continuous positive region
        from itertools import groupby
        regions = [(k, sum(1 for _ in g)) for k, g in groupby(pos_regions)]
        pos_regions = [(i, n) for i, (k, n) in enumerate(regions) if k]
        
        if not pos_regions:
            return start_idx, end_idx
            
        # Use the longest positive region
        best_region = max(pos_regions, key=lambda x: x[1])
        
        # Convert region indices to absolute indices
        region_start = sum(r[1] for r in regions[:best_region[0]])
        region_end = region_start + best_region[1]
        
        return start_idx + region_start, start_idx + region_end
        
    def calculate_velocity(self, processed_data: ProcessedData, rep: RepData) -> float:
        """
        Calculate peak velocity for a given rep.
        
        Args:
            processed_data: Processed IMU data
            rep: Rep data containing start and end indices
            
        Returns:
            Peak velocity during the concentric phase
        """
        # Extract vertical acceleration
        vert_accel = processed_data.world_accel[:, 2]
        
        # Find concentric phase
        conc_start, conc_end = self._find_concentric_phase(
            vert_accel, rep.start_idx, rep.end_idx
        )
        
        # Get time and acceleration data for the concentric phase
        time = processed_data.timestamp[conc_start:conc_end]
        accel = vert_accel[conc_start:conc_end]
        
        if len(time) < 2:
            return 0.0
            
        # Integrate acceleration to get velocity
        velocity = cumtrapz(accel, time, initial=0)
        
        if self.drift_correction:
            # Apply simple drift correction by ensuring velocity returns to zero
            drift = velocity[-1] / len(velocity)
            correction = np.arange(len(velocity)) * drift
            velocity = velocity - correction
            
        # Return peak velocity
        return np.max(velocity) 