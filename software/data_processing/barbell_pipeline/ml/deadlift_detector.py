"""
Deadlift-specific rep detector using state machine and ML-based validation.
"""
import numpy as np
from collections import deque
from scipy.signal import savgol_filter
from ..core.interfaces import RepDetector, ProcessedData, RepData

# Deadlift phase states
DL_STATE_IDLE = 0       # Between reps, bar relatively still
DL_STATE_SETUP = 1      # Initial slight movements before pull
DL_STATE_PULL = 2       # Initial acceleration (breaking the floor)
DL_STATE_CONCENTRIC = 3 # Main lifting phase (upward movement)
DL_STATE_LOCKOUT = 4    # Brief transition at top of movement
DL_STATE_ECCENTRIC = 5  # Lowering phase
DL_STATE_FINISH = 6     # Rep finishing, returning to baseline

class DeadliftRepDetector(RepDetector):
    def __init__(
        self,
        # Rep detection parameters
        start_acc_threshold: float = 1.3,
        concentric_min_vel: float = 0.15,
        peak_vel_threshold: float = 0.05,
        min_rep_duration: int = 15,
        max_rep_duration: int = 250,
        
        # Signal processing
        smooth_window: int = 21,
        smooth_poly: int = 3,
        sampling_freq: float = 104
    ):
        """Initialize the deadlift rep detector."""
        self.start_acc_threshold = start_acc_threshold
        self.concentric_min_vel = concentric_min_vel
        self.peak_vel_threshold = peak_vel_threshold
        self.min_rep_duration = min_rep_duration
        self.max_rep_duration = max_rep_duration
        
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.sampling_freq = sampling_freq
        
        # State tracking
        self.state = DL_STATE_IDLE
        self.rep_start_idx = 0
        self.pull_start_idx = 0
        self.concentric_start_idx = 0
        self.lockout_start_idx = 0
        self.eccentric_start_idx = 0
        self.rep_end_idx = 0
        
        # Rep metrics
        self.current_rep_duration = 0
        self.peak_velocity = 0.0
        self.concentric_duration = 0
        self.eccentric_duration = 0
        self.rep_rom = 0.0  # Range of motion
        
        # Buffers
        self.vel_buffer = deque(maxlen=5)
        self.concentric_vel_samples = []
        self.eccentric_vel_samples = []
        
    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to signal."""
        if len(signal) > self.smooth_window:
            return savgol_filter(signal, self.smooth_window, self.smooth_poly)
        return signal
        
    def detect_reps(self, processed_data: ProcessedData) -> list[RepData]:
        """Detect deadlift reps in the processed data."""
        # Extract vertical components
        vert_acc = processed_data.world_accel[:, 2]
        
        # Calculate velocity through integration
        vert_vel = np.cumsum(vert_acc) / self.sampling_freq
        
        # Smooth signals
        vert_acc = self.smooth_signal(vert_acc)
        vert_vel = self.smooth_signal(vert_vel)
        
        detected_reps = []
        n_samples = len(vert_acc)
        
        for i in range(n_samples):
            # Buffer velocity for peak detection
            self.vel_buffer.append(vert_vel[i])
            
            # State machine transitions
            prev_state = self.state
            
            if self.state == DL_STATE_IDLE:
                # Reset metrics
                self.peak_velocity = 0.0
                self.rep_rom = 0.0
                
                # Transition to setup if we detect movement
                if abs(vert_acc[i]) > self.start_acc_threshold/2:
                    self.state = DL_STATE_SETUP
                    self.rep_start_idx = i
                    self.current_rep_duration = 0
                    self.concentric_vel_samples = []
                    self.eccentric_vel_samples = []
                    
            elif self.state == DL_STATE_SETUP:
                # Wait for initial pull (significant upward acceleration)
                if vert_acc[i] > self.start_acc_threshold:
                    self.state = DL_STATE_PULL
                    self.pull_start_idx = i
                # Timeout if setup takes too long
                elif i - self.rep_start_idx > 60:
                    self.state = DL_STATE_IDLE
                    
            elif self.state == DL_STATE_PULL:
                # Transition to concentric phase when velocity becomes significant
                if vert_vel[i] > self.concentric_min_vel:
                    self.state = DL_STATE_CONCENTRIC
                    self.concentric_start_idx = i
                # Reset if pull doesn't lead to movement
                elif i - self.pull_start_idx > 30:
                    self.state = DL_STATE_IDLE
                    
            elif self.state == DL_STATE_CONCENTRIC:
                # Track velocity during concentric phase
                self.concentric_vel_samples.append(vert_vel[i])
                
                # Update peak velocity
                if vert_vel[i] > self.peak_velocity:
                    self.peak_velocity = vert_vel[i]
                    
                # Range of motion estimation
                self.rep_rom += vert_vel[i] / self.sampling_freq
                
                # Detect lockout (velocity approaches zero after peak)
                if (len(self.vel_buffer) == self.vel_buffer.maxlen and
                    max(self.vel_buffer) < self.peak_vel_threshold and
                    self.peak_velocity > self.concentric_min_vel * 2):
                    
                    self.state = DL_STATE_LOCKOUT
                    self.lockout_start_idx = i
                    self.concentric_duration = i - self.concentric_start_idx
                    
            elif self.state == DL_STATE_LOCKOUT:
                # Check for beginning of descent
                if vert_vel[i] < -self.peak_vel_threshold:
                    self.state = DL_STATE_ECCENTRIC
                    self.eccentric_start_idx = i
                # Timeout if lockout lasts too long
                elif i - self.lockout_start_idx > 50:
                    self.state = DL_STATE_FINISH
                    self.rep_end_idx = i
                    
            elif self.state == DL_STATE_ECCENTRIC:
                # Track velocity during eccentric phase
                self.eccentric_vel_samples.append(vert_vel[i])
                
                # End rep when velocity approaches zero
                if (abs(vert_vel[i]) < self.peak_vel_threshold and 
                    abs(vert_acc[i]) < self.start_acc_threshold/2):
                    self.state = DL_STATE_FINISH
                    self.rep_end_idx = i
                    self.eccentric_duration = i - self.eccentric_start_idx
                    
            elif self.state == DL_STATE_FINISH:
                # Validate the rep
                self.current_rep_duration = i - self.rep_start_idx
                
                if (self.current_rep_duration >= self.min_rep_duration and
                    self.current_rep_duration <= self.max_rep_duration and
                    len(self.concentric_vel_samples) > 5 and
                    len(self.eccentric_vel_samples) > 5 and
                    self.peak_velocity > self.concentric_min_vel):
                    
                    # Calculate rep quality metrics
                    ideal_ratio = 0.6  # Ideal concentric:eccentric time ratio
                    actual_ratio = self.concentric_duration / max(1, self.eccentric_duration)
                    time_score = 1 - min(1, abs(actual_ratio - ideal_ratio) / ideal_ratio)
                    
                    vel_consistency = 1 - np.std(self.concentric_vel_samples) / max(0.01, self.peak_velocity)
                    
                    quality_score = 0.7 * time_score + 0.3 * vel_consistency
                    
                    # Create rep data
                    rep = RepData(
                        start_idx=self.rep_start_idx,
                        end_idx=self.rep_end_idx,
                        peak_velocity=self.peak_velocity
                    )
                    detected_reps.append(rep)
                
                # Return to idle state
                self.state = DL_STATE_IDLE
                
            # Update duration counter
            if self.state != DL_STATE_IDLE:
                self.current_rep_duration = i - self.rep_start_idx
                
            # Timeout safety
            if (self.state != DL_STATE_IDLE and 
                self.current_rep_duration > self.max_rep_duration):
                self.state = DL_STATE_IDLE
                
        return detected_reps 