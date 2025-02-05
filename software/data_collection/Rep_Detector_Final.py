#!/usr/bin/env python3
"""
Robust End-to-End Deadlift Repetition Segmentation with Zero-Crossing Adjusted Boundaries

This script:
  1. Loads a CSV file (tab-separated) with columns:
         time_s          : time stamps (in seconds)
         vertical_accel_g: vertical acceleration (in g) from your Arduino.
  2. Estimates the sampling frequency.
  3. Applies a lowpass Butterworth filter to smooth the signal.
  4. Detects major rep peaks (assumed to be the concentric peaks) using preset criteria.
  5. Computes filtered turning points (local extrema) and applies a minimum time gap filter.
  6. For each detected major peak, defines preliminary rep boundaries by computing midpoints:
         • rep_start = midpoint between the turning point immediately before the peak and the next turning point.
         • rep_end   = midpoint between the turning point immediately after the peak and its following turning point.
  7. Then, for each preliminary boundary, the script searches within a small window for a zero crossing
     (where the acceleration changes sign) and uses linear interpolation to adjust the boundary to a time
     when acceleration is effectively 0.
  8. Finally, the script outputs the adjusted rep boundaries and saves the full rep’s time and filtered acceleration
     data to separate CSV files.
     
Adjustable parameters (e.g. cutoff frequency, min_gap_sec for peaks, peak height/prominence, min_turn_gap_sec,
and the zero-crossing search window) can be tuned as needed.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, argrelextrema
import matplotlib.pyplot as plt
import os

# -------------------------------
# Lowpass Filter Function
# -------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# -------------------------------
# Peak Detection Function
# -------------------------------
def detect_rep_peaks(time, accel_filt, fs, min_gap_sec=3.0, height=0.3, prominence=0.1):
    """
    Detects major rep peaks (concentric phase peaks) in the filtered acceleration.
    """
    min_distance = int(min_gap_sec * fs)
    peak_indices, properties = find_peaks(accel_filt, distance=min_distance,
                                            height=height, prominence=prominence)
    return peak_indices, properties

# -------------------------------
# Turning Point Extraction with Time Filter
# -------------------------------
def get_filtered_turning_points(time, accel_filt, min_turn_gap_sec=0.5):
    """
    Computes local extrema (both minima and maxima) and then filters them so that consecutive
    turning points are separated by at least min_turn_gap_sec.
    """
    local_max = argrelextrema(accel_filt, np.greater)[0]
    local_min = argrelextrema(accel_filt, np.less)[0]
    all_tps = np.sort(np.concatenate((local_max, local_min)))
    
    filtered_tps = []
    last_tp = None
    for tp in all_tps:
        if last_tp is None:
            filtered_tps.append(tp)
            last_tp = tp
        else:
            if time[tp] - time[last_tp] >= min_turn_gap_sec:
                filtered_tps.append(tp)
                last_tp = tp
    return np.array(filtered_tps)

# -------------------------------
# Adjust Boundary to Zero Crossing
# -------------------------------
def adjust_boundary_to_zero(time, accel, idx, window_size=10):
    """
    Searches within a window around the given index for a zero crossing.
    If a sign change is found between two consecutive samples, the function
    computes an interpolated zero-crossing time.
    
    Parameters:
      time        : full time array (numpy array)
      accel       : full filtered acceleration array (numpy array)
      idx         : index around which to search
      window_size : number of samples to check before and after idx
      
    Returns:
      (new_idx, t_zero): new index (for reference) and interpolated zero-crossing time.
      If no zero crossing is found, returns the original index and time.
    """
    start = max(0, idx - window_size)
    end = min(len(accel) - 2, idx + window_size)  # -2 to allow i+1 access
    for i in range(start, end):
        # Check for sign change between sample i and i+1
        if accel[i] * accel[i+1] < 0:
            # Linear interpolation:
            t0 = time[i]
            t1 = time[i+1]
            a0 = accel[i]
            a1 = accel[i+1]
            # Calculate zero-crossing time: t_zero = t0 - a0*(t1-t0)/(a1-a0)
            t_zero = t0 - a0 * (t1 - t0) / (a1 - a0)
            return i, t_zero
    # If no crossing found, return original
    return idx, time[idx]

# -------------------------------
# Adjusted Full Rep Extraction Function
# -------------------------------
def extract_full_rep_from_peak_adjusted(time, accel_filt, peak_idx, min_turn_gap_sec=0.5):
    """
    For a detected major rep peak, first compute filtered turning points.
    Then, define preliminary boundaries as midpoints:
      - rep_start_time = midpoint between the turning point immediately before the peak (T1)
                         and the next turning point (T2).
      - rep_end_time   = midpoint between the turning point immediately after the peak (T3)
                         and its following turning point (T4).
    After that, each boundary is adjusted to the nearest zero crossing in the filtered signal.
    
    Parameters:
      time             : full time array (numpy array)
      accel_filt       : full filtered acceleration array (numpy array)
      peak_idx         : index of the detected major rep peak
      min_turn_gap_sec : minimum time gap between turning points
      
    Returns:
      rep_start_time_adj : adjusted rep start time (when acceleration is ~0)
      rep_end_time_adj   : adjusted rep end time (when acceleration is ~0)
      rep_tps            : the raw filtered turning point indices used for the rep.
      
    Returns (None, None, None) if insufficient turning points are available.
    """
    tps = get_filtered_turning_points(time, accel_filt, min_turn_gap_sec)
    if len(tps) < 4:
        return None, None, None
    diffs = np.abs(tps - peak_idx)
    j = np.argmin(diffs)
    if j - 1 < 0 or j + 2 >= len(tps):
        return None, None, None
    # Preliminary boundaries as midpoints:
    rep_start_time = (time[tps[j-1]] + time[tps[j]]) / 2.0
    rep_end_time   = (time[tps[j+1]] + time[tps[j+2]]) / 2.0
    # Find indices closest to these times:
    rep_start_idx = np.argmin(np.abs(time - rep_start_time))
    rep_end_idx   = np.argmin(np.abs(time - rep_end_time))
    
    # Adjust each boundary to the nearest zero crossing:
    _, rep_start_time_adj = adjust_boundary_to_zero(time, accel_filt, rep_start_idx, window_size=10)
    _, rep_end_time_adj = adjust_boundary_to_zero(time, accel_filt, rep_end_idx, window_size=10)
    
    rep_tps = tps[j-1 : j+3]
    return rep_start_time_adj, rep_end_time_adj, rep_tps

# -------------------------------
# Main Analysis Routine
# -------------------------------
def main():
    # Step 1: Load CSV data.
    filename = 'rep_detector_test_v1.csv'
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return
    df = pd.read_csv(filename, sep=',')
    if 'time_s' not in df.columns or 'vertical_accel_g' not in df.columns:
        print("CSV file must contain columns 'time_s' and 'vertical_accel_g'.")
        return
    time = df['time_s'].values
    accel = df['vertical_accel_g'].values

    # Step 2: Estimate sampling frequency.
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    print(f"Estimated sampling frequency: {fs:.2f} Hz")

    # Step 3: Filter the acceleration signal.
    cutoff = 5.0  # Hz; adjust if necessary.
    accel_filt = butter_lowpass_filter(accel, cutoff, fs, order=2)

    # Step 4: Detect major rep peaks.
    min_gap_sec = 3.0
    peak_height = 0.3
    peak_prominence = 0.1
    peak_indices, properties = detect_rep_peaks(time, accel_filt, fs,
                                                 min_gap_sec=min_gap_sec,
                                                 height=peak_height,
                                                 prominence=peak_prominence)
    if len(peak_indices) == 0:
        print("No rep peaks detected. Adjust detection parameters.")
        return
    print(f"Detected {len(peak_indices)} rep peaks.")

    # Step 5: For each detected peak, extract the full rep with boundaries adjusted to zero crossings.
    min_turn_gap_sec = 0.5  # Minimum time between turning points.
    rep_results = []
    for rep_num, peak_idx in enumerate(peak_indices, start=1):
        result = extract_full_rep_from_peak_adjusted(time, accel_filt, peak_idx, min_turn_gap_sec)
        if result[0] is None:
            print(f"Rep {rep_num}: Not enough filtered turning points around peak at index {peak_idx}; skipping rep.")
            continue
        rep_start_time_adj, rep_end_time_adj, rep_tps = result
        
        # Find indices corresponding to adjusted times (for slicing the original time array)
        rep_start_idx = np.argmin(np.abs(time - rep_start_time_adj))
        rep_end_idx   = np.argmin(np.abs(time - rep_end_time_adj))
        
        rep_info = {
            'rep_number': rep_num,
            'rep_start_time': rep_start_time_adj,
            'rep_end_time': rep_end_time_adj,
            'turning_points': rep_tps,
            'rep_time': time[rep_start_idx: rep_end_idx+1],
            'rep_accel': accel_filt[rep_start_idx: rep_end_idx+1]
        }
        rep_results.append(rep_info)

    # Step 6: Output results.
    print("\nDetected Full Repetitions (Zero-Crossing Adjusted Boundaries):")
    for rep in rep_results:
        print(f"Rep {rep['rep_number']}:")
        print(f"  Start Time: {rep['rep_start_time']:.3f}s (acceleration ≈ 0)")
        print(f"  End Time  : {rep['rep_end_time']:.3f}s (acceleration ≈ 0)")
        out_filename = f"rep_{rep['rep_number']}_full.csv"
        out_df = pd.DataFrame({
            'time_s': rep['rep_time'],
            'vertical_accel_g': rep['rep_accel']
        })
        out_df.to_csv(out_filename, index=False)
        print(f"  Data saved to {out_filename}\n")

    # Step 7: Visualization.
    plt.figure(figsize=(14, 7))
    plt.plot(time, accel_filt, label='Filtered Acceleration')
    plt.plot(time[peak_indices], accel_filt[peak_indices], 'rv', markersize=10, label='Detected Peaks')
    tps_filtered = get_filtered_turning_points(time, accel_filt, min_turn_gap_sec)
    plt.plot(time[tps_filtered], accel_filt[tps_filtered], 'ko', markersize=5, label='Filtered Turning Points')
    for rep in rep_results:
        plt.axvline(rep['rep_start_time'], color='purple', linestyle='--', label='Rep Start' if rep['rep_number']==1 else "")
        plt.axvline(rep['rep_end_time'], color='brown', linestyle='--', label='Rep End' if rep['rep_number']==1 else "")
        plt.fill_between(rep['rep_time'], rep['rep_accel'], alpha=0.3, label='Full Rep' if rep['rep_number']==1 else "")
    plt.xlabel('Time (s)')
    plt.ylabel('Vertical Acceleration (g)')
    plt.title('Deadlift Repetition Segmentation with Zero-Crossing Adjusted Boundaries')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
