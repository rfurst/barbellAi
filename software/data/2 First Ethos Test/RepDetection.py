#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.signal as ssig

def load_csv(filename):
    """
    Load a CSV with columns [Timestamp, VerticalAccel_g].
    Assumes 'Timestamp' is in seconds or ms, we will handle it.
    Returns a DataFrame with columns: ['time_s','acc_g'].
    """
    df = pd.read_csv(filename)
    # Ensure the column names match your actual CSV header
    # For example, if your CSV has 'Timestamp_s' and 'VerticalAccel_g', rename them:
    if 'Timestamp' in df.columns:
        df.rename(columns={'Timestamp': 'time_s'}, inplace=True)
    if 'VerticalAccel_g' in df.columns:
        df.rename(columns={'VerticalAccel_g': 'acc_g'}, inplace=True)
    
    # If your Timestamp is in ms, convert to seconds:
    # df['time_s'] *= 1e-3
    
    df = df[['time_s','acc_g']].copy()
    df.sort_values('time_s', inplace=True)  # sort by time
    df.reset_index(drop=True, inplace=True)
    return df

def filter_acceleration(acc, fs, cutoff=5.0):
    """
    Optional: low-pass filter the acceleration data to reduce noise.
    cutoff=5 Hz for example.
    fs = sampling frequency in Hz.
    """
    # Design a Butterworth filter
    nyq = 0.5 * fs
    normal_cut = cutoff / nyq
    b, a = ssig.butter(4, normal_cut, btype='low', analog=False)
    # Apply filtfilt
    filtered = ssig.filtfilt(b, a, acc)
    return filtered

def detect_reps(time_s, velocity, v_thresh=0.1, min_hold=0.2):
    """
    Very simplistic rep detection:
    - We look for times the bar's velocity crosses near zero
      and changes direction (i.e. from negative to positive or vice versa).
    - A real implementation might look for min velocity around -X, or zero crossing
      plus a dwell time, etc.

    Returns a list of (rep_start_idx, rep_end_idx).
    v_thresh = small threshold near zero to consider "turnaround"
    min_hold = minimal time in seconds between rep boundaries (to avoid double counting).
    """
    reps = []
    N = len(time_s)
    i = 0
    last_boundary_idx = 0
    in_rep = False
    
    while i < N-1:
        # Check sign change
        v_now = velocity[i]
        v_next = velocity[i+1]
        if abs(v_now) < v_thresh and np.sign(v_now) != np.sign(v_next):
            # possible boundary
            t_cur = time_s[i]
            if (t_cur - time_s[last_boundary_idx]) > min_hold:
                if not in_rep:
                    # we found a rep start
                    rep_start = i
                    in_rep = True
                else:
                    # we found a rep end
                    rep_end = i
                    reps.append((rep_start, rep_end))
                    in_rep = False
                last_boundary_idx = i
        i += 1
    return reps

def segmentwise_integration(time_s, acc_g, g_to_mss=9.81):
    """
    Integrate acceleration(g) -> velocity(m/s).
    We'll do a naive numeric integration with trapezoidal rule.
    Return velocity array, same length as acc_g.
    We do NOT do any rep-based resetting here; we just do full continuous integration
    from the start, assuming v(0)=0.
    """
    # Convert g to m/s^2:
    acc_mss = acc_g * g_to_mss
    
    v = np.zeros_like(acc_mss)
    for i in range(1, len(acc_mss)):
        dt = time_s[i] - time_s[i-1]
        # trapezoid rule
        v[i] = v[i-1] + 0.5*(acc_mss[i] + acc_mss[i-1])*dt
    return v

def apply_rep_zeroing(time_s, v, reps):
    """
    For each rep boundary, we forcibly set velocity to 0 (or near 0)
    at the boundaries, then re-integrate forward from there.
    This kills drift in each rep segment. We do it in-place for simplicity.
    """
    v_adj = v.copy()
    # For each rep (start_idx, end_idx), set v to 0 at start_idx, end_idx
    # Then linearly re-shift the segment in between so the integration
    # is continuous.  Easiest is to just do partial re-integration segment by segment.
    for (start_i, end_i) in reps:
        if start_i < 0 or end_i >= len(v):
            continue
        # force velocity = 0 at start & end
        v_adj[start_i] = 0.0
        v_adj[end_i]   = 0.0
        
        # re-integrate from start_i to end_i
        for i in range(start_i+1, end_i+1):
            dt = time_s[i] - time_s[i-1]
            # keep existing acceleration info by differencing the velocity array
            # or re-run the original acceleration integration, but we no longer
            # have that in this function. So let's do a simpler "shift" approach:
            dv = v_adj[i] - v_adj[i-1]
            # but that doesn't fix drift well. Real approach: we'd store the original
            # acceleration, re-integrate inside each segment with v=0 at the start.
            # We'll demonstrate a simpler approach: just set everything to the same offset:
            v_adj[i] = v_adj[i-1] + dv
        # Now we ensure the end is forced to 0
        offset = v_adj[end_i]
        # subtract this offset from entire segment
        v_adj[start_i:end_i+1] -= offset
    
    return v_adj

def compute_rep_velocities(time_s, v, reps):
    """
    For each rep, compute something like max velocity, average velocity, etc.
    Return a list of dictionaries, one per rep.
    """
    results = []
    for (start_i, end_i) in reps:
        rep_time = time_s[end_i] - time_s[start_i]
        seg_v = v[start_i:end_i+1]
        v_max = np.max(seg_v)
        v_min = np.min(seg_v)
        v_mean = np.mean(seg_v)
        results.append({
            'start_idx': start_i,
            'end_idx':   end_i,
            'rep_time_s': rep_time,
            'v_max': v_max,
            'v_min': v_min,
            'v_mean':v_mean
        })
    return results

def estimate_1RM_from_velocity(load_kg, velocity_m_s):
    """
    Very rough velocity-based 1RM estimate:
    - Suppose user did reps with known load_kg
    - We measure an average or peak velocity (velocity_m_s).
    - Then we do a naive linear extrapolation. One simple heuristic says:
         If we assume velocity -> 0 at 1RM, we can pick a "critical velocity"
         at submax load and do a linear fit. This is extremely approximate.

    Example formula (some coaches do):
        1RM = load_kg / (1 - velocity_m_s / V0)
    where V0 might be a reference velocity intercept for that exercise (like 1.5 m/s).
    But you need actual data for that.

    We'll just do a toy approach: 1RM = load_kg * (some factor).
    """
    # Without a known load-velocity profile or calibration, there's no universal formula.
    # Let's demonstrate a made-up approach:
    V_crit = 0.15  # assume that at 1RM, bar speed is ~0.15 m/s
    if velocity_m_s <= V_crit:
        # obviously heavier than we guessed
        estimated_1rm = load_kg * 1.05  # minimal guess
    else:
        # naive ratio
        estimated_1rm = load_kg * (velocity_m_s / V_crit)
    return estimated_1rm


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process_barbell_data.py <filename.csv>")
        sys.exit(1)
    csv_file = sys.argv[1]

    # 1) Load data
    df = load_csv(csv_file)
    # Suppose your data is ~100 Hz, or measure from consecutive timestamps
    # to get an average sampling frequency
    time_s = df['time_s'].values.astype(float)
    acc_g  = df['acc_g'].values.astype(float)

    # estimate sampling freq
    dt_array = np.diff(time_s)
    fs_est = 1.0 / np.median(dt_array)
    print(f"Estimated sampling freq ~ {fs_est:.1f} Hz")

    # 2) (Optional) Filter
    acc_filt = filter_acceleration(acc_g, fs_est, cutoff=5.0)

    # 3) Integrate to velocity (continuous)
    vel_full = segmentwise_integration(time_s, acc_filt, g_to_mss=9.81)

    # 4) Detect reps from the velocity sign changes
    #    (We could also do it from the raw or filtered acceleration, etc.)
    #    For demonstration, let's do from vel_full:
    rep_boundaries = detect_reps(time_s, vel_full, v_thresh=0.1, min_hold=0.2)
    print(f"Detected {len(rep_boundaries)} reps (start_idx,end_idx):", rep_boundaries)

    # 5) Re‐zero velocity in each rep to kill drift
    vel_corrected = vel_full.copy()
    # A more robust way is to re‐integrate inside each rep with v=0 at boundaries,
    # which requires storing the original acceleration array. Let's demonstrate that:
    vel_segmented = np.zeros_like(vel_corrected)
    start_global = 0
    for i, (start_i, end_i) in enumerate(rep_boundaries):
        # integrate from the last boundary to this boundary with v( boundary )=0
        # Actually, let's define "rep" as that chunk:
        if end_i <= start_i:
            continue
        # We'll re‐integrate the acceleration from [start_i .. end_i],
        # forcing v=0 at start and end.
        # This is the best approach to kill drift in each rep segment.

        # do a partial integration
        vel_segment = np.zeros(end_i - start_i + 1)
        for idx in range(start_i+1, end_i+1):
            dt = time_s[idx] - time_s[idx-1]
            # acceleration in m/s^2:
            a_mss = acc_filt[idx] * 9.81
            a_mss_prev = acc_filt[idx-1] * 9.81
            vel_segment[idx - start_i] = vel_segment[idx - start_i - 1] + 0.5*(a_mss + a_mss_prev)*dt

        # Now we have a velocity trace that ends at some nonzero value. Force end=0:
        final_offset = vel_segment[-1]
        vel_segment -= final_offset  # shift entire segment so that final = 0

        # Put it into vel_segmented array
        vel_segmented[start_i:end_i+1] = vel_segment

    # The rest of the data outside these rep boundaries we can just leave as is
    # or set to zero.  For demonstration, let's set them to zero:
    # find all covered indices
    covered_idx = []
    for (si, ei) in rep_boundaries:
        covered_idx.extend(range(si, ei+1))
    mask = np.ones(len(vel_segmented), dtype=bool)
    mask[covered_idx] = False
    vel_segmented[mask] = 0.0

    # 6) Compute rep velocities
    rep_stats = compute_rep_velocities(time_s, vel_segmented, rep_boundaries)
    for rs in rep_stats:
        print(rs)

    # 7) Suppose the user was lifting load_kg for these reps,
    #    and we want to guess a 1RM. We'll pick, say, the 3rd rep's peak velocity:
    load_kg = 61.0
    if len(rep_stats) > 0:
        example_rep = rep_stats[-1]  # last rep
        v_peak = example_rep['v_max']
        est_1rm = estimate_1RM_from_velocity(load_kg, v_peak)
        print(f"\nUser lifted {load_kg} kg with peak velocity ~ {v_peak:.3f} m/s")
        print(f"Estimated 1RM ~ {est_1rm:.1f} kg (toy example)")

if __name__ == '__main__':
    main()
