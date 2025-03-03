##############################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barbell IMU Processing - Enhanced Gating & Rep Detection
--------------------------------------------------------
Flow Outline:

1) Read & calibrate the IMU data.
2) Initialize orientation from the first stationary segment (N_CALIB samples).
3) For each sample, do an EKF predict step and then a gating-based update:
   - "Full Reinit" if we just became still (to reset orientation).
   - Gravity update if we remain still.
   - Possibly skip or partial updates if rolling or big impact, etc.
4) Lever-arm compensation each step => final corrected acceleration.
5) Integrate corrected acceleration => velocity at bar center.
6) Optionally forcibly zero final velocity & smooth.
7) Output final vertical acceleration & velocity plus debug logs.

"""

import numpy as np
import pandas as pd
import math
from collections import deque
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

##############################################################################
# ------------------------ USER TUNABLES -------------------------------------
##############################################################################

INPUT_CSV  = "imu_data_Farm1.csv"
VERT_CSV   = "vertical_output.csv"
DEBUG_CSV  = "debug_output.csv"

N_CALIB = 50
GRAVITY_SIGN = -1.0
SENSOR_OFFSET = (0.0, 0.0, 0.05)

STILLNESS_ACC_STD     = 0.02
STILLNESS_GYRO_STD    = 0.02
STILLNESS_WINDOW_SIZE = 10
SKIP_IMPACT_THRESHOLD_G = 3.0

# Zero-velocity update thresholds (not used, but we keep them)
VEL_MEAS_STD   = 0.1
ACC_MEAS_STD   = 1.5
GYRO_NOISE_STD = 0.03
BIAS_NOISE_STD = 0.0005
VEL_NOISE_STD  = 0.2

# Rep detection placeholders
REP_START_ACC_G  = 1.3
MIN_REP_SAMPLES  = 10
COOLDOWN_SAMPLES = 50

FORCE_ZUPT_AT_END = True
SMOOTH_VELOCITY   = True
SMOOTH_WINDOW     = 21
SMOOTH_POLY       = 3

##############################################################################
# Global helper to track stillness transitions
was_still_previous = False

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
    ax_raw= df["ax"].values
    ay_raw= df["ay"].values
    az_raw= df["az"].values
    gx_raw= df["gx"].values
    gy_raw= df["gy"].values
    gz_raw= df["gz"].values

    # Calibration offsets
    axb = np.mean(ax_raw[:N_CALIB])
    ayb = np.mean(ay_raw[:N_CALIB])
    azb = np.mean(az_raw[:N_CALIB])
    gxb = np.mean(gx_raw[:N_CALIB])
    gyb = np.mean(gy_raw[:N_CALIB])
    gzb = np.mean(gz_raw[:N_CALIB])

    # Subtract offsets
    ax0 = ax_raw - axb
    ay0 = ay_raw - ayb
    az0 = az_raw - azb
    gx0 = gx_raw - gxb
    gy0 = gy_raw - gyb
    gz0 = gz_raw - gzb

    # Convert to physical units
    ax0 *= 9.81
    ay0 *= 9.81
    az0 *= 9.81
    gx0 *= math.pi/180.0
    gy0 *= math.pi/180.0
    gz0 *= math.pi/180.0

    dt_array = np.diff(t_ms)/1000.
    dt_array = np.concatenate(([0.01], dt_array))

    #------------------------------------------------------------------
    # 1) Initialize orientation from first N_CALIB samples
    #------------------------------------------------------------------
    ax_init = np.mean(ax0[:N_CALIB])
    ay_init = np.mean(ay0[:N_CALIB])
    az_init = np.mean(az0[:N_CALIB])
    q = initialize_quaternion_from_acc(ax_init, ay_init, az_init)

    b_g = np.zeros(3)
    v   = np.zeros(3)

    P = np.eye(9)*0.01
    Q_gyro = (GYRO_NOISE_STD**2)*np.eye(3)
    Q_bias = (BIAS_NOISE_STD**2)*np.eye(3)
    Q_vel  = (VEL_NOISE_STD**2)*np.eye(3)
    Q = np.block([
        [Q_gyro,          np.zeros((3,3)), np.zeros((3,3))],
        [np.zeros((3,3)), Q_bias,          np.zeros((3,3))],
        [np.zeros((3,3)), np.zeros((3,3)), Q_vel]
    ])
    R_acc = (ACC_MEAS_STD**2)*np.eye(3)
    R_vel = (VEL_MEAS_STD**2)*np.eye(3)

    r_body = np.array(SENSOR_OFFSET)
    old_omega_world = np.zeros(3)
    v_corrected = np.zeros(3)

    # Rolling buffers for STILLNESS
    recent_ax = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_ay = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_az = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gx = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gy = deque(maxlen=STILLNESS_WINDOW_SIZE)
    recent_gz = deque(maxlen=STILLNESS_WINDOW_SIZE)

    global was_still_previous
    was_still_previous = False

    # Rep detection placeholders
    is_rep_in_progress = False
    rep_cooldown_count = 0
    consecutive_rep_acc_samples = 0

    final_data = []
    debug_data = []

    # Helper function for a "full re-init" of orientation from stillness
    def reinit_orientation_from_acc(q, b_g, v, P, ax_array, ay_array, az_array):
        ax_mean = np.mean(ax_array)
        ay_mean = np.mean(ay_array)
        az_mean = np.mean(az_array)
        new_q = initialize_quaternion_from_acc(ax_mean, ay_mean, az_mean)
        # keep b_g, v, P or reset them if you prefer
        return new_q, b_g, v, P

    # If you want to force rep_in_progress = False:
    def is_rep_in_progress_function():
        return False

    #----------------------------------------------
    # Main Loop
    #----------------------------------------------
    for i in range(len(t_ms)):
        tstamp = t_ms[i]
        dt = dt_array[i]

        a_sens = np.array([ax0[i], ay0[i], az0[i]])
        g_sens = np.array([gx0[i], gy0[i], gz0[i]])

        # EKF predict
        if i > 0:
            q, b_g, v, P = predict_9d(q, b_g, v, P, g_sens, a_sens, dt, Q)

        # Initialize gating flags each loop
        skip_impact      = False
        do_full_reinit   = False
        do_gravity_update= False
        do_zupt          = False  # we won't use ZUPT for now

        # Quick net accel
        a_mag_g = np.linalg.norm(a_sens)/9.81

        # Overwrite rep detection with a dummy that returns False
        is_rep_in_progress = is_rep_in_progress_function()

        # STILLNESS detection
        recent_ax.append(a_sens[0])
        recent_ay.append(a_sens[1])
        recent_az.append(a_sens[2])
        recent_gx.append(g_sens[0])
        recent_gy.append(g_sens[1])
        recent_gz.append(g_sens[2])

        std_ax = std_ay = std_az = std_gx = std_gy = std_gz = float('nan')
        is_still = False
        if len(recent_ax) == STILLNESS_WINDOW_SIZE:
            std_ax = np.std(recent_ax)
            std_ay = np.std(recent_ay)
            std_az = np.std(recent_az)
            std_gx = np.std(recent_gx)
            std_gy = np.std(recent_gy)
            std_gz = np.std(recent_gz)
            if (std_ax < STILLNESS_ACC_STD and
                std_ay < STILLNESS_ACC_STD and
                std_az < STILLNESS_ACC_STD and
                std_gx < STILLNESS_GYRO_STD and
                std_gy < STILLNESS_GYRO_STD and
                std_gz < STILLNESS_GYRO_STD):
                is_still = True

        # If we detect a large impact:
        if a_mag_g > SKIP_IMPACT_THRESHOLD_G:
            skip_impact = True

        # Now gating logic
        if is_still and (not was_still_previous):
            # We just became still => do full re-init
            do_full_reinit = True
        elif is_still:
            # We remain still => do normal gravity update
            do_gravity_update = True

        # if not still => do nothing

        # Apply measurement updates
        if not skip_impact:
            if do_full_reinit:
                q, b_g, v, P = reinit_orientation_from_acc(q, b_g, v, P,
                                            recent_ax, recent_ay, recent_az)
            elif do_gravity_update:
                q, b_g, v, P = update_gravity_3d(q, b_g, v, P, a_sens, R_acc)
            # do_zupt is always False here

        # update was_still_previous
        was_still_previous = is_still

        # lever-arm compensation + velocity integration
        g_val = 9.81
        a_world = rotate_vector(q, a_sens) - np.array([0., 0., GRAVITY_SIGN*g_val])
        gyro_corr = g_sens - b_g
        omega_world = rotate_vector(q, gyro_corr)

        if i == 0:
            alpha_world_est = np.zeros(3)
        else:
            alpha_world_est = (omega_world - old_omega_world)/dt
        old_omega_world = omega_world.copy()

        r_world = rotate_vector(q, r_body)
        aC = -np.cross(omega_world, np.cross(omega_world, r_world))
        aT = np.cross(alpha_world_est, r_world)
        a_corrected = a_world - (aC + aT)

        v_corrected += a_corrected*dt
        vertical_acc = a_corrected[2]
        vertical_vel = v_corrected[2]

        # Store final_data
        final_data.append((tstamp, vertical_acc, vertical_vel))

        # Debug flags & info appended
        rollDeg, pitchDeg, yawDeg = quat_to_euler(q)
        ekf_speed = np.linalg.norm(v)

        # IMPORTANT: The order of booleans must match debug_cols exactly
        debug_data.append((
            tstamp,
            rollDeg, pitchDeg, yawDeg,
            skip_impact,               # skip_impact
            do_full_reinit,           # full_reinit
            do_gravity_update,        # gravity_update
            do_zupt,                  # zupt
            is_rep_in_progress,       # rep_in_progress
            a_mag_g,
            np.linalg.norm(gyro_corr),
            ekf_speed,
            a_corrected[0], a_corrected[1], a_corrected[2],
            v_corrected[0], v_corrected[1], v_corrected[2],
            vertical_acc, vertical_vel,
            std_ax, std_ay, std_az, std_gx, std_gy, std_gz,
            consecutive_rep_acc_samples, rep_cooldown_count
        ))

    #------------------------------------------------------------------
    # 3) Optional forced zero-velocity at end
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

    #------------------------------------------------------------------
    # 4) Optional smoothing
    #------------------------------------------------------------------
    if SMOOTH_VELOCITY:
        arr_vel = np.array([r[2] for r in final_data])
        if len(arr_vel) >= SMOOTH_WINDOW:
            arr_vel_smooth = savgol_filter(arr_vel, SMOOTH_WINDOW, SMOOTH_POLY)
            new_final = []
            for i, row in enumerate(final_data):
                new_final.append((row[0], row[1], arr_vel_smooth[i]))
            final_data = new_final

    #------------------------------------------------------------------
    # 5) Write out CSVs
    #------------------------------------------------------------------
    df_vert = pd.DataFrame(final_data, columns=["tMillis","vertical_acc","vertical_vel"])
    df_vert.to_csv(VERT_CSV, index=False)
    print(f"Saved final vertical results to {VERT_CSV}.")

    # EXACT matching order for debug_data
    debug_cols = [
        "tMillis",
        "roll_deg","pitch_deg","yaw_deg",
        "skip_impact","full_reinit","gravity_update","zupt","rep_in_progress",
        "acc_mag_g","gyro_mag","ekf_speed",
        "acc_world_x","acc_world_y","acc_world_z",
        "vel_world_x","vel_world_y","vel_world_z",
        "vertical_acc","vertical_vel",
        "std_ax","std_ay","std_az","std_gx","std_gy","std_gz",
        "rep_acc_count","rep_cooldown_count"
    ]
    df_dbg = pd.DataFrame(debug_data, columns=debug_cols)
    df_dbg.to_csv(DEBUG_CSV, index=False)
    print(f"Saved debug info to {DEBUG_CSV}.")

    plot_results(final_data, df_dbg)

##############################################################################
# EKF & Utility Functions
##############################################################################

def reinit_orientation_from_current_acc(q, b_g, v, P, ax_array, ay_array, az_array):
        """Use the average of the rolling window to create a new quaternion from scratch."""
        ax_mean = np.mean(ax_array)
        ay_mean = np.mean(ay_array)
        az_mean = np.mean(az_array)
        new_q = initialize_quaternion_from_acc(ax_mean, ay_mean, az_mean)
        # You also might want to reset velocity or partial states if you prefer
        return new_q, b_g, v, P  # keep b_g, v, P as is, or reset them if needed


def is_rep_in_progress_function():
        return False


def force_reinit_orientation_from_acc(q, b_g, v, P,
                                      ax_buf, ay_buf, az_buf,
                                      GRAVITY_SIGN):
    """
    Completely re-initialize orientation from the *averaged* acceleration
    in ax_buf, ay_buf, az_buf. That effectively resets roll/pitch to 0,0
    in the new frame. Also resets velocity if desired, and re-inits P.

    You can tweak whether you keep the bias or zero it, etc.
    """
    if len(ax_buf) == 0:
        return q, b_g, v, P  # do nothing if no data

    ax_mean = np.mean(ax_buf)
    ay_mean = np.mean(ay_buf)
    az_mean = np.mean(az_buf)

    q_new = initialize_quaternion_from_acc(ax_mean, ay_mean, az_mean)
    # optionally reset velocity:
    v_new = np.zeros(3)

    # re-init covariance
    P_new = np.eye(9)*0.01
    # optionally keep b_g or zero it out:
    b_g_new = b_g  # or b_g_new = np.zeros(3) if you prefer

    return q_new, b_g_new, v_new, P_new


def predict_9d(q, b_g, v, P, gyro_sens, accel_sens, dt, Q):
    """ Standard 9D predict: orientation, gyro bias, velocity. """
    gyro_corr = gyro_sens - b_g
    # orientation integration
    q_pred = integrate_quaternion(q, gyro_corr, dt)
    # velocity integration (world frame)
    g_val = 9.81
    a_world = rotate_vector(q, accel_sens) - np.array([0.,0., GRAVITY_SIGN*g_val])
    v_pred = v + a_world*dt

    # Covariance
    F11 = np.eye(3) - skew(gyro_corr)*dt
    F12 = -np.eye(3)*dt
    F13 = np.zeros((3,3))

    F21 = np.zeros((3,3))
    F22 = np.eye(3)
    F23 = np.zeros((3,3))

    acc_world = rotate_vector(q, accel_sens)
    F31 = -skew(acc_world)*dt
    F32 = np.zeros((3,3))
    F33 = np.eye(3)

    F = np.block([
        [F11, F12, F13],
        [F21, F22, F23],
        [F31, F32, F33]
    ])
    P_pred = F @ P @ F.T + Q

    return q_pred, b_g, v_pred, P_pred


def update_gravity_3d(q, b_g, v, P, a_sens, R_acc):
    """ Full gravity vector alignment. """
    g_val= 9.81
    a_pred = rotate_vector(q, np.array([0.,0., GRAVITY_SIGN*g_val]))
    r = a_sens - a_pred

    H_att = -skew(a_pred)
    H_bg  = np.zeros((3,3))
    H_v   = np.zeros((3,3))
    H = np.block([[H_att, H_bg, H_v]])

    S = H @ P @ H.T + R_acc
    S_reg = S + 1e-9*np.eye(3)
    if np.linalg.cond(S_reg) > 1e12:
        return q, b_g, v, P

    K = P @ H.T @ np.linalg.inv(S_reg)
    dx = K @ r
    dtheta = dx[0:3]
    dbg    = dx[3:6]
    dv     = dx[6:9]

    dq= quat_from_small_angle(dtheta)
    q_new = quat_normalize(quat_multiply(q, dq))
    b_g_new = b_g + dbg
    v_new   = v + dv

    I9= np.eye(9)
    P_new= (I9 - K@H) @ P
    return q_new, b_g_new, v_new, P_new


def update_zupt_9d(q, b_g, v, P, R_vel):
    """ Zero-velocity update. """
    r = v
    H_att= np.zeros((3,3))
    H_bg = np.zeros((3,3))
    H_v  = np.eye(3)
    H= np.block([[H_att,H_bg,H_v]])

    S= H@P@H.T + R_vel
    S_reg= S + 1e-9*np.eye(3)
    if np.linalg.cond(S_reg)>1e12:
        return q,b_g,v,P

    K= P@ H.T @ np.linalg.inv(S_reg)
    dx= K@(-r)
    dtheta= dx[:3]
    dbg   = dx[3:6]
    dv    = dx[6:9]

    dq= quat_from_small_angle(dtheta)
    q_new= quat_normalize(quat_multiply(q,dq))
    b_g_new= b_g + dbg
    v_new= v + dv

    I9= np.eye(9)
    P_new= (I9 - K@H)@P
    return q_new,b_g_new,v_new,P_new


def skew(v):
    x,y,z= v
    return np.array([
        [   0, -z,  y],
        [  z,   0, -x],
        [ -y,  x,   0]
    ])


def integrate_quaternion(q, gyro, dt):
    """ Integrate orientation given gyro in rad/s. """
    gq= np.array([0., gyro[0], gyro[1], gyro[2]])
    dq= quat_multiply(q, gq)*0.5*dt
    return quat_normalize(q + dq)


def quat_from_small_angle(dtheta):
    """ For small angles, dq ~ [1, 0.5*dtheta]. """
    dq= np.hstack(([1.], 0.5*dtheta))
    return quat_normalize(dq)

def quat_multiply(q, r):
    w1,x1,y1,z1= q
    w2,x2,y2,z2= r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q):
    n= np.linalg.norm(q)
    if n<1e-12:
        return np.array([1.,0.,0.,0.])
    return q/n

def rotate_vector(q, v):
    """ Rotate vector v from sensor frame to world frame by quaternion q. """
    qv = np.hstack(([0.], v))
    return quat_multiply(quat_multiply(q,qv), quat_conjugate(q))[1:]

def quat_conjugate(q):
    return q*np.array([1., -1., -1., -1.])

def initialize_quaternion_from_acc(ax, ay, az):
    """ Initialize orientation so that sensor's measured accel
        aligns with [0,0,GRAVITY_SIGN*9.81] in world frame.
    """
    a= np.array([ax, ay, az])
    norm_a= np.linalg.norm(a)
    if norm_a<1e-6:
        return np.array([1.,0.,0.,0.])
    a_norm= a/norm_a

    g_vec= np.array([0.,0., GRAVITY_SIGN*9.81])
    cross_ = np.cross(a_norm, g_vec)
    s= np.linalg.norm(cross_)
    c= np.dot(a_norm, g_vec)/9.81
    if s<1e-12:
        # means a_norm is basically parallel to g_vec
        return np.array([1.,0.,0.,0.])
    angle= math.atan2(s, c)
    axis= cross_/s

    half= angle*0.5
    w= math.cos(half)
    xyz= axis*math.sin(half)
    return quat_normalize(np.hstack(([w], xyz)))


def quat_to_euler(q):
    """ Convert quaternion to (roll, pitch, yaw) in degrees. """
    w,x,y,z = q
    # roll
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2*(w*y - z*x)
    if abs(sinp)>1.0:
        pitch = math.copysign(math.pi/2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


##############################################################################
# Simple Plot
##############################################################################
def plot_results(final_data, df_dbg):
    """
    Quick visualization at the end – now with an extra subplot for net accel (G).
    """
    times   = np.array([row[0] for row in final_data])
    vertAcc = np.array([row[1] for row in final_data])
    vertVel = np.array([row[2] for row in final_data])

    # Retrieve debug info
    roll_deg      = df_dbg["roll_deg"].values
    pitch_deg     = df_dbg["pitch_deg"].values
    yaw_deg       = df_dbg["yaw_deg"].values

    skip_impact   = df_dbg["skip_impact"].values.astype(float)
    gravity_update= df_dbg["gravity_update"].values.astype(float)
    zupt          = df_dbg["zupt"].values.astype(float)
    rep_in_prog   = df_dbg["rep_in_progress"].values.astype(float)

    acc_world_x   = df_dbg["acc_world_x"].values
    acc_world_y   = df_dbg["acc_world_y"].values
    acc_world_z   = df_dbg["acc_world_z"].values

    vel_world_x   = df_dbg["vel_world_x"].values
    vel_world_y   = df_dbg["vel_world_y"].values
    vel_world_z   = df_dbg["vel_world_z"].values

    # The net acceleration in "g" that we used to detect reps:
    acc_mag_g     = df_dbg["acc_mag_g"].values
    
    # If you keep your threshold in code, you can store it here too.
    # For example, if you had REP_START_ACC_G = 1.3, you can define:
    REP_START_ACC_G = 1.3  

    # We'll make 6 subplots now, up from 5
    fig, axes = plt.subplots(6,1, figsize=(12,18), sharex=True)
    fig.suptitle("Barbell IMU Debug Plots")

    # (1) Orientation
    ax1 = axes[0]
    ax1.plot(times, roll_deg,  label="Roll (deg)")
    ax1.plot(times, pitch_deg, label="Pitch (deg)")
    ax1.plot(times, yaw_deg,   label="Yaw (deg)")
    ax1.grid(True); ax1.legend()
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Orientation")

    # (2) Gating flags
    ax2 = axes[1]
    ax2.step(times, skip_impact,    label="SkipImpact",    where="post")
    ax2.step(times, gravity_update, label="GravityUpdate", where="post")
    ax2.step(times, zupt,           label="ZUPT",          where="post")
    ax2.step(times, rep_in_prog,    label="RepInProgress", where="post")
    ax2.grid(True); ax2.legend()
    ax2.set_ylim([-0.1,1.1])
    ax2.set_ylabel("Gating (0/1)")
    ax2.set_title("Gating Flags + Rep Detection")

    # (3) 3D Acceleration (world)
    ax3 = axes[2]
    ax3.plot(times, acc_world_x, label="accX (m/s^2)")
    ax3.plot(times, acc_world_y, label="accY (m/s^2)")
    ax3.plot(times, acc_world_z, label="accZ (m/s^2)")
    ax3.grid(True); ax3.legend()
    ax3.set_ylabel("m/s^2")
    ax3.set_title("World-Frame Acceleration (Lever-Arm Corrected)")

    # (4) 3D Velocity (world)
    ax4 = axes[3]
    ax4.plot(times, vel_world_x, label="velX (m/s)")
    ax4.plot(times, vel_world_y, label="velY (m/s)")
    ax4.plot(times, vel_world_z, label="velZ (m/s)")
    ax4.grid(True); ax4.legend()
    ax4.set_ylabel("m/s")
    ax4.set_title("World-Frame Velocity (Integrated)")

    # (5) Final vertical Acc & Vel
    ax5 = axes[4]
    ax5.plot(times, vertAcc, label="Vertical Acc (m/s^2)", color="blue")
    ax5.plot(times, vertVel, label="Vertical Vel (m/s)",   color="red")
    ax5.grid(True); ax5.legend()
    ax5.set_ylabel("m/s^2 / m/s")
    ax5.set_title("Final Vertical Acc & Vel")

    # (6) Net acceleration in G (rep detection metric)
    ax6 = axes[5]
    ax6.plot(times, acc_mag_g, label="Net Acc (G)", color="darkorange")
    # We can draw a horizontal line at the rep threshold for reference:
    ax6.axhline(REP_START_ACC_G, color="red", linestyle="--", label="Rep Start Threshold")
    ax6.grid(True); ax6.legend()
    ax6.set_ylabel("Net Acc (g)")
    ax6.set_xlabel("Time (ms)")
    ax6.set_title("Net Accel in G (Trigger for Rep Detection)")

    plt.tight_layout()
    plt.savefig("barbell_debug_plots.png")
    plt.show()



def quat_conjugate(q):
    return q*np.array([1., -1., -1., -1.])

##############################################################################

if __name__=="__main__":
    main()
