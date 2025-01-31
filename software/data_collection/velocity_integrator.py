# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:09:18 2025

@author: rkfurst
"""

import asyncio
import csv
import sys
import time
from datetime import datetime

from bleak import BleakScanner, BleakClient
import numpy as np
from scipy.signal import savgol_filter

SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
CHAR_UUID    = "19B10001-E8F2-537E-4F6C-D104768A1214"

global_data = []  # will store (time_s, accel_g)


def notification_handler(sender: int, data: bytearray):
    """
    Called every time we get a BLE notification with the CSV string "<millis>,<accel_g>".
    We'll parse and store as (time_in_seconds, accel_in_g).
    """
    txt = data.decode('utf-8').strip()
    try:
        millis_str, accel_str = txt.split(',')
        t_millis = float(millis_str)
        accel_g  = float(accel_str)

        t_sec = t_millis / 1000.0
        global_data.append((t_sec, accel_g))

    except:
        print(f"[Warning] Bad data format: {txt}")


async def run_ble_logging():
    print("Scanning for BLE devices (5 seconds)...")
    devices = await BleakScanner.discover(timeout=5.0)
    target_device = None
    for d in devices:
        devname = d.name or ""  # fallback to empty string if None
        if "Nano33BLE-SingleAxis" in devname:
            target_device = d
            break
    
    if not target_device:
        print("Could not find device named 'Nano33BLE-OffAxis'. Exiting.")
        sys.exit(1)
    
    print(f"Found device: {target_device.name}, {target_device.address}")
    async with BleakClient(target_device) as client:
        print("Connected. Subscribing to notifications...")
        await client.start_notify(CHAR_UUID, notification_handler)
        print("Press Ctrl+C to stop logging.")
        try:
            while True:
                await asyncio.sleep(0.2)
        except KeyboardInterrupt:
            pass

        print("Stopping notifications...")
        await client.stop_notify(CHAR_UUID)


def save_to_csv_and_process():
    """
    After logging ends, save the data to CSV, then run the velocity analysis.
    """
    out_filename = "data_log.csv"
    with open(out_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "vertical_accel_g"])
        writer.writerows(global_data)

    print(f"Saved {len(global_data)} samples to {out_filename}.")

    # Now do the velocity analysis
    process_data(global_data)


def process_data(data):
    """
    data is a list of (t_s, accel_g).
    We'll:
      1) Convert g->m/s^2
      2) Optionally filter
      3) Rep detection w/ threshold + min rep duration
      4) Integrate velocity, clamp to 0 at end
      5) Print stats
    """
    if len(data) < 5:
        print("Not enough samples to process.")
        return

    times = np.array([d[0] for d in data])
    acc_g = np.array([d[1] for d in data])
    acc_ms2 = acc_g * 9.81

    # --- A) Filtering ---
    # We'll do a small Savitzky-Golay (window=9)
    # If your sample rate is ~100 Hz, that's a ~0.09s window
    # If you get errors about window>data length, reduce window_length or gather more data
    window_len = min(9, len(acc_ms2) - (len(acc_ms2)%2==0))  # ensure odd
    if window_len < 5:
        window_len = 5  # fallback
    filt_acc = savgol_filter(acc_ms2, window_length=window_len, polyorder=3)

    # --- B) Rep detection ---
    # We'll define thresholds in m/s^2
    # For your data, let's say crossing above +2g => start
    # Then after a negative spike, we confirm the bar is at rest => end
    threshold_up   = 0.8 * 9.81    # ~7.85 m/s^2
    threshold_down = -0.8 * 9.81   # ~-7.85 m/s^2

    rep_starts = []
    rep_ends   = []
    in_rep = False
    start_idx = None

    # We'll define a min rep duration, e.g. 0.3 s, to avoid false quick triggers
    MIN_REP_DURATION = 0.2

    def is_near_zero(idx, margin=0.3*9.81, dur=0.10):
        """
        Check if the next ~dur seconds of data are all within ±margin
        """
        if idx >= len(filt_acc):
            return False
        dt_mean = np.mean(np.diff(times)) if len(times)>1 else 0.01
        n_check = int(dur / dt_mean)
        end_i = min(idx + n_check, len(filt_acc))
        subset = filt_acc[idx:end_i]
        return np.all(np.abs(subset) < margin)

    for i in range(len(filt_acc)):
        a = filt_acc[i]
        if not in_rep:
            # look for crossing above threshold_up
            if a > threshold_up:
                start_idx = i
                in_rep = True
        else:
            # we're in a rep, watch for big negative + near-zero
            if a < threshold_down:
                # once we detect negative spike, see if bar rests
                if i+1 < len(filt_acc) and is_near_zero(i+1):
                    # also check min duration
                    if times[i] - times[start_idx] >= MIN_REP_DURATION:
                        rep_starts.append(start_idx)
                        rep_ends.append(i+1)
                    in_rep = False

    # --- C) Integrate velocity per rep ---
    # We'll clamp velocity=0 at both start and end
    # so each rep is a self-contained "bar from rest to rest"
    for r in range(len(rep_starts)):
        sidx = rep_starts[r]
        eidx = rep_ends[r]
        t_slice = times[sidx:eidx+1]
        a_slice = filt_acc[sidx:eidx+1]

        v = np.zeros_like(a_slice)
        for k in range(1, len(v)):
            dt = t_slice[k] - t_slice[k-1]
            v[k] = v[k-1] + a_slice[k]*dt

        # clamp end velocity to 0
        # we'll do a linear "drift fix" from the final velocity down to 0
        final_vel = v[-1]
        drift = final_vel
        n = len(v)
        for k in range(n):
            frac = k/(n-1) if n>1 else 1
            v[k] -= drift*frac  # remove the linear drift across the rep

        peak_vel = np.max(v)
        avg_vel  = np.mean(v)
        duration= t_slice[-1] - t_slice[0]

        print(f"\nRep {r+1}:")
        print(f"  Start time : {t_slice[0]:.3f} s, End time: {t_slice[-1]:.3f} s, Duration={duration:.3f}s")
        print(f"  Peak velocity = {peak_vel:.3f} m/s")
        print(f"  Avg velocity  = {avg_vel:.3f} m/s")

    if len(rep_starts)==0:
        print("\nNo reps detected. Try adjusting thresholds or min rep duration.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_ble_logging())
    finally:
        save_to_csv_and_process()
