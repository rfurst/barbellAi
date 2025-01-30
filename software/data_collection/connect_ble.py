# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:49:42 2025

@author: rkfurst
"""
import asyncio
import csv
import time
from bleak import BleakClient

# -------------------------------------------------------------------
# Adjust these to match your device and characteristic UUID
# -------------------------------------------------------------------
DEVICE_ADDRESS = "b5:db:86:24:a7:1c"  # Bluetooth MAC address (Windows/Linux) or UUID (on macOS)
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
CHAR_UUID    = "19B10001-E8F2-537E-4F6C-D104768A1214"

# Name of the CSV file where data will be saved
CSV_FILENAME = "vertical_accel_data.csv"


# -------------------------------------------------------------------
# Notification Callback
# -------------------------------------------------------------------
# This function is called every time the BLE peripheral sends new data.
async def notification_handler(sender: int, data: bytearray):
    """
    Parse the incoming BLE data (string float value), get current time,
    and append it to the CSV file.
    """
    # Decode the data as a UTF-8 string
    accel_str = data.decode("utf-8").strip()
    
    # Try to convert to float, though it may be a string like "0.1234"
    try:
        vertical_accel = float(accel_str)
    except ValueError:
        vertical_accel = None

    # Get current system timestamp (seconds since epoch)
    timestamp = time.time()  # or time.perf_counter(), depending on your needs

    # Append row to CSV
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if vertical_accel is not None:
            writer.writerow([timestamp, vertical_accel])
        else:
            # If parsing failed, log the raw data for debugging
            writer.writerow([timestamp, "PARSE_ERROR", accel_str])

    # (Optional) Print to console for real-time viewing
    print(f"Time: {timestamp:.2f}, VerticalAccel: {accel_str}")


# -------------------------------------------------------------------
# Main Async Function
# -------------------------------------------------------------------
async def main(address: str):
    print(f"Connecting to {address} ...")
    async with BleakClient(address) as client:
        print("Connected.")

        # Check if the service/characteristic is found
        svcs = await client.get_services()
        if SERVICE_UUID not in [s.uuid for s in svcs]:
            print(f"Warning: Service {SERVICE_UUID} not found on the device.")
        else:
            print(f"Service {SERVICE_UUID} found.")

        # Start notification on the characteristic
        print(f"Subscribing to characteristic {CHAR_UUID}...")
        await client.start_notify(CHAR_UUID, notification_handler)

        print("Collecting data... Press Ctrl+C to stop.")
        try:
            # Keep running until user stops (Ctrl+C)
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping notifications...")
            await client.stop_notify(CHAR_UUID)
    print("Disconnected.")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Create/Open the CSV file and write a header row if desired
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "VerticalAccel_g"])

    # Run the event loop for Bleak
    asyncio.run(main(DEVICE_ADDRESS))
