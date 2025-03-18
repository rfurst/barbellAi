# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:41:50 2025

@author: rkfurst
"""

import asyncio
import struct
import csv
from bleak import BleakScanner, BleakClient

# --------- Adjust these as needed to match your Arduino sketch ---------
DEVICE_NAME = "Nano33_RawIMU"
SERVICE_UUID = "12340000-0000-0000-0000-1234567890AB"
CHAR_UUID    = "12340001-0000-0000-0000-1234567890AB"
OUTPUT_CSV   = "imu_data_Ethos_Jacob_3plateAnd25_1rep.csv"
# -----------------------------------------------------------------------

# Each IMU sample is 28 bytes total:
#   [ 4 bytes tMillis (uint32_t) ]
#   [ 4 bytes ax (float) ]
#   [ 4 bytes ay (float) ]
#   [ 4 bytes az (float) ]
#   [ 4 bytes gx (float) ]
#   [ 4 bytes gy (float) ]
#   [ 4 bytes gz (float) ]
SAMPLE_SIZE_BYTES = 28

# Global buffer
leftover_bytes = b''

async def main():
    print("Scanning for BLE devices...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name == DEVICE_NAME
    )
    if not device:
        print(f"Device '{DEVICE_NAME}' not found. Make sure it's advertising.")
        return

    print(f"Found device: {device.address}. Connecting...")

    # Open CSV file for writing
    with open(OUTPUT_CSV, mode='w', newline='') as f_csv:
        csv_writer = csv.writer(f_csv)
        # Write header row if desired
        csv_writer.writerow(["tMillis", "ax", "ay", "az", "gx", "gy", "gz"])

        def handle_notification(sender: int, data: bytearray):
            """
            Callback for handling incoming BLE notifications.
            'sender' is the characteristic handle,
            'data' is a bytes-like object with the raw payload.
            """
            global leftover_bytes

            # Combine leftover + newly received data
            buffer = leftover_bytes + data

            offset = 0
            # Parse as many whole 28-byte samples as we can
            while (offset + SAMPLE_SIZE_BYTES) <= len(buffer):
                chunk = buffer[offset : offset + SAMPLE_SIZE_BYTES]

                # Decode 1 sample
                # Little-endian (<):
                #   I = 4-byte unsigned int
                #   f = 4-byte float
                # struct format: <I f f f f f f  => 4 + (6*4) = 28 bytes
                tMillis, ax, ay, az, gx, gy, gz = struct.unpack('<Iffffff', chunk)

                # Write CSV row
                csv_writer.writerow([tMillis, ax, ay, az, gx, gy, gz])

                offset += SAMPLE_SIZE_BYTES

            # Save any leftover partial data for next time
            leftover_bytes = buffer[offset:]

        # Connect and start notifications
        async with BleakClient(device) as client:
            print("Connected. Subscribing to characteristic notifications...")

            # Start listening
            await client.start_notify(CHAR_UUID, handle_notification)
            print(f"Subscribed to {CHAR_UUID}. Logging data to {OUTPUT_CSV}...")
            print("Press Ctrl+C to stop.")

            # Keep running until user stops
            try:
                while True:
                    await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                print("\nStopping...")

            # Cleanup: stop notify
            await client.stop_notify(CHAR_UUID)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())