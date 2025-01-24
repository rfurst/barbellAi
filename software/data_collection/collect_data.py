# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:29:28 2025

@author: rkfurst
"""
import serial
import time
import csv

# Connect to Arduino
ser = serial.Serial('COM3', 115200)  # Windows: COM*, Linux: /dev/ttyUSB0

with open('barbell_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'rep_label', 'exercise_id'])
    
    try:
        while True:
            input("Press Enter to start recording a rep...")
            start_time = time.time()
            
            while time.time() - start_time < 5:  # Record 5s per rep
                line = ser.readline().decode().strip()
                if line:
                    ax, ay, az, gx, gy, gz = map(float, line.split(','))
                    # Label: 1=rep active, 0=rest; exercise_id (0=deadlift, 1=bench, etc.)
                    writer.writerow([ax, ay, az, gx, gy, gz, 1, 0])  
    except KeyboardInterrupt:
        ser.close()