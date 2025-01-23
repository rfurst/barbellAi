import serial
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
PORT = 'COM3'  # Windows: COM*, Linux: /dev/ttyUSB0000
BAUD_RATE = 115200
EXERCISES = {0: "deadlift", 1: "bench_press", 2: "squat"}

# Initialize
ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
exercise_id = int(input(f"Enter exercise ID {list(EXERCISES.keys())}: "))
filename = f"{EXERCISES[exercise_id]}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

# Setup plot
plt.ion()
fig, ax = plt.subplots()
timestamps, ax_values = [], []
line, = ax.plot([], [], 'r-')
ax.set_title("Real-Time Acceleration (X-Axis)")

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'rep_phase'])
    
    try:
        start_time = time.time()
        rep_phase = 0  # 0=rest, 1=concentric, 2=eccentric
        
        while True:
            # User control
            cmd = input("Press 's' (start rep), 'e' (end rep), 'q' (quit): ").lower()
            if cmd == 'q': break
            rep_phase = 1 if cmd == 's' else 2 if cmd == 'e' else rep_phase
            
            # Read serial
            line = ser.readline().decode().strip()
            if not line: continue
            
            # Process data
            try:
                values = list(map(float, line.split(',')))
                if len(values) != 6: continue
            except:
                print(f"Bad data: {line}")
                continue
            
            # Log and plot
            timestamp = time.time() - start_time
            writer.writerow([timestamp, *values, rep_phase])
            
            timestamps.append(timestamp)
            ax_values.append(values[0])
            line.set_xdata(timestamps)
            line.set_ydata(ax_values)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"Data saved to {filename}")