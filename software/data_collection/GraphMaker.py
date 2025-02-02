# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 21:05:11 2025

@author: rkfurst
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1) Save the data above to a CSV file.
#    Ensure the CSV has headers: Timestamp,VerticalAccel_g
#    For example: data.csv

# 2) Read the CSV file
df = pd.read_csv(r"\\deturbansci07\ford\DBI\Project Management\RKF\Ruben_velocityvalidated_5pausedeadlift_NOROLL.csv")

# 3) Optionally sort by timestamp in case any rows are out of order
df.sort_values(by="time_s", inplace=True)

# 4) Plot every point. 
#    scatter() will show each marker. You can also do plot() to connect them with lines.
plt.figure(figsize=(10, 6))
plt.scatter(df["time_s"], df["vertical_accel_g"], s=8, c="blue", alpha=0.7)

# 5) Label axes and show the plot
plt.title("Vertical Acceleration over Time")
plt.xlabel("Timestamp (s)")
plt.ylabel("Vertical Acceleration (g)")
plt.grid(True)
plt.show()