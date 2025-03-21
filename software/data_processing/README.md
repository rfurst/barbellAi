# Barbell Pipeline

A modular framework for processing IMU data from a smart barbell clip to detect reps and calculate velocities.

## Overview

The pipeline consists of three main components:

1. **Physics Processing**: Converts raw IMU data to world frame coordinates
   - Uses complementary filter for orientation tracking
   - Handles gravity compensation
   - Provides stable world frame acceleration data

2. **Rep Detection**: ML-based approach to identify individual reps
   - Learns from labeled training data
   - Uses peak detection with adaptive parameters
   - Identifies rep boundaries accurately

3. **Velocity Calculation**: Computes peak velocity for each rep
   - Integrates acceleration during concentric phase
   - Applies drift correction
   - Handles edge cases and noise

## Data Format

### Input Data Format
Raw IMU data should be in CSV format with the following columns:
- `timestamp`: Time in seconds
- `accel_x`, `accel_y`, `accel_z`: Accelerometer readings in m/s²
- `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope readings in rad/s

Example:
```csv
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0.000,-0.123,9.81,0.456,0.001,-0.002,0.003
...
```

### Label Format
Labels should be in JSON format with the following structure:
```json
{
  "reps": [
    {
      "start_idx": 100,
      "end_idx": 200,
      "peak_velocity": 1.5
    },
    ...
  ]
}
```

### File Naming Convention
- Raw data: `*_imu.csv`
- Labels: `*_labels.json` (matching prefix with raw data)

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training the Pipeline
```bash
python -m barbell_pipeline.main train --data-dir /path/to/data --model-dir /path/to/save/models
```

This will:
1. Load all training data pairs (`*_imu.csv` and `*_labels.json`)
2. Train the rep detector on the labeled data
3. Save the trained model to the specified directory

### Processing New Data
```bash
python -m barbell_pipeline.main process --input /path/to/new_data.csv --model-dir /path/to/models
```

This will:
1. Load the trained model
2. Process the new IMU data
3. Output detected reps and their peak velocities

## Components

### PhysicsProcessor
- `ComplementaryFilterProcessor`: Fuses accelerometer and gyroscope data
  - `alpha`: Weight for gyroscope integration (default: 0.96)
  - `dt`: Time step between measurements (default: 1/100 s)

### RepDetector
- `MLRepDetector`: Machine learning based rep detection
  - `window_size`: Smoothing window size (default: 25)
  - `prominence`: Peak prominence threshold (learned from data)
  - `distance`: Minimum distance between peaks (learned from data)

### VelocityCalculator
- `TrapzVelocityCalculator`: Trapezoidal integration for velocity
  - `drift_correction`: Whether to apply drift correction (default: True)

## Extending the Pipeline

### Adding New Physics Processors
Implement the `PhysicsProcessor` interface:
```python
class CustomPhysicsProcessor(PhysicsProcessor):
    def process_orientation(self, imu_data: ImuData) -> ProcessedData:
        # Your implementation here
        pass
```

### Adding New Rep Detectors
Implement the `RepDetector` interface:
```python
class CustomRepDetector(RepDetector):
    def detect_reps(self, processed_data: ProcessedData) -> List[RepData]:
        # Your implementation here
        pass
```

### Adding New Velocity Calculators
Implement the `VelocityCalculator` interface:
```python
class CustomVelocityCalculator(VelocityCalculator):
    def calculate_velocity(self, processed_data: ProcessedData, rep: RepData) -> float:
        # Your implementation here
        pass
``` 