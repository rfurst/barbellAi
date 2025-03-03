# Barbell IMU Processing Pipeline

A modular framework for processing barbell IMU data, with components for physics processing, rep detection, and velocity calculation.

## Project Structure

```
software/data_processing/barbell_pipeline/
├── core/
│   └── interfaces.py      # Core interfaces and data classes
├── physics/
│   ├── deadlift_processor.py  # Physics processing for deadlifts
│   └── velocity_calculator.py # Velocity calculation with drift correction
├── ml/
│   └── deadlift_detector.py   # Deadlift rep detection
├── data/
│   └── data_loader.py     # Data loading utilities
└── main.py               # Main script for running the pipeline
```

## Features

- **Physics Processing**: 
  - Advanced complementary filter for orientation tracking
  - Stillness detection and gyro bias estimation
  - Gravity compensation
  - Signal filtering and smoothing

- **Rep Detection**:
  - State machine approach for deadlift phase detection
  - Quality metrics for rep validation
  - Configurable thresholds for different movement patterns

- **Velocity Calculation**:
  - Integration with drift correction
  - Signal smoothing
  - Peak velocity detection

## Data Format

### IMU Data Files
Expected CSV format with columns:
- `tMillis`: Timestamp in milliseconds
- `ax`, `ay`, `az`: Accelerometer data
- `gx`, `gy`, `gz`: Gyroscope data

### Label Files
JSON format with structure:
```json
{
  "reps": [
    {
      "rep_number": 1,
      "peak_velocity": 0.88
    }
  ],
  "source_imu_file": "imu_data_example.csv",
  "source_velocity_file": "velocity_validated.xlsx"
}
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r software/data_processing/barbell_pipeline/requirements.txt
```

## Usage

### Training the Pipeline
```bash
python -m software.data_processing.barbell_pipeline.main train \
    --data-dir path/to/data \
    --model-dir path/to/models
```

### Processing New Data
```bash
python -m software.data_processing.barbell_pipeline.main process \
    --input path/to/imu_data.csv \
    --model-dir path/to/models
```

## Customization

The pipeline is designed to be modular. You can extend or replace any component by implementing the relevant interface:

- `PhysicsProcessor`: For custom physics processing
- `RepDetector`: For custom rep detection algorithms
- `TrapzVelocityCalculator`: For custom velocity calculation methods

### Example: Custom Rep Detector

```python
from barbell_pipeline.core.interfaces import RepDetector, ProcessedData, RepData

class CustomRepDetector(RepDetector):
    def detect_reps(self, processed_data: ProcessedData) -> list[RepData]:
        # Your custom rep detection logic here
        pass
```

## Parameters

### Physics Processor
- `gravity_sign`: Sign of gravity compensation (-1.0 or 1.0)
- `sensor_offset`: (x, y, z) offset of sensor from center of mass
- `stillness_acc_std`: Standard deviation threshold for accelerometer stillness
- `stillness_gyro_std`: Standard deviation threshold for gyroscope stillness
- `cf_still_alpha`: Complementary filter alpha during stillness
- `cf_pull_alpha`: Complementary filter alpha during pulling

### Rep Detector
- `start_acc_threshold`: Acceleration threshold for rep start
- `concentric_min_vel`: Minimum velocity for concentric phase
- `peak_vel_threshold`: Threshold for peak velocity detection
- `min_rep_duration`: Minimum duration for valid rep
- `max_rep_duration`: Maximum duration for valid rep

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 