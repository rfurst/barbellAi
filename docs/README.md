# BarbellAI: Smart Barbell Tracker *(Internal Repository)*

**Proprietary Project**  
*This repository contains confidential code and documentation for the BarbellAI smart barbell tracking system. Access is restricted to authorized team members only.*

---

## Project Overview
BarbellAI is a hardware-software system that attaches to barbells to track velocity, acceleration, and repetitions during weightlifting. It provides real-time feedback and analytics for athletes and trainers, including 1RM estimation and progress tracking. This repository houses all code, schematics, and documentation for development.

### Key Goals
1. Build a hardware prototype for motion tracking.
2. Develop a companion mobile app with real-time BLE data streaming.
3. Implement machine learning for rep detection and exercise classification.
4. Create a dashboard for trainers to monitor athlete performance.

---

## Features
### Hardware
- **Arduino Nano 33 BLE Sense**: Handles IMU data collection and BLE communication.
- **3D-Printed Enclosure**: Securely attaches to barbells (design files in `/hardware/3d_models`).
- **LiPo Battery & Charging**: TP4056 module for safe battery management.

### Firmware
- **IMU Data Processing**: Real-time acceleration/velocity calculations.
- **BLE Communication**: Transmits data to the mobile app.
- **TensorFlow Lite Model**: Edge inference for rep detection and exercise classification.

### Software
- **Mobile App (React Native)**: Real-time dashboard for athletes and trainers.
- **Backend (Firebase)**: Stores user profiles, session data, and analytics.
- **Data Pipeline**: Python scripts for training ML models (`/machine_learning`).

---

## Repository Structure
BarbellAI/
├── hardware/ # Circuit schematics, 3D models, BOM
├── firmware/ # Arduino code, TFLite models, libraries
├── software/ # Mobile app, backend, data collection scripts
├── machine_learning/ # Jupyter notebooks, datasets, trained models
├── docs/ # Internal documentation
│ ├── hardware_setup.md # Assembly and testing guide
│ ├── firmware_guide.md # Flashing and calibration
│ └── api_specs.md # BLE/service protocols
└── .gitignore

---

## Documentation
### For Developers
1. **Hardware Setup**:  
   See [`/docs/hardware_setup.md`](docs/hardware_setup.md) for:  
   - Circuit assembly instructions  
   - 3D printing guidelines  
   - Power management testing  

2. **Firmware Workflow**:  
   - Flash Arduino code using PlatformIO (configuration in `/firmware/platformio.ini`)  
   - Calibrate IMU offsets using [`/firmware/calibration_tool`](firmware/calibration_tool)  

3. **Mobile App**:  
   - Set up React Native environment (see [`/software/mobile_app/README.md`](software/mobile_app/README.md))  
   - Configure Firebase keys in `.env`  

4. **ML Model Training**:  
   - Follow steps in [`/machine_learning/train_model.ipynb`](machine_learning/train_model.ipynb)  
   - Export models to TFLite and update `/firmware/tflite_models`  

---

## Development Team
### Roles & Access
| Name             | Role                        | Permissions       |
|------------------|-----------------------------|-------------------|
| Matthew Pacas    | CEO                         | Admin             |
| Jacob Handloser  | CMO                         | Read/Write        |
| Ruben Furst      | CTO                         | Read/Write        |
| Aidan O'Boyle    | CFO                         | Read/Write        |

### Workflow
1. **Branching**: Use `feature/` or `bugfix/` prefixes.  
2. **Code Reviews**: Required for merges to `main`.  
3. **Issues**: Label with `hardware`, `firmware`, or `software`.  

---


*Last Updated: January 2025*  