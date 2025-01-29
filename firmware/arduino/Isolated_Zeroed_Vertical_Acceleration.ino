/*
  Arduino Nano 33 BLE Sense (Rev2) + Arduino_BMI270_BMM150
  --------------------------------------------------------
  Sends ONLY the vertical (Z-axis) acceleration, with gravity subtracted
  so that when the board is flat & still (Z-axis up), the reading is ~0 g.

  - Calibration: Averages ~1 g on Z at rest, then subtracts it so output is ~0.
  - BLE: Uses a custom BLE Service & Characteristic. Sends the Z value as a string.
*/

#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>  // Library for the BMI270 & BMM150

// Custom BLE service & characteristic UUIDs
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");

// We'll send the Z-accel as a string. Allow up to 20 bytes for the float text.
BLECharacteristic vertAccelCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214",
                                          BLERead | BLENotify,
                                          20);

float offsetZ = 0.0;                  // Z-axis offset to subtract gravity
const int CALIBRATION_SAMPLES = 200;  // number of samples to take for calibration

void setup() {
  Serial.begin(115200);
  while (!Serial) ;  // wait for Serial Monitor

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("Nano33BLE-VertAccel");
  BLE.setAdvertisedService(accelService);

  // Add our single characteristic for vertical accel
  accelService.addCharacteristic(vertAccelCharacteristic);
  BLE.addService(accelService);

  // Start advertising
  BLE.advertise();
  Serial.println("BLE advertising started...");

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize BMI270/BMM150!");
    while (1);
  }
  Serial.println("IMU initialized.");

  // Calibrate the Z axis so we get ~0 g at rest
  calibrateZAxis();
}

void loop() {
  // Check for BLE central connection
  BLEDevice central = BLE.central();

  if (central) {
    // Once connected, continuously read acceleration
    if (IMU.accelerationAvailable()) {
      float x, y, z;
      IMU.readAcceleration(x, y, z);

      // Remove the gravity offset
      float verticalAccel = z - offsetZ;

      // Convert to string
      char buffer[20];
      snprintf(buffer, sizeof(buffer), "%.4f", verticalAccel);

      // Write the string to the characteristic
      vertAccelCharacteristic.writeValue((uint8_t*)buffer, strlen(buffer));

      // For debugging
      Serial.print("Vertical Accel (g): ");
      Serial.println(buffer);
    }
  }
}

/****************************************************************
  Calibrate the Z-axis:
    - Assume the board is flat, Z-axis up.
    - Collect multiple samples, average them, store in offsetZ.
    - offsetZ ~ 1 g if truly flat & still. Then z-offsetZ ~ 0.
****************************************************************/
void calibrateZAxis() {
  Serial.println("Calibrating Z Axis... Keep the board flat and still.");

  float sumZ = 0.0;

  for (int i = 0; i < CALIBRATION_SAMPLES; i++) {
    while (!IMU.accelerationAvailable());
    float x, y, z;
    IMU.readAcceleration(x, y, z);

    sumZ += z;
    delay(10);
  }

  float avgZ = sumZ / CALIBRATION_SAMPLES;
  offsetZ = avgZ;

  Serial.print("Computed Z offset = ");
  Serial.println(offsetZ, 4);
  Serial.println("Calibration complete. The vertical reading should be ~0 g at rest.");
  Serial.println("-------------------------------------------------------------");
}
