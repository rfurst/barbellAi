#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // Library for the onboard BMI270 sensor

// BLE Service and Characteristic UUIDs
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic accelChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 20);

void setup() {
  // Initialize IMU
  if (!IMU.begin()) {
    while (1); // Halt program if IMU initialization fails
  }

  // Initialize BLE
  if (!BLE.begin()) {
    while (1); // Halt program if BLE initialization fails
  }

  // Configure BLE
  BLE.setLocalName("Nano33BLE_Accel");
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(accelChar);
  BLE.addService(accelService);

  BLE.advertise(); // Start BLE advertising
}

void loop() {
  // Wait for a central device to connect
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      // Collect acceleration data and send over BLE
      float x, y, z;

      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);

        // Format the data and send via BLE
        char dataBuffer[20];
        snprintf(dataBuffer, sizeof(dataBuffer), "X:%.2f,Y:%.2f,Z:%.2f", x, y, z);
        accelChar.writeValue(dataBuffer);

        delay(100); // Update frequency: 100 ms (10 Hz)
      }
    }
  }
}
