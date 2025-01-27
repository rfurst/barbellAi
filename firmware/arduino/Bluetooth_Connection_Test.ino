#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // Library for the onboard BMI270 sensor

// BLE Service and Characteristic UUIDs
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic accelChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 20);

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }

  // Configure BLE
  BLE.setLocalName("Nano33BLE_Accel");
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(accelChar);
  BLE.addService(accelService);

  BLE.advertise();
  Serial.println("BLE device is now advertising...");
}

void loop() {
  // Wait for a central device to connect
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to central device: ");
    Serial.println(central.address());

    while (central.connected()) {
      // Collect acceleration data and send over BLE
      float x, y, z;

      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);

        char dataBuffer[20];
        snprintf(dataBuffer, sizeof(dataBuffer), "X:%.2f,Y:%.2f,Z:%.2f", x, y, z);
        accelChar.writeValue(dataBuffer);

        Serial.println(dataBuffer);
        delay(100); // Adjust the update frequency as needed
      }
    }

    Serial.println("Disconnected from central device.");
  }
}
