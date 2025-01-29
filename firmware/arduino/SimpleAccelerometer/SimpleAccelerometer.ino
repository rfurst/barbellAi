/*
  Arduino Nano 33 BLE Sense (Rev2) + Madgwick Filter
  --------------------------------------------------
  Fuses accelerometer, gyroscope, and magnetometer data to compute vertical
  acceleration in the *world frame*, accounting for barbell rotation.
*/

#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <MadgwickAHRS.h>  // Install via Library Manager

// BLE Setup (same as before)
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic vertAccelCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214",
                                          BLERead | BLENotify, 20);

Madgwick filter;  // Sensor fusion filter
const float sensorRate = 104.0;  // Hz (BMI270 default accelerometer rate)
float beta = 0.1;  // Madgwick filter parameter (tune for responsiveness vs. stability)

// Gravity in world frame (z-axis)
const float gravity = 9.81;  // m/s²

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Initialize BLE (same as before)
  if (!BLE.begin()) {
    Serial.println("BLE failed!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-VertAccel");
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(vertAccelCharacteristic);
  BLE.addService(accelService);
  BLE.advertise();

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("IMU failed!");
    while (1);
  }

  // Configure Madgwick filter
  filter.begin(sensorRate);
  filter.setBeta(beta);

  Serial.println("Calibrating... Keep device still for 1 second.");
  delay(1000);  // Let IMU stabilize
}

void loop() {
  static unsigned long prevMillis = 0;
  BLEDevice central = BLE.central();

  if (central && IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
    // Read sensors
    float aX, aY, aZ;  // Accelerometer (m/s²)
    float gX, gY, gZ;  // Gyroscope (rad/s)
    float mX, mY, mZ;  // Magnetometer (μT)
    IMU.readAcceleration(aX, aY, aZ);
    IMU.readGyroscope(gX, gY, gZ);
    IMU.readMagneticField(mX, mY, mZ);

    // Convert gyro to rad/s (BMI270 reports in degrees/s)
    gX *= DEG_TO_RAD;
    gY *= DEG_TO_RAD;
    gZ *= DEG_TO_RAD;

    // Update Madgwick filter with sensor data
    filter.update(gX, gY, gZ, aX, aY, aZ, mX, mY, mZ);

    // Get orientation quaternion
    float q0, q1, q2, q3;
    filter.getQuaternion(&q0, &q1, &q2, &q3);

    // Rotate accelerometer to world frame
    float worldAccelX = (1 - 2*q2*q2 - 2*q3*q3)*aX + (2*q1*q2 - 2*q0*q3)*aY + (2*q1*q3 + 2*q0*q2)*aZ;
    float worldAccelY = (2*q1*q2 + 2*q0*q3)*aX + (1 - 2*q1*q1 - 2*q3*q3)*aY + (2*q2*q3 - 2*q0*q1)*aZ;
    float worldAccelZ = (2*q1*q3 - 2*q0*q2)*aX + (2*q2*q3 + 2*q0*q1)*aY + (1 - 2*q1*q1 - 2*q2*q2)*aZ;

    // Subtract gravity (world Z-axis) to get linear acceleration
    float linearVertAccel = worldAccelZ - gravity;

    // Convert to string and send via BLE
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%.4f", linearVertAccel);
    vertAccelCharacteristic.writeValue((uint8_t*)buffer, strlen(buffer));

    // Debug output
    Serial.print("Vertical Accel (m/s²): ");
    Serial.println(linearVertAccel);
  }

  // Control loop rate (adjust based on your sensor rate)
  while (millis() - prevMillis < (1000 / sensorRate));
  prevMillis = millis();
}