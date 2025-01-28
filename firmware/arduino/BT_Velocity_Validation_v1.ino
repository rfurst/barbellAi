#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // Library for the onboard BMI270 sensor
#include <MadgwickAHRS.h>          // Sensor fusion library for orientation correction

// BLE Service and Characteristic UUIDs
BLEService velocityService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic velocityChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 50);

Madgwick filter;                // Sensor fusion filter
float velocity[3] = {0, 0, 0};  // Velocity in X, Y, Z
float accelBias[3] = {0, 0, 0}; // Acceleration bias correction
float lastAccel[3] = {0, 0, 0}; // Last accelerometer readings (filtered)
unsigned long prevTime = 0;     // Previous timestamp for time delta calculation
const float alpha = 0.8;        // Low-pass filter coefficient

void calibrateBias() {
  float sum[3] = {0, 0, 0};
  int samples = 100;

  for (int i = 0; i < samples; i++) {
    if (IMU.accelerationAvailable()) {
      float x, y, z;
      IMU.readAcceleration(x, y, z);
      sum[0] += x;
      sum[1] += y;
      sum[2] += z;
    }
    delay(10);
  }

  accelBias[0] = sum[0] / samples;
  accelBias[1] = sum[1] / samples;
  accelBias[2] = sum[2] / samples;
}

void setup() {
  // Initialize IMU
  if (!IMU.begin()) {
    while (1);
  }

  // Calibrate accelerometer bias
  calibrateBias();

  // Initialize BLE
  if (!BLE.begin()) {
    while (1);
  }

  // Configure BLE
  BLE.setLocalName("Nano33BLE_Velocity");
  BLE.setAdvertisedService(velocityService);
  velocityService.addCharacteristic(velocityChar);
  BLE.addService(velocityService);

  BLE.advertise(); // Start BLE advertising

  // Initialize the Madgwick filter for sensor fusion
  filter.begin(50); // Sampling frequency in Hz
  prevTime = micros(); // Initialize the timestamp with microsecond precision
}

void loop() {
  // Wait for a central device to connect
  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      // Check if IMU data is available
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        float ax, ay, az, gx, gy, gz;

        // Read IMU data
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);

        // Apply bias correction and low-pass filter
        ax = alpha * lastAccel[0] + (1 - alpha) * (ax - accelBias[0]);
        ay = alpha * lastAccel[1] + (1 - alpha) * (ay - accelBias[1]);
        az = alpha * lastAccel[2] + (1 - alpha) * (az - accelBias[2]);
        lastAccel[0] = ax;
        lastAccel[1] = ay;
        lastAccel[2] = az;

        // Update Madgwick filter for orientation correction
        filter.updateIMU(gx, gy, gz, ax, ay, az);

        // Calculate velocity using integration
        unsigned long currentTime = micros();
        float dt = (currentTime - prevTime) / 1e6; // Time delta in seconds
        prevTime = currentTime;

        velocity[0] += ax * dt;
        velocity[1] += ay * dt;
        velocity[2] += az * dt;

        // Send velocity data over BLE
        char velocityData[50];
        snprintf(velocityData, sizeof(velocityData), "%.2f,%.2f,%.2f", velocity[0], velocity[1], velocity[2]);
        velocityChar.writeValue(velocityData);
      }
    }
  }
}

