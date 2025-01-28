#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <MadgwickAHRS.h>

BLEService velocityService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic velocityChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify, 12); // Store 3 floats

Madgwick filter;
float velocity[3] = {0, 0, 0};
float accelBias[3] = {0, 0, 0};
float gyroBias[3] = {0, 0, 0};
unsigned long prevTime = 0;
const float alpha = 0.90; // More aggressive low-pass
const float beta = 0.98;  // Complementary filter for drift reduction
const float G = 9.80665;  // Standard gravity

void calibrateSensors() {
  const int samples = 500;
  float aSum[3] = {0}, gSum[3] = {0};

  for (int i = 0; i < samples; i++) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      float ax, ay, az, gx, gy, gz;
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      
      aSum[0] += ax;
      aSum[1] += ay;
      aSum[2] += az;
      gSum[0] += gx;
      gSum[1] += gy;
      gSum[2] += gz;
    }
    delayMicroseconds(500);
  }

  for (int i = 0; i < 3; i++) {
    accelBias[i] = aSum[i] / samples;
    gyroBias[i] = gSum[i] / samples;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("IMU failure!");
    while (1);
  }

  // Configure IMU parameters (modify based on your sensor's capabilities)
  IMU.setAccelerationRange(4);  // ±4g
  IMU.setGyroscopeRange(500);   // ±500°/s
  IMU.setAccelerationODR(BMI270_ODR_100Hz);
  IMU.setGyroscopeODR(BMI270_ODR_100Hz);

  calibrateSensors();
  filter.begin(100); // Match sample rate to IMU ODR

  if (!BLE.begin()) {
    Serial.println("BLE failure!");
    while (1);
  }

  BLE.setLocalName("Nano33BLE_Velocity");
  BLE.setAdvertisedService(velocityService);
  velocityService.addCharacteristic(velocityChar);
  BLE.addService(velocityService);
  BLE.advertise();

  prevTime = micros();
}

void loop() {
  BLEDevice central = BLE.central();
  if (central && central.connected()) {
    while (central.connected()) {
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        // Raw sensor readings
        float ax, ay, az, gx, gy, gz;
        IMU.readAcceleration(ax, ay, az);
        IMU.readGyroscope(gx, gy, gz);

        // Bias correction
        ax = (ax - accelBias[0]) * G; // Convert to m/s²
        ay = (ay - accelBias[1]) * G;
        az = (az - accelBias[2]) * G;
        gx -= gyroBias[0];
        gy -= gyroBias[1];
        gz -= gyroBias[2];

        // Update orientation filter
        filter.updateIMU(gx, gy, gz, ax, ay, az);

        // Get gravity vector in sensor frame
        float qw, qx, qy, qz;
        filter.getQuaternion(qw, qx, qy, qz);
        float gravityX = 2 * (qx*qz - qw*qy);
        float gravityY = 2 * (qw*qx + qy*qz);
        float gravityZ = qw*qw - qx*qx - qy*qy + qz*qz;

        // Subtract gravity from accelerometer readings
        float linearAccel[3] = {
          ax - gravityX * G,
          ay - gravityY * G,
          az - gravityZ * G
        };

        // Apply low-pass filter to linear acceleration
        static float filteredAccel[3] = {0};
        for (int i = 0; i < 3; i++) {
          filteredAccel[i] = alpha * filteredAccel[i] + (1 - alpha) * linearAccel[i];
        }

        // Time calculation with overflow protection
        unsigned long currentTime = micros();
        float dt = (currentTime - prevTime) / 1e6;
        prevTime = currentTime;

        // Integrate acceleration with drift compensation
        for (int i = 0; i < 3; i++) {
          velocity[i] += filteredAccel[i] * dt;
          // Apply high-pass characteristic to reduce drift
          velocity[i] *= beta; 
        }

        // Send binary float data (more efficient than string)
        velocityChar.writeValue(velocity, sizeof(velocity));
      }
    }
    // Reset velocity when disconnected
    memset(velocity, 0, sizeof(velocity));
  }
}
