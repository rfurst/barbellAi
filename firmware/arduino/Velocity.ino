#include <Arduino_BMI270_BMM150.h> // IMU library for the BMI270/BMM150 sensor
#include <MadgwickAHRS.h>          // Sensor fusion library for orientation correction

Madgwick filter;                // Sensor fusion filter
float velocity[3] = {0, 0, 0};  // Velocity in X, Y, Z
float accelBias[3] = {0, 0, 0}; // Acceleration bias correction (can be tuned)
unsigned long prevTime = 0;     // Previous timestamp for time delta calculation

void setup() {
  Serial.begin(115200); // Start serial communication
  while (!Serial);

  // Initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Initialize the Madgwick filter for sensor fusion
  filter.begin(50); // Sampling frequency in Hz (adjust based on your application)

  prevTime = millis(); // Initialize the timestamp

  // Print CSV header to Serial
  Serial.println("Timestamp,VelocityX,VelocityY,VelocityZ");
}

void loop() {
  // Check if IMU data is available
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    // Read acceleration and gyroscope data
    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // Correct orientation using the Madgwick filter
    unsigned long currTime = millis();
    float dt = (currTime - prevTime) / 1000.0; // Convert milliseconds to seconds
    prevTime = currTime;

    // Update the filter with gyroscope and accelerometer data
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    float roll = filter.getRoll();
    float pitch = filter.getPitch();
    float yaw = filter.getYaw();

    // Align acceleration to the global frame
    float accelGlobal[3];
    alignAccelerationToGlobal(ax, ay, az, roll, pitch, yaw, accelGlobal);

    // Remove acceleration bias (if any)
    accelGlobal[0] -= accelBias[0];
    accelGlobal[1] -= accelBias[1];
    accelGlobal[2] -= accelBias[2];

    // Integrate acceleration to compute velocity
    velocity[0] += accelGlobal[0] * dt;
    velocity[1] += accelGlobal[1] * dt;
    velocity[2] += accelGlobal[2] * dt;

    // Output the timestamp and velocity data in CSV format
    Serial.print(currTime);             // Timestamp in milliseconds
    Serial.print(",");
    Serial.print(velocity[0], 4);       // Velocity X with 4 decimal places
    Serial.print(",");
    Serial.print(velocity[1], 4);       // Velocity Y
    Serial.print(",");
    Serial.println(velocity[2], 4);     // Velocity Z

    delay(50); // Adjust to match your desired sampling rate (e.g., 50Hz)
  }
}

// Function to align acceleration to the global frame
void alignAccelerationToGlobal(float ax, float ay, float az, float roll, float pitch, float yaw, float* accelGlobal) {
  // Compute rotation matrix from roll, pitch, yaw
  float cosR = cos(roll), sinR = sin(roll);
  float cosP = cos(pitch), sinP = sin(pitch);
  float cosY = cos(yaw), sinY = sin(yaw);

  float R[3][3] = {
    {cosP * cosY, cosY * sinP * sinR - sinY * cosR, cosY * sinP * cosR + sinY * sinR},
    {cosP * sinY, sinY * sinP * sinR + cosY * cosR, sinY * sinP * cosR - cosY * sinR},
    {-sinP, cosP * sinR, cosP * cosR}
  };

  // Rotate acceleration into the global frame
  accelGlobal[0] = R[0][0] * ax + R[0][1] * ay + R[0][2] * az;
  accelGlobal[1] = R[1][0] * ax + R[1][1] * ay + R[1][2] * az;
  accelGlobal[2] = R[2][0] * ax + R[2][1] * ay + R[2][2] * az;
}
