/*
   Single-Axis IMU with BLE Sending "millis,vertical_accel_g"
   ----------------------------------------------------------
   1) Reads the BMI270 accelerometer & gyroscope on an Arduino Nano 33 BLE Sense (or similar).
   2) Uses a single-axis complementary filter to track rotation about X (the bar axis).
   3) Rotates local (y,z) into "vertical" to isolate motion from gravity.
   4) Subtracts gravity and a centripetal term (if sensor offset).
   5) Sends "<millis>,<vertical_accel_g>" via BLE notifications.

   Requirements:
     - Arduino Nano 33 BLE Sense (with BMI270_BMM150).
     - Libraries: 
         * Arduino_BMI270_BMM150
         * ArduinoBLE
     - Python side expects CSV string: "millis,accel_g"
*/

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // for accelerometer/gyro

// ---------- BLE SERVICE/CHARACTERISTIC UUIDS -----------
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic accelChar(
    "19B10001-E8F2-537E-4F6C-D104768A1214", 
    BLERead | BLENotify, 
    20 // enough bytes to hold our CSV string
);

// ---------- COMPLEMENTARY FILTER SETTINGS --------------
static const float LOOP_HZ           = 100.0f; // target update rate ~100 Hz
static const float COMPLEMENTARY_ALPHA = 0.02f; 
  // small alpha => trust gyro more, big alpha => trust accel more

// ---------- SENSOR OFFSET FROM ROTATION AXIS -----------
static const float rSensorZ = 0.053f; // e.g. sensor offset by +5.3 cm from bar axis in local Z

// ---------- GLOBALS FOR TRACKING -----------------------
static float angleX_deg = 0.0f;      // single-axis angle about X
static unsigned long prevMillis = 0; // to keep track of loop timing

// ---------- CONSTANTS ----------------------------------
static const float G_MAG = 1.0f;     // we measure acceleration in g
static const float DEG_TO_RAD = PI / 180.0f;

// -------------------------------------------------------

void setup()
{
  Serial.begin(115200);
  while(!Serial) { /* wait for Serial Monitor to open */ }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-SingleAxis"); // so we can find it easily
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(accelChar);
  BLE.addService(accelService);
  BLE.advertise();

  Serial.println("BLE advertising started.");

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize BMI270/BMM150 IMU!");
    while (1);
  }
  Serial.println("IMU started.");

  prevMillis = millis();
  Serial.println("Setup complete.");
}

//
// Helper function: single-axis complementary filter
//
void updateAngleXComplementary(float gx_deg_s, float ay_g, float az_g, float dt_s, float &angle_deg_inout)
{
  // 1) Integrate gyro for angle
  float angle_gyro = angle_deg_inout + (gx_deg_s * dt_s);

  // 2) Acc-based angle about X = -atan2(ay, az) in degrees
  float angle_acc_rad = -atan2(ay_g, az_g);
  float angle_acc_deg = angle_acc_rad * (180.0f / PI);

  // 3) Combine
  float fused = (1.0f - COMPLEMENTARY_ALPHA) * angle_gyro
                + COMPLEMENTARY_ALPHA * angle_acc_deg;

  angle_deg_inout = fused;
}

//
// Helper function: rotate local (y,z) by -angleX => vertical component
//   verticalAccel = cos(angleX)*localZ + sin(angleX)*localY
//
float rotateYzToVertical(float localY, float localZ, float angleX_deg)
{
  float ax_rad = angleX_deg * DEG_TO_RAD;
  float c = cosf(ax_rad);
  float s = sinf(ax_rad);
  return c*localZ + s*localY;
}

void loop()
{
  // Let BLE process inbound connection requests, etc.
  BLE.poll();

  // Try to run at ~100 Hz
  unsigned long now = millis();
  if (now - prevMillis < 10) {
    return;
  }
  float dt_s = (now - prevMillis)*1e-3; // convert ms->seconds
  prevMillis = now;

  // Read sensor
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax_g, ay_g, az_g;
    float gx_deg_s, gy_deg_s, gz_deg_s;

    IMU.readAcceleration(ax_g, ay_g, az_g);   // in g
    IMU.readGyroscope(gx_deg_s, gy_deg_s, gz_deg_s); // in deg/s

    // 1) Complementary filter for angle about X
    updateAngleXComplementary(gx_deg_s, ay_g, az_g, dt_s, angleX_deg);

    // 2) Rotate local (y,z) to find net vertical accel
    float verticalAccel_g = rotateYzToVertical(ay_g, az_g, angleX_deg);

    // 3) Subtract gravity => net acceleration
    verticalAccel_g -= G_MAG;

    // 4) Subtract centripetal if sensor offset in Z
    float wX_rad_s = gx_deg_s * DEG_TO_RAD;
    float centrip_m_s2 = wX_rad_s * wX_rad_s * rSensorZ; // in m/s^2
    float centrip_g = centrip_m_s2 / 9.81f; // convert to g

    // offset is along +Z => centripetal along ±Y => rotate that Y into vertical
    float localY_centrip = centrip_g; 
    float verticalCentrip_g = rotateYzToVertical(localY_centrip, 0.0f, angleX_deg);
    verticalAccel_g -= verticalCentrip_g;

    // 5) Send <millis>,<verticalAccel_g> over BLE
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%lu,%.4f", now, verticalAccel_g);
    accelChar.writeValue((uint8_t*)buffer, strlen(buffer));

    // Debug
    Serial.print("t="); Serial.print(now);
    Serial.print(" ms, angleX="); Serial.print(angleX_deg, 2);
    Serial.print(", rawAcc=["); Serial.print(ax_g,2); Serial.print(", ");
    Serial.print(ay_g,2);      Serial.print(", ");
    Serial.print(az_g,2);      Serial.print("], vertAccel=");
    Serial.println(verticalAccel_g, 4);
  }
}
