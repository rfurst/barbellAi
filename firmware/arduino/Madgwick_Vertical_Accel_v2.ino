/*
  Off-Axis IMU Correction with Official Arduino Madgwick Library
  --------------------------------------------------------------
  1) Uses only accelerometer + gyro (no magnetometer).
  2) Madgwick internal sampling freq = 100 Hz (approx).
  3) We transform data to Earth frame via roll/pitch/yaw (Euler angles).
  4) Remove gravity, then subtract centripetal & tangential accelerations 
     due to bar rotation if the sensor is 5.3 cm off-axis.

  NOTE: Because the library's quaternion is private and updateIMU()
        has no dt parameter, we do a best-effort approach with Euler angles.
        Also, 'beta' is private, so we rely on library defaults.

  Requirements:
    - Arduino_BMI270_BMM150
    - ArduinoBLE
    - Madgwick (official Arduino library, which has updateIMU(gx,gy,gz,ax,ay,az))
*/

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>  // for accelerometer/gyro
#include <MadgwickAHRS.h>           // official Arduino Madgwick

// ---------------------- USER SETTINGS --------------------------

// Desired loop/filter frequency (Hz). Must match your actual loop rate fairly closely:
static const float SAMPLE_FREQ = 100.0f;

// If your sensor is physically 5.3 cm ABOVE the bar's axis along 
// the sensor's local +Z, you might define rSensor = (0,0,+0.053).
// But if local +Z is "upward away from the bar," that means the bar is below
// the sensor along -Z, so then you'd do (0,0,-0.053). Adjust accordingly!
float rSensorX = 0.0f;
float rSensorY = 0.0f;
float rSensorZ = +0.053f; // Example: +Z is above the bar by 5.3 cm

// Assume Earth-frame Z is +1 g at rest. If your rest reading is -1 g, use -1.0f
static const float GRAVITY_SIGN = +1.0f;

// ---------------------- BLE SETUP -----------------------------
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic verticalAccelChar("19B10001-E8F2-537E-4F6C-D104768A1214",
                                    BLERead | BLENotify,
                                    20); // sending a string float

// ---------------------- GLOBALS -------------------------------
Madgwick filter; 
unsigned long lastLoopTime = 0; // for measuring loop rate if desired

// We'll store the Earth-frame gyro from the previous loop to estimate alpha
float oldOmegaEx = 0.0f, oldOmegaEy = 0.0f, oldOmegaEz = 0.0f;

// -------------- EULER ANGLE -> ROTATION MATRIX ----------------
// The official Arduino Madgwick library defines roll (x), pitch (y), yaw (z) in degrees.
//
// We'll define a function that takes RPY in degrees, builds the matrix for 
// Rz(yaw)*Ry(pitch)*Rx(roll), and multiplies it by a local vector (vx, vy, vz).
// This yields the Earth-frame vector (ex, ey, ez).

void rotateVectorByEuler(float vx, float vy, float vz,
                         float rollDeg, float pitchDeg, float yawDeg,
                         float &ex, float &ey, float &ez)
{
  // Convert degrees to radians
  float rollRad  = rollDeg  * DEG_TO_RAD;
  float pitchRad = pitchDeg * DEG_TO_RAD;
  float yawRad   = yawDeg   * DEG_TO_RAD;

  // Precompute sines & cosines
  float cr = cosf(rollRad);
  float sr = sinf(rollRad);
  float cp = cosf(pitchRad);
  float sp = sinf(pitchRad);
  float cy = cosf(yawRad);
  float sy = sinf(yawRad);

  // We'll define the combined rotation:
  //   Rz(yaw)*Ry(pitch)*Rx(roll)*[vx, vy, vz]
  // In column-major form (assuming vectors are column).

  // Step 1: v1 = Rx(roll)*v
  float v1x = vx;
  float v1y = cr*vy - sr*vz;
  float v1z = sr*vy + cr*vz;

  // Step 2: v2 = Ry(pitch)*v1
  float v2x =  cp*v1x + sp*v1z;
  float v2y =  v1y;
  float v2z = -sp*v1x + cp*v1z;

  // Step 3: v3 = Rz(yaw)*v2
  float v3x =  cy*v2x - sy*v2y;
  float v3y =  sy*v2x + cy*v2y;
  float v3z =  v2z;

  ex = v3x;
  ey = v3y;
  ez = v3z;
}

// -------------- SETUP -----------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial);

  // BLE
  if (!BLE.begin()) {
    Serial.println("Failed to init BLE!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-OffAxis");
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(verticalAccelChar);
  BLE.addService(accelService);
  BLE.advertise();
  Serial.println("BLE advertising started.");

  // IMU
  if (!IMU.begin()) {
    Serial.println("Failed to init BMI270/BMM150!");
    while (1);
  }
  Serial.println("IMU started.");

  // Initialize Madgwick
  // We assume ~100 Hz loop. Adjust if your actual loop is different.
  filter.begin(SAMPLE_FREQ);

  // Can't set 'beta' directly because it's private in the official library.
  // We'll rely on the library's internal calculation.

  lastLoopTime = millis();
  Serial.println("Setup complete.");
}

// -------------- MAIN LOOP --------------------------------------
void loop() {
  // keep BLE alive
  BLEDevice central = BLE.central();

  // Attempt a stable ~100 Hz. (Optional: you can use delay(10) or so.)
  static unsigned long prevMillis = 0;
  unsigned long nowMillis = millis();
  if (nowMillis - prevMillis < 10) {
    return; // ~100 Hz
  }
  prevMillis = nowMillis;

  // Check if new data is available
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax, ay, az;
    float gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // The official Madgwick library wants gyro in deg/s (which we have).
    // So no need to convert to rad/s.

    // 1) Update filter
    // No magnetometer used => updateIMU
    filter.updateIMU(gx, gy, gz, ax, ay, az);

    // 2) Get Euler angles from the filter
    float rollDeg  = filter.getRoll();   // rotation about X, degrees
    float pitchDeg = filter.getPitch();  // rotation about Y, degrees
    float yawDeg   = filter.getYaw();    // rotation about Z, degrees

    // 3) Rotate raw acceleration to Earth frame
    float eAx, eAy, eAz;
    rotateVectorByEuler(ax, ay, az, rollDeg, pitchDeg, yawDeg, eAx, eAy, eAz);

    // 4) Remove gravity from Earth frame
    float aNoGravX = eAx;
    float aNoGravY = eAy;
    float aNoGravZ = eAz - GRAVITY_SIGN;

    // 5) Rotate IMU gyro to Earth frame (to get Earth-frame angular velocity)
    float omegaEx, omegaEy, omegaEz;
    rotateVectorByEuler(gx, gy, gz, rollDeg, pitchDeg, yawDeg, omegaEx, omegaEy, omegaEz);

    // 6) Approximate angular acceleration alphaE = [omegaE - oldOmegaE]*SAMPLE_FREQ
    static bool firstLoop = true;
    float alphaEx = 0.0f, alphaEy = 0.0f, alphaEz = 0.0f;
    if (!firstLoop) {
      alphaEx = (omegaEx - oldOmegaEx) * SAMPLE_FREQ;
      alphaEy = (omegaEy - oldOmegaEy) * SAMPLE_FREQ;
      alphaEz = (omegaEz - oldOmegaEz) * SAMPLE_FREQ;
    } else {
      // skip on first loop to avoid large spike
      firstLoop = false;
    }

    // store for next iteration
    oldOmegaEx = omegaEx;
    oldOmegaEy = omegaEy;
    oldOmegaEz = omegaEz;

    // 7) Rotate the sensor offset vector rSensor to Earth frame
    float rEx, rEy, rEz;
    rotateVectorByEuler(rSensorX, rSensorY, rSensorZ, rollDeg, pitchDeg, yawDeg, rEx, rEy, rEz);

    // 8) Compute centripetal: aC = -omegaE x (omegaE x rE)
    //    cross(omegaE, rE) => temp
    float tempX = omegaEy*rEz - omegaEz*rEy;
    float tempY = omegaEz*rEx - omegaEx*rEz;
    float tempZ = omegaEx*rEy - omegaEy*rEx;

    // then cross(omegaE, temp) => aC, then negate
    float aCx = -(omegaEy*tempZ - omegaEz*tempY);
    float aCy = -(omegaEz*tempX - omegaEx*tempZ);
    float aCz = -(omegaEx*tempY - omegaEy*tempX);

    // 9) Compute tangential: aT = alphaE x rE
    float aTx = alphaEy*rEz - alphaEz*rEy;
    float aTy = alphaEz*rEx - alphaEx*rEz;
    float aTz = alphaEx*rEy - alphaEy*rEx;

    // 10) Subtract them from aNoGrav => final corrected
    float aCorrectedX = aNoGravX - (aCx + aTx);
    float aCorrectedY = aNoGravY - (aCy + aTy);
    float aCorrectedZ = aNoGravZ - (aCz + aTz);

    // 11) The vertical acceleration is Earth-frame Z
    float verticalAccel = aCorrectedZ;

    // Send via BLE
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%.4f", verticalAccel);
    verticalAccelChar.writeValue((uint8_t*)buffer, strlen(buffer));

    // Debug
    Serial.print("VertAccel(g)= "); Serial.print(verticalAccel, 4);
    Serial.print(" | eAz= "); Serial.print(eAz, 3);
    Serial.print(" | aCx= "); Serial.print(aCx, 3);
    Serial.print(" | aTx= "); Serial.print(aTx, 3);
    Serial.println();
  }
}
