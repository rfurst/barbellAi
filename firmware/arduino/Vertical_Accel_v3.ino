/*
  Off-Axis IMU Correction with Official Arduino Madgwick Library
  --------------------------------------------------------------
  Changes vs. prior version:
    - Measure actual dt each loop, pass 1/dt to Madgwick's .begin() each time
      so the library can keep its internal time step updated.
    - Use dt to compute alphaEx, alphaEy, alphaEz properly.
    - Output the same CSV with <millis,verticalAccel>.
*/

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <MadgwickAHRS.h>

static const float GRAVITY_SIGN = +1.0f; // assume Earth-frame +Z is up at rest

// The sensor is physically +5.3 cm above the bar axis along the sensor's +Z:
float rSensorX = 0.0f;
float rSensorY = 0.0f;
float rSensorZ = +0.028f; // meters

// BLE
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic verticalAccelChar("19B10001-E8F2-537E-4F6C-D104768A1214",
                                    BLERead | BLENotify,
                                    40);

// Madgwick filter
Madgwick filter;
float oldOmegaEx = 0.0f, oldOmegaEy = 0.0f, oldOmegaEz = 0.0f;
bool firstLoop = true;

// We'll track the previous time to measure dt
unsigned long lastTimeMs = 0;

// rotation utility
void rotateVectorByEuler(float vx, float vy, float vz,
                         float rollDeg, float pitchDeg, float yawDeg,
                         float &ex, float &ey, float &ez)
{
  float rollRad  = rollDeg  * DEG_TO_RAD;
  float pitchRad = pitchDeg * DEG_TO_RAD;
  float yawRad   = yawDeg   * DEG_TO_RAD;

  float cr = cosf(rollRad),  sr = sinf(rollRad);
  float cp = cosf(pitchRad), sp = sinf(pitchRad);
  float cy = cosf(yawRad),   sy = sinf(yawRad);

  // Rx(roll)
  float v1x = vx;
  float v1y = cr*vy - sr*vz;
  float v1z = sr*vy + cr*vz;

  // Ry(pitch)
  float v2x =  cp*v1x + sp*v1z;
  float v2y =  v1y;
  float v2z = -sp*v1x + cp*v1z;

  // Rz(yaw)
  float v3x =  cy*v2x - sy*v2y;
  float v3y =  sy*v2x + cy*v2y;
  float v3z =  v2z;

  ex = v3x;  ey = v3y;  ez = v3z;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!BLE.begin()) {
    Serial.println("Failed to init BLE!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-OffAxis");
  BLE.setAdvertisedService(accelService);
  accelService.addCharacteristic(verticalAccelChar);
  BLE.addService(accelService);
  BLE.advertise();

  if (!IMU.begin()) {
    Serial.println("Failed to init BMI270!");
    while (1);
  }

  // We'll initialize Madgwick at some nominal freq (100), but will keep updating it dynamically
  filter.begin(100.0f);

  lastTimeMs = millis();

  Serial.println("Setup complete.");
}

void loop() {
  BLEDevice central = BLE.central();

  // measure actual dt
  unsigned long nowMs = millis();
  float dt = (nowMs - lastTimeMs) * 0.001f; // in seconds
  if (dt < 0.005f) {
    // skip if not enough time has passed
    return;
  }
  lastTimeMs = nowMs;

  // update Madgwick's internal frequency to match dt (the library doesn't have an updateIMU with dt)
  float currentFreq = 1.0f / dt;
  filter.begin(currentFreq);

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax, ay, az;
    float gxDeg, gyDeg, gzDeg;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gxDeg, gyDeg, gzDeg);

    // feed the filter (which wants deg/s)
    filter.updateIMU(gxDeg, gyDeg, gzDeg, ax, ay, az);

    float rollDeg  = filter.getRoll();
    float pitchDeg = filter.getPitch();
    float yawDeg   = filter.getYaw();

    // rotate accel to Earth frame
    float eAx, eAy, eAz;
    rotateVectorByEuler(ax, ay, az, rollDeg, pitchDeg, yawDeg, eAx, eAy, eAz);

    // remove gravity
    float aNoGravX = eAx;
    float aNoGravY = eAy;
    float aNoGravZ = eAz - GRAVITY_SIGN;

    // rotate gyro to Earth, but first convert deg/s -> rad/s for cross products
    float gxRad = gxDeg * DEG_TO_RAD;
    float gyRad = gyDeg * DEG_TO_RAD;
    float gzRad = gzDeg * DEG_TO_RAD;

    float omegaEx, omegaEy, omegaEz;
    rotateVectorByEuler(gxRad, gyRad, gzRad, rollDeg, pitchDeg, yawDeg,
                        omegaEx, omegaEy, omegaEz);

    // alpha in Earth frame
    float alphaEx = 0.0f, alphaEy = 0.0f, alphaEz = 0.0f;
    if (!firstLoop) {
      alphaEx = (omegaEx - oldOmegaEx) / dt;
      alphaEy = (omegaEy - oldOmegaEy) / dt;
      alphaEz = (omegaEz - oldOmegaEz) / dt;
    } else {
      firstLoop = false;
    }
    oldOmegaEx = omegaEx;
    oldOmegaEy = omegaEy;
    oldOmegaEz = omegaEz;

    // offset vector
    float rEx, rEy, rEz;
    rotateVectorByEuler(rSensorX, rSensorY, rSensorZ,
                        rollDeg, pitchDeg, yawDeg, rEx, rEy, rEz);

    // centripetal aC = -(omega x (omega x r))
    float tempX = omegaEy*rEz - omegaEz*rEy;
    float tempY = omegaEz*rEx - omegaEx*rEz;
    float tempZ = omegaEx*rEy - omegaEy*rEx;
    float aCx = -(omegaEy*tempZ - omegaEz*tempY);
    float aCy = -(omegaEz*tempX - omegaEx*tempZ);
    float aCz = -(omegaEx*tempY - omegaEy*tempX);

    // tangential aT = alpha x r
    float aTx = alphaEy*rEz - alphaEz*rEy;
    float aTy = alphaEz*rEx - alphaEx*rEz;
    float aTz = alphaEx*rEy - alphaEy*rEx;

    float aCorrectedX = aNoGravX - (aCx + aTx);
    float aCorrectedY = aNoGravY - (aCy + aTy);
    float aCorrectedZ = aNoGravZ - (aCz + aTz);

    float verticalAccel = aCorrectedZ; // still in g

    // send CSV: "<millis>,<verticalAccel>"
    char buffer[40];
    snprintf(buffer, sizeof(buffer), "%lu,%.5f", nowMs, verticalAccel);
    verticalAccelChar.writeValue((uint8_t*)buffer, strlen(buffer));
    // for debug:
    Serial.println(buffer);
  }
}
