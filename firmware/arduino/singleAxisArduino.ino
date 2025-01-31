/* 
   Single-Axis Barbell IMU with BLE
   --------------------------------
   This code:
   1) Uses a single-axis complementary filter to track spin about the bar's X axis.
   2) Rotates local (y,z) into "vertical" to remove gravity.
   3) Subtracts centripetal acceleration from bar spin if sensor is offset from bar center.
   4) Sends final vertical accel (in g) via BLE.

   Requirements:
     - Arduino_BMI270_BMM150 (or similar) for IMU
     - ArduinoBLE for BLE
     - (No Madgwick needed here)
*/

#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // for accelerometer/gyro

// ------------------ USER SETTINGS -------------------

// BLE: same as your old code
BLEService accelService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic verticalAccelChar("19B10001-E8F2-537E-4F6C-D104768A1214",
                                    BLERead | BLENotify,
                                    20);

// Suppose ~100 Hz loop
static const float LOOP_HZ = 100.0f;
static const float DT_MS   = 1000.0f / LOOP_HZ;

// Single-axis complementary filter parameter
// small alpha => trust gyro more, big alpha => trust accel more
static const float COMPLEMENTARY_ALPHA = 0.02f;

// If sensor is offset 5.3 cm along local Z from bar axis, set:
static const float rSensorX = 0.0f;
static const float rSensorY = 0.0f;
static const float rSensorZ = 0.053f; // e.g. +5.3 cm in local Z

// We'll define bar's X as the spin axis, so the sensor can revolve around that axis
// local Y and Z might see centripetal acceleration from the spin

// Gravity in g:
static const float G_MAG = 1.0f; // we measure in g, so 1.0f

// We'll keep track of angle about X in degrees:
static float angleX_deg = 0.0f;

// We'll store the previous time to measure dt accurately
static unsigned long prevMillis = 0;

// ---------- HELPER FUNCTIONS ---------------

// Single-axis complementary filter update:
//   - input: gyro_x_deg_s,  accel y,z in g
//   - dt_s: time step in s
//   - modifies angleX_deg_inout
void updateAngleXComplementary(float gyro_x_deg_s,
                               float ay_g,
                               float az_g,
                               float dt_s,
                               float &angleX_deg_inout)
{
  // 1) Gyro integration
  float angle_gyro = angleX_deg_inout + (gyro_x_deg_s * dt_s);

  // 2) Acc-based angle about X:
  //    If bar is horizontal, gravity is mostly in (y,z).
  //    angleX_acc = -atan2( a_y, a_z ) in radians => convert to deg
  //    sign depends on your axis definitions.
  float angle_acc_rad = -atan2(ay_g, az_g); 
  float angle_acc_deg = angle_acc_rad * (180.0f / PI);

  // 3) Combine with complementary filter
  float fused = (1.0f - COMPLEMENTARY_ALPHA)*angle_gyro
                + (COMPLEMENTARY_ALPHA)*angle_acc_deg;

  angleX_deg_inout = fused;
}

// Rotate (localY, localZ) by -angleX to find "vertical" acceleration
// effectively: 
//    verticalAccel = cos(angleX)*localZ + sin(angleX)*localY
float rotateYzToVertical(float localY, float localZ, float angleX_deg)
{
  float ax_rad = angleX_deg * (PI / 180.0f);
  float c = cosf(ax_rad);
  float s = sinf(ax_rad);
  // If angleX=0 => bar is unspun => gravity ~ -Z
  // We'll define vertical = c*Z + s*Y
  float vert = c*localZ + s*localY;
  return vert;
}

void setup()
{
  Serial.begin(115200);
  while (!Serial) { }

  // BLE
  if (!BLE.begin()) {
    Serial.println("Failed to init BLE!");
    while (1);
  }
  BLE.setLocalName("Nano33BLE-SingleAxis");
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

  prevMillis = millis();
  Serial.println("Single-axis barbell IMU setup complete.");
}

void loop()
{
  // Keep BLE alive
  BLE.poll();

  // Attempt ~100 Hz
  unsigned long now = millis();
  if (now - prevMillis < 10) {
    return;
  }
  float dt_s = (now - prevMillis)*1e-3;
  prevMillis = now;

  // IMU read
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax_g, ay_g, az_g;  // in g
    float gx_deg_s, gy_deg_s, gz_deg_s; // in deg/s
    IMU.readAcceleration(ax_g, ay_g, az_g);
    IMU.readGyroscope(gx_deg_s, gy_deg_s, gz_deg_s);

    // 1) Update single-axis angle about X
    //    We only use the gyro about X (gx_deg_s)
    updateAngleXComplementary(gx_deg_s, ay_g, az_g, dt_s, angleX_deg);

    // 2) Rotate local (y,z) to find net vertical accel (before gravity subtraction)
    float verticalAccel_g = rotateYzToVertical(ay_g, az_g, angleX_deg);

    // 3) Subtract gravity
    verticalAccel_g -= G_MAG; // if bar is stationary & horizontal, this yields 0.

    // 4) Remove centripetal if sensor is off-axis and bar spins about X
    //    centripetal =  (omega_x^2)*r, direction normal to X
    //    We only want the portion that projects into "verticalAccel".
    //    If r=(0,0,rSensorZ) and bar spins about X => the radial direction is local +/- Y
    //    Then rotate that into vertical as well. Or for simplicity, if r is purely Z,
    //    the centripetal is mostly local +/- Y => let's do a minimal approach:

    float wX_rad_s = gx_deg_s * (PI / 180.0f); // bar spin in rad/s
    float centrip_g = (wX_rad_s * wX_rad_s * rSensorZ) / 9.81f; 
    // This is total centrip in "some direction" if offset is purely in +Z.
    // That direction is local +/-Y, so let's see how that shows up in "vertical."

    // We'll do a partial transform: local Y => vertical = s*( localY ), so if
    // centripetal is in +Y, that rotates by angleX into vertical as well.
    // => "vertical portion of centrip" = s*(centrip_g).
    // sign depends on direction. We'll assume r is +Z, so spin => centrip in -Y or +Y?
    // For demonstration, let's assume it's in negative Y if spin is positive. 
    // We'll do a magnitude & let the rotation handle sign:
    // localY_centrip = ± centrip_g. Then vertical part = rotateYzToVertical(localY_centrip,0).
    
    float localY_centrip = centrip_g; 
    // (You could detect spin sign or verify if it's negative Y, etc.)

    float verticalCentrip_g = rotateYzToVertical(localY_centrip, 0.0f, angleX_deg);
    // subtract it
    verticalAccel_g -= verticalCentrip_g;

    // 5) Done. Now verticalAccel_g is hopefully free of gravity & spin offsets.

    // 6) Send BLE
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%.4f", verticalAccel_g);
    verticalAccelChar.writeValue((uint8_t*)buffer, strlen(buffer));

    // 7) Debug
    Serial.print("dt="); Serial.print(dt_s,3);
    Serial.print(" angleX="); Serial.print(angleX_deg,2);
    Serial.print(" rawAcc(g)=["); Serial.print(ax_g,2);
    Serial.print(",");            Serial.print(ay_g,2);
    Serial.print(",");            Serial.print(az_g,2);
    Serial.print("] vertAccel="); Serial.println(verticalAccel_g,4);
  }
}